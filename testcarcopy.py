import os
import math
import time
import subprocess
import numpy as np
import tempfile
import pathlib
import importlib
from collections import defaultdict, deque

try:
    import cv2  # type: ignore
except (ModuleNotFoundError, ImportError):
    cv2 = None


def _attempt_fix_cv2():
    import sys

    if not _running_in_streamlit():
        return

    # Streamlit Cloud runtimes commonly run in a non-writable environment.
    # Do not attempt runtime package surgery unless explicitly allowed.
    if os.environ.get("ALLOW_RUNTIME_PIP_FIX", "").strip() not in {"1", "true", "TRUE", "yes", "YES"}:
        return

    try:
        # `uv pip uninstall` doesn't accept `-y` on some versions; keep this best-effort.
        subprocess.check_call(["uv", "pip", "uninstall", "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"])
        subprocess.check_call(["uv", "pip", "install", "--no-cache-dir", "--force-reinstall", "opencv-python-headless==4.11.0.86"])
        return
    except Exception:
        pass

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "opencv-python-headless==4.11.0.86"])
    except Exception:
        return


def _require_cv2():
    global cv2

    if cv2 is None:
        try:
            cv2 = importlib.import_module("cv2")
        except Exception:
            _attempt_fix_cv2()
            try:
                cv2 = importlib.import_module("cv2")
            except Exception as e:
                raise ModuleNotFoundError(
                    "Missing dependency: `cv2` (OpenCV).\n\n"
                    "On Streamlit Cloud, prefer `opencv-python-headless` (and avoid installing `opencv-python`).\n"
                    "Note: `ultralytics` often pulls in `opencv-python` by default; this repo's `postBuild` forces headless.\n"
                    "Reboot the app to trigger a fresh build.\n\n"
                    f"Original import error: {e}"
                ) from e

    return cv2


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: `ultralytics`.\n\n"
            "Install it in your current environment, e.g.:\n"
            "  python -m pip install ultralytics\n\n"
            "If the import fails due to `torch` missing, install a compatible PyTorch wheel for your Python version."
        ) from e

    return YOLO


# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "yolo11n.pt"
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=muijHPW82vI"

CONF = 0.25
IOU = 0.60
INFERENCE_SIZE = 1280

# Physics
PIXELS_PER_METER = 35.0
MOTION_WINDOW = 10
STOP_SPEED_THRESHOLD = 2.0

# Heatmap
HEATMAP_DECAY = 0.995
HEATMAP_ALPHA = 0.45
HEATMAP_RADIUS = 24
CONGESTION_THRESHOLD = 150

VEHICLE_CLASSES = [1, 2, 3, 5, 7]


# ===============================
# UTILITIES
# ===============================
def extract_youtube_stream(url):
    """Extract direct m3u8 stream using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--js-runtime", "node",
        "-f", "best[protocol*=m3u8]/best",
        "-g", url
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return next((line for line in out.splitlines() if line.startswith("http")), None)
    except Exception as e:
        print(f"Error extracting stream: {e}")
        return None


def _looks_like_hls_stream(url: str) -> bool:
    u = (url or "").lower()
    return u.startswith(("http://", "https://")) and (u.endswith(".m3u8") or "hls_playlist" in u or "/hls_" in u)


def _heat_colormap_bgr01(v: np.ndarray) -> np.ndarray:
    """Map normalized heat [0..1] to BGR colors in [0..1], without OpenCV/matplotlib."""
    v = np.clip(v, 0.0, 1.0)
    r = np.zeros_like(v, dtype=np.float32)
    g = np.zeros_like(v, dtype=np.float32)
    b = np.zeros_like(v, dtype=np.float32)

    m1 = v < 0.25
    m2 = (v >= 0.25) & (v < 0.5)
    m3 = (v >= 0.5) & (v < 0.75)
    m4 = v >= 0.75

    # blue -> cyan
    b[m1] = 1.0
    g[m1] = (v[m1] / 0.25)

    # cyan -> green
    g[m2] = 1.0
    b[m2] = 1.0 - ((v[m2] - 0.25) / 0.25)

    # green -> yellow
    g[m3] = 1.0
    r[m3] = ((v[m3] - 0.5) / 0.25)

    # yellow -> red
    r[m4] = 1.0
    g[m4] = 1.0 - ((v[m4] - 0.75) / 0.25)

    return np.stack([b, g, r], axis=-1)


class HeatmapAccumulator:
    def __init__(self, *, decay: float = HEATMAP_DECAY, radius: int = HEATMAP_RADIUS):
        self.decay = float(decay)
        self.radius = int(radius)
        self.heatmap = None
        self._kernel = None
        self._kernel_radius = None

    def _ensure_heatmap(self, h: int, w: int) -> None:
        if self.heatmap is None or self.heatmap.shape[:2] != (h, w):
            self.heatmap = np.zeros((h, w), dtype=np.float32)

    def _ensure_kernel(self) -> None:
        r = max(1, int(self.radius))
        if self._kernel is not None and self._kernel_radius == r:
            return
        sigma = max(1.0, r / 2.0)
        ax = np.arange(-r, r + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel /= max(1e-6, float(kernel.max()))
        self._kernel = kernel.astype(np.float32)
        self._kernel_radius = r

    def step(self, frame_h: int, frame_w: int) -> None:
        self._ensure_heatmap(frame_h, frame_w)
        self._ensure_kernel()
        self.heatmap *= self.decay

    def add_point(self, x: int, y: int, *, strength: float = 1.0) -> None:
        if self.heatmap is None:
            return
        r = int(self._kernel_radius or self.radius or 1)
        h, w = self.heatmap.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w, x + r + 1)
        y2 = min(h, y + r + 1)

        kx1 = x1 - (x - r)
        ky1 = y1 - (y - r)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)

        self.heatmap[y1:y2, x1:x2] += float(strength) * self._kernel[ky1:ky2, kx1:kx2]

    def value_255(self, x: int, y: int) -> float:
        if self.heatmap is None:
            return 0.0
        h, w = self.heatmap.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        mx = float(np.max(self.heatmap))
        if mx <= 1e-6:
            return 0.0
        return float((self.heatmap[y, x] / mx) * 255.0)

    def overlay_bgr(self, frame_bgr: np.ndarray, *, alpha: float = HEATMAP_ALPHA) -> np.ndarray:
        if self.heatmap is None:
            return frame_bgr
        mx = float(np.max(self.heatmap))
        if mx <= 1e-6:
            return frame_bgr

        v = (self.heatmap / mx).astype(np.float32)
        cmap_bgr = (_heat_colormap_bgr01(v) * 255.0).astype(np.float32)

        base = frame_bgr.astype(np.float32)
        a = float(np.clip(alpha, 0.0, 1.0))
        out = base * (1.0 - a) + cmap_bgr * a
        return np.clip(out, 0.0, 255.0).astype(np.uint8)


# ===============================
# FRAME PROCESSING
# ===============================
def process_frame(
    frame,
    *,
    model,
    heatmap_acc,
    track_history,
    fps,
    conf=CONF,
    iou=IOU,
    imgsz=INFERENCE_SIZE,
    heat_alpha=HEATMAP_ALPHA,
):
    """Run YOLO tracking + a simple accumulated heatmap, returning an annotated BGR image and counters."""
    cv2 = _require_cv2()
    stopped_ids = set()
    moving_ids = set()

    results = model.track(
        frame,
        persist=True,
        conf=conf,
        iou=iou,
        classes=VEHICLE_CLASSES,
        imgsz=imgsz,
        verbose=False,
    )[0]

    h, w = frame.shape[:2]
    heatmap_acc.step(h, w)
    frame_for_overlay = frame.copy()
    draw_items = []

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            track_history[track_id].append((cx, cy))
            points = list(track_history[track_id])
            heatmap_acc.add_point(cx, cy, strength=1.0)

            if len(points) > 1:
                dist = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
                speed_px = dist / len(points)
                speed_mps = (speed_px * fps) / PIXELS_PER_METER
            else:
                speed_mps = 0.0

            heat_value = heatmap_acc.value_255(cx, cy)

            is_stopped = speed_mps < STOP_SPEED_THRESHOLD
            is_congested = heat_value > CONGESTION_THRESHOLD

            if is_stopped:
                stopped_ids.add(track_id)
            else:
                moving_ids.add(track_id)

            if is_stopped:
                if is_congested:
                    label = f"JAM {int(heat_value)}"
                    color = (0, 0, 255)
                else:
                    label = "STOP"
                    color = (0, 0, 255)
            else:
                label = f"{speed_mps:.1f} m/s"
                color = (0, 255, 0)

            draw_items.append((x1, y1, x2, y2, label, color))

    frame_with_heatmap = heatmap_acc.overlay_bgr(frame_for_overlay, alpha=heat_alpha)

    for x1, y1, x2, y2, label, color in draw_items:
        cv2.rectangle(frame_with_heatmap, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_with_heatmap,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    cv2.putText(
        frame_with_heatmap,
        f"Mode: YOLO + Accum Heatmap | {imgsz}px",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame_with_heatmap,
        f"Stopped: {len(stopped_ids)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame_with_heatmap,
        f"Moving: {len(moving_ids)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return frame_with_heatmap, len(stopped_ids), len(moving_ids)


# ===============================
# CLI (OpenCV window)
# ===============================
def run_cli():
    try:
        cv2 = _require_cv2()
        YOLO = _require_ultralytics()
    except Exception as e:
        print(str(e))
        return

    print(f"Loading model {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    url = input("Enter YouTube URL: ").strip()
    source = extract_youtube_stream(url)
    if not source:
        print("❌ Failed to extract stream.")
        return

    if _looks_like_hls_stream(source):
        print("❌ This YouTube stream is HLS (.m3u8). OpenCV VideoCapture often can't open HLS URLs.")
        print("   Use a downloaded/uploaded video file instead.")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Failed to open stream.")
        return

    ret, frame = cap.read()
    if not ret:
        print("❌ No frames received.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    track_history = defaultdict(lambda: deque(maxlen=MOTION_WINDOW))
    heatmap_acc = HeatmapAccumulator(decay=HEATMAP_DECAY, radius=HEATMAP_RADIUS)

    print(f"✅ Processing at {INFERENCE_SIZE}px inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream interrupted. Reconnecting...")
            time.sleep(2)
            cap = cv2.VideoCapture(source)
            continue

        frame_with_heatmap, _, _ = process_frame(
            frame,
            model=model,
            heatmap_acc=heatmap_acc,
            track_history=track_history,
            fps=fps,
            conf=CONF,
            iou=IOU,
            imgsz=INFERENCE_SIZE,
        )

        cv2.imshow("Smart Traffic Heatmap", frame_with_heatmap)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# Streamlit (Ultralytics live inference-style UI)
# ===============================
def _running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_streamlit():
    import streamlit as st

    try:
        cv2 = _require_cv2()
        YOLO = _require_ultralytics()
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.set_page_config(page_title="Traffic Heatmap (YOLO)", layout="wide")
    st.title("Traffic analysis heatmap (YOLO tracking + accumulated heatmap)")

    weights = sorted(str(p) for p in pathlib.Path(".").glob("*.pt"))
    default_weight = MODEL_PATH if MODEL_PATH in weights else (weights[0] if weights else MODEL_PATH)

    with st.sidebar:
        st.header("Settings")
        weight_path = st.selectbox("Model", options=weights or [MODEL_PATH], index=(weights.index(default_weight) if default_weight in weights else 0))
        conf = st.slider("Confidence", 0.0, 1.0, float(CONF), 0.01)
        iou = st.slider("IOU", 0.0, 1.0, float(IOU), 0.01)
        imgsz = st.select_slider("Inference size", options=[640, 960, 1280, 1600], value=int(INFERENCE_SIZE))
        frames_per_run = st.select_slider("Frames per refresh", options=[1, 2, 4, 8, 16], value=4)
        heat_alpha = st.slider("Heat overlay alpha", 0.0, 1.0, float(HEATMAP_ALPHA), 0.01)
        heat_radius = st.select_slider("Heat radius", options=[8, 12, 16, 24, 32, 40], value=int(HEATMAP_RADIUS))
        heat_decay = st.slider("Heat decay", 0.90, 0.999, float(HEATMAP_DECAY), 0.001)

        st.divider()
        input_mode = st.radio("Source", options=["Webcam (streamlit-webrtc)", "YouTube URL", "Upload video"])

    if hasattr(st, "cache_resource"):
        @st.cache_resource
        def _load_model(path):
            return YOLO(path)
    else:
        def _load_model(path):
            return YOLO(path)

    model = _load_model(weight_path)

    if "track_history" not in st.session_state:
        st.session_state.track_history = defaultdict(lambda: deque(maxlen=MOTION_WINDOW))
    if "heatmap_acc" not in st.session_state:
        st.session_state.heatmap_acc = HeatmapAccumulator(decay=heat_decay, radius=heat_radius)
    st.session_state.heatmap_acc.decay = float(heat_decay)
    st.session_state.heatmap_acc.radius = int(heat_radius)

    col1, col2 = st.columns([2, 1])
    frame_slot = col1.empty()
    with col2:
        stopped_slot = st.empty()
        moving_slot = st.empty()
        status_slot = st.empty()
        stopped_slot.metric("Stopped", 0)
        moving_slot.metric("Moving", 0)

    if input_mode == "Webcam (streamlit-webrtc)":
        try:
            from streamlit_webrtc import webrtc_streamer
            import av
        except Exception:
            st.error("Install webcam streaming deps: `pip install streamlit-webrtc av`")
            st.stop()

        class VideoProcessor:
            def __init__(self):
                self.model = model
                self.track_history = defaultdict(lambda: deque(maxlen=MOTION_WINDOW))
                self.heatmap_acc = HeatmapAccumulator(decay=heat_decay, radius=heat_radius)
                self.last_ts = None
                self.fps = 30.0

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                now = time.time()
                if self.last_ts is not None:
                    dt = max(1e-3, now - self.last_ts)
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                self.last_ts = now

                out, stopped, moving = process_frame(
                    img,
                    model=self.model,
                    heatmap_acc=self.heatmap_acc,
                    track_history=self.track_history,
                    fps=self.fps,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    heat_alpha=heat_alpha,
                )
                return av.VideoFrame.from_ndarray(out, format="bgr24")

        status_slot.caption("Webcam mode (live)")
        webrtc_streamer(key="traffic-heatmap", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})
        st.stop()

    # Non-webcam modes: process in small chunks per rerun so the UI stays responsive.
    def _stop():
        st.session_state.running = False

    if "running" not in st.session_state:
        st.session_state.running = False

    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "tmpfile" not in st.session_state:
        st.session_state.tmpfile = None
    if "fps" not in st.session_state:
        st.session_state.fps = 30.0

    start = False
    with st.sidebar:
        if st.session_state.running:
            st.button("Stop", on_click=_stop, type="primary")
        else:
            start = st.button("Start", type="primary")

    source = None
    if input_mode == "YouTube URL":
        url = DEFAULT_YOUTUBE_URL
        st.text_input("YouTube URL", value=url, disabled=True)
        if start:
            source = extract_youtube_stream(url.strip())
            if not source:
                st.error("Failed to extract stream. Ensure `yt-dlp` is installed and the URL is valid.")
                st.stop()
            if _looks_like_hls_stream(source):
                st.error(
                    "This YouTube source is an HLS stream (.m3u8). OpenCV on Streamlit Cloud usually can't open it.\n\n"
                    "Use **Upload video** (recommended) or **Webcam** mode instead."
                )
                st.stop()
    elif input_mode == "Upload video":
        upload = st.file_uploader("Upload .mp4/.mov", type=["mp4", "mov", "mkv", "avi"])
        if start:
            if upload is None:
                st.error("Upload a video file first.")
                st.stop()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{upload.name}")
            tmp.write(upload.getbuffer())
            tmp.flush()
            tmp.close()
            st.session_state.tmpfile = tmp.name
            source = tmp.name

    if start:
        if st.session_state.cap is not None:
            try:
                st.session_state.cap.release()
            except Exception:
                pass
        st.session_state.track_history = defaultdict(lambda: deque(maxlen=MOTION_WINDOW))
        st.session_state.heatmap_acc = HeatmapAccumulator(decay=heat_decay, radius=heat_radius)
        st.session_state.cap = cv2.VideoCapture(source)
        if not st.session_state.cap.isOpened():
            st.session_state.cap = None
            st.error("Failed to open video source.")
            st.stop()

        fps = st.session_state.cap.get(cv2.CAP_PROP_FPS) or 30.0
        st.session_state.fps = float(fps)

        st.session_state.running = True

    if st.session_state.running and st.session_state.cap is not None:
        status_slot.caption("Processing…")
        for _ in range(int(frames_per_run)):
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.session_state.running = False
                status_slot.caption("Done.")
                break

            processed, stopped, moving = process_frame(
                frame,
                model=model,
                heatmap_acc=st.session_state.heatmap_acc,
                track_history=st.session_state.track_history,
                fps=st.session_state.fps,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                heat_alpha=heat_alpha,
            )
            frame_slot.image(processed, channels="BGR", use_container_width=True)
            stopped_slot.metric("Stopped", stopped)
            moving_slot.metric("Moving", moving)

        if st.session_state.running:
            time.sleep(0.001)
            st.rerun()

    if not st.session_state.running:
        if st.session_state.cap is not None:
            try:
                st.session_state.cap.release()
            except Exception:
                pass
            st.session_state.cap = None
        if st.session_state.tmpfile:
            try:
                os.unlink(st.session_state.tmpfile)
            except Exception:
                pass
            st.session_state.tmpfile = None


if __name__ == "__main__":
    if _running_in_streamlit():
        run_streamlit()
    else:
        run_cli()
