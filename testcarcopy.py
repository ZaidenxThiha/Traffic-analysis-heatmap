import os
import math
import time
import subprocess
import numpy as np
import tempfile
import pathlib
from collections import defaultdict, deque

try:
    import cv2  # type: ignore
except (ModuleNotFoundError, ImportError):
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise ModuleNotFoundError(
            "Missing dependency: `cv2` (OpenCV).\n\n"
            "For Streamlit Cloud, add `opencv-python-headless` to `requirements.txt` and redeploy."
        )

    return cv2


def _require_ultralytics():
    import sys

    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "Ultralytics requires PyTorch, which (currently) does not provide wheels for Python 3.13. "
            "Use Python 3.12 (or 3.11), recreate your virtualenv, then install deps."
        )

    try:
        from ultralytics import YOLO  # type: ignore
        from ultralytics.solutions import Heatmap  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: `ultralytics`.\n\n"
            "Install it in your current environment, e.g.:\n"
            "  python -m pip install ultralytics\n\n"
            "If you are on Python 3.13, create a Python 3.12 venv instead (PyTorch wheels are not available for 3.13)."
        ) from e

    return YOLO, Heatmap


# ===============================
# CONFIGURATION
# ===============================
OUTPUT_VIDEO = "traffic_analysis_heatmap.mp4"

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


# ===============================
# FRAME PROCESSING
# ===============================
def process_frame(
    frame,
    *,
    model,
    yolo_heatmap,
    track_history,
    fps,
    conf=CONF,
    iou=IOU,
    imgsz=INFERENCE_SIZE,
):
    """Run YOLO tracking + Ultralytics Heatmap, returning an annotated BGR image and counters."""
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

    solution_results = yolo_heatmap.process(frame)
    frame_with_heatmap = solution_results.plot_im

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)

        heat_arr = getattr(yolo_heatmap, "heatmap", None)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            track_history[track_id].append((cx, cy))
            points = list(track_history[track_id])

            if len(points) > 1:
                dist = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
                speed_px = dist / len(points)
                speed_mps = (speed_px * fps) / PIXELS_PER_METER
            else:
                speed_mps = 0.0

            heat_value = 0.0
            if heat_arr is not None:
                hh, ww = heat_arr.shape[:2]
                if 0 <= cy < hh and 0 <= cx < ww:
                    pixel = heat_arr[cy, cx]
                    heat_value = float(np.mean(pixel))

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
        f"Mode: Ultralytics Heatmap | {imgsz}px",
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
        YOLO, Heatmap = _require_ultralytics()
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

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Failed to open stream.")
        return

    ret, frame = cap.read()
    if not ret:
        print("❌ No frames received.")
        return

    H, W = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    # Ultralytics native heatmap (current API uses .process())
    yolo_heatmap = Heatmap(
        model=MODEL_PATH,
        classes=VEHICLE_CLASSES,
        colormap=cv2.COLORMAP_TURBO,
    )

    track_history = defaultdict(lambda: deque(maxlen=MOTION_WINDOW))

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
            yolo_heatmap=yolo_heatmap,
            track_history=track_history,
            fps=fps,
            conf=CONF,
            iou=IOU,
            imgsz=INFERENCE_SIZE,
        )

        cv2.imshow("Smart Traffic Heatmap", frame_with_heatmap)
        writer.write(frame_with_heatmap)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
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
        YOLO, Heatmap = _require_ultralytics()
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.set_page_config(page_title="Traffic Heatmap (YOLO + Ultralytics Heatmap)", layout="wide")
    st.title("Traffic analysis heatmap (YOLO tracking + Ultralytics Heatmap)")

    weights = sorted(str(p) for p in pathlib.Path(".").glob("*.pt"))
    default_weight = MODEL_PATH if MODEL_PATH in weights else (weights[0] if weights else MODEL_PATH)

    with st.sidebar:
        st.header("Settings")
        weight_path = st.selectbox("Model", options=weights or [MODEL_PATH], index=(weights.index(default_weight) if default_weight in weights else 0))
        conf = st.slider("Confidence", 0.0, 1.0, float(CONF), 0.01)
        iou = st.slider("IOU", 0.0, 1.0, float(IOU), 0.01)
        imgsz = st.select_slider("Inference size", options=[640, 960, 1280, 1600], value=int(INFERENCE_SIZE))
        frames_per_run = st.select_slider("Frames per refresh", options=[1, 2, 4, 8, 16], value=4)
        save_video = st.checkbox("Save processed video", value=False)
        output_path = st.text_input("Output path", value=OUTPUT_VIDEO, disabled=not save_video)

        st.divider()
        input_mode = st.radio("Source", options=["Webcam (streamlit-webrtc)", "YouTube URL", "Upload video"])

    if hasattr(st, "cache_resource"):
        @st.cache_resource
        def _load_model(path):
            return YOLO(path)

        @st.cache_resource
        def _load_heatmap(path):
            return Heatmap(model=path, classes=VEHICLE_CLASSES, colormap=cv2.COLORMAP_TURBO)
    else:
        def _load_model(path):
            return YOLO(path)

        def _load_heatmap(path):
            return Heatmap(model=path, classes=VEHICLE_CLASSES, colormap=cv2.COLORMAP_TURBO)

    model = _load_model(weight_path)
    yolo_heatmap = _load_heatmap(weight_path)

    if "track_history" not in st.session_state:
        st.session_state.track_history = defaultdict(lambda: deque(maxlen=MOTION_WINDOW))

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
                self.yolo_heatmap = yolo_heatmap
                self.track_history = defaultdict(lambda: deque(maxlen=MOTION_WINDOW))
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
                    yolo_heatmap=self.yolo_heatmap,
                    track_history=self.track_history,
                    fps=self.fps,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
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
    if "writer" not in st.session_state:
        st.session_state.writer = None
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
        if st.session_state.writer is not None:
            try:
                st.session_state.writer.release()
            except Exception:
                pass
        st.session_state.track_history = defaultdict(lambda: deque(maxlen=MOTION_WINDOW))
        st.session_state.cap = cv2.VideoCapture(source)
        if not st.session_state.cap.isOpened():
            st.session_state.cap = None
            st.error("Failed to open video source.")
            st.stop()

        fps = st.session_state.cap.get(cv2.CAP_PROP_FPS) or 30.0
        st.session_state.fps = float(fps)

        if save_video:
            ret, first = st.session_state.cap.read()
            if not ret:
                st.error("Could not read the first frame.")
                st.session_state.cap.release()
                st.session_state.cap = None
                st.stop()
            h, w = first.shape[:2]
            st.session_state.writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), st.session_state.fps, (w, h))
            processed, stopped, moving = process_frame(
                first,
                model=model,
                yolo_heatmap=yolo_heatmap,
                track_history=st.session_state.track_history,
                fps=st.session_state.fps,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
            )
            st.session_state.writer.write(processed)
            frame_slot.image(processed, channels="BGR", use_container_width=True)
            stopped_slot.metric("Stopped", stopped)
            moving_slot.metric("Moving", moving)

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
                yolo_heatmap=yolo_heatmap,
                track_history=st.session_state.track_history,
                fps=st.session_state.fps,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
            )
            frame_slot.image(processed, channels="BGR", use_container_width=True)
            stopped_slot.metric("Stopped", stopped)
            moving_slot.metric("Moving", moving)
            if st.session_state.writer is not None:
                st.session_state.writer.write(processed)

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
        if st.session_state.writer is not None:
            try:
                st.session_state.writer.release()
            except Exception:
                pass
            st.session_state.writer = None
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
