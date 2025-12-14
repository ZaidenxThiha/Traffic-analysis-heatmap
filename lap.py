"""
Minimal stub of the optional `lap` dependency.

Ultralytics' tracking utilities import `lap` for linear assignment. This app does not
use Ultralytics trackers (it uses detection + a lightweight tracker), but some
Ultralytics versions may still import `lap` at import-time in tracking modules.

Keeping this stub prevents a hard crash on platforms where `lap` wheels are not
available (e.g. Streamlit Cloud with newer Python versions).
"""

from __future__ import annotations

__all__ = ["lapjv", "__version__"]

__version__ = "0.0.0-stub"


def lapjv(*args, **kwargs):
    raise RuntimeError(
        "Ultralytics tracking requires the optional `lap` package, which is not installed.\n"
        "This app is configured to avoid Ultralytics trackers; if you still hit this error,\n"
        "remove any calls to `YOLO.track()` and use detection-only inference instead."
    )

