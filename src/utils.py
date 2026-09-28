"""
Marine Plastic Detection - Utility Functions

Image I/O, validation, result serialization, metric formatting,
and general helper functions used across the project.
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.config import (
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    MAX_IMAGE_DIMENSION,
    DETECTIONS_DIR,
    METRICS_DIR,
    COMPARISONS_DIR,
)


# ──────────────────────────────────────────────
# Image I/O & Validation
# ──────────────────────────────────────────────

class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation."""
    pass


def validate_image_file(file_path: str | Path) -> Path:
    """
    Validate that *file_path* points to a supported, non-corrupt image.

    Returns the resolved Path on success; raises ImageValidationError otherwise.
    """
    path = Path(file_path)

    if not path.exists():
        raise ImageValidationError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ImageValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB."
        )

    # Try loading to catch corrupt files
    try:
        img = cv2.imread(str(path))
        if img is None:
            raise ImageValidationError("Image could not be decoded (possibly corrupt).")
    except Exception as exc:
        raise ImageValidationError(f"Error reading image: {exc}") from exc

    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({w}×{h}) exceed maximum ({MAX_IMAGE_DIMENSION} px)."
        )

    return path.resolve()


def validate_uploaded_bytes(file_bytes: bytes, file_name: str) -> np.ndarray:
    """
    Validate raw bytes from a Streamlit file uploader.

    Returns the decoded image as a BGR numpy array.
    """
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ImageValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB."
        )

    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ImageValidationError("Image could not be decoded (possibly corrupt).")

    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({w}×{h}) exceed maximum ({MAX_IMAGE_DIMENSION} px)."
        )

    return img


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def resize_for_display(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Resize an image so its longest side is at most *max_side* pixels."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ──────────────────────────────────────────────
# Detection-Result Drawing
# ──────────────────────────────────────────────

# Ocean-themed palette (BGR)
BBOX_COLORS = [
    (230, 159, 14),   # cerulean-ish
    (60, 180, 75),    # green
    (0, 130, 255),    # orange
    (200, 80, 200),   # magenta
    (0, 210, 210),    # yellow-cyan
]


def draw_detections(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of *img*.

    Each detection dict must contain:
        class_name, confidence, bbox (x_min, y_min, x_max, y_max).
    """
    canvas = img.copy()
    for i, det in enumerate(detections):
        color = BBOX_COLORS[i % len(BBOX_COLORS)]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det["confidence"]
        label = f"{det['class_name']} {conf:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            canvas, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return canvas


# ──────────────────────────────────────────────
# Result Serialization
# ──────────────────────────────────────────────

def _make_id(prefix: str = "det") -> str:
    """Generate a short unique id like det_a3f1b2."""
    ts = str(time.time()).encode()
    return f"{prefix}_{hashlib.md5(ts).hexdigest()[:8]}"


def save_detection_result(
    image_name: str,
    detections: List[Dict[str, Any]],
    model_info: Dict[str, Any],
    enhancement: Optional[str] = None,
    inference_time_ms: float = 0.0,
) -> Path:
    """
    Persist a detection result as a JSON file in the detections directory.

    Returns the path to the saved file.
    """
    result = {
        "result_id": _make_id("det"),
        "timestamp": datetime.now().isoformat(),
        "image_name": image_name,
        "model": model_info,
        "enhancement": enhancement,
        "inference_time_ms": round(inference_time_ms, 2),
        "num_detections": len(detections),
        "detections": detections,
    }
    out_path = DETECTIONS_DIR / f"{result['result_id']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return out_path


def save_comparison_result(
    original_result: Dict[str, Any],
    enhanced_result: Dict[str, Any],
    enhancement_method: str,
) -> Path:
    """Save a side-by-side comparison record."""
    comparison = {
        "comparison_id": _make_id("cmp"),
        "timestamp": datetime.now().isoformat(),
        "enhancement_method": enhancement_method,
        "original": original_result,
        "enhanced": enhanced_result,
    }
    out_path = COMPARISONS_DIR / f"{comparison['comparison_id']}.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    return out_path


def save_metrics(
    metrics: Dict[str, float],
    condition: str,
    model_info: Dict[str, Any],
) -> Path:
    """Save evaluation metrics for a specific condition."""
    record = {
        "evaluation_id": _make_id("eval"),
        "timestamp": datetime.now().isoformat(),
        "condition": condition,
        "model": model_info,
        "metrics": metrics,
    }
    out_path = METRICS_DIR / f"{record['evaluation_id']}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return out_path


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    return json.loads(Path(path).read_text())


# ──────────────────────────────────────────────
# Metric Helpers
# ──────────────────────────────────────────────

def compute_f1(precision: float, recall: float) -> float:
    """Compute F1-score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics dict as a human-readable string."""
    lines = []
    for k, v in metrics.items():
        lines.append(f"  {k:<16s}: {v:.4f}")
    return "\n".join(lines)
