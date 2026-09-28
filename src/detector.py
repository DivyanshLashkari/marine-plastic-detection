"""
Marine Plastic Detection - YOLOv8n Detector Module

Wraps Ultralytics YOLOv8 for single-image inference.
Provides model loading with caching, detection, and
result formatting as standardized dicts.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    DEFAULT_WEIGHTS,
    PRETRAINED_WEIGHTS,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    CLASS_NAMES,
    MODEL_NAME,
)


class DetectorError(Exception):
    """Raised when the detector encounters an unrecoverable error."""
    pass


class MarineDebrisDetector:
    """
    Lightweight YOLOv8n-based marine debris / plastic detector.

    Usage
    -----
    >>> detector = MarineDebrisDetector("weights/best.pt")
    >>> results = detector.detect(image_bgr)
    >>> for det in results["detections"]:
    ...     print(det["class_name"], det["confidence"], det["bbox"])
    """

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        confidence: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        img_size: int = IMAGE_SIZE,
    ):
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        self.confidence = confidence
        self.iou = iou
        self.img_size = img_size
        self._model = None

    # ── Model lifecycle ───────────────────────

    def load_model(self) -> None:
        """
        Load YOLO model weights.

        If the custom weights file does not exist, falls back to the
        Ultralytics pretrained checkpoint so the app can still start
        (useful during development before training completes).
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "Ultralytics package not installed. "
                "Run: pip install ultralytics"
            ) from exc

        if self.weights_path.exists():
            self._model = YOLO(str(self.weights_path))
            self._weights_source = str(self.weights_path)
        else:
            # Fallback: pretrained COCO checkpoint (won't know marine_debris
            # but keeps the pipeline testable before fine-tuning)
            self._model = YOLO(PRETRAINED_WEIGHTS)
            self._weights_source = PRETRAINED_WEIGHTS

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the currently loaded model."""
        return {
            "model_name": MODEL_NAME,
            "weights": self._weights_source if self.is_loaded else None,
            "confidence_threshold": self.confidence,
            "iou_threshold": self.iou,
            "image_size": self.img_size,
            "loaded": self.is_loaded,
        }

    # ── Inference ─────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run detection on a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format.
        confidence : float, optional
            Override confidence threshold for this call.
        iou : float, optional
            Override IoU threshold for this call.

        Returns
        -------
        dict
            {
              "detections": [ {class_name, class_id, confidence, bbox}, ... ],
              "num_detections": int,
              "inference_time_ms": float,
              "image_shape": (H, W),
            }
        """
        if not self.is_loaded:
            raise DetectorError(
                "Model not loaded. Call detector.load_model() first."
            )

        conf = confidence if confidence is not None else self.confidence
        iou_thr = iou if iou is not None else self.iou

        start = time.perf_counter()
        try:
            results = self._model.predict(
                source=image,
                conf=conf,
                iou=iou_thr,
                imgsz=self.img_size,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"Detection failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000

        detections = self._parse_results(results)

        h, w = image.shape[:2]
        return {
            "detections": detections,
            "num_detections": len(detections),
            "inference_time_ms": round(elapsed_ms, 2),
            "image_shape": (h, w),
        }

    def _parse_results(self, results) -> List[Dict[str, Any]]:
        """Convert Ultralytics results to a list of standardized dicts."""
        detections: List[Dict[str, Any]] = []

        model_names = getattr(self._model, "names", {})

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Dynamic model class name lookup with fallback
                if isinstance(model_names, dict) and cls_id in model_names:
                    class_name = model_names[cls_id]
                elif isinstance(model_names, (list, tuple)) and 0 <= cls_id < len(model_names):
                    class_name = model_names[cls_id]
                else:
                    class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 1), round(y1, 1),
                             round(x2, 1), round(y2, 1)],
                })

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections


# ──────────────────────────────────────────────
# Module-level convenience
# ──────────────────────────────────────────────

_cached_detector: Optional[MarineDebrisDetector] = None


def get_detector(
    weights_path: Optional[str | Path] = None,
    **kwargs,
) -> MarineDebrisDetector:
    """
    Return a cached detector instance.

    Avoids reloading weights on every Streamlit rerun.
    """
    global _cached_detector

    wp = Path(weights_path) if weights_path else DEFAULT_WEIGHTS

    if _cached_detector is not None and _cached_detector.weights_path == wp:
        return _cached_detector

    detector = MarineDebrisDetector(weights_path=wp, **kwargs)
    detector.load_model()
    _cached_detector = detector
    return detector
