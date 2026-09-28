"""
Marine Plastic Detection - Configuration Constants

Centralizes all project-wide settings: paths, model parameters,
enhancement defaults, and evaluation thresholds.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DETECTIONS_DIR = OUTPUTS_DIR / "detections"
COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"
METRICS_DIR = OUTPUTS_DIR / "metrics"
CHALLENGE_DIR = OUTPUTS_DIR / "challenge_sets"

# Ensure output directories exist
for _d in [DETECTIONS_DIR, COMPARISONS_DIR, METRICS_DIR, CHALLENGE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "yolov8n"  # Lightweight YOLOv8 nano
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"
PRETRAINED_WEIGHTS = "yolov8n.pt"  # Ultralytics pretrained checkpoint

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640

# Class definition (TrashCan ICRA19 marine debris classes)
CLASS_NAMES = {
    0: "plastic",
    1: "bio",
    2: "rov",
}
NUM_CLASSES = 3


# ──────────────────────────────────────────────
# Image Validation
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_IMAGE_DIMENSION = 8192  # pixels

# ──────────────────────────────────────────────
# Enhancement Defaults
# ──────────────────────────────────────────────
ENHANCEMENT_METHODS = ["clahe", "gamma", "denoise", "contrast"]

CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

GAMMA_VALUE = 1.5  # >1 brightens, <1 darkens

DENOISE_H = 10
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

CONTRAST_ALPHA = 1.3  # contrast multiplier
CONTRAST_BETA = 10    # brightness offset

# ──────────────────────────────────────────────
# Challenge-Set Generation Parameters
# ──────────────────────────────────────────────
CHALLENGE_CONDITIONS = {
    "normal": {},
    "low_light": {"gamma": 0.4},
    "blur": {"kernel_size": 15, "sigma": 5.0},
    "haze": {"intensity": 0.55},
    "noise": {"mean": 0, "sigma": 35},
}

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
METRIC_NAMES = ["precision", "recall", "f1_score", "mAP50", "mAP50_95"]

# ──────────────────────────────────────────────
# Training Defaults
# ──────────────────────────────────────────────
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
TRAIN_IMAGE_SIZE = 640
TRAIN_PATIENCE = 20  # early-stopping patience
