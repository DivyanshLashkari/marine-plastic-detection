"""
Marine Plastic Detection - Unit Tests

Tests for the core modules: utils, enhancement, detector, and config.
Run with:   pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_image():
    """Create a synthetic 200×300 BGR test image."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    # Add some color variation
    img[50:150, 80:220] = [180, 120, 60]   # rectangle (ocean-ish)
    img[70:130, 100:200] = [40, 40, 200]   # debris-like blob
    return img


@pytest.fixture
def sample_detections():
    """Sample detection results."""
    return [
        {
            "class_id": 0,
            "class_name": "marine_debris",
            "confidence": 0.92,
            "bbox": [100.0, 70.0, 200.0, 130.0],
        },
        {
            "class_id": 0,
            "class_name": "marine_debris",
            "confidence": 0.75,
            "bbox": [50.0, 30.0, 90.0, 60.0],
        },
    ]


@pytest.fixture
def tmp_image_file(tmp_path, sample_image):
    """Write a sample image to a temp file and return its path."""
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), sample_image)
    return img_path


# ──────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────

class TestConfig:
    def test_directories_exist(self):
        from src.config import DETECTIONS_DIR, COMPARISONS_DIR, METRICS_DIR
        assert DETECTIONS_DIR.exists()
        assert COMPARISONS_DIR.exists()
        assert METRICS_DIR.exists()

    def test_class_names_defined(self):
        from src.config import CLASS_NAMES, NUM_CLASSES
        assert len(CLASS_NAMES) == NUM_CLASSES
        assert 0 in CLASS_NAMES
        assert CLASS_NAMES[0] == "plastic"

    def test_challenge_conditions_defined(self):
        from src.config import CHALLENGE_CONDITIONS
        assert "normal" in CHALLENGE_CONDITIONS
        assert "low_light" in CHALLENGE_CONDITIONS
        assert "blur" in CHALLENGE_CONDITIONS
        assert "haze" in CHALLENGE_CONDITIONS


# ──────────────────────────────────────────────
# Utils Tests
# ──────────────────────────────────────────────

class TestValidation:
    def test_validate_valid_file(self, tmp_image_file):
        from src.utils import validate_image_file
        result = validate_image_file(tmp_image_file)
        assert result.exists()

    def test_validate_missing_file(self):
        from src.utils import validate_image_file, ImageValidationError
        with pytest.raises(ImageValidationError, match="not found"):
            validate_image_file("/nonexistent/image.jpg")

    def test_validate_unsupported_extension(self, tmp_path):
        from src.utils import validate_image_file, ImageValidationError
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not an image")
        with pytest.raises(ImageValidationError, match="Unsupported"):
            validate_image_file(bad_file)

    def test_validate_uploaded_bytes(self, sample_image):
        from src.utils import validate_uploaded_bytes
        _, buf = cv2.imencode(".jpg", sample_image)
        result = validate_uploaded_bytes(buf.tobytes(), "test.jpg")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3

    def test_validate_uploaded_bytes_bad_ext(self):
        from src.utils import validate_uploaded_bytes, ImageValidationError
        with pytest.raises(ImageValidationError, match="Unsupported"):
            validate_uploaded_bytes(b"data", "test.xyz")


class TestImageHelpers:
    def test_bgr_to_rgb(self, sample_image):
        from src.utils import bgr_to_rgb
        rgb = bgr_to_rgb(sample_image)
        assert rgb.shape == sample_image.shape
        # Blue and red channels should be swapped
        np.testing.assert_array_equal(rgb[:, :, 0], sample_image[:, :, 2])

    def test_resize_for_display_small(self, sample_image):
        from src.utils import resize_for_display
        # Image is 300×200, max_side=800 → no change
        result = resize_for_display(sample_image, max_side=800)
        assert result.shape == sample_image.shape

    def test_resize_for_display_large(self):
        from src.utils import resize_for_display
        big = np.zeros((2000, 3000, 3), dtype=np.uint8)
        result = resize_for_display(big, max_side=800)
        assert max(result.shape[:2]) <= 800


class TestDrawDetections:
    def test_draw_returns_copy(self, sample_image, sample_detections):
        from src.utils import draw_detections
        result = draw_detections(sample_image, sample_detections)
        # Should be a different array (copy)
        assert result is not sample_image
        assert result.shape == sample_image.shape

    def test_draw_empty_detections(self, sample_image):
        from src.utils import draw_detections
        result = draw_detections(sample_image, [])
        np.testing.assert_array_equal(result, sample_image)


class TestSerialization:
    def test_save_detection_result(self, sample_detections, tmp_path, monkeypatch):
        from src import config
        monkeypatch.setattr(config, "DETECTIONS_DIR", tmp_path)
        # Re-import to pick up patched path
        from src.utils import save_detection_result
        # Monkey-patch the module-level import too
        import src.utils as utils_mod
        original_dir = utils_mod.DETECTIONS_DIR
        utils_mod.DETECTIONS_DIR = tmp_path

        path = save_detection_result(
            "test.jpg",
            sample_detections,
            {"model_name": "test"},
            inference_time_ms=42.5,
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["num_detections"] == 2
        assert data["inference_time_ms"] == 42.5

        utils_mod.DETECTIONS_DIR = original_dir

    def test_compute_f1(self):
        from src.utils import compute_f1
        assert compute_f1(1.0, 1.0) == 1.0
        assert compute_f1(0.0, 0.0) == 0.0
        assert abs(compute_f1(0.8, 0.6) - 0.6857) < 0.01


# ──────────────────────────────────────────────
# Enhancement Tests
# ──────────────────────────────────────────────

class TestEnhancement:
    def test_clahe(self, sample_image):
        from src.enhancement import apply_clahe
        result = apply_clahe(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_gamma_correction(self, sample_image):
        from src.enhancement import apply_gamma_correction
        result = apply_gamma_correction(sample_image, gamma=1.5)
        assert result.shape == sample_image.shape

    def test_denoise(self, sample_image):
        from src.enhancement import apply_denoise
        result = apply_denoise(sample_image)
        assert result.shape == sample_image.shape

    def test_contrast(self, sample_image):
        from src.enhancement import apply_contrast
        result = apply_contrast(sample_image)
        assert result.shape == sample_image.shape

    def test_enhance_dispatcher(self, sample_image):
        from src.enhancement import enhance_image
        for method in ["clahe", "gamma", "denoise", "contrast"]:
            result = enhance_image(sample_image, method)
            assert result.shape == sample_image.shape

    def test_enhance_invalid_method(self, sample_image):
        from src.enhancement import enhance_image
        with pytest.raises(ValueError, match="Unknown"):
            enhance_image(sample_image, "invalid_method")

    def test_available_methods(self):
        from src.enhancement import get_available_methods
        methods = get_available_methods()
        assert "clahe" in methods
        assert "gamma" in methods
        assert len(methods) >= 4


# ──────────────────────────────────────────────
# Detector Tests (unit-level, no model loading)
# ──────────────────────────────────────────────

class TestDetector:
    def test_detector_init(self):
        from src.detector import MarineDebrisDetector
        det = MarineDebrisDetector()
        assert not det.is_loaded
        assert det.confidence > 0

    def test_detector_not_loaded_error(self, sample_image):
        from src.detector import MarineDebrisDetector, DetectorError
        det = MarineDebrisDetector()
        with pytest.raises(DetectorError, match="not loaded"):
            det.detect(sample_image)

    def test_model_info_before_load(self):
        from src.detector import MarineDebrisDetector
        det = MarineDebrisDetector()
        info = det.get_model_info()
        assert info["loaded"] is False
        assert info["model_name"] == "yolov8n"


# ──────────────────────────────────────────────
# Challenge Set Tests
# ──────────────────────────────────────────────

class TestChallengeGeneration:
    def test_simulate_low_light(self, sample_image):
        from scripts.create_challenge_sets import simulate_low_light
        result = simulate_low_light(sample_image, gamma=0.4)
        assert result.shape == sample_image.shape
        # Low-light image should be darker overall
        assert result.mean() <= sample_image.mean() + 1  # allow rounding

    def test_simulate_blur(self, sample_image):
        from scripts.create_challenge_sets import simulate_blur
        result = simulate_blur(sample_image, kernel_size=15, sigma=5.0)
        assert result.shape == sample_image.shape

    def test_simulate_haze(self, sample_image):
        from scripts.create_challenge_sets import simulate_haze
        result = simulate_haze(sample_image, intensity=0.5)
        assert result.shape == sample_image.shape
        # Haze should make image brighter
        assert result.mean() >= sample_image.mean()

    def test_simulate_noise(self, sample_image):
        from scripts.create_challenge_sets import simulate_noise
        result = simulate_noise(sample_image, mean=0, sigma=35)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
