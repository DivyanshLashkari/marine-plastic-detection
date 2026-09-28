"""
Marine Plastic Detection - Challenge Set Generator

Creates controlled degraded versions of test images to evaluate
detector robustness under diverse environmental conditions.

    python scripts/create_challenge_sets.py --source datasets/marine_debris_yolo/images/test

Produces sub-directories under outputs/challenge_sets/:
    normal/      — unmodified copies
    low_light/   — simulated low-light (gamma darkening)
    blur/        — Gaussian blur
    haze/        — simulated haze overlay
    noise/       — additive Gaussian noise
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHALLENGE_CONDITIONS, CHALLENGE_DIR


# ──────────────────────────────────────────────
# Degradation Functions
# ──────────────────────────────────────────────

def simulate_low_light(img: np.ndarray, gamma: float = 0.4) -> np.ndarray:
    """Darken the image using inverse gamma correction."""
    inv = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)


def simulate_blur(
    img: np.ndarray, kernel_size: int = 15, sigma: float = 5.0
) -> np.ndarray:
    """Apply Gaussian blur."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def simulate_haze(img: np.ndarray, intensity: float = 0.55) -> np.ndarray:
    """
    Overlay a semi-transparent white layer to simulate haze.

    intensity ∈ [0, 1]:  0 = no haze,  1 = fully white.
    """
    white = np.ones_like(img, dtype=np.uint8) * 255
    return cv2.addWeighted(img, 1 - intensity, white, intensity, 0)


def simulate_noise(
    img: np.ndarray, mean: float = 0, sigma: float = 35
) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


_TRANSFORM = {
    "low_light": simulate_low_light,
    "blur": simulate_blur,
    "haze": simulate_haze,
    "noise": simulate_noise,
}


# ──────────────────────────────────────────────
# Main Generator
# ──────────────────────────────────────────────

def create_challenge_sets(source_dir: Path, output_root: Path):
    """
    Iterate over images in *source_dir* and create one sub-folder
    per challenge condition under *output_root*.
    """
    images = sorted(
        p for p in source_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    if not images:
        print(f"[ERROR] No images found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images in {source_dir}\n")

    for condition, params in CHALLENGE_CONDITIONS.items():
        cond_dir = output_root / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

        print(f"  -> {condition:<14s}  params={params}")

        for img_path in images:
            if condition == "normal":
                # Just copy the original
                shutil.copy2(img_path, cond_dir / img_path.name)
            else:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"    [WARN] Skipped (unreadable): {img_path.name}")
                    continue
                transform_fn = _TRANSFORM[condition]
                degraded = transform_fn(img, **params)
                cv2.imwrite(str(cond_dir / img_path.name), degraded)

    print(f"\n[OK] Challenge sets written to {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate challenge-condition image sets."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directory containing clean test images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CHALLENGE_DIR),
        help="Root directory for challenge sets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)

    if not source.is_dir():
        print(f"[ERROR] Source directory not found: {source}")
        sys.exit(1)

    print("=" * 60)
    print("  Marine Debris Detection - Challenge Set Generation")
    print("=" * 60)
    create_challenge_sets(source, output)



if __name__ == "__main__":
    main()
