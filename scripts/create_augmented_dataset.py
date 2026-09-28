"""
Multi-Condition Environmental Dataset Augmentation Script

Implements Option B:
Expands the training set with physically realistic underwater degradations:
- Low-light simulation (gamma darkening)
- Haze / turbidity simulation (light scattering)
- Optical blur (water scattering / defocus)
- Contrast equalization (CLAHE enhancement)

This forces the YOLOv8 model to learn condition-invariant representations,
making detection robust across murky, turbid, nocturnal, or degraded waters.
"""

import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "datasets" / "merged"
OUT_DIR = PROJECT_ROOT / "datasets" / "augmented"

random.seed(42)
np.random.seed(42)


def simulate_low_light(img: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Darken the image using inverse gamma."""
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def simulate_haze(img: np.ndarray, intensity: float = 0.45) -> np.ndarray:
    """Overlay water backscatter haze."""
    # Slightly bluish/greenish haze for realistic water turbidity
    tint = np.array([210, 200, 170], dtype=np.uint8)  # BGR
    haze_layer = np.full_like(img, tint)
    return cv2.addWeighted(img, 1.0 - intensity, haze_layer, intensity, 0)


def simulate_blur(img: np.ndarray, ksize: int = 9, sigma: float = 3.0) -> np.ndarray:
    """Simulate optical scattering and water turbidity blur."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE enhancement on the L channel of LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


def build_augmented_dataset():
    if not SRC_DIR.exists():
        print(f"[ERROR] Source merged dataset not found: {SRC_DIR}")
        return

    print("=" * 60)
    print("  Creating Multi-Condition Environmentally Augmented Dataset")
    print("=" * 60)

    # Clean existing output
    shutil.rmtree(OUT_DIR, ignore_errors=True)

    # 1. Validation split: copy as-is (clean benchmark)
    val_img_out = OUT_DIR / "images" / "val"
    val_lbl_out = OUT_DIR / "labels" / "val"
    val_img_out.mkdir(parents=True, exist_ok=True)
    val_lbl_out.mkdir(parents=True, exist_ok=True)

    for p in (SRC_DIR / "images" / "val").glob("*.jpg"):
        link_or_copy(p, val_img_out / p.name)
        lbl_p = SRC_DIR / "labels" / "val" / f"{p.stem}.txt"
        if lbl_p.exists():
            link_or_copy(lbl_p, val_lbl_out / lbl_p.name)

    print(f"[VAL] Preserved {len(list(val_img_out.glob('*.jpg')))} clean validation benchmark images.")

    # 2. Train split: link original + generate condition variations
    tr_img_out = OUT_DIR / "images" / "train"
    tr_lbl_out = OUT_DIR / "labels" / "train"
    tr_img_out.mkdir(parents=True, exist_ok=True)
    tr_lbl_out.mkdir(parents=True, exist_ok=True)

    train_imgs = list((SRC_DIR / "images" / "train").glob("*.jpg"))
    print(f"[TRAIN] Processing {len(train_imgs)} base images...")

    orig_count = 0
    aug_count = 0

    for img_p in train_imgs:
        # First, keep the clean original
        link_or_copy(img_p, tr_img_out / img_p.name)
        txt_p = SRC_DIR / "labels" / "train" / f"{img_p.stem}.txt"
        if txt_p.exists():
            link_or_copy(txt_p, tr_lbl_out / txt_p.name)
        orig_count += 1

        # Check if image contains plastic debris (class 0)
        has_debris = False
        label_content = ""
        if txt_p.exists():
            label_content = txt_p.read_text(encoding="utf-8").strip()
            if any(line.startswith("0 ") for line in label_content.splitlines()):
                has_debris = True

        # Only apply degradation augmentations to images with debris (or selective background)
        if has_debris or random.random() < 0.25:
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            # Randomly select 1 or 2 condition transformations
            augs_to_apply = []
            if random.random() < 0.50:
                augs_to_apply.append(("lowlight", lambda im: simulate_low_light(im, gamma=random.uniform(0.40, 0.60))))
            if random.random() < 0.50:
                augs_to_apply.append(("haze", lambda im: simulate_haze(im, intensity=random.uniform(0.35, 0.50))))
            if random.random() < 0.40:
                augs_to_apply.append(("blur", lambda im: simulate_blur(im, ksize=random.choice([7, 9, 11]), sigma=random.uniform(2.5, 4.0))))
            if random.random() < 0.35:
                augs_to_apply.append(("clahe", apply_clahe))

            for aug_name, fn in augs_to_apply:
                try:
                    degraded = fn(img)
                    aug_filename = f"{img_p.stem}_{aug_name}.jpg"
                    aug_lblname = f"{img_p.stem}_{aug_name}.txt"

                    cv2.imwrite(str(tr_img_out / aug_filename), degraded)
                    (tr_lbl_out / aug_lblname).write_text(label_content, encoding="utf-8")
                    aug_count += 1
                except Exception:
                    pass

    total_train = len(list(tr_img_out.glob("*.jpg")))
    print(f"\n[OK] Environmental Augmentation Complete!")
    print(f"  Base Images:      {orig_count}")
    print(f"  Augmented Images: {aug_count}")
    print(f"  Total Train:      {total_train}")
    print(f"  Total Val:        {len(list(val_img_out.glob('*.jpg')))}")

    # 3. Generate data/data_augmented.yaml
    yaml_content = f"""# Multi-Condition Environmentally Augmented Marine Plastic Dataset
path: {OUT_DIR.as_posix()}
train: images/train
val: images/val

nc: 3
names:
  0: plastic
  1: bio
  2: rov
"""
    yaml_p = PROJECT_ROOT / "data" / "data_augmented.yaml"
    yaml_p.write_text(yaml_content, encoding="utf-8")
    print(f"[OK] Generated dataset config: {yaml_p}")


if __name__ == "__main__":
    build_augmented_dataset()
