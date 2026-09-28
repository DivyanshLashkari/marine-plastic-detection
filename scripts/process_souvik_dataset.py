"""
Process SOUVIK Classification Dataset into YOLO Object Detection Format.

1. Auto-annotates plastic images using YOLOv8 predictions + adaptive saliency fallback.
2. Incorporates no-plastic images as official YOLO background/negative samples (empty .txt).
3. Merges SOUVIK with trash_ICRA19 into a unified, diverse training dataset:
   - High-diversity underwater ROV debris (trash_ICRA19)
   - Real floating ocean plastic bags/bottles (SOUVIK)
   - Clean underwater background samples to minimize false positives
"""

import os
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOUVIK_RAW = Path(r"C:\Users\chara\.cache\kagglehub\datasets\surajit651\souvikdataset\versions\1\SOUVIK")
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "souvik_yolo"


def get_saliency_bbox(img: np.ndarray):
    """Fallback foreground bounding box using Otsu thresholding + contours."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Try both standard and inverted Otsu
    _, t1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, t2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    boxes = []
    for thresh in [t1, t2]:
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            # Filter reasonable debris sizes (3% to 90% of image area)
            if 0.03 * w * h < area < 0.90 * w * h:
                x, y, bw, bh = cv2.boundingRect(c)
                # Convert to YOLO normalized format: x_center, y_center, w, h
                xc = (x + bw / 2.0) / w
                yc = (y + bh / 2.0) / h
                nw = bw / float(w)
                nh = bh / float(h)
                boxes.append((xc, yc, nw, nh))
    
    return boxes[:3] if boxes else [(0.5, 0.5, 0.6, 0.6)]


def process_souvik():
    if not SOUVIK_RAW.exists():
        print(f"[ERROR] SOUVIK raw directory not found: {SOUVIK_RAW}")
        return

    print("=" * 60)
    print("  Processing SOUVIK Dataset -> YOLO Object Detection Format")
    print("=" * 60)

    # Use best trained weights or fallback to yolov8n.pt for pseudo-labeling
    weights_path = PROJECT_ROOT / "weights" / "best.pt"
    if weights_path.exists():
        model = YOLO(str(weights_path))
        print(f"Using trained model: {weights_path}")
    else:
        model = YOLO("yolov8n.pt")
        print("Using base YOLOv8n model.")

    for split in ["train", "test"]:
        img_out = OUTPUT_DIR / "images" / split
        lbl_out = OUTPUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        plastic_dir = SOUVIK_RAW / split / "plastic"
        noplastic_dir = SOUVIK_RAW / split / "no-plastic"

        # 1. Process Plastic Images
        plastic_imgs = list(plastic_dir.glob("*.jpg"))
        print(f"[{split.upper()}] Annotating {len(plastic_imgs)} plastic images...")

        annotated_count = 0
        for img_p in plastic_imgs:
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            h, w = img.shape[:2]
            # Copy image
            dest_img = img_out / img_p.name
            shutil.copy2(img_p, dest_img)

            # Detect with YOLO
            results = model.predict(img, conf=0.20, verbose=False)
            boxes = []

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for b in r.boxes:
                        # Map to class 0 (plastic)
                        xc, yc, bw, bh = b.xywhn[0].tolist()
                        boxes.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # If YOLO found nothing, use saliency fallback
            if not boxes:
                s_boxes = get_saliency_bbox(img)
                for xc, yc, bw, bh in s_boxes:
                    boxes.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # Write label file
            txt_dest = lbl_out / f"{img_p.stem}.txt"
            with open(txt_dest, "w", encoding="utf-8") as f:
                f.write("\n".join(boxes) + "\n")
            annotated_count += 1

        # 2. Process No-Plastic Background Images (Negative Samples)
        # We take a balanced number (e.g. 200 in train, 50 in test)
        n_bg = 200 if split == "train" else 50
        noplastic_imgs = list(noplastic_dir.glob("*.jpg"))[:n_bg]
        print(f"[{split.upper()}] Adding {len(noplastic_imgs)} negative background samples...")

        for img_p in noplastic_imgs:
            dest_img = img_out / f"bg_{img_p.name}"
            shutil.copy2(img_p, dest_img)
            # Empty label file = true negative background image in YOLO
            txt_dest = lbl_out / f"bg_{img_p.stem}.txt"
            with open(txt_dest, "w", encoding="utf-8") as f:
                pass

    print("\n[OK] SOUVIK pseudo-annotation complete!")


def build_merged_dataset():
    """Merge SOUVIK with trash_ICRA19 for maximum detection robustness."""
    merged_dir = PROJECT_ROOT / "datasets" / "merged"
    shutil.rmtree(merged_dir, ignore_errors=True)

    icra_dir = PROJECT_ROOT / "datasets" / "trash_ICRA19" / "yolo"
    souvik_dir = OUTPUT_DIR

    for split in ["train", "val"]:
        m_img = merged_dir / "images" / split
        m_lbl = merged_dir / "labels" / split
        m_img.mkdir(parents=True, exist_ok=True)
        m_lbl.mkdir(parents=True, exist_ok=True)

        icra_split = split
        souvik_split = "train" if split == "train" else "test"

        # 1. Link ICRA19 images (600 in train, 100 in val)
        n_icra = 700 if split == "train" else 100
        icra_imgs = list((icra_dir / "images" / icra_split).glob("obj*.jpg"))[:n_icra]
        for ip in icra_imgs:
            os.link(str(ip), str(m_img / f"icra_{ip.name}"))
            lp = icra_dir / "labels" / icra_split / f"{ip.stem}.txt"
            if lp.exists():
                os.link(str(lp), str(m_lbl / f"icra_{ip.stem}.txt"))

        # 2. Link SOUVIK images
        n_souvik = 500 if split == "train" else 100
        s_imgs = list((souvik_dir / "images" / souvik_split).glob("*.jpg"))[:n_souvik]
        for sp in s_imgs:
            os.link(str(sp), str(m_img / f"svk_{sp.name}"))
            slp = souvik_dir / "labels" / souvik_split / f"{sp.stem}.txt"
            if slp.exists():
                os.link(str(slp), str(m_lbl / f"svk_{sp.stem}.txt"))

        print(f"[MERGED {split.upper()}] {len(icra_imgs)} ICRA19 + {len(s_imgs)} SOUVIK = {len(icra_imgs) + len(s_imgs)} total.")

    # Write data/data_merged.yaml
    yaml_content = f"""# Unified Marine Plastic Detection Dataset (ICRA19 + SOUVIK)
path: {merged_dir.as_posix()}
train: images/train
val: images/val

nc: 3
names:
  0: plastic
  1: bio
  2: rov
"""
    yaml_p = PROJECT_ROOT / "data" / "data_merged.yaml"
    with open(yaml_p, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[OK] Generated unified dataset config: {yaml_p}")


if __name__ == "__main__":
    process_souvik()
    build_merged_dataset()
