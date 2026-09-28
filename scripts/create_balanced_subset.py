"""
Create a plastic-debris-focused balanced training subset:
- 400 plastic debris images (obj*.jpg) + 100 marine bio images (bio*.jpg) for training
- 80 plastic debris images + 20 bio images for validation
Total: 500 train, 100 val with high plastic density.
"""

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
yolo_dir = PROJECT_ROOT / "datasets" / "trash_ICRA19" / "yolo"
sample_dir = PROJECT_ROOT / "datasets" / "trash_ICRA19" / "sample"

# Clear existing sample
shutil.rmtree(sample_dir, ignore_errors=True)

def link_file(src: Path, dst: Path):
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))

for split in ["train", "val"]:
    img_dest = sample_dir / "images" / split
    lbl_dest = sample_dir / "labels" / split
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)

    n_obj = 400 if split == "train" else 80
    n_bio = 100 if split == "train" else 20

    obj_imgs = list((yolo_dir / "images" / split).glob("obj*.jpg"))[:n_obj]
    bio_imgs = list((yolo_dir / "images" / split).glob("bio*.jpg"))[:n_bio]

    selected = obj_imgs + bio_imgs
    for img_p in selected:
        link_file(img_p, img_dest / img_p.name)
        txt_p = yolo_dir / "labels" / split / f"{img_p.stem}.txt"
        if txt_p.exists():
            link_file(txt_p, lbl_dest / txt_p.name)

    print(f"[{split.upper()}] {len(obj_imgs)} plastic debris + {len(bio_imgs)} bio images = {len(selected)} total.")

print("Done! Balanced subset ready.")
