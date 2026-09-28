"""
Marine Plastic Detection - Dataset Preparation Script

Prepares the trash_ICRA19 dataset into standard YOLOv8 directory structure:
    datasets/trash_ICRA19/yolo/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/

Uses hardlinks (zero disk space duplication, instant) with fallback to file copying.
Also generates data/data.yaml and data/trash_icra19_sample.yaml for quick training.
"""

import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def link_or_copy(src: Path, dst: Path):
    """Link file if possible, else copy."""
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except (OSError, AttributeError):
        shutil.copy2(str(src), str(dst))


def prepare_yolo_structure(raw_dir: Path, out_dir: Path):
    """
    Separates mixed .jpg and .txt files from raw_dir splits into
    standard YOLO images/ and labels/ subdirectories.
    """
    splits = ["train", "val", "test"]
    stats = {}

    for split in splits:
        src_split_dir = raw_dir / split
        if not src_split_dir.exists():
            print(f"[WARN] Split directory not found: {src_split_dir}")
            continue

        img_dest = out_dir / "images" / split
        lbl_dest = out_dir / "labels" / split
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        img_count = 0
        lbl_count = 0

        for file_path in src_split_dir.iterdir():
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                link_or_copy(file_path, img_dest / file_path.name)
                img_count += 1
            elif file_path.suffix.lower() == ".txt":
                link_or_copy(file_path, lbl_dest / file_path.name)
                lbl_count += 1

        stats[split] = {"images": img_count, "labels": lbl_count}
        print(f"  [{split.upper()}] Linked {img_count} images and {lbl_count} label files.")

    return stats


def generate_yaml_configs(yolo_dir: Path):
    """Generate YOLO data.yaml configuration files."""
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    yolo_dir_posix = yolo_dir.as_posix()

    full_yaml_content = f"""# Marine Plastic / Underwater Trash ICRA19 Dataset Configuration
path: {yolo_dir_posix}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: plastic
  1: bio
  2: rov
"""

    main_yaml_path = data_dir / "data.yaml"
    with open(main_yaml_path, "w", encoding="utf-8") as f:
        f.write(full_yaml_content)
    print(f"[OK] Generated full dataset config: {main_yaml_path}")

    # Also generate sample mini-subset config for fast CPU training/testing
    sample_dir = PROJECT_ROOT / "datasets" / "trash_ICRA19" / "sample"
    for split, count in [("train", 300), ("val", 80)]:
        src_imgs = list((yolo_dir / "images" / split).glob("*.jpg"))[:count]
        s_img_dir = sample_dir / "images" / split
        s_lbl_dir = sample_dir / "labels" / split
        s_img_dir.mkdir(parents=True, exist_ok=True)
        s_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_p in src_imgs:
            link_or_copy(img_p, s_img_dir / img_p.name)
            txt_p = yolo_dir / "labels" / split / f"{img_p.stem}.txt"
            if txt_p.exists():
                link_or_copy(txt_p, s_lbl_dir / txt_p.name)

    sample_yaml_content = f"""# Marine Plastic / Underwater Trash ICRA19 Sample (Fast CPU Experimentation)
path: {sample_dir.as_posix()}
train: images/train
val: images/val

nc: 3
names:
  0: plastic
  1: bio
  2: rov
"""
    sample_yaml_path = data_dir / "data_sample.yaml"
    with open(sample_yaml_path, "w", encoding="utf-8") as f:
        f.write(sample_yaml_content)
    print(f"[OK] Generated sample dataset config (300 train / 80 val): {sample_yaml_path}")


def main():
    print("=" * 60)
    print("  Marine Plastic Detection - Dataset Preparation")
    print("=" * 60)

    raw_dir = PROJECT_ROOT / "datasets" / "trash_ICRA19" / "dataset"
    if not raw_dir.exists():
        print(f"[ERROR] Raw dataset folder not found at: {raw_dir}")
        sys.exit(1)

    out_dir = PROJECT_ROOT / "datasets" / "trash_ICRA19" / "yolo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing dataset from: {raw_dir}")
    print(f"Output YOLO directory:   {out_dir}")

    stats = prepare_yolo_structure(raw_dir, out_dir)
    generate_yaml_configs(out_dir)

    print("\n[OK] Dataset preparation complete!")
    print(f"  Train : {stats.get('train', {}).get('images', 0)} images")
    print(f"  Val   : {stats.get('val', {}).get('images', 0)} images")
    print(f"  Test  : {stats.get('test', {}).get('images', 0)} images")
    print("=" * 60)


if __name__ == "__main__":
    main()

