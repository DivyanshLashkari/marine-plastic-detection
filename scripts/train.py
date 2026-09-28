"""
Marine Plastic Detection - Training Script

Trains a YOLOv8n model on the prepared marine debris dataset.
Run from the project root:

    python scripts/train.py                       # defaults
    python scripts/train.py --epochs 50 --batch 8 # override
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR,
    WEIGHTS_DIR,
    PRETRAINED_WEIGHTS,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_IMAGE_SIZE,
    TRAIN_PATIENCE,
    MODEL_NAME,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n on the marine debris dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_DIR / "data.yaml"),
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=PRETRAINED_WEIGHTS,
        help="Pretrained weights to start from (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAIN_EPOCHS,
        help=f"Training epochs (default: {TRAIN_EPOCHS}).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=TRAIN_BATCH_SIZE,
        help=f"Batch size (default: {TRAIN_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TRAIN_IMAGE_SIZE,
        help=f"Training image size (default: {TRAIN_IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=TRAIN_PATIENCE,
        help=f"Early-stopping patience (default: {TRAIN_PATIENCE}).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(WEIGHTS_DIR.parent / "runs"),
        help="Directory for training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="marine_debris_train",
        help="Run name inside project directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if __import__("torch").cuda.is_available() else "cpu",
        help="Device to train on (e.g. 0, cpu).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader workers (default 2 to prevent RAM exhaustion).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=False,
        help="Generate training plots (default False).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from weights checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate data config exists
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[ERROR] Dataset config not found: {data_yaml}")
        print("  → Prepare the dataset first (see README.md § Dataset Preparation).")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics not installed.  pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("  Marine Debris Detection - Training")
    print("=" * 60)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Weights     : {args.weights}")
    print(f"  Data config : {args.data}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Patience    : {args.patience}")
    print(f"  Device      : {args.device}")
    print(f"  Output dir  : {args.project}/{args.name}")
    print("=" * 60)

    # Load model
    model = YOLO(args.weights)

    # Train
    if args.resume:
        print("[INFO] Resuming training from checkpoint...")
        results = model.train(resume=True, workers=args.workers, plots=args.plots, device=args.device)
    else:
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            plots=args.plots,
            project=args.project,
            name=args.name,
            exist_ok=True,
            verbose=True,
        )

    # Copy best weights to the standard location
    run_dir = Path(args.project) / args.name
    best_pt = run_dir / "weights" / "best.pt"
    dest = WEIGHTS_DIR / "best.pt"

    if best_pt.exists():
        import shutil
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dest)
        print(f"\n[OK] Best weights copied to {dest}")
    else:
        print(f"\n[WARN] best.pt not found at {best_pt}")

    print("\nTraining complete.")



if __name__ == "__main__":
    main()
