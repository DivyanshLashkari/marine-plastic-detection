"""
Marine Plastic Detection - Validation / Evaluation Script

Evaluates a trained YOLOv8n model on the test set and reports
precision, recall, F1, mAP@50, and mAP@50-95.

    python scripts/validate.py                         # defaults
    python scripts/validate.py --weights weights/best.pt --data data/data.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR,
    DEFAULT_WEIGHTS,
    METRICS_DIR,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    MODEL_NAME,
)
from src.utils import compute_f1, format_metrics_table, save_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the marine debris detector."
    )
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data", type=str, default=str(DATA_DIR / "data.yaml"))
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument(
        "--condition",
        type=str,
        default="normal",
        help="Label this evaluation run (e.g. 'normal', 'low_light').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics not installed.  pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("  Marine Debris Detection - Validation")
    print("=" * 60)
    print(f"  Weights   : {weights}")
    print(f"  Data      : {args.data}")
    print(f"  Condition : {args.condition}")
    print("=" * 60)

    model = YOLO(str(weights))

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
    )

    # Extract metrics
    precision = float(results.results_dict.get("metrics/precision(B)", 0))
    recall = float(results.results_dict.get("metrics/recall(B)", 0))
    f1 = compute_f1(precision, recall)
    map50 = float(results.results_dict.get("metrics/mAP50(B)", 0))
    map50_95 = float(results.results_dict.get("metrics/mAP50-95(B)", 0))

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
    }

    model_info = {
        "model_name": MODEL_NAME,
        "weights": str(weights),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
    }

    print("\n-- Results ------------------------------")
    print(format_metrics_table(metrics))

    # Persist
    out_path = save_metrics(metrics, args.condition, model_info)
    print(f"\nMetrics saved to {out_path}")



if __name__ == "__main__":
    main()
