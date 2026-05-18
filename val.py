"""
ARF-YOLO Validation Script.

Evaluates ARF-YOLO on VisDrone2019 or UAVDT validation/test sets.

Usage:
    python val.py --weights runs/train/arf_yolo/weights/best.pt --data configs/visdrone.yaml
    python val.py --weights runs/train/arf_yolo/weights/best.pt --data configs/uavdt.yaml
"""

import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from utils.register_modules import register_custom_modules


def parse_args():
    parser = argparse.ArgumentParser(description="ARF-YOLO Validation")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--data", type=str, default="configs/visdrone.yaml",
                        help="Dataset configuration file")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU threshold")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-class metrics")
    parser.add_argument("--save-json", action="store_true",
                        help="Save results in COCO JSON format")
    parser.add_argument("--project", type=str, default="runs/val",
                        help="Project directory")
    parser.add_argument("--name", type=str, default="arf_yolo",
                        help="Experiment name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Register custom modules
    register_custom_modules()

    # Load model
    print(f"Loading model from {args.weights}")
    model = YOLO(args.weights)

    # Validate
    val_args = {
        "data": args.data,
        "imgsz": args.img_size,
        "batch": args.batch_size,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "split": args.split,
        "verbose": args.verbose,
        "save_json": args.save_json,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }

    print("=" * 60)
    print("ARF-YOLO Validation Configuration:")
    print(f"  Weights: {args.weights}")
    print(f"  Dataset: {args.data}")
    print(f"  Split: {args.split}")
    print(f"  Image size: {args.img_size}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  NMS IoU threshold: {args.iou}")
    print("=" * 60)

    results = model.val(**val_args)

    # Print summary
    print("\nValidation Results:")
    print(f"  mAP@0.5:      {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision:     {results.box.mp:.4f}")
    print(f"  Recall:        {results.box.mr:.4f}")

    if args.verbose:
        print("\nPer-class AP@0.5:")
        for i, ap in enumerate(results.box.ap50):
            print(f"  Class {i}: {ap:.4f}")

    return results


if __name__ == "__main__":
    main()
