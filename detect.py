"""
ARF-YOLO Inference Script.

Runs inference on images, videos, or directories.

Usage:
    python detect.py --weights runs/train/arf_yolo/weights/best.pt --source path/to/images
    python detect.py --weights runs/train/arf_yolo/weights/best.pt --source path/to/video.mp4
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
    parser = argparse.ArgumentParser(description="ARF-YOLO Inference")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Input source (image, video, directory, webcam)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save detection results")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save results in txt format")
    parser.add_argument("--show", action="store_true",
                        help="Display results")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Project directory")
    parser.add_argument("--name", type=str, default="arf_yolo",
                        help="Experiment name")
    parser.add_argument("--classes", nargs="+", type=int, default=None,
                        help="Filter by class index")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum detections per image")
    return parser.parse_args()


def main():
    args = parse_args()

    # Register custom modules
    register_custom_modules()

    # Load model
    print(f"Loading model from {args.weights}")
    model = YOLO(args.weights)

    # Predict
    predict_args = {
        "source": args.source,
        "imgsz": args.img_size,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "save": args.save,
        "save_txt": args.save_txt,
        "show": args.show,
        "project": args.project,
        "name": args.name,
        "classes": args.classes,
        "max_det": args.max_det,
        "exist_ok": True,
    }

    print("=" * 60)
    print("ARF-YOLO Inference Configuration:")
    print(f"  Weights: {args.weights}")
    print(f"  Source: {args.source}")
    print(f"  Image size: {args.img_size}")
    print(f"  Confidence: {args.conf}")
    print(f"  NMS IoU: {args.iou}")
    print("=" * 60)

    results = model.predict(**predict_args)

    print(f"\nInference completed! Results saved to {args.project}/{args.name}/")

    return results


if __name__ == "__main__":
    main()
