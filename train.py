"""
ARF-YOLO Training Script.

Trains ARF-YOLO on VisDrone2019 or UAVDT datasets using the Ultralytics
framework with custom modules (AGRH, AMFF, FSRA) and ARF-SALoss.

Usage:
    python train.py --cfg configs/arf_yolo.yaml --data configs/visdrone.yaml
    python train.py --cfg configs/arf_yolo.yaml --data configs/uavdt.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from models import AGRH, AMFF, FSRA, MPFA, ARFScaleAwareLoss
from utils.register_modules import register_custom_modules


def parse_args():
    parser = argparse.ArgumentParser(description="ARF-YOLO Training")
    parser.add_argument("--cfg", type=str, default="configs/arf_yolo.yaml",
                        help="Model configuration file")
    parser.add_argument("--data", type=str, default="configs/visdrone.yaml",
                        help="Dataset configuration file")
    parser.add_argument("--weights", type=str, default="yolo11m.pt",
                        help="Pretrained weights (default: YOLOv11m COCO)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device(s), e.g., '0' or '0,1'")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Project directory")
    parser.add_argument("--name", type=str, default="arf_yolo",
                        help="Experiment name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    # ARF-SALoss hyperparameters
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="ARF-SALoss alpha parameter")
    parser.add_argument("--beta", type=float, default=3.0,
                        help="ARF-SALoss beta parameter")
    parser.add_argument("--lambda-sa", type=float, default=0.5,
                        help="ARF-SALoss lambda weight")
    return parser.parse_args()


def main():
    args = parse_args()

    # Register custom modules with Ultralytics
    register_custom_modules()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load model
    print(f"Loading model from {args.cfg} with pretrained weights: {args.weights}")
    model = YOLO(args.cfg)

    # Training configuration following the paper:
    # SGD with momentum=0.937, weight_decay=5e-4
    # Initial lr=0.01, cosine annealing, 3-epoch warmup
    # Mosaic augmentation for first 190 epochs (disabled last 10)
    # EMA decay=0.9999
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.img_size,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "seed": args.seed,
        # Optimizer
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,  # Final lr = lr0 * lrf (cosine annealing)
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        # Augmentation (standard YOLO protocol)
        "mosaic": 1.0,
        "close_mosaic": 10,  # Disable mosaic in last 10 epochs
        "flipud": 0.0,
        "fliplr": 0.5,
        "scale": 0.5,  # Multi-scale jitter
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "copy_paste": 0.1,
        # Other
        "pretrained": True,
        "cos_lr": True,  # Cosine annealing
        "val": True,
        "plots": True,
        "save": True,
        "exist_ok": True,
    }

    # Train
    print("=" * 60)
    print("ARF-YOLO Training Configuration:")
    print(f"  Model config: {args.cfg}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.img_size}")
    print(f"  Device: {args.device}")
    print(f"  ARF-SALoss: alpha={args.alpha}, beta={args.beta}, lambda={args.lambda_sa}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    results = model.train(**train_args)

    print("\nTraining completed!")
    print(f"Best weights saved to: {args.project}/{args.name}/weights/best.pt")

    return results


if __name__ == "__main__":
    main()
