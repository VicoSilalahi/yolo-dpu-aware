#!/usr/bin/env python3
"""
train.py - Fine-tuning / Training script for YOLOv8-seg models (Base or DPU-modified)
Optimized with defaults suitable for 4GB VRAM (GTX 1650).

Usage examples:
    # 1. Standard fine-tuning on COCO:
    python train.py --weights yolov8n-seg.pt --data coco.yaml --epochs 50 --batch 4 --imgsz 640

    # 2. Fast recovery fine-tuning with subset of data (e.g. 10% or 20% of COCO):
    python train.py --weights yolov8n_relu_fused.pt --data coco.yaml --fraction 0.2 --epochs 20

    # 3. Quick test with coco8-seg or coco128-seg:
    python train.py --weights yolov8n-seg.pt --data coco8-seg.yaml --epochs 5
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train or fine-tune a YOLOv8-seg model.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n-seg.pt",
        help="Initial weights (.pt) to start fine-tuning from",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco.yaml",
        help="Dataset YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (default: 4; recommended for 4GB GTX 1650 VRAM to prevent OOM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (e.g., 640 or 416 for FPGA/DPU targets)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or 'cpu' (default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Dataset fraction to use (e.g., 0.1 for 10%% of COCO for quick recovery training)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001 for recovery fine-tuning)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/recovery",
        help="Directory to save training artifacts and checkpoints (default: runs/recovery)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="hardswish",
        help="Run subfolder name inside project (default: hardswish)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists() and not str(weights_path).endswith(".pt"):
        print(f"Error: Weights file '{weights_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("\n========================================================")
    print("Starting YOLOv8-seg Training/Fine-tuning")
    print(f"Model:      {weights_path}")
    print(f"Dataset:    {args.data}")
    print(f"Epochs:     {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")
    print(f"Fraction:   {args.fraction * 100:.1f}% | Device: {args.device}")
    print("========================================================\n")

    model = YOLO(str(weights_path))

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
