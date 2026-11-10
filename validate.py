#!/usr/bin/env python3
"""
validate.py - Flexible evaluation script for YOLOv8-seg models (Base, DPU-aware, or fine-tuned)
Computes and reports mAP50 and mAP50-95 for both bounding boxes and segmentation masks.

NOTE ON SPEED (COCO val2017):
    - Use '--data coco_full_fast.yaml' (default) instead of 'coco.yaml'.
    - Why? Ultralytics has hardcoded logic in val.py that forces 'save_json=True' whenever
      the validation file is named 'val2017.txt'. For segmentation, this forces full-resolution
      CPU RLE mask encoding for every detected instance, stalling validation from ~3.7 it/s down
      to ~3.0 s/it (14+ minutes vs <1 minute).
    - 'coco_full_fast.yaml' validates the exact same 5,000 val2017 images but uses 'val_full.txt',
      bypassing the forced CPU RLE stall.

Usage examples:
    # 1. Validate base YOLOv8n-seg model on full COCO 5,000 images (<1 minute):
    python validate.py --weights yolov8n-seg.pt --data coco_full_fast.yaml --batch 16

    # 2. Validate custom / DPU-aware weights and save JSON metrics for comparison:
    python validate.py --weights yolov8n_relu_fused.pt --data coco_full_fast.yaml --output-json results_dpu.json

    # 3. Quick test on 500 images (~15 seconds):
    python validate.py --weights yolov8n-seg.pt --data coco_val500.yaml
"""

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a YOLOv8-seg model on COCO or custom dataset.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n-seg.pt",
        help="Path to model weights (.pt checkpoint)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco_full_fast.yaml",
        help="Dataset YAML config (default: coco_full_fast.yaml for fast <1m validation of all 5000 images)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Validation image size (default: 640; use 416 or 640 depending on your DPU target)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker threads (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or 'cpu' (default: 0)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate ('val', 'test', or 'train')",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/val",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (defaults to model file stem)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save summary metrics dictionary in JSON format",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=None,
        help="Force FP16 half precision inference on CUDA",
    )
    parser.add_argument(
        "--full-precision",
        "--fp32",
        action="store_true",
        default=False,
        help="Force FP32 full precision inference (half=False)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        help="Generate full COCO RLE predictions JSON (warning: significantly slows down val step on CPU)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Model weights not found at: {weights_path}", file=sys.stderr)
        sys.exit(1)

    run_name = args.name or f"val_{weights_path.stem}_{args.imgsz}"

    print("\n========================================================")
    print(f"Model:   {weights_path}")
    print(f"Dataset: {args.data} (split: {args.split})")
    print(f"Image:   {args.imgsz}x{args.imgsz} | Batch: {args.batch} | Device: {args.device}")
    print("========================================================\n")

    # Load model
    model = YOLO(str(weights_path))

    # Precision selection
    val_half = None
    if args.full_precision:
        val_half = False
    elif args.half:
        val_half = True

    # Run validation
    val_kwargs = {
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": run_name,
        "save_json": False,
        "plots": False,
    }
    if val_half is not None:
        val_kwargs["half"] = val_half

    metrics = model.val(**val_kwargs)

    box_map50_95 = float(metrics.box.map)
    box_map50 = float(metrics.box.map50)
    box_map75 = float(metrics.box.map75)

    seg_map50_95 = float(metrics.seg.map)
    seg_map50 = float(metrics.seg.map50)
    seg_map75 = float(metrics.seg.map75)

    print("\n" + "=" * 56)
    print("                 VALIDATION SUMMARY                     ")
    print("=" * 56)
    print(f" Model:            {weights_path.name}")
    print(" Bounding Box:")
    print(f"   - mAP50-95:     {box_map50_95:.4f} ({box_map50_95 * 100:.2f}%)")
    print(f"   - mAP50:        {box_map50:.4f} ({box_map50 * 100:.2f}%)")
    print(f"   - mAP75:        {box_map75:.4f} ({box_map75 * 100:.2f}%)")
    print(" Segmentation Mask:")
    print(f"   - mAP50-95:     {seg_map50_95:.4f} ({seg_map50_95 * 100:.2f}%)")
    print(f"   - mAP50:        {seg_map50:.4f} ({seg_map50 * 100:.2f}%)")
    print(f"   - mAP75:        {seg_map75:.4f} ({seg_map75 * 100:.2f}%)")
    print("=" * 56)

    # Optional JSON output for automated comparisons
    if args.output_json:
        summary = {
            "model": str(weights_path),
            "data": args.data,
            "split": args.split,
            "imgsz": args.imgsz,
            "box": {
                "map50_95": box_map50_95,
                "map50": box_map50,
                "map75": box_map75,
            },
            "seg": {
                "map50_95": seg_map50_95,
                "map50": seg_map50,
                "map75": seg_map75,
            },
        }
        output_file = Path(args.output_json)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved metrics summary to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
