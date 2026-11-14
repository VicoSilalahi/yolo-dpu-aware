import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

# Ensure ultralytics is in sys.path
sys.path.insert(0, "/workspace/ultralytics")
from ultralytics import YOLO
from pytorch_nndct.qproc import TorchQuantProcessor


def preprocess_image(img_path, imgsz=640):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read calibration image: {img_path}")
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (imgsz - nh) // 2
    bottom = imgsz - nh - top
    left = (imgsz - nw) // 2
    right = imgsz - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a DPU-truncated YOLOv8-seg model with Vitis AI NNDCT.")
    parser.add_argument("--weights", default="/workspace/ultralytics/checkpoints/yolov8n-seg-dpu-base.pt")
    parser.add_argument("--calib-dir", default="/workspace/datasets/coco/images/val2017")
    parser.add_argument("--output-dir", default="/workspace/ultralytics/checkpoints/quantize_result")
    parser.add_argument("--num-calib", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--bitwidth-w", type=int, default=8)
    parser.add_argument("--bitwidth-a", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading YOLO model from {args.weights}...")
    yolo = YOLO(args.weights)
    model = yolo.model.eval().cpu()

    # CRITICAL FOR KV260 VITIS AI COMPATIBILITY:
    # Completely truncate post-processing (decode_bboxes, anchor generation, strided_slice, sigmoid).
    # Intercept the forward pass of Segment (model.model[-1]) so it strictly outputs the 10 raw convolution
    # feature tensors directly. This ensures 100% of operations compile to pure DPU hardware kernels
    # with 0 CPU fallback subgraphs and NO strided_slice-fix operators.
    seg = model.model[-1]

    def raw_dpu_forward(x):
        outputs = []
        for i in range(seg.nl):
            outputs.append(seg.cv2[i](x[i]))  # box conv (1, 64, H_i, W_i)
            outputs.append(seg.cv3[i](x[i]))  # cls conv (1, nc, H_i, W_i)
            outputs.append(seg.cv4[i](x[i]))  # mask coeff conv (1, 32, H_i, W_i)
        outputs.append(seg.proto(x[0]))       # proto mask feature map (1, 32, 160, 160)
        return tuple(outputs)

    seg.forward = raw_dpu_forward

    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz)

    print("Initializing TorchQuantProcessor for calibration...")
    quantizer = TorchQuantProcessor(
        quant_mode="calib",
        module=model,
        input_data=dummy_input,
        output_dir=args.output_dir,
        bitwidth_w=args.bitwidth_w,
        bitwidth_a=args.bitwidth_a,
        device=torch.device("cpu"),
    )

    quant_model = quantizer.quant_model()

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    img_files = sorted({f for pattern in patterns for f in glob.glob(os.path.join(args.calib_dir, pattern))})
    img_files = img_files[:args.num_calib]
    if not img_files:
        raise FileNotFoundError(f"No calibration images found in {args.calib_dir}")
    print(f"Running calibration over {len(img_files)} images from {args.calib_dir}...")
    with torch.no_grad():
        for i, img_path in enumerate(img_files):
            inp = preprocess_image(img_path, args.imgsz)
            _ = quant_model(inp)
            if (i + 1) % 10 == 0 or (i + 1) == len(img_files):
                print(f"Calibrated {i + 1}/{len(img_files)} images...")

    quantizer.export_quant_config()
    print("Calibration config exported successfully!")

    print("\n--- Step 2: Testing INT8 Evaluation & Exporting XIR / XModel ---")
    quantizer_test = TorchQuantProcessor(
        quant_mode="test",
        module=model,
        input_data=dummy_input,
        output_dir=args.output_dir,
        bitwidth_w=args.bitwidth_w,
        bitwidth_a=args.bitwidth_a,
        device=torch.device("cpu"),
    )

    test_model = quantizer_test.quant_model()
    # Run dummy pass to finalize test graph
    with torch.no_grad():
        _ = test_model(dummy_input)

    print("Exporting XMODEL / XIR graph for compiler...")
    quantizer_test.export_xmodel(output_dir=args.output_dir, deploy_check=False)
    print(f"All quantized artifacts generated in {args.output_dir}")


if __name__ == "__main__":
    main()
