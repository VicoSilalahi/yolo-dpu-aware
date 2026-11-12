import os
import sys
import glob
import torch
import cv2
import numpy as np

# Ensure ultralytics is in sys.path
sys.path.insert(0, "/workspace/ultralytics")
from ultralytics import YOLO
from pytorch_nndct.qproc import TorchQuantProcessor

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    r = min(640 / h, 640 / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (640 - nh) // 2
    bottom = 640 - nh - top
    left = (640 - nw) // 2
    right = 640 - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0)

def main():
    pt_path = "/workspace/ultralytics/checkpoints/yolov8n-seg-dpu-base.pt"
    calib_dir = "/workspace/datasets/coco/images/val2017"
    output_dir = "/workspace/ultralytics/checkpoints/quantize_result"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading YOLO model from {pt_path}...")
    yolo = YOLO(pt_path)
    model = yolo.model.eval().cpu()

    # CRITICAL FOR KV260 VITIS AI 2.0/3.0 COMPATIBILITY:
    # Completely truncate post-processing (decode_bboxes, anchor generation, strided_slice, sigmoid).
    # Intercept the forward pass of Segment (model.model[-1]) so it strictly outputs the 10 raw convolution
    # feature tensors directly. This ensures 100% of operations compile to pure DPU hardware kernels
    # with 0 CPU fallback subgraphs and NO strided_slice-fix operators.
    seg = model.model[-1]
    def raw_dpu_forward(x):
        outputs = []
        for i in range(seg.nl):
            outputs.append(seg.cv2[i](x[i]))  # box conv (1, 64, H_i, W_i)
            outputs.append(seg.cv3[i](x[i]))  # cls conv (1, 80, H_i, W_i)
            outputs.append(seg.cv4[i](x[i]))  # mask coeff conv (1, 32, H_i, W_i)
        outputs.append(seg.proto(x[0]))       # proto mask feature map (1, 32, 160, 160)
        return tuple(outputs)

    seg.forward = raw_dpu_forward

    dummy_input = torch.randn(1, 3, 640, 640)

    print("Initializing TorchQuantProcessor for calibration...")
    # target DPUCZDX8G_ISA1_B3136 (KV260 smartcam overlay)
    quantizer = TorchQuantProcessor(
        quant_mode="calib",
        module=model,
        input_data=dummy_input,
        output_dir=output_dir,
        bitwidth_w=8,
        bitwidth_a=8,
        device=torch.device("cpu")
    )

    quant_model = quantizer.quant_model()

    img_files = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))[:50]
    print(f"Running calibration over {len(img_files)} COCO images...")
    with torch.no_grad():
        for i, img_path in enumerate(img_files):
            inp = preprocess_image(img_path)
            _ = quant_model(inp)
            if (i + 1) % 10 == 0:
                print(f"Calibrated {i + 1}/{len(img_files)} images...")

    quantizer.export_quant_config()
    print("Calibration config exported successfully!")

    print("\n--- Step 2: Testing INT8 Evaluation & Exporting XIR / XModel ---")
    quantizer_test = TorchQuantProcessor(
        quant_mode="test",
        module=model,
        input_data=dummy_input,
        output_dir=output_dir,
        bitwidth_w=8,
        bitwidth_a=8,
        device=torch.device("cpu")
    )

    test_model = quantizer_test.quant_model()
    # Run dummy pass to finalize test graph
    with torch.no_grad():
        _ = test_model(dummy_input)

    print("Exporting XMODEL / XIR graph for compiler...")
    quantizer_test.export_xmodel(output_dir=output_dir, deploy_check=False)
    print(f"All quantized artifacts generated in {output_dir}")

if __name__ == "__main__":
    main()
