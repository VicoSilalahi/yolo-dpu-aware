# YOLOv8-Seg DPU Porting Context & Handoff Guide

> **Target Platform:** AMD Kria KV260 Vision AI Starter Kit (`DPUCZDX8G_ISA1_B4096`)  
> **Workspace Root:** `/home/vicosilalahi/programming/yolo-fpga/ultralytics`  
> **Python Virtualenv:** `/home/vicosilalahi/programming/yolo-fpga/venv` (`/home/vicosilalahi/programming/yolo-fpga/venv/bin/python`)  
> **Strict Constraint:** Do **NOT** inspect, modify, or reference `/home/vicosilalahi/programming/yolo-fpga/kv260-yolo-dpu`.

---

## 1. Project Overview & Progress Status

We adapted the official **Ultralytics YOLOv8-seg** model to make it **100% hardware-compatible** with AMD/Xilinx DPU (`DPUCZDX8G`). All unsupported operators (`SiLU`, dynamic `torch.chunk` slicing, and Proto `ConvTranspose2d`) have been replaced, fine-tuned, exported to ONNX, INT8-quantized, and compiled into an `.xmodel` hardware binary.

### Verification Matrix (COCO `val2017`):

| Model Stage | File | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 | DPU Compatibility |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Official Baseline** | `checkpoints/yolov8n-seg.pt` | 36.19% | 51.56% | 30.24% | 48.84% | ❌ Incompatible (SiLU, chunk, deconv) |
| **1. Hardswish Replacement** | `checkpoints/yolov8n-seg-hardswish-init.pt` | 28.62% | 42.80% | 22.68% | 39.92% | ⚠️ Activation ready |
| **2. C2f 1x1 Identity Conv**| `checkpoints/yolov8n-seg-hardswish-c2fsplit.pt` | 28.62% | 42.80% | 22.68% | 39.92% | ⚠️ Slices eliminated |
| **3. C2f Warmup** | `checkpoints/yolov8n-seg-hardswish-c2fsplit-finetuned.pt` | 31.95% | 46.75% | 26.50% | 43.97% | ⚠️ Proto deconv left |
| **4. DPU Base Ready (FP32)**| `checkpoints/yolov8n-seg-dpu-base.pt` | **30.66%** | **45.64%** | **23.84%** | **41.64%** | ✅ **100% DPU Graph Ready** |
| **5. INT8 Quantized (SIL)** | `checkpoints/yolov8n-seg-dpu-base-int8.onnx` | **27.61%** | **43.35%** | **23.10%** | **40.34%** | ✅ **96.9% Mask Retention** |
| **6. Hardware Binary** | `checkpoints/compiled_kv260/yolov8n_seg_kv260.xmodel` | — | — | — | — | ✅ **Compiled for KV260 B4096** |

---

## 2. Exact Codebase Modifications Applied

All changes were made in-place inside the editable install at `/home/vicosilalahi/programming/yolo-fpga/ultralytics`:

1. **Activation Replacement (`SiLU` $\rightarrow$ `Hardswish`)**:
   - File: `ultralytics/nn/modules/conv.py` (line ~58)
   - Changed: `default_act = nn.Hardswish(inplace=True)` across `Conv`, `ConvTranspose`, and `RepConv`.
2. **C2f Chunking Replacement (`torch.chunk` $\rightarrow$ Frozen $1 \times 1$ Identity Convolutions)**:
   - File: `ultralytics/nn/modules/block.py` (`C2f` class, lines ~291–345)
   - Replaced `torch.chunk(feat, 2, dim=1)` with `self.split0(feat)` and `self.split1(feat)` using weights initialized to identity matrix with `requires_grad=False`.
   - File: `ultralytics/engine/trainer.py` (line ~395)
   - Added `".split0", ".split1"` to `always_freeze_names` to prevent training loop unfreezing.
3. **Proto Upsampling (`nn.ConvTranspose2d` $\rightarrow$ `nn.Upsample` Nearest)**:
   - File: `ultralytics/nn/modules/block.py` (`Proto` class, line ~98)
   - Replaced deconvolution `nn.ConvTranspose2d(c_, c_, 2, 2, bias=True)` with `nn.Upsample(scale_factor=2, mode="nearest")`.

---

## 3. Key Artifacts & Deployment Package

- **Base Checkpoint (Ready for Custom Dataset Training)**:
  `checkpoints/yolov8n-seg-dpu-base.pt`
- **ONNX Model (Fixed Shape 1x3x640x640)**:
  `checkpoints/yolov8n-seg-dpu-base.onnx`
- **INT8 Quantized ONNX Model**:
  `checkpoints/yolov8n-seg-dpu-base-int8.onnx`
- **Intermediate Hardware Graph**:
  `checkpoints/quantize_result/SegmentationModel_int.xmodel`
- **Compiled KV260 Hardware Binary**:
  `checkpoints/compiled_kv260/yolov8n_seg_kv260.xmodel` (5.2 MB)
- **Deployment Tarball**:
  `deploy_kv260.tar.gz` (contains `yolov8n_seg_kv260.xmodel`, `run_kv260_dpu.py`, and `test.jpg`)

---

## 4. Current Task & Immediate Next Steps

The user has connected the **Kria KV260 board** and is transferring `deploy_kv260.tar.gz` via their laptop.

### On the Kria KV260:
1. **Network**: KV260 USB tether IP is `10.21.42.164`.
2. **Transfer**:
   ```bash
   scp deploy_kv260.tar.gz ubuntu@10.21.42.164:~/
   ```
3. **Extract & Load Firmware**:
   ```bash
   tar -xzvf deploy_kv260.tar.gz
   cd deploy_kv260
   sudo xmutil loadapp kv260-benchmark   # or smartcam
   ```
4. **Run DPU Inference**:
   ```bash
   python3 run_kv260_dpu.py yolov8n_seg_kv260.xmodel test.jpg
   ```

### Subsequent Phase (Fine-Tuning on Custom Dataset):
1. Load user custom dataset YAML.
2. Train using base checkpoint `checkpoints/yolov8n-seg-dpu-base.pt`.
3. Run `dpu_workflow/quantize_pytorch.py` on calibration images from target dataset.
4. Run `vai_c_xir` inside Docker (`xilinx/vitis-ai-pytorch-cpu`) to produce target `.xmodel`.
