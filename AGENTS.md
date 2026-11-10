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

## 4. Current Status, Issues Encountered & Immediate Next Steps

### What Has Been Accomplished:
1. **Board Provisioning Verified on Ubuntu 22.04 LTS (Jammy)**:
   - Successfully added the official PPA: `sudo add-apt-repository -y ppa:xilinx-apps && sudo apt update`.
   - Installed the DPU bitstream package: `sudo apt install -y xlnx-firmware-kv260-smartcam`.
   - Installed the runtime libraries: `sudo apt install -y vitis-ai-library vitis-ai-runtime python3-opencv python3-numpy`.
   - Replaced flat starter kit overlay with smartcam overlay:
     ```bash
     sudo xmutil unloadapp 0
     sudo xmutil loadapp kv260-smartcam
     ```
   - Hardware detection confirmed via `xbutil examine`:
     ```text
     Devices present: [0000:00:00.0] edge user(inst=0) Yes
     Driver node: /dev/dri/renderD128 is active.
     ```

### What Went Wrong & Root Cause:
When attempting to execute `python3 run_kv260_dpu.py yolov8n_seg_kv260.xmodel test.jpg` on the KV260, the VART deserializer crashed with:
```text
[UNILOG][FATAL][XIR_INVALID_ATTR_OCCUR][Invalid attr occurence!] 
xir::Op{name = SegmentationModel__SegmentationModel_Segment_model__Segment_22__18300, type = strided_slice-fix} : 
Attr shape has type REQUIRED, but not set.
```
- **Root Cause**: 
  1. The target board runs **Vitis AI 2.0 runtime (`vitis-ai-runtime 2.0-0ubuntu1`)**, while compilation was done with **Vitis AI 3.x (`vai_c_xir`)**.
  2. More critically, the initial quantization script traced the **entire end-to-end forward graph**, which included the anchor decoding loop and channel slicing inside `model.22` (`Segment`).
  3. These post-processing slice operations became an untyped CPU fallback op (`strided_slice-fix`) that Vitis AI 2.0 cannot deserialize.
  4. Note: `xdputil query` also segfaults on Ubuntu 22.04 due to an upstream Python ctypes struct mapping bug in Canonical's package, but this is bypassed once the `.xmodel` is clean.

### What is Required to Be Done Next (Host / Dev PC with Vitis AI Docker):

Because compilation requires the Vitis AI Docker container (`xilinx/vitis-ai-pytorch-cpu`), run the following on your Linux / WSL dev machine:

1. **Enable Raw Conv Export Mode during Quantization** (Already patched in `dpu_workflow/quantize_pytorch.py`):
   ```python
   # Sets Segment head to return raw conv tensors, skipping anchor slicing & strided_slice-fix
   model.model[-1].export = True
   ```
2. **Re-run INT8 Calibration & Export inside Docker**:
   ```bash
   python3 dpu_workflow/quantize_pytorch.py
   ```
3. **Compile Clean DPU Binary**:
   ```bash
   vai_c_xir \
     -x checkpoints/quantize_result/SegmentationModel_int.xmodel \
     -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
     -o checkpoints/compiled_kv260 \
     -n yolov8n_seg_kv260
   ```
   *Result*: A pure DPU hardware binary without CPU fallback subgraphs or `strided_slice-fix` operators.
4. **Push to GitHub**:
   Commit and push `yolov8n_seg_kv260.xmodel` to GitHub so the KV260 can download it via `wget`.
5. **Update Post-Processing in `run_kv260_dpu.py`**:
   Anchor decoding and non-maximum suppression (NMS) are handled cleanly on the ARM CPU in Python/NumPy.

---

## 5. Board Deployment Quick Reference (KV260)

```bash
# 1. Unload & load DPU overlay
sudo xmutil unloadapp 0
sudo xmutil loadapp kv260-smartcam

# 2. Download new xmodel
cd ~/deploy_kv260
wget -O yolov8n_seg_kv260.xmodel https://raw.githubusercontent.com/VicoSilalahi/yolo-dpu-aware/main/deploy_kv260/yolov8n_seg_kv260.xmodel

# 3. Run inference
python3 run_kv260_dpu.py yolov8n_seg_kv260.xmodel test.jpg
```
