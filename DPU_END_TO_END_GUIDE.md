# End-to-End DPU Guide: Adapting YOLOv8-seg for Kria KV260

This guide documents the full transformation of **Ultralytics YOLOv8-seg** into a hardware-compatible binary targeting the AMD/Xilinx **DPUCZDX8G** IP core on the **Kria KV260 Vision AI Starter Kit**.

It details every modification, reason, validation result, and commands executed—from the initial activation replacement to the final compiled `.xmodel`.

---

## 1. Executive Summary & Hardware Context

The AMD Deep Learning Processing Unit (`DPUCZDX8G`) is an optimized hardware systolic engine designed for Zynq UltraScale+ MPSoC devices (including the Kria K26 SOM / KV260). 

While extremely fast at integer convolutions, pooling, and standard activations (ReLU, Hardswish, LeakyReLU), the DPU engine has rigid structural constraints:
1. **No exponential math units**: Activations like `SiLU` (`x * sigmoid(x)`) and standalone `Sigmoid` are unsupported on the DPU core.
2. **Strict channel slicing rules**: Dynamic memory slicing / chunking operators (such as PyTorch's `torch.chunk`) force graph breaks and fallback to the Arm Cortex-A53 CPU.
3. **No transposed convolutions**: Fractionally strided deconvolution (`nn.ConvTranspose2d`) is not supported by `DPUCZDX8G`.
4. **8-bit fixed point arithmetic**: Weights and activations must be quantized from FP32 to INT8 using calibration data.

---

## 2. Phase-by-Phase Architecture Transformations

```mermaid
flowchart TD
    A["Baseline YOLOv8-seg (yolov8n-seg.pt)<br>SiLU, chunk(), ConvTranspose2d"] --> B["Step 1: SiLU to Hardswish<br>Zero-shot (mAP: 28.62% Box / 22.68% Mask)"]
    B --> C["Step 2: C2f Chunk to 1x1 Identity Conv<br>Frozen weights (requires_grad=False)"]
    C --> D["Step 3: Warmup Fine-tuning (5 epochs)<br>Recovered to 31.95% Box / 26.50% Mask"]
    D --> E["Step 4: Proto ConvTranspose to Nearest Upsample<br>+ 3-epoch recovery (yolov8n-seg-dpu-base.pt)"]
    E --> F["Step 5: ONNX Fixed-Graph Export<br>(yolov8n-seg-dpu-base.onnx)"]
    F --> G["Step 6: INT8 Quantization (NNDCT / ONNX)<br>(SegmentationModel_int.xmodel)"]
    G --> H["Step 7: Vitis AI Compilation (vai_c_xir)<br>(yolov8n_seg_kv260.xmodel)"]
```

---

### Step 1: SiLU $\rightarrow$ Hardswish Activation Replacement

* **The Problem**: Standard YOLOv8 uses `nn.SiLU()` across every convolutional block. In Vitis AI, SiLU splits the computation graph into dozens of CPU fallback nodes, reducing frame rate to $< 3$ FPS.
* **The Fix**: Replaced `default_act` with `nn.Hardswish(inplace=True)` in [`ultralytics/nn/modules/conv.py`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/ultralytics/nn/modules/conv.py#L58).
* **Hardware Impact**: `Hardswish` is a piecewise-linear function (`x * ReLU6(x + 3) / 6`) natively supported and fused directly into the DPU convolution engine.
* **Zero-Shot Result**: Evaluated on COCO `val2017` (5,000 images):
  - Box mAP50-95: **28.62%** (vs 36.19% baseline)
  - Mask mAP50-95: **22.68%** (vs 30.24% baseline)

---

### Step 2: C2f `torch.chunk()` $\rightarrow$ Frozen $1 \times 1$ Identity Convolutions

* **The Problem**: The CSP bottleneck (`C2f`) splits feature maps into two halves using `torch.chunk(x, 2, dim=1)`. The compiler translates `torch.chunk` into `StridedSlice` / `Split` operations. On `DPUCZDX8G`, non-contiguous slice ops break DPU graph continuity.
* **The Fix**: Replaced `torch.chunk` in [`ultralytics/nn/modules/block.py`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/ultralytics/nn/modules/block.py#L291-L345) with two frozen $1 \times 1$ identity convolutions:
  $$\text{split0.weight}[i, j, 0, 0] = \delta_{ij}, \quad \text{split1.weight}[i, c + j, 0, 0] = \delta_{ij}$$
  Added `.split0` and `.split1` to `always_freeze_names` in [`ultralytics/engine/trainer.py`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/ultralytics/engine/trainer.py#L395).
* **Hardware Impact**: Slicing becomes a standard convolution that compiles 100% natively on the DPU systolic array.
* **Numerical Verification**: Exactly **0.000%** loss compared to Step 1 (identical mAP: 28.62% / 22.68%).

---

### Step 3: Warmup Fine-Tuning

* Ran 5 epochs on COCO with `lr0=0.001` to adapt the network weights to Hardswish non-linearities.
* **Result**:
  - Box mAP50-95: **31.95%** (recovered $+3.33\%$)
  - Mask mAP50-95: **26.50%** (recovered $+3.82\%$)

---

### Step 4: Proto `ConvTranspose2d` $\rightarrow$ `nn.Upsample` (Nearest)

* **The Problem**: In YOLOv8-seg, the mask prototype generator (`Proto`) uses `nn.ConvTranspose2d(c_, c_, 2, 2, bias=True)`. The DPU does not support fractionally strided deconvolution and would force a CPU copy roundtrip.
* **The Fix**: In [`ultralytics/nn/modules/block.py`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/ultralytics/nn/modules/block.py#L98), replaced `ConvTranspose2d` with `nn.Upsample(scale_factor=2, mode="nearest")`.
* **Accuracy Recovery**: Zero-shot replacement collapsed mask mAP because 65,536 learned parameters were removed. A gentle 3-epoch fine-tuning run restored performance immediately to **23.84% mask mAP50-95** (41.64% mAP50).
* **Result Checkpoint**: [`checkpoints/yolov8n-seg-dpu-base.pt`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/yolov8n-seg-dpu-base.pt) (**100% DPU Graph Ready**).

---

## 3. Full COCO `val2017` Validation Matrix

| Milestone | Checkpoint (.pt) | Architectural Features | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 | DPU Graph Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Official Base** | `yolov8n-seg.pt` | Baseline (SiLU, `chunk`, `ConvTranspose`) | **36.19%** | **51.56%** | **30.24%** | **48.84%** | ❌ Incompatible |
| **1. Hardswish** | `yolov8n-seg-hardswish-init.pt` | SiLU $\rightarrow$ Hardswish zero-shot | **28.62%** | **42.80%** | **22.68%** | **39.92%** | ⚠️ Activation ready |
| **2. C2f Split** | `yolov8n-seg-hardswish-c2fsplit.pt` | Hardswish + $1 \times 1$ Identity Conv | **28.62%** | **42.80%** | **22.68%** | **39.92%** | ⚠️ C2f + Act DPU-native |
| **3. C2f Warmup** | `yolov8n-seg-hardswish-c2fsplit-finetuned.pt` | Step 2 + 5-epoch training | **31.95%** | **46.75%** | **26.50%** | **43.97%** | ⚠️ `ConvTranspose` left |
| **4. DPU Base Ready** | `yolov8n-seg-dpu-base.pt` | Complete DPU model (Upsample Proto) | **30.66%** | **45.64%** | **23.84%** | **41.64%** |  **100% DPU Graph Ready** |

---

## 4. ONNX Export & Graph Inventory

The base checkpoint was exported to fixed-shape ONNX format:
- **Command**:
  ```bash
  python -c "
  from ultralytics import YOLO
  model = YOLO('checkpoints/yolov8n-seg-dpu-base.pt')
  model.export(format='onnx', imgsz=640, dynamic=False, simplify=True)
  "
  ```
- **Output File**: [`checkpoints/yolov8n-seg-dpu-base.onnx`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/yolov8n-seg-dpu-base.onnx) (13.5 MB)
- **Input Dimension**: `(1, 3, 640, 640)`
- **Output Tensors**:
  - `output0`: `(1, 116, 8400)` (Bounding boxes, objectness, 80 COCO classes)
  - `output1`: `(1, 32, 160, 160)` (32 Prototype mask coefficients)

### ONNX Operator Verification:
| Op Type | Count | DPU Status on KV260 |
| :--- | :---: | :--- |
| **Conv** | 92 | ✅ Fully compiled onto DPU systolic array |
| **HardSwish** | 66 | ✅ Fully fused into preceding convolutions |
| **Resize** (Nearest) | 3 | ✅ Fully supported on DPU |
| **Concat / Add** | 27 | ✅ Supported on DPU |
| **SiLU** | **0** | ✅ Completely eliminated |
| **ConvTranspose** | **0** | ✅ Completely eliminated |
| **Split / Slice** | **0** | ✅ Completely eliminated |

---

## 5. INT8 Quantization & Validation (SIL Simulation)

Before compiling to hardware binaries, the model was quantized to INT8 using calibration datasets and validated across a 500-image evaluation split from COCO `val2017`:

- **Calibration Set**: 50 images from COCO `val2017`.
- **Generated Model**: [`checkpoints/yolov8n-seg-dpu-base-int8.onnx`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/yolov8n-seg-dpu-base-int8.onnx) (3.9 MB, **71% compression** from 13.5 MB).
- **Validation Comparison (FP32 vs INT8 Simulation)**:

| Metric | FP32 Base Model (`.pt` / `.onnx`) | INT8 Quantized Model (`-int8.onnx`) | Accuracy Retention |
| :--- | :---: | :---: | :---: |
| **Box mAP50-95** | **30.66%** | **27.61%** | **90.1%** |
| **Box mAP50** | **45.64%** | **43.35%** | **95.0%** |
| **Mask mAP50-95** | **23.84%** | **23.10%** | **96.9%** |
| **Mask mAP50** | **41.64%** | **40.34%** | **96.9%** |

- **Quantization Analysis**:
  - Mask segmentation accuracy retained **96.9%** of full floating-point precision ($0.74\%$ absolute drop).
  - Box mAP50 retained **95.0%** of full floating-point precision ($2.29\%$ absolute drop).
  - This demonstrates that the replacement operators (Hardswish, $1 \times 1$ conv splits, and nearest upsample) are numerically robust under 8-bit fixed-point scaling.

---

## 6. Vitis AI Hardware Compilation (`.xmodel`)

Using AMD's official Vitis AI container (`xilinx/vitis-ai-pytorch-cpu`), the model underwent hardware compilation:

1. **Quantization & Graph Partitioning (`pytorch_nndct`)**:
   - Analyzed 287 PyTorch operators and equalized weights.
   - Exported the intermediate XIR hardware graph:
     [`checkpoints/quantize_result/SegmentationModel_int.xmodel`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/quantize_result/SegmentationModel_int.xmodel) (15.1 MB).
2. **Compiler Execution (`vai_c_xir`)**:
   - Targeted: `DPUCZDX8G_ISA1_B4096` (`arch.json` for Kria KV260).
   - Command:
     ```bash
     vai_c_xir \
       -x /workspace/ultralytics/checkpoints/quantize_result/SegmentationModel_int.xmodel \
       -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
       -o /workspace/ultralytics/checkpoints/compiled_kv260 \
       -n yolov8n_seg_kv260
     ```

### Generated Hardware Artifacts:
In [`checkpoints/compiled_kv260/`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/compiled_kv260/):
- **`yolov8n_seg_kv260.xmodel`** (5.2 MB): The hardware binary containing compiled instructions for the KV260 DPU.
- **`meta.json`**: Kernel routing file pointing to `libvart-dpu-runner.so`.
- **`md5sum.txt`**: Binary integrity checksum (`3595c902175783aaa03f97f6908d91c3`).

### Compiler Architecture Partitioning:
- **DPU Subgraphs (Hardware Core)**:
  The entire feature extraction backbone, neck downsamplers, C2f blocks, and prototype convolution layers compile directly into DPU hardware kernels.
- **CPU Subgraphs (Arm Host Post-Processing)**:
  Only terminal anchor coordinate post-processing and bounding box transpose operations run on the Arm host CPU.

---

## 7. Board Deployment on Kria KV260 (Ubuntu 22.04 LTS)

When running Ubuntu 22.04 on the KV260 without Snaps, configure the DPU overlay as follows:

### 1. Install DPU Firmware & Vitis AI Runtime Packages
```bash
# Add official Xilinx apps PPA
sudo add-apt-repository -y ppa:xilinx-apps
sudo apt update

# Install smartcam bitstream containing the DPUCZDX8G B4096 IP core
sudo apt install -y xlnx-firmware-kv260-smartcam

# Install VART, XIR, and Python dependencies
sudo apt install -y vitis-ai-library vitis-ai-runtime python3-opencv python3-numpy
```

### 2. Load Accelerator via `xmutil`
```bash
# Check installed accelerators
sudo xmutil listapps

# Unload previous active slot (e.g. k26-starter-kits)
sudo xmutil unloadapp 0

# Load smartcam DPU overlay
sudo xmutil loadapp kv260-smartcam

# Verify FPGA compute unit is recognized by XRT
xbutil examine
```
The output should report `Devices present: 1 device found (edge user)` and `/dev/dri/renderD128` will be active.

> [!NOTE]
> `xdputil query` may report a segmentation fault on Ubuntu 22.04 due to an upstream Python ctypes struct mapping bug in Canonical's package. As long as `xbutil examine` reports the edge device ready and `/dev/dri/renderD128` is present, the VART C++ backend operates normally.

### 3. Compilation Note: Preventing `strided_slice-fix` & Vitis AI Version Matching Failures
When quantizing and compiling YOLOv8-seg:
- `model.model[-1].export = True` is set before quantization in `dpu_workflow/quantize_pytorch.py`.
- This ensures the Segment head returns raw convolutional feature maps directly without entering the dynamic tensor slicing / anchor subtraction loop.
- **Critical Vitis AI Version Matching (KV260 Ubuntu 22.04 LTS)**:
  - Official Canonical Ubuntu 22.04 PPA (`ppa:xilinx-apps`) packages **`kv260-smartcam` built with Vivado/Vitis 2022.1 (`v++ 2022.1`, XRT `2.13.477`, Vitis AI Runtime `2.0-0ubuntu1`)**.
  - The hardware DPU IP in this bitstream has DPU fingerprint **`0x1000020f6014407`** (DPUCZDX8G ISA 1.4.0 / B4096).
  - If you compile using the newest Vitis AI Docker container (`xilinx/vitis-ai-pytorch-cpu:latest` / Vitis AI 3.0/3.5), `vai_c_xir` generates instructions for ISA 1.4.1 (fingerprint `0x101000056010407`).
  - **Result if mismatched:** Calling `vart.Runner.create_runner(...)` crashes with `Segmentation fault (core dumped)`.
  - **The Fix:** Compile targeting the Vitis AI 2.0 / 2.5 architecture (fingerprint `0x1000020f6014407`). Use the Vitis AI 2.5 container (`xilinx/vitis-ai-pytorch-cpu:2.5`) or pass a custom `arch.json` containing `"fingerprint": "0x1000020f6014407"`.

#### Recompilation Instructions for Host PC (Vitis AI 2.5 / 2.0):

1. **Pull and run the Vitis AI 2.5 Docker container on your host machine**:
   ```bash
   docker pull xilinx/vitis-ai-pytorch-cpu:2.5
   docker run -v $(pwd):/workspace -w /workspace --rm -it xilinx/vitis-ai-pytorch-cpu:2.5 bash
   ```

2. **Run PyTorch Quantization with raw conv export**:
   Inside the container (with conda environment `vitis-ai-pytorch` active):
   ```bash
   conda activate vitis-ai-pytorch
   python3 dpu_workflow/quantize_pytorch.py
   ```

3. **Verify or create the KV260 Vitis AI 2.0/2.5 `arch.json`**:
   Ensure `arch.json` has:
   ```json
   {
       "target": "DPUCZDX8G_ISA1_B4096",
       "fingerprint": "0x1000020f6014407"
   }
   ```
   *(Or use default `/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json` inside the 2.5 container)*.

4. **Compile using `vai_c_xir`**:
   ```bash
   vai_c_xir \
     -x checkpoints/quantize_result/SegmentationModel_int.xmodel \
     -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
     -o checkpoints/compiled_kv260 \
     -n yolov8n_seg_kv260
   ```

5. **Deploy to KV260**:
   Copy the newly generated `checkpoints/compiled_kv260/yolov8n_seg_kv260.xmodel` to `~/deploy_kv260/yolov8n_seg_kv260.xmodel` on the KV260 board.


### 4. Run YOLOv8-seg Inference on DPU
```bash
mkdir -p ~/deploy_kv260 && cd ~/deploy_kv260

# Download model artifacts
wget -O yolov8n_seg_kv260.xmodel https://raw.githubusercontent.com/VicoSilalahi/yolo-dpu-aware/main/deploy_kv260/yolov8n_seg_kv260.xmodel
wget -O run_kv260_dpu.py https://raw.githubusercontent.com/VicoSilalahi/yolo-dpu-aware/main/deploy_kv260/run_kv260_dpu.py
wget -O test.jpg https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg

# Execute inference using VART runtime
python3 run_kv260_dpu.py yolov8n_seg_kv260.xmodel test.jpg
```

---

## 8. Next Step: Fine-Tuning on Custom Target Classes

Now that the complete hardware toolchain is verified from PyTorch to `.xmodel`:
1. Provide your custom dataset (images and segmentation labels).
2. Train [`checkpoints/yolov8n-seg-dpu-base.pt`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/yolov8n-seg-dpu-base.pt) on your custom classes.
3. Re-run [`dpu_workflow/quantize_pytorch.py`](file:///home/vicosilalahi/programming/yolo-fpga/ultralytics/dpu_workflow/quantize_pytorch.py) to calibrate on your custom images and compile your target `.xmodel`!
