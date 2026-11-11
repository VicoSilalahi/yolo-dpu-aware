# Instructions for Recompiling YOLOv8-seg for Kria KV260 (Vitis AI 2.5)

## Problem Summary
The Kria KV260 board runs Ubuntu 22.04 with `kv260-smartcam` built with Vivado 2022.1 / Vitis AI 2.0 (`DPUCZDX8G_ISA1_B4096`, fingerprint `0x1000020f6014407`). 

Because Vitis AI 3.x dropped support for this ISA version and generates `0x101...` binaries, `vart.Runner.create_runner` crashes with a native `Segmentation fault (core dumped)` on the KV260. We must quantize and compile using the **Vitis AI 2.5 Docker image** (`xilinx/vitis-ai-pytorch-cpu:2.5`).

---

## Commands to Run on the Host PC (Linux / WSL)

Inside the repository directory (`/home/vicosilalahi/programming/yolo-fpga/ultralytics`):

### 1. Launch the Vitis AI 2.5 Docker Container
```bash
docker run -v $(pwd):/workspace -w /workspace --rm -it xilinx/vitis-ai-pytorch-cpu:2.5 bash
```

### 2. Quantize PyTorch Model (Raw Conv Head Mode)
Inside the container:
```bash
conda activate vitis-ai-pytorch
python3 dpu_workflow/quantize_pytorch.py
```
*(Ensure `model.model[-1].export = True` is active in `quantize_pytorch.py` so raw conv feature maps are output without dynamic slicing).*

### 3. Compile with Native KV260 Architecture
```bash
vai_c_xir \
  -x checkpoints/quantize_result/SegmentationModel_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o checkpoints/compiled_kv260 \
  -n yolov8n_seg_kv260
```

### 4. Verify Fingerprint Matches the KV260 Hardware
```bash
python3 -c "import xir; g=xir.Graph.deserialize('checkpoints/compiled_kv260/yolov8n_seg_kv260.xmodel'); dpu=[s for s in g.get_root_subgraph().toposort_child_subgraph() if s.has_attr('device') and s.get_attr('device')=='DPU'][0]; print('Fingerprint:', hex(dpu.get_attr('dpu_fingerprint')))"
```
> **Requirement**: The output **MUST** start with `0x100...` (e.g., `0x1000020f6014407` or `0x1000020f6014406`), NOT `0x101...`.

### 5. Deploy Artifacts and Push to GitHub
```bash
cp checkpoints/compiled_kv260/yolov8n_seg_kv260.xmodel deploy_kv260/yolov8n_seg_kv260.xmodel
git add deploy_kv260/yolov8n_seg_kv260.xmodel checkpoints/compiled_kv260/
git commit -m "build: compile yolov8n_seg_kv260.xmodel for KV260 using Vitis AI 2.5 (fingerprint 0x100...)"
git push
```

---

## On-Board Verification (KV260)
After pushing, run on the KV260:
```bash
cd ~/deploy_kv260
wget -O yolov8n_seg_kv260.xmodel https://raw.githubusercontent.com/VicoSilalahi/yolo-dpu-aware/main/deploy_kv260/yolov8n_seg_kv260.xmodel
python3 run_kv260_dpu.py yolov8n_seg_kv260.xmodel test.jpg
```
