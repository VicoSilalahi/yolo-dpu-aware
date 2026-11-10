# Training on Google Colab (Custom Dataset)

You can train your DPU-compatible YOLOv8-seg model using a free **T4 GPU** on Google Colab.
**You do NOT need the 18 GB COCO dataset anymore.** You only need your custom dataset.

---

## 1. Setup in Colab (Cell 1)

Make sure your Colab runtime is set to **GPU** (`Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`).

```python
# 1. Clone your private repository (use personal access token if private)
!git clone https://<YOUR_GITHUB_TOKEN>@github.com/VicoSilalahi/yolo-dpu-aware.git
%cd yolo-dpu-aware

# 2. Install dependencies & install the modified codebase in editable mode
!pip install -e .
!pip install onnx onnxruntime
```

---

## 2. Prepare Your Custom Dataset (Cell 2)

Your custom dataset should follow standard YOLO segmentation format:

```text
custom_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Create your dataset configuration YAML (`custom_data.yaml`):

```python
yaml_content = """
path: /content/custom_dataset  # dataset root dir
train: images/train
val: images/val

# Classes
names:
  0: class_1
  1: class_2
  # add your classes here
"""

with open("custom_data.yaml", "w") as f:
    f.write(yaml_content)
```

_(Tip: Upload your dataset as a `.zip` file to Google Drive or Colab, then `!unzip -q my_dataset.zip -d /content/custom_dataset`)_

---

## 3. Fine-Tune Starting from DPU Base Model (Cell 3)

We start training directly from `checkpoints/yolov8n-seg-dpu-base.pt`. Because the $1 \times 1$ split convolutions and Hardswish activations are already embedded in the architecture, your custom dataset training will preserve 100% DPU compatibility automatically!

```python
from ultralytics import YOLO

# Load the 100% DPU-compatible pretrained base model
model = YOLO("checkpoints/yolov8n-seg-dpu-base.pt")

# Train on your custom dataset
results = model.train(
    data="custom_data.yaml",
    epochs=50,  # adjust based on dataset size
    imgsz=640,
    batch=16,  # Colab T4 (16GB VRAM) easily handles batch 16-32
    device=0,
    lr0=0.01,
    project="runs_custom",
    name="dpu_custom_model",
)
```

---

## 4. Export to ONNX on Colab (Cell 4)

Once training finishes, export the best checkpoint to DPU-ready ONNX directly in Colab:

```python
best_model = YOLO("runs_custom/dpu_custom_model/weights/best.pt")

# Export to fixed-shape ONNX (640x640)
onnx_path = best_model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
print("Exported ONNX:", onnx_path)
```

---

## 5. Download the Trained Model Back to PC/KV260

Download `best.pt` and `best.onnx` back to your PC or upload to Google Drive:

```python
from google.colab import files

files.download("runs_custom/dpu_custom_model/weights/best.pt")
files.download("runs_custom/dpu_custom_model/weights/best.onnx")
```

---

## 6. Compile to `.xmodel` for KV260

Back on your PC (or laptop with Docker):

1. Place the new `best.pt` in `checkpoints/`.
2. Run quantization calibration on ~50 images from your custom dataset:
   ```bash
   docker run --rm -v $(pwd):/workspace xilinx/vitis-ai-pytorch-cpu:latest \
     bash -c "source /opt/vitis_ai/conda/bin/activate vitis-ai-pytorch && \
     python /workspace/dpu_workflow/quantize_pytorch.py"
   ```
3. Compile to `.xmodel`:
   ```bash
   docker run --rm -v $(pwd):/workspace xilinx/vitis-ai-pytorch-cpu:latest \
     bash -c "source /opt/vitis_ai/conda/bin/activate vitis-ai-pytorch && \
     vai_c_xir -x /workspace/checkpoints/quantize_result/SegmentationModel_int.xmodel \
               -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
               -o /workspace/checkpoints/compiled_kv260 \
               -n custom_yolov8n_seg"
   ```
4. Copy `custom_yolov8n_seg.xmodel` to your KV260 and run!
