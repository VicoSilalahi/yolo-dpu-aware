# YOLOv8-Seg on AMD Kria KV260 (DPUCZDX8G)

A DPU-aware fork of Ultralytics YOLOv8-seg, modified so the full network compiles to an
AMD/Xilinx `DPUCZDX8G` hardware binary (`.xmodel`) and runs on the Kria KV260.

**Verified on hardware:** both B3136 and B4096 run on a KV260. Steady-state DPU-only
latency is **42.16 ms p50 (23.7 FPS)** for B3136 and **27.55 ms p50 (36.3 FPS)** for
B4096 — a **1.53x** speedup. CPU preprocessing is currently the pipeline bottleneck.

![YOLOv8-seg result on KV260](assets/result_b3136.jpg)

```
bus      0.88
person   0.82
person   0.78
person   0.68
```

---

## Verified target

| Item | Value |
| :--- | :--- |
| Board | AMD Kria KV260 Vision AI Starter Kit |
| DPU IP | `DPUCZDX8G_ISA1_B3136` (smartcam) / `DPUCZDX8G_ISA1_B4096` (benchmark) |
| DPU fingerprint | `0x101000016010406` B3136 / `0x101000016010407` B4096 |
| Runtime | Vitis AI / VART / XIR **2.5.0** (`xilinx/smartcam:2022.1`) |
| Compiler target | `dpu_workflow/arch_kv260_isa1_b3136.json` / `arch_kv260_isa1_b4096.json` |
| Model input | `[1, 640, 640, 3]` xint8 |
| Model outputs | 10 raw convolution tensors (box, class, mask coefficients, proto) |
| Measured DPU latency (p50) | **B3136: 42.16 ms (23.7 FPS) · B4096: 27.55 ms (36.3 FPS)** |
| Xmodel | `checkpoints/compiled_kv260_b3136/` and `checkpoints/compiled_kv260_b4096/` |
| Xmodel MD5 | B3136 `b29b1b05c03571d5860d8887f4a26c8e` · B4096 `8bbacbee9e7f67219f4808ddc70159bc` |

> The earlier **79.62 ms** figure was a cold single-shot run. Steady-state p50 after
> warmup is **42.16 ms** on B3136. Both B3136 and B4096 produced identical detections
> on the reference image.

### Measured performance

300 iterations after 30 warmup iterations, same input, DPU-only p50:

| Configuration | DPU p50 | DPU FPS | Preprocess | Full p50 | Full FPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| B3136 + smartcam | 42.16 ms | 23.7 | 52.6 ms (float) | 94.81 ms | 10.6 |
| B4096 + benchmark | 27.55 ms | 36.3 | 53.6 ms (float) | 81.14 ms | 12.3 |
| B3136 + LUT preprocess | 42.20 ms | 23.7 | 5.16 ms | **47.63 ms** | **21.0** |
| **B4096 + LUT preprocess** | **27.61 ms** | **36.2** | **5.18 ms** | **33.11 ms** | **30.2** |

The LUT rows are measured closed-loop (preprocess + DPU per frame), not estimates.
The integer LUT path is **bit-exact** with the float path and ~9x faster on the A53.
See [PERFORMANCE_AND_B4096.md](PERFORMANCE_AND_B4096.md).

> `B3136` is the verified smartcam path. `B4096` is the DPU-only benchmark overlay;
> use the B3136 smartcam bitstream if you need the camera/display pipeline.

---

## Accuracy

COCO `val2017` (5,000 images, 640x640):

| Model stage | File | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Official baseline | `yolov8n-seg.pt` | 36.19% | 51.56% | 30.24% | 48.84% |
| + Hardswish | `yolov8n-seg-hardswish-init.pt` | 28.62% | 42.80% | 22.68% | 39.92% |
| + C2f identity split | `yolov8n-seg-hardswish-c2fsplit.pt` | 28.62% | 42.80% | 22.68% | 39.92% |
| + warmup fine-tune | `yolov8n-seg-hardswish-c2fsplit-finetuned.pt` | 31.95% | 46.75% | 26.50% | 43.97% |
| **DPU base (FP32)** | `yolov8n-seg-dpu-base.pt` | **30.66%** | **45.64%** | **23.84%** | **41.64%** |
| **INT8 quantized** | `yolov8n-seg-dpu-base-int8.onnx` | **27.61%** | **43.35%** | **23.10%** | **40.34%** |

The INT8 model retains **96.9%** of the FP32 mask mAP and **90.1%** of the box mAP.
Per-run metrics are stored in `metrics/`.

> `.pt` and `.onnx` files are gitignored. Download or regenerate them before training,
> quantization, or compilation.

### Crack detection (crack-01)

Roboflow [`crack-01-rshs1`](https://universe.roboflow.com/crack-detection-slmz7/crack-01-rshs1) v17
(4,347 train / 388 val / 242 test images, single class `crack`, 640x640, FP32), fine-tuned from
the DPU base for 50 epochs:

| Split | File | Images | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| val | `runs/best.pt` | 388 | 47.64% | 63.87% | 26.08% | 59.31% |
| **test (held out)** | `runs/best.pt` | 242 | **55.10%** | **74.13%** | **29.51%** | **62.45%** |

Box precision/recall is 80.1/60.7 on val and 77.0/70.1 on test; masks 74.8/58.7 and 72.5/61.5.
Mask mAP50-95 is low because crack masks are thin — the model finds the crack (mAP50) but tight
IoU thresholds punish the boundaries. These numbers are not comparable to the COCO rows above.

Reproduce with `validate.py --weights runs/best.pt --data <crack-01>/data.yaml` (add `--split test`
for the held-out split); the per-run JSONs are `metrics/results_crack01_fp32.json` and
`metrics/results_crack01_test_fp32.json`.

#### Crack + spalling (2-class)

[`notebooks/train_crack_spalling_kaggle.ipynb`](notebooks/train_crack_spalling_kaggle.ipynb) trains a
two-class (`crack`, `spalling`) YOLOv8n-seg model on Kaggle (T4 ×2, ~2 h); the
[Colab](notebooks/train_crack_spalling_colab.ipynb) and
[local](notebooks/train_crack_spalling_local.ipynb) variants run the same pipeline. They download the
datasets, merge them by class name into one YOLO-seg dataset, warm-start from the 1-class `best.pt`
(the class head is rebuilt; backbone, neck and the class-agnostic mask branch transfer), then validate
`val` and the held-out `test` split, write per-class JSON, and **sync `last.pt`/`best.pt` to
`runs_backup/<run>_live/` and export a zip** so a killed session cannot lose the results.

Run `concrete-2cls` (2026-11-13, warm start from the 1-class `crack-best.pt`): 100 epochs planned,
early-stopped after 96 with the best checkpoint at epoch 66; batch 16 per GPU, 640x640, AdamW
`lr0=1e-3`, `lrf=0.01`, seed 0, AMP. Weights:
[`yolov8n-seg-concrete-2cls.pt`](https://github.com/VicoSilalahi/yolo-dpu-aware/releases/download/weights-v2/yolov8n-seg-concrete-2cls.pt)
(7.4 MB, SHA256 `EF4534F3E531683FE42D188AC1D60D2E3A519901D5C591ACCE7EDA709C1D848D`), attached to
release [`weights-v2`](https://github.com/VicoSilalahi/yolo-dpu-aware/releases/tag/weights-v2);
`*.pt` stays gitignored. An ONNX export of the same checkpoint is attached as
`yolov8n-seg-concrete-2cls.onnx` (exported locally with Ultralytics 8.4.208, opset 12, static
1x3x640x640, and numerically checked against the `.pt`).

| Split | Images | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| val | — | 63.88% | 78.92% | 51.52% | 76.41% |
| **test (held out)** | 414 | **71.25%** | **86.29%** | **55.31%** | **79.88%** |

Per class on the test split:

| Class | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| crack | 83.5% | 67.2% | 74.66% | 56.36% | 76.0% | 60.8% | 61.83% | 28.21% |
| spalling | 99.5% | 94.0% | 97.93% | 86.13% | 99.5% | 93.7% | 97.93% | 82.40% |

Caveats: the test split is only fully native for `crack-01-rshs1`; the two spalling sources ship no
test split, so a deterministic third of their `val` images is moved into `test` (never trained on,
but not an independent external set), and `test` scoring above `val` reflects that mix. Mask
mAP50-95 on `crack` is low for the same thin-mask reason as the single-class model above. These are
FP32 numbers. The checkpoint has since been quantized and compiled:

- `checkpoints/quantize_result_concrete_2cls/` — Vitis AI 3.5 NNDCT, 50 calibration images.
- `checkpoints/compiled_kv260_b3136_concrete_2cls/yolov8n_seg_concrete_2cls_kv260_b3136.xmodel`
  — target `DPUCZDX8G_ISA1_B3136`, MD5 `29326065c560dc51a1a45883390209ce`.
- `checkpoints/compiled_kv260_b4096_concrete_2cls/yolov8n_seg_concrete_2cls_kv260_b4096.xmodel`
  — target `DPUCZDX8G_ISA1_B4096`, MD5 `583a2e90fe93b82dbd4fe89a79d9f0a5`.

Both binaries use the same 1-USER / 1-DPU / 10-CPU subgraph layout as the base model; the class
outputs have 2 channels and the input is `[1,640,640,3] xint8 fix_point=6`.

Reproduce with `python validate.py --weights runs_backup/concrete-2cls_live/best.pt --data
<merged>/data.yaml --split test --full-precision`. The per-run JSONs are
`metrics/results_concrete_2cls_val_fp32.json`, `metrics/results_concrete_2cls_test_fp32.json` and
`metrics/results_concrete_2cls_test_per_class_fp32.json`. Restore the checkpoint with:

```bash
wget -q https://github.com/VicoSilalahi/yolo-dpu-aware/releases/download/weights-v2/yolov8n-seg-concrete-2cls.pt -O runs_backup/concrete-2cls_live/best.pt
```

**Training progress.** The released `best.pt` is the epoch-66 checkpoint, not the final one. This
fork ranks checkpoints by `box mAP50-95 + mask mAP50-95`, and the checkpoint's stored
`train_metrics` (63.94% / 78.94% box, 51.67% / 76.45% mask) exactly match the epoch-66 row of
`results.csv`. The run stopped at 96 of the 100 planned epochs: fitness peaked at epoch 66, did not
improve for the next 30 epochs (patience 30), and early stopping fired at 96. The epoch-96
checkpoint was slightly worse (63.48% / 77.84% box, 51.29% / 75.54% mask), so `last.pt` is not a
substitute. Mask mAP50-95 alone peaked later, at epoch 90 (51.80% vs 51.67%), and box mAP50 alone at
epoch 54 (79.51% vs 78.94%), but epoch 66 wins the combined score. The val row above is the
post-training full-precision re-validation, so it differs slightly from these AMP training-time
values. The unmodified per-epoch log is attached to the release as `concrete-2cls-results.csv`.

`*.pt` files are gitignored, so the clone has no weights: upload `runs/best.pt` (7.1 MB) as a Kaggle
dataset and add it with *Add Input* (Colab: put it in Drive under `MyDrive/yolo-dpu-aware/`). The
notebook scans `/kaggle/input` for `*best*.pt`, keeps only checkpoints whose classes are
`crack`/`spalling`, and refuses to start from an unrelated (e.g. COCO) model.

The merge is name-based, so source class ids do not have to agree:

| Source | Version | Images | Classes | License | Attribution |
| :--- | :---: | :---: | :--- | :--- | :--- |
| [`crack-01-rshs1`](https://universe.roboflow.com/crack-detection-slmz7/crack-01-rshs1) | v17 | 4,977 | `crack` | MIT | Crack Detection (Roboflow Universe) |
| [`concrete-crack-spalling`](https://universe.roboflow.com/1-7frt7/concrete-crack-spalling) | v2 | 1,390 | `crack`, `spalling` | CC BY 4.0 | Roboflow Universe workspace `1-7frt7` |
| [`concrete-spalling-sm5v3`](https://universe.roboflow.com/1-7frt7/concrete-spalling-sm5v3) (optional) | v1 | 999 | `spalling` | CC BY 4.0 | Roboflow Universe workspace `1-7frt7` |

`concrete-crack-spalling` and `concrete-spalling-sm5v3` are **CC BY 4.0**: keep the attribution above
in any published model card, dataset or paper that uses them. Neither project ships a test split, so
the notebook moves a deterministic third of each one's `val` images into `test`. The notebook never
stores a Roboflow API key — it prompts for one at runtime.

---

## DPU compatibility changes

All three changes are already applied in this repository:

| Original | Problem on DPUCZDX8G | Replacement |
| :--- | :--- | :--- |
| `nn.SiLU` | No exponential ALU; CPU fallback | `nn.Hardswish` in `ultralytics/nn/modules/conv.py` |
| `torch.chunk` in `C2f` | Dynamic slice/split breaks the DPU graph | Two frozen 1x1 identity convs (`split0`, `split1`) in `ultralytics/nn/modules/block.py` |
| `nn.ConvTranspose2d` in `Proto` | Transposed convolution unsupported | `nn.Upsample(scale_factor=2, mode="nearest")` in `ultralytics/nn/modules/block.py` |

The frozen split layers are registered in `always_freeze_names` inside
`ultralytics/engine/trainer.py` so fine-tuning cannot unfreeze them.

Resulting ONNX operator inventory: `Conv` 92, `HardSwish` 66, `Resize` 3,
`SiLU` 0, `ConvTranspose` 0, `Split/Slice` 0.

### Extending to YOLO11 / YOLO26

The vendored `ultralytics` package keeps all model families, including YOLO11 and
YOLO26. To target the DPU with those models, apply the same three substitutions to
their building blocks (notably the `C3k2` split path) and re-run the pipeline below.

Karki, Ahmed, and Jungeblut do exactly this for YOLOv11n and YOLOv26n on AMD DPUs
(B512-B4096): they replace unsupported activations, map split operations to 1x1
convolutions, and approximate spatial attention with convolution and element-wise
operations, reaching 34.05 FPS for YOLOv26n. See the references in
[PERFORMANCE_AND_B4096.md](PERFORMANCE_AND_B4096.md#6-references).

---

## Repository layout

```
├── README.md
├── assets/                        # result images
├── DPU_END_TO_END_GUIDE.md        # detailed technical guide and root-cause notes
├── PERFORMANCE_AND_B4096.md       # ground-station split and B4096 targeting notes
├── notebooks/
│   ├── train_crack_spalling_kaggle.ipynb  # 2-class run on T4 x2 (~2 h)
│   ├── train_crack_spalling_colab.ipynb   # same pipeline on Colab + Drive export
│   └── train_crack_spalling_local.ipynb   # same pipeline on the 4 GB GTX 1650
├── train.py                       # fine-tuning entry point
├── validate.py                    # COCO mAP evaluation
├── coco_*.yaml                    # dataset configs (full / 500 / train)
├── ultralytics/                   # modified Ultralytics source (v8/v11/v26 models)
├── dpu_workflow/
│   ├── arch_kv260_isa1_b3136.json # compiler target for this board
│   ├── quantize_pytorch.py        # Vitis AI NNDCT INT8 quantization
│   ├── quantize_onnx.py           # optional ONNX Runtime INT8 path
│   ├── verify_simulation.py       # FP32 vs INT8 ONNX comparison
│   └── eval_int8_coco.py          # INT8 COCO evaluation
├── deploy_kv260/
│   ├── run_kv260_dpu.py           # Python VART runner (raw tensor output + timing)
│   ├── run_kv260_dpu.cpp          # readable C++ VART/XIR runner (build with make)
│   ├── Makefile                   # builds the C++ runner
│   └── postprocess_b3136.py       # DFL/NMS/mask decode -> annotated JPEG
├── checkpoints/
│   ├── compiled_kv260_b3136/      # verified hardware binary + meta.json
│   └── quantize_result/           # INT8 intermediate XIR graph
└── metrics/                       # JSON validation results
```

---

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install -e .
pip install onnx onnxruntime
```

Vitis AI quantization and compilation use two locally cached AMD images:

| Stage | Image | Toolchain |
| :--- | :--- | :--- |
| NNDCT quantization | `xilinx/vitis-ai-pytorch-cpu:latest` (digest `sha256:baf43fd9d6e1ea5a2c12b727ac3aafd15ead936172bcd636d7af98b0cca49298`) | Vitis AI 3.5.0, Python 3.8, torch 1.13.1 |
| `vai_c_xir` compile | `xilinx/vitis-ai-cpu:2.5` | Vitis AI 2.5.0 |

The `3.5.0` image is needed because the vendored Ultralytics 8.4.208 source requires
Python 3.8+. The 2.5 image ships Python 3.7 and cannot import it. Reuse the cached
images; do not pull a newer tag.

---

## 1. Train / fine-tune

Start from the DPU-compatible base checkpoint so the architecture stays hardware-ready:

```bash
python train.py \
  --weights checkpoints/yolov8n-seg-dpu-base.pt \
  --data custom_data.yaml \
  --epochs 50 --batch 4 --imgsz 640
```

`train.py` defaults are tuned for a 4 GB GPU (batch 4). For large GPUs or Colab,
raise `--batch` to 16-32.

## 2. Validate

```bash
python validate.py --weights checkpoints/yolov8n-seg-dpu-base.pt --data coco_full_fast.yaml
python validate.py --weights checkpoints/yolov8n-seg-dpu-base-int8.onnx --data coco_full_fast.yaml
```

`coco_full_fast.yaml` validates all 5,000 `val2017` images while avoiding the
forced COCO RLE JSON step that makes segmentation validation very slow.

## 3. Export ONNX

```bash
python -c "
from ultralytics import YOLO
YOLO('checkpoints/yolov8n-seg-dpu-base.pt').export(
    format='onnx', imgsz=640, dynamic=False, simplify=True
)"
```

Expected outputs: `(1, 116, 8400)` boxes/classes and `(1, 32, 160, 160)` prototypes.

## 4. Quantize (INT8)

```bash
# Run from the directory that contains both ultralytics/ and datasets/.
docker run --rm -it -v "$PWD":/workspace \
  xilinx/vitis-ai-pytorch-cpu:latest \
  bash -lc "source /opt/vitis_ai/conda/bin/activate vitis-ai-pytorch && \
    python /workspace/ultralytics/dpu_workflow/quantize_pytorch.py \
      --weights /workspace/ultralytics/checkpoints/concrete-2cls/yolov8n-seg-concrete-2cls.pt \
      --calib-dir /workspace/datasets/concrete-2cls/val/images \
      --output-dir /workspace/ultralytics/checkpoints/quantize_result_concrete_2cls \
      --num-calib 50"
```

The script takes `--weights`, `--calib-dir`, `--output-dir`, `--num-calib`,
`--imgsz`, `--bitwidth-w`, and `--bitwidth-a`. For the base COCO model, omit the
arguments and it uses the defaults in `dpu_workflow/quantize_pytorch.py`. The script
truncates the detection head so it emits the 10 raw convolution tensors, keeping the
whole graph on the DPU.

> **The shipped model is already quantized.** `yolov8n_seg_kv260_b3136.xmodel` is an
> INT8 graph: all weights and activations are 8-bit (`quant_info.json` records
> `bitwidth_w=8`, `bitwidth_a=8`, `bias_corrected=true`), and the DPU input is
> `xint8` with `fix_point=6`. The 2-class binaries in
> `checkpoints/compiled_kv260_*_concrete_2cls/` have the same INT8 input contract.
> Nothing in `deploy_kv260/` re-quantizes the model. The integer LUT there only
> converts 8-bit camera pixels into that model's int8 input domain; it does not touch
> weights or activations.

## 5. Compile the xmodel

```bash
# Run from the ultralytics/ repository root.
docker run --rm -it -v "$PWD":/workspace \
  xilinx/vitis-ai-cpu:2.5 \
  bash -lc "source /opt/vitis_ai/conda/bin/activate vitis-ai-pytorch && \
    vai_c_xir \
      -x /workspace/checkpoints/quantize_result/SegmentationModel_int.xmodel \
      -a /workspace/dpu_workflow/arch_kv260_isa1_b3136.json \
      -o /workspace/checkpoints/compiled_kv260_b3136 \
      -n yolov8n_seg_kv260_b3136"
```

`vai_c_xir` comes from the 2.5 image; the quantized XIR graph is produced by the 3.5
NNDCT. For the 2-class model use
`checkpoints/quantize_result_concrete_2cls/SegmentationModel_int.xmodel`,
`checkpoints/compiled_kv260_b3136_concrete_2cls`, and
`yolov8n_seg_concrete_2cls_kv260_b3136`.

Verify `meta.json` reports `"target": "DPUCZDX8G_ISA1_B3136"` and compare
`md5sum.txt` with the generated binary.

---

## 6. Deploy on the KV260

```bash
# On the board: load the DPU overlay
sudo xmutil unloadapp 0
sudo xmutil loadapp kv260-smartcam
xbutil examine          # device should report Ready: Yes

# Copy deploy_kv260/ and the .xmodel to ~/deploy_kv260 on the board, then:
cd ~/deploy_kv260
docker run --rm -it --privileged \
  -v /dev:/dev -v /sys:/sys \
  -v /lib/firmware/xilinx:/lib/firmware/xilinx \
  -v "$PWD":/workspace -w /workspace \
  -e PYTHONPATH=/workspace/python_libs \
  xilinx/smartcam:2022.1 bash
```

Inside the container:

```bash
# Raw tensors + latency (Python)
python3 run_kv260_dpu.py yolov8n_seg_kv260_b3136.xmodel test.jpg

# Annotated segmentation result (Python)
python3 postprocess_b3136.py yolov8n_seg_kv260_b3136.xmodel test.jpg result.jpg

# Readable C++ runner: decode, NMS, masks, timing, annotated JPEG
make
./run_kv260_dpu yolov8n_seg_concrete_2cls_kv260_b3136.xmodel test.jpg \
  --classes crack,spalling
```

The C++ runner is a single file, `run_kv260_dpu.cpp`. It prints the DPU input and
output layout, runs warmup plus timed iterations (`--bench`, `--warmup`), reports
mean/p50/p90/FPS, then decodes DFL boxes, class scores, mask coefficients, and the
proto mask. Useful options:

```text
-o/--output <path>   annotated image (default result_cpp.jpg)
-c/--conf <float>    confidence threshold (default 0.25)
-i/--iou <float>     NMS IoU threshold (default 0.45)
--classes a,b,c      class names; otherwise CLASSES or class_0...
--inspect            print the DPU I/O layout and exit (works without a DPU)
--no-draw            benchmark only, write no image
```

The Makefile detects the system VART/XIR headers first, then the Vitis AI conda
environment, and accepts either `pkg-config opencv4` or `pkg-config opencv`.
`--inspect` is useful on a workstation because it reads the XIR metadata without
opening `/dev/dpu`.

Build on the KV260 itself (aarch64), not on the x86 workstation: the resulting binary
must match the board's architecture. The build environment needs `g++`, `make`,
OpenCV development headers, and the VART/XIR headers. If the `smartcam` container does
not provide them, build in the aarch64 Vitis AI SDK/sysroot that matches the board
runtime, then copy the binary and source next to the `.xmodel`.

`postprocess_b3136.py` performs DFL box decoding, sigmoid scoring, class-aware NMS,
and `mask_coeff x proto` mask generation on the board, then writes an annotated JPEG.
It reads the class count from the xmodel; set `CLASSES=crack,spalling` for the
2-class concrete model.

---

## Optional: B4096 and ground-station post-processing

The KV260 can also load AMD's DPU-only **B4096 benchmark overlay**
(`xlnx-firmware-kv260-benchmark-b4096`). On this board it reports
`DPUCZDX8G_ISA1_B4096`, fingerprint `0x101000016010407`, 300 MHz, ISA1/ALU.

Measured results (same model, same input, 300 iterations after 30 warmup):

| Configuration | DPU p50 | DPU FPS | Speedup |
| :--- | ---: | ---: | ---: |
| B3136 smartcam | 42.16 ms | 23.7 | 1.00x |
| **B4096 benchmark** | **27.55 ms** | **36.3** | **1.53x** |

Both overlays produced identical detections on the reference image. The practical
bottleneck is now CPU preprocessing: the Python float path costs ~53 ms/frame, while
an integer LUT path costs **5.16 ms** and is bit-exact. With the LUT path the
B4096 pipeline should be about **32.7 ms (~30.6 FPS)**.

At 36 FPS, raw int8 output is ~83 MB/s (~667 Mbps), so a ground-station link needs
compression or a reduced payload.

See [PERFORMANCE_AND_B4096.md](PERFORMANCE_AND_B4096.md) for:

- full measured tables and the fast-preprocess implementation,
- raw-output bandwidth and packet format,
- B4096 resource, ISA, and fingerprint caveats,
- loading the B4096 overlay and compiling with
  `dpu_workflow/arch_kv260_isa1_b4096.json`,
- benchmark method and references.

`B3136` remains the verified camera/display path; `B4096` is the DPU-only benchmark
overlay. Both binaries are kept in `checkpoints/`.

## Troubleshooting

- **First run hangs / times out:** the DPU/KDS can get stuck after a previous failed
  model. A clean reboot is the reliable fix; reloading the overlay alone may not be
  enough.
  ```bash
  sudo reboot
  # after boot
  sudo xmutil unloadapp 0
  sudo xmutil loadapp kv260-smartcam
  ```
- **Segfault in `std::_Rb_tree_decrement` on bare-metal Ubuntu 22.04:** native Xilinx
  packages were built with GCC 9 and conflict with Ubuntu 22.04's libstdc++. Run
  inference inside the `xilinx/smartcam:2022.1` container instead.
- **Compilation fails on Hardswish:** confirm the target is `DPUCZDX8G_ISA1_B3136`.
  ISA0 devices have no ALU engine and only support ReLU/LeakyReLU.
- **Accuracy drop after quantization:** rebuild the calibration set from images that
  match your deployment domain, and keep the letterbox preprocessing consistent.

See [DPU_END_TO_END_GUIDE.md](DPU_END_TO_END_GUIDE.md) for the full root-cause
analysis, per-stage benchmarks, and compiler details.

---

## References

DPU-aware attention approximation and the substitution strategy this repository builds on:

> S. Karki, Q. A. Ahmed, and T. Jungeblut, "No Attention, No Problem: DPU-Aware
> Attention Approximation in Modern YOLO on FPGA," in *Proc. 2026 29th Euromicro
> Conf. Digital System Design (DSD)*, 2026, pp. 106-113,
> doi: 10.23919/DSDCPS00089.2026.00025.

```bibtex
@inproceedings{karki2026noattention,
  author    = {Karki, Suraj and Ahmed, Qazi Arbab and Jungeblut, Thorsten},
  title     = {No Attention, No Problem: {DPU}-Aware Attention Approximation in Modern {YOLO} on {FPGA}},
  booktitle = {2026 29th Euromicro Conference on Digital System Design (DSD)},
  pages     = {106--113},
  year      = {2026},
  doi       = {10.23919/DSDCPS00089.2026.00025}
}
```

The paper evaluates YOLOv8n/YOLOv11n/YOLOv26n on AMD DPUs (ZCU104, B512-B4096) with the same
DPU-compatibility substitutions used here — unsupported activations replaced, split operations
mapped to 1x1 convolutions, and spatial attention approximated with convolution and element-wise
operations — and reports 34.05 FPS for YOLOv26n at roughly 3x lower power than the state of the
art. Full notes and the arXiv entry are in
[PERFORMANCE_AND_B4096.md](PERFORMANCE_AND_B4096.md#6-references).

Defect datasets used by the crack and crack + spalling models:

- [`crack-01-rshs1`](https://universe.roboflow.com/crack-detection-slmz7/crack-01-rshs1) v17,
  Crack Detection, Roboflow Universe, MIT license.
- [`concrete-crack-spalling`](https://universe.roboflow.com/1-7frt7/concrete-crack-spalling) v2,
  Roboflow Universe workspace `1-7frt7`, CC BY 4.0.
- [`concrete-spalling-sm5v3`](https://universe.roboflow.com/1-7frt7/concrete-spalling-sm5v3) v1,
  Roboflow Universe workspace `1-7frt7`, CC BY 4.0.
- G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023, AGPL-3.0:
  <https://github.com/ultralytics/ultralytics>.

---

## License

The modified Ultralytics source is distributed under **AGPL-3.0**. See `LICENSE`.
