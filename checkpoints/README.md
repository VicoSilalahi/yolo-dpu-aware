# Model artifacts

This directory contains the tracked base-model artifacts and locally downloaded
release artifacts. The 2-class concrete model is published on GitHub Releases
because its compiled and quantized files add ~28 MB.

Release: [`kv260-2cls-v1`](https://github.com/VicoSilalahi/yolo-dpu-aware/releases/tag/kv260-2cls-v1)

## Artifact index

| Model | Classes | Target | Artifact | Where | MD5 | Status |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| COCO DPU base | 80 | B3136 | `yolov8n_seg_kv260_b3136.xmodel` | `checkpoints/compiled_kv260_b3136/` | `b29b1b05c03571d5860d8887f4a26c8e` | hardware verified |
| COCO DPU base | 80 | B4096 | `yolov8n_seg_kv260_b4096.xmodel` | `checkpoints/compiled_kv260_b4096/` | `8bbacbee9e7f67219f4808ddc70159bc` | hardware verified |
| COCO DPU base | 80 | — | NNDCT XIR `SegmentationModel_int.xmodel` | `checkpoints/quantize_result/` | — | source for both base builds |
| concrete-2cls | 2 | B3136 | `yolov8n_seg_concrete_2cls_kv260_b3136.xmodel` | release asset | `29326065c560dc51a1a45883390209ce` | compiled; I/O structure verified |
| concrete-2cls | 2 | B4096 | `yolov8n_seg_concrete_2cls_kv260_b4096.xmodel` | release asset | `583a2e90fe93b82dbd4fe89a79d9f0a5` | compiled; I/O structure verified |
| concrete-2cls | 2 | — | `quantize_result_concrete_2cls.tar.gz` | release asset | SHA256 `3847b8f95cff3bd681223d74a2336c880073329c982912c7754299d6b5afa0d8` | NNDCT output |

The 2-class binaries have the same graph layout as the base model: one USER
subgraph, one DPU subgraph, ten CPU subgraphs, input `[1,640,640,3] xint8
fix_point=6`, and ten raw convolution outputs. The class tensors have two
channels.

## Download the 2-class release

From the repository root:

```bash
gh release download kv260-2cls-v1 --repo VicoSilalahi/yolo-dpu-aware \
  --pattern 'yolov8n_seg_concrete_2cls_kv260_b3136*' \
  --pattern 'yolov8n_seg_concrete_2cls_kv260_b4096*' \
  --pattern 'quantize_result_concrete_2cls.tar.gz' \
  --pattern 'SHA256SUMS' \
  --dir checkpoints
```

The release contains both xmodels, their `meta.json` and `md5sum.txt` files, the
quantized XIR tarball, and `SHA256SUMS` for the release assets.

Or download a single binary directly:

```bash
wget \
  https://github.com/VicoSilalahi/yolo-dpu-aware/releases/download/kv260-2cls-v1/yolov8n_seg_concrete_2cls_kv260_b3136.xmodel \
  -O checkpoints/yolov8n_seg_concrete_2cls_kv260_b3136.xmodel
```

To restore the quantized XIR graph:

```bash
tar -xzf quantize_result_concrete_2cls.tar.gz -C checkpoints
```

## Provenance

| Item | Value |
| :--- | :--- |
| Source checkpoint | [`yolov8n-seg-concrete-2cls.pt`](https://github.com/VicoSilalahi/yolo-dpu-aware/releases/download/weights-v2/yolov8n-seg-concrete-2cls.pt) |
| Checkpoint SHA256 | `EF4534F3E531683FE42D188AC1D60D2E3A519901D5C591ACCE7EDA709C1D848D` |
| Calibration set | 50 images from `datasets/concrete-2cls/val/images` |
| Quantizer | Vitis AI 3.5 NNDCT (`3.5.0+60df3f1+torch1.13.1`) |
| Compiler | Vitis AI 2.5 `vai_c_xir` |
| Input | `[1, 640, 640, 3] xint8`, `fix_point=6` |
| Class outputs | 2 channels (`crack`, `spalling`) |
| Base-model input contract | identical |

The base COCO model remains in the git tree. Do not replace the tracked base
binaries with the 2-class builds; they have different class outputs.

## Regenerate

The commands are in [README.md](../README.md#4-quantize-int8) and
[README.md](../README.md#5-compile-the-xmodel). Use the cached Vitis AI 3.5
NNDCT image for quantization and the cached 2.5 image for compilation.
