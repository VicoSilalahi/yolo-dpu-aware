# KV260 deployment

Copy this directory and the `.xmodel` you want to run into `~/deploy_kv260` on
the KV260. The C++ runner is the primary path; the Python scripts are small
fallbacks.

| File | Purpose |
| :--- | :--- |
| `run_kv260_dpu.cpp` | VART/XIR runner with DFL decode, NMS, masks, timing, and annotated JPEG output |
| `Makefile` | Builds the C++ runner; `make inspect` reads an xmodel without a DPU |
| `run_kv260_dpu.py` | Lightweight raw-tensor sanity runner |
| `postprocess_b3136.py` | Standalone Python DFL/NMS/mask decoder |

## Get an xmodel

The base COCO builds are tracked in `../checkpoints/`:

```text
checkpoints/compiled_kv260_b3136/yolov8n_seg_kv260_b3136.xmodel
checkpoints/compiled_kv260_b4096/yolov8n_seg_kv260_b4096.xmodel
```

The 2-class concrete builds are release assets:

```bash
gh release download kv260-2cls-v1 --repo VicoSilalahi/yolo-dpu-aware \
  --pattern 'yolov8n_seg_concrete_2cls_kv260_b3136.xmodel' \
  --dir checkpoints
```

See [`checkpoints/README.md`](../checkpoints/README.md) for the full artifact
table, hashes, and download commands.

## Build the C++ runner

Build on the KV260 (`aarch64`) **inside** the `xilinx/smartcam:2022.1` container
so the binary links against the container's `libstdc++` and OpenCV ABI (a binary
compiled on the Ubuntu 22.04 host will segfault inside the container):

```bash
cd /workspace
apt-get update && apt-get install -y g++ make libopencv-dev libgoogle-glog-dev
make clean && make
```

You only need to install those build packages once; the compiled `run_kv260_dpu`
binary is saved in `~/deploy_kv260` on the host and persists across container
restarts.

The Makefile prefers the system VART/XIR headers and falls back to the Vitis AI
conda environment. It accepts either `pkg-config opencv4` or
`pkg-config opencv`. Override the environment if needed:

```bash
make VAI_ENV=/path/to/vitis-ai-env
```

`--inspect` only reads XIR metadata and works without `/dev/dpu`, so it is useful
on a workstation:

```bash
make inspect MODEL=../checkpoints/compiled_kv260_b3136/yolov8n_seg_kv260_b3136.xmodel
```

## Run on the board

```bash
./run_kv260_dpu yolov8n_seg_concrete_2cls_kv260_b3136.xmodel "$IMAGE" \
  -o result_concrete_2cls.jpg \
  --classes crack,spalling
```

Useful options:

```text
-o/--output <path>   annotated image (default result_cpp.jpg)
-c/--conf <float>    confidence threshold (default 0.25)
-i/--iou <float>     NMS IoU threshold (default 0.45)
--classes a,b,c      class names; otherwise CLASSES or class_0...
--inspect            print the DPU I/O layout and exit
--no-draw            benchmark only, write no image
```

The runner prints the DPU input/output layout, warmup and timed iterations
(`--bench`, `--warmup`), mean/p50/p90/FPS, then writes the annotated JPEG.

Observed hardware benchmarks (KV260, `test.jpg`):

| Runner | Model | Target | DPU Latency | FPS |
| :--- | :--- | :--- | :--- | :---: |
| C++ (`run_kv260_dpu`, 10 runs) | 2-class concrete | B3136 | 42.80 ms p50 (mean 42.79 ms, p90 42.83 ms) | **23.37** |
| C++ (`run_kv260_dpu`, 10 runs) | 2-class concrete | B4096 | 27.97 ms p50 (mean 27.96 ms, p90 28.03 ms) | **35.77** |
| Python (`run_kv260_dpu.py`) | 2-class concrete | B3136 | 72.79 ms | **13.7** |
| Python (`run_kv260_dpu.py`) | 80-class COCO | B3136 | 80.33 ms | **12.4** |

## Python fallbacks

Raw tensors and latency:

```bash
python3 run_kv260_dpu.py yolov8n_seg_concrete_2cls_kv260_b3136.xmodel "$IMAGE"
```

Decoded result:

```bash
python3 postprocess_b3136.py \
  yolov8n_seg_concrete_2cls_kv260_b3136.xmodel "$IMAGE" \
  -o result_concrete_2cls.jpg \
  --classes crack,spalling
```

`postprocess_b3136.py` reads the class count from the xmodel. `--classes` and
the `CLASSES` environment variable are interchangeable; both fall back to COCO
names when unset. The file name is historical: it works with the B3136 and B4096
builds and with any class count.

## Container

```bash
cd ~/deploy_kv260
docker run --rm -it --privileged \
  -v /dev:/dev -v /sys:/sys \
  -v /lib/firmware/xilinx:/lib/firmware/xilinx \
  -v "$PWD":/workspace -w /workspace \
  -e PYTHONPATH=/workspace/python_libs \
  -e XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-smartcam/kv260-smartcam.xclbin \
  xilinx/smartcam:2022.1 bash
```
*(For `kv260-benchmark-b4096`, set `XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin`.)*

Load the overlay before running:

```bash
sudo xmutil unloadapp 0
sudo xmutil loadapp kv260-smartcam        # B3136
# or: sudo xmutil loadapp kv260-benchmark-b4096
xbutil examine
```

## Sample image

`test.jpg` is a local, gitignored convenience file. Use your own image through
`$IMAGE`, or the committed 2-class example under `assets/` when it is available.
Do not assume a fresh clone contains `deploy_kv260/test.jpg`.

## Target notes

- B3136 matches `kv260-smartcam`, fingerprint `0x101000016010406`.
- B4096 matches the DPU-only `kv260-benchmark-b4096` overlay, fingerprint
  `0x101000016010407`.
- Match the xmodel to the loaded overlay. `--inspect` prints the target recorded
  in `meta.json`; `xdputil query` prints the target on the board.
