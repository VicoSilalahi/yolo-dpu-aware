# Implementation Plan: Segmentation Data Export, Unique-Fault Snapshots, and Live Stream

## Goal Description

Downstream subsystems need **segmentation geometry** (to compute metric **spalling area** using RealSense depth, and **crack length/orientation**), **unique-fault visual snapshots**, and an optional **downscaled live video feed**—while keeping the DPU runner (`deploy_kv260/run_kv260_stream.cpp`) fast, decoupled, and focused on inference + tracking.

---

## Proposed Changes

### 1. Compact Segmentation Data Export

#### [MODIFY] [run_kv260_stream.cpp](run_kv260_stream.cpp)
- **Segmentation mask export (for `is_new_unique == true` or `--export-masks`)**:
  - Encode each detection's binary `cv::Mat mask` as a **lossless 1-channel PNG** (`cv::imencode(".png", d.mask, buf)` — typically **1–3 KB** per defect).
  - Send via **ZeroMQ multipart message** (`[JSON_metadata, mask_png_0, mask_png_1, ...]`) when `--zmq-pub` is active, or save alongside snapshots when `--save-snapshots <dir>` is set (`fault_<track_id>_<class>_mask.png`).
  - Downstream ROS 2 / Python nodes decode with `cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)` to compute exact **spalling area** (mask pixels × depth²) and **crack length/orientation** (skeletonization / PCA).
- **Lightweight pixel geometry in JSON**:
  - Keep `area_px` (exact mask pixel count) and optionally add `orientation_deg` + `major_minor_px` from `cv::minAreaRect` on the mask contour.

---

### 2. Unique-Fault Annotated JPEG Snapshots

#### [MODIFY] [run_kv260_stream.cpp](run_kv260_stream.cpp)
- Add `--save-snapshots <dir>`:
  - Only when at least one detection in the frame has `is_new_unique == true`, render `draw_annotations()` onto a copy of the frame and write `<dir>/fault_<track_id>_<class>.jpg` (and optionally publish the JPEG bytes as a ZeroMQ multipart frame for immediate ground-station transmission).

---

### 3. Optional Downscaled Live Video Stream

#### [MODIFY] [run_kv260_stream.cpp](run_kv260_stream.cpp)
- Add `--stream-udp <host:port>` (or `--stream-scale 0.5 --stream-fps 10`):
  - Downscales the annotated frame (e.g. `320×240`) and writes at a capped rate (e.g. every 2nd/3rd frame) to a GStreamer `cv::VideoWriter` UDP/H.264 pipeline (`appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=400 ! rtph264pay ! udpsink host=<host> port=<port>`).

---

## Verification Plan

### Hardware Verification on KV260
- Rebuild `make stream` (and `make stream USE_ZMQ=1` when `libzmq3-dev` is installed).
- Run on `test_concrete_pan.mp4` with `--track --save-snapshots snapshots/ --no-json --report-interval 1.0`.
- Verify:
  1. `snapshots/` contains exactly **5** annotated JPEGs + 5 lossless mask PNGs (one per unique defect).
  2. Sustained FPS remains **~14–16 FPS** on B4096.
