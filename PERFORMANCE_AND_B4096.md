# Performance Notes: Ground-Station Post-Processing and KV260 B4096

This note covers two related decisions:

1. Moving segmentation post-processing from the KV260 to the ground station.
2. Targeting the larger `DPUCZDX8G_ISA1_B4096` DPU instead of the current `B3136`.

See [README.md](README.md) for the current verified B3136 setup and
[DPU_END_TO_END_GUIDE.md](DPU_END_TO_END_GUIDE.md) for the compilation flow.

---

## 1. TL;DR

| Question | Answer |
| :--- | :--- |
| Does ground-station post-processing make the DPU faster? | **No.** DPU time is unchanged. It removes A53 work and simplifies the board, which helps end-to-end latency/throughput if post-processing was a bottleneck. |
| Can the KV260 run B4096? | **Yes.** AMD ships a DPU-only B4096 benchmark firmware/overlay for the KV260. It is the right fit for a board that only runs the DPU and streams raw tensors. |
| Is B4096 faster? | **Yes.** Measured on this board: **27.55 ms p50 (36.3 FPS)** vs **42.16 ms p50 (23.7 FPS)** for B3136, a **1.53x** DPU speedup. The theoretical peak ratio is 1.23 TOPS / 0.92 TOPS = 1.33x, so the 16x16 channel alignment helps. |
| Does B4096 support Hardswish? | Yes for this overlay: it reports fingerprint `0x101000016010407`, i.e. **ISA1 with ALU**. Verify the fingerprint of the exact overlay you load. |
| Biggest risk? | Fingerprint/ISA mismatch. Never compile B4096 for a board running a different B4096 build. Always read `xdputil query` first and compile against that fingerprint. |

> **Status on this board:** both overlays are now verified. B3136 runs through
> `kv260-smartcam`; B4096 runs through `kv260-benchmark-b4096`. B4096 reported
> `IP version v4.0.0`, `DPUCZDX8G_ISA1_B4096`, 300 MHz, fingerprint
> `0x101000016010407`. Both produced identical detections on the reference image.

---

## 2. Board / ground-station split

A clean split is:

**Board (KV260)**
1. Receive a frame (USB camera, Ethernet, or VCU-decoded stream).
2. Letterbox/resize to 640x640.
3. Quantize to int8 using the DPU input `fix_point`.
4. Run the DPU.
5. Copy the 10 raw output tensors into a packet.
6. Transmit the packet; do **no** DFL decode, sigmoid, NMS, mask assembly, or drawing.

**Ground station**
1. Parse the binary packet.
2. Dequantize each tensor using its `fix_point`.
3. DFL box decode, class sigmoid, class-aware NMS.
4. Mask assembly from `mask_coeff x proto`.
5. Draw and display/store the result.

This keeps the A53 free and removes OpenCV/Python post-processing jitter from the
board. It also makes the B4096 DPU-only benchmark overlay a natural target, because
no camera/display/pre-processing IP is required in the FPGA.

### 2.1 Raw output size

For the current YOLOv8n-seg INT8 model at 640x640, the DPU returns int8 tensors:

| Tensor | Shape (NHWC) | Bytes |
| :--- | :--- | ---: |
| Proto | `1 x 160 x 160 x 32` | 819,200 |
| P3 box + class + mask coeff | `1 x 80 x 80 x 176` | 1,126,400 |
| P4 box + class + mask coeff | `1 x 40 x 40 x 176` | 281,600 |
| P5 box + class + mask coeff | `1 x 20 x 20 x 176` | 70,400 |
| **Total** | | **2,297,600 B (~2.19 MiB)** |

Bandwidth at different frame rates:

| FPS | Raw payload |
|---:|---:|
| 12.6 | ~29.0 MB/s (~232 Mbps) |
| 23.7 (B3136 DPU p50) | ~54.5 MB/s (~436 Mbps) |
| 30 | ~68.9 MB/s (~551 Mbps) |
| 36.3 (B4096 DPU p50) | ~83.4 MB/s (~667 Mbps) |

A 1 GbE link can carry this, but at 36 FPS it is about two-thirds of the link and
leaves little headroom for protocol overhead. Compression is effectively mandatory
if you want to stream every frame at B4096 speed. If bandwidth is tight:

- compress with LZ4/Zstd before transmission,
- transmit int8 only (never convert to float on the board),
- include `fix_point`, shapes, tensor IDs, frame ID, and letterbox metadata in a binary header,
- optionally run the very cheap DFL decode and sigmoid on the board and send only
  top-k boxes/classes/scores plus mask coefficients and proto,
- reduce proto resolution to 80x80 (see performance notes) to cut 0.8 MB to 0.2 MB.

Do not send JSON-wrapped base64 payloads; use a fixed binary header and raw buffers.

### 2.2 What this does and does not improve

- **Does not** reduce DPU latency: the same 6.1 GMAC model still runs on the DPU.
- **Does** remove the post-processing tail from the pipeline, which matters if the
  A53 post-processing was adding tens of milliseconds and causing frame drops.
- **Does** free the board to use a DPU-only bitstream with no camera/display IP.

Measure DPU-only time and end-to-end time separately before attributing gains.

---

## 3. Measured benchmark

The earlier `79.62 ms` figure was a cold single execution. With 30 warmup
iterations and 300 measured iterations, reusing buffers and reporting p50:

| Configuration | DPU p50 | DPU p90 | DPU FPS (p50) | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| B3136 + `kv260-smartcam` | **42.16 ms** | 42.23 ms | **23.7** | 1.00x |
| B4096 + `kv260-benchmark-b4096` | **27.55 ms** | 27.62 ms | **36.3** | **1.53x** |

Both runs used the same input, `yolov8n_seg_kv260_b*_b*.xmodel`, and the same
`xilinx/smartcam:2022.1` runtime. B4096 produced the same four detections as B3136:
`bus 0.88`, `person 0.82`, `person 0.78`, `person 0.68`.

At 6.106 GMAC/frame:

- B3136: 42.16 ms -> ~145 GOPS, ~15.7% of 0.92 TOPS peak.
- B4096: 27.55 ms -> ~222 GOPS, ~18.0% of 1.23 TOPS peak.

The absolute utilization is still low, so model-level slimming (head, proto,
resolution, class count) remains valuable.

### 3.1 The real bottleneck is CPU preprocessing

The benchmark separates DPU time from preprocessing. The current Python float
preprocess costs about **52-53 ms/frame** on the A53:

| Configuration | DPU | Preprocess | Full p50 | Full FPS |
| :--- | ---: | ---: | ---: | ---: |
| B3136 + float preprocess | 42.16 ms | 52.6 ms | 94.81 ms | 10.6 |
| B4096 + float preprocess | 27.55 ms | 53.6 ms | 81.14 ms | 12.3 |
| B3136 + LUT preprocess | 42.20 ms | 5.16 ms | **47.63 ms** | **21.0** |
| **B4096 + LUT preprocess** | **27.61 ms** | **5.18 ms** | **33.11 ms** | **30.2** |

The float rows were measured with the original Python file-based loop (including
`cv2.imread`). The LUT rows are closed-loop in-memory preprocess + DPU, 300 iterations
after 30 warmup, and are the numbers to use for pipeline planning.

An integer LUT path is **bit-exact** and ~9x faster:

```python
lut = np.clip(np.rint(np.arange(256, dtype=np.float32) / 255.0 * (2 ** fix)),
              0, 127).astype(np.uint8)
padded = letterbox(img, 640)
inp = cv2.LUT(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), lut).view(np.int8)[None]
```

`lut` is a 256-entry table that precomputes the DPU input quantization
(`round(pixel / 255 * 2 ** fix)`, clamped to `[0, 127]`) for every possible 8-bit
pixel value. The float path repeats that multiply/divide/round for each of the
1.2M pixels per frame; `cv2.LUT` does one byte-indexed table read per pixel in
optimized C. It is bit-exact because an 8-bit input has only 256 distinct values.
The `view(np.int8)` reinterprets the bytes as signed without copying, and is safe
because the table is clamped to 127.

This is only the *input encoding* step. Three separate quantizations are involved:

| Step | When | What | Where |
| :--- | :--- | :--- | :--- |
| Model quantization | compile time | weights and activations -> int8 | `quantize_pytorch.py` (NNDCT) |
| Input encoding | runtime | uint8 pixels -> int8 fixed point (`fix_point=6`) | the LUT above |
| Output decoding | ground station | int8 raw tensors -> floats via `2 ** -fix_point` | post-processing |

The model is therefore fully quantized before it ever reaches the board. The LUT
does not change the model; it only feeds the quantized model correctly and quickly.

Measured preprocessing on the same 1080x810 image:

| Implementation | p50 |
| :--- | ---: |
| Original float path | 48.35 ms |
| Float conversion component alone | 21.29 ms |
| Letterbox resize + border | 2.33 ms |
| `cv2.LUT` | 2.40 ms |
| **Integer LUT + letterbox + RGB** | **5.16 ms** |

Closed-loop measured pipeline (preprocess + DPU per frame, 300 iterations after 30
warmup):

| Configuration | Full p50 | Full p90 | Full FPS |
| :--- | ---: | ---: | ---: |
| B3136 + LUT | **47.63 ms** | 47.72 ms | **21.0** |
| **B4096 + LUT** | **33.11 ms** | 33.23 ms | **30.2** |

The LUT path was verified bit-exact against the float path on the reference image
(`maxdiff = 0`, `nonzero = 0`). The full-loop gain from B4096 is 47.63 / 33.11 =
**1.44x**, because preprocessing and other fixed overheads do not shrink with the DPU.

If the camera is not physically on the board, the ground station can also send
already-letterboxed int8 input. In that case the board only runs the DPU, and the
B4096 DPU-only figure (27.55 ms, 36.3 FPS) is the pipeline number.

---

## 4. B4096 on the KV260

### 4.1 What `B4096` means

From AMD PG338, DPUCZDX8G parallelism:

| Architecture | Pixel parallelism | Input channel parallelism | Output channel parallelism | Peak ops/cycle |
| :--- | ---: | ---: | ---: | ---: |
| B3136 | 8 | 14 | 14 | 3136 |
| B4096 | 8 | 16 | 16 | 4096 |

At 300 MHz:

| DPU | Peak |
| :--- | ---: |
| B3136 | ~0.92 TOPS |
| B4096 | ~1.23 TOPS |

The B4096 advantage is not only +33% peak. Its 16x16 channel parallelism aligns
cleanly with 64-, 128-, and 256-channel convolutions, whereas 14x14 leaves lane
waste on the same layers. That is consistent with the measured **1.53x** DPU speedup,
which exceeds the 1.33x peak ratio.

### 4.2 Resource reality on K26

The K26 SOM has 1,248 DSP slices, 144 BRAM, and 64 UltraRAM. AMD's generic PG338
B4096 utilization example is about 710 DSP and 68 URAM (or 255 BRAM). That is at or
over the K26 budget, which is why the standard KV260 smartcam application uses
B3136 with pre-processing, display, and VCU IP in the same design.

B4096 fits when the overlay is **DPU-only**. AMD provides exactly that:

- PPA package: `xlnx-firmware-kv260-benchmark-b4096` (jammy)
- Accelerator name: `kv260-benchmark-b4096`
- Upstream firmware: `Xilinx/kria-apps-firmware`, branch `xlnx_rel_v2022.2`,
  `boards/kv260/benchmark-b4096`

The package we inspected contains a ~7.8 MB bitstream, a 38,898-byte xclbin built
with Vitis 2022.1, `shell.json`, and a device-tree overlay. The xclbin runs the DPU
kernel at 300 MHz and contains no camera/display pre-processing IP.

That is a good match for this project: the board runs the DPU, streams raw int8
tensors, and the ground station does the rest.

### 4.3 ISA and fingerprint

A 2022.2 rebuild of the KV260 B4096 benchmark overlay (LogicTronix) reports:

```json
{"fingerprint": "0x101000016010407"}
```

This is `ISA1` with the ALU engine, so `Hardswish` is supported. Do not assume this
for every B4096 package: older 2021.1/1.4 B4096 benchmark builds are `ISA0` and have
no ALU. They would require ReLU-only models.

Always verify on the actual loaded overlay:

```bash
# inside the Vitis AI container
xdputil query
```

Look for `"DPU Arch": "DPUCZDX8G_ISA1_B4096"` and the exact fingerprint. Compile
against that fingerprint, not against a guess.

### 4.4 Install and load the B4096 overlay

Option A, PPA package:

```bash
sudo apt install -y xlnx-firmware-kv260-benchmark-b4096
sudo xmutil listapps
sudo xmutil unloadapp 0
sudo xmutil loadapp kv260-benchmark-b4096
```

Option B, upstream 2022.2 firmware (if the PPA build reports an ISA0 fingerprint):

```bash
git clone --depth 1 -b xlnx_rel_v2022.2 https://github.com/Xilinx/kria-apps-firmware
sudo mkdir -p /lib/firmware/xilinx/kv260-benchmark-b4096
cd kria-apps-firmware/boards/kv260/benchmark-b4096

sudo cp kv260-benchmark-b4096.xclbin shell.json /lib/firmware/xilinx/kv260-benchmark-b4096/
sudo cp kv260-benchmark-b4096.bit /lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.bit.bin

# The dtsi sets firmware-name = "kv260-benchmark-b4096.bit.bin"
dtc -I dts -O dtb -o kv260-benchmark-b4096.dtbo kv260-benchmark-b4096.dtsi
sudo cp kv260-benchmark-b4096.dtbo /lib/firmware/xilinx/kv260-benchmark-b4096/

# Point the runtime at the new xclbin
echo "firmware: /lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin" | \
  sudo tee /etc/vart.conf

sudo xmutil listapps
sudo xmutil unloadapp 0
sudo xmutil loadapp kv260-benchmark-b4096
```

Then verify in the runtime container:

```bash
xbutil examine
xdputil query
```

> On this board the first `loadapp` after unloading smartcam returned `load Error: -1`;
> a second `sudo xmutil loadapp kv260-benchmark-b4096` succeeded. If this happens,
> retry once before rebooting.

### 4.5 Compile the model for B4096

An arch file for the 2022.2 rebuild fingerprint is included at
`dpu_workflow/arch_kv260_isa1_b4096.json`. **Replace it with the fingerprint reported
by your loaded overlay if it differs.**

```json
{"fingerprint": "0x101000016010407"}
```

Compile the same quantized XIR graph:

```bash
vai_c_xir \
  -x /workspace/checkpoints/quantize_result/SegmentationModel_int.xmodel \
  -a /workspace/dpu_workflow/arch_kv260_isa1_b4096.json \
  -o /workspace/checkpoints/compiled_kv260_b4096 \
  -n yolov8n_seg_kv260_b4096
```

Then check `meta.json` and run the same benchmark you used for B3136. Keep both
binaries so the comparison is controlled.

> The current repository's verified binary remains
> `checkpoints/compiled_kv260_b3136/yolov8n_seg_kv260_b3136.xmodel`. Do not overwrite
> it when testing B4096.

### 4.6 Measured outcome

| Target | Peak | DPU p50 | DPU FPS | Measured speedup | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| B3136 | 0.92 TOPS | 42.16 ms | 23.7 | 1.00x | full smartcam overlay |
| **B4096** | **1.23 TOPS** | **27.55 ms** | **36.3** | **1.53x** | DPU-only benchmark overlay |

The measured 1.53x gain exceeds the 1.33x peak ratio, consistent with the 16x16
channel alignment being a better fit for 64/128-channel YOLO layers.

Full-pipeline results depend heavily on preprocessing:

| Configuration | Full p50 | Full FPS |
| :--- | ---: | ---: |
| B3136 + Python float preprocess | 94.81 ms | 10.6 |
| B4096 + Python float preprocess | 81.14 ms | 12.3 |
| B3136 + integer LUT preprocess | **47.63 ms** | **21.0** |
| **B4096 + integer LUT preprocess** | **33.11 ms** | **30.2** |

The LUT rows are closed-loop measurements. With the LUT path, the DPU is again the
dominant per-frame cost on B4096 (27.6 ms of 33.1 ms).

### 4.7 When not to use B4096

- You need the smartcam camera/display pipeline and cannot rebuild it for B4096.
  The stock smartcam overlay is B3136; a custom B4096 smartcam overlay requires a
  Vivado/Vitis rebuild (Avnet has a reference patch).
- You need a production-hardened design. The benchmark overlay is intended for
  testing, not a full product; a custom platform is the long-term answer.
- Power/thermal headroom is tight. B4096 uses more DSPs and routing; validate the
  KV260 heatsink and power budget under sustained load.
- You cannot match the fingerprint. An ISA0 B4096 overlay cannot run Hardswish.

---

## 5. Recommended plan

1. ~~Benchmark B3136 properly~~ — done: 42.16 ms p50, 23.7 FPS.
2. ~~Load B4096 benchmark overlay and verify fingerprint~~ — done:
   `DPUCZDX8G_ISA1_B4096`, `0x101000016010407`, 300 MHz.
3. ~~A/B benchmark B3136 vs B4096~~ — done: 27.55 ms p50, 36.3 FPS, 1.53x.
4. ~~Replace the Python float preprocess with the integer LUT path~~ — done in
   `deploy_kv260/` and measured: full loop **47.63 ms (B3136)** and
   **33.11 ms (B4096)**.
5. **Move post-processing off-board** and measure raw-output bandwidth/compression.
   At 30 FPS the raw payload is ~69 MB/s (~551 Mbps); at 36 FPS it is ~83 MB/s
   (~667 Mbps), so compress or reduce it.
6. Then invest in model slimming (class count, resolution, head/proto) on B4096.

Model-level changes still matter: the detection head is 54.6% of compute, Proto is
20.2%, and the P3 branch alone is 30.7%. A custom class count and a 512x512 input
can outweigh the B4096 gain. The B4096 result also means the DPU now has less
headroom to hide a slow preprocessing path.

---

## 6. References

- AMD PG338, DPUCZDX8G architecture and resource utilization:
  <https://docs.amd.com/r/en-US/pg338-dpu/Architecture-of-the-DPUCZDX8G>
  and <https://docs.amd.com/r/en-US/pg338-dpu/Resource-Utilization>
- KV260 smartcam hardware architecture (B3136 resource use, 0.92 TOPS):
  <https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/smartcamera/docs/hw_arch_accel.html>
- KV260 benchmark B4096 firmware:
  <https://github.com/Xilinx/kria-apps-firmware/tree/xlnx_rel_v2022.2/boards/kv260/benchmark-b4096>
- Xilinx KV260 AI benchmark:
  <https://github.com/Xilinx/kria-kv260-ai-benchmark>
- LogicTronix B4096 rebuild and arch fingerprint `0x101000016010407`:
  <https://github.com/LogicTronixInc/kria-ai-benchmark-dpu-b4096>
- Vitis AI 3.0 KV260 B4096 fingerprint example `0x101000056010407`:
  <https://adaptivesupport.amd.com/s/question/0D54U000088eNqJSAU/>
- Vitis AI DPU fingerprint explanation:
  <https://fpga.eetrend.com/content/2024/100580713.html>
- Avnet smartcam B4096 patch (rebuilding the full camera design):
  <https://xterra2.avnet.com/xilinx/example-board-projects/kv260/kv260-demos-workflow-example/-/tree/main>
- KV260 starter kit B4096 performance (1.23 TOPS):
  <https://docs.amd.com/r/2.5-English/ug1354-xilinx-ai-sdk/KV260-Vision-AI-Starter-Kit>
- S. Karki, Q. A. Ahmed, and T. Jungeblut, "No Attention, No Problem: DPU-Aware
  Attention Approximation in Modern YOLO on FPGA," arXiv:2607.13106 [cs.AR],
  14 Jul. 2026. doi: [10.48550/arXiv.2607.13106](https://doi.org/10.48550/arXiv.2607.13106).
  <https://arxiv.org/abs/2607.13106>
  - Also published at DSD 2026, pp. 106-113,
    doi: [10.23919/DSDCPS00089.2026.00025](https://doi.org/10.23919/DSDCPS00089.2026.00025).
  - Directly relevant here: it evaluates YOLOv8n/YOLOv11n/YOLOv26n on AMD DPUs
    (ZCU104, B512-B4096) and uses the same DPU-compatibility substitutions this
    repository relies on — replacing unsupported activation functions, mapping split
    operations to 1x1 convolutions, and approximating spatial attention with
    convolution and element-wise operations. Reported results include 34.05 FPS for
    YOLOv26n (29.55 FPS for YOLOv26n-OBB) with roughly 3x lower power than the
    state of the art.
