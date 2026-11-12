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
| Is B4096 faster? | The theoretical peak is **1.23 TOPS vs 0.92 TOPS** for B3136 (+33%). Real gain is typically 20-35%, possibly more for YOLO because 16x16 channel parallelism aligns better with 64/128-channel layers. Must be measured. |
| Does B4096 support Hardswish? | The 2022.2 KV260 B4096 design uses fingerprint `0x101000016010407`, i.e. **ISA1 with ALU**, so yes. Verify the fingerprint of the exact overlay you load. |
| Biggest risk? | Fingerprint/ISA mismatch. Never compile B4096 for a board running a different B4096 build. Always read `xdputil query` first and compile against that fingerprint. |

> **Status on this board:** B3136 is verified end-to-end. B4096 has not yet been
> loaded and measured on this specific KV260. The B4096 steps below are based on AMD's
> published firmware, the PPA package contents, and public B4096 designs; run the
> verification steps before trusting the numbers.

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
| 12.6 (current measured) | ~29.0 MB/s (~232 Mbps) |
| 15 | ~34.5 MB/s (~276 Mbps) |
| 20 | ~46.0 MB/s (~368 Mbps) |
| 30 | ~68.9 MB/s (~551 Mbps) |

A 1 GbE link can carry this, but it is not negligible. If bandwidth is tight:

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

## 3. Benchmark the current B3136 correctly first

The earlier `79.62 ms` measurement was a cold single execution. Warm up the runner,
reuse buffers, and report percentiles:

```python
for _ in range(10):
    job = runner.execute_async([inp], out_bufs)
    runner.wait(job)

ts = []
for _ in range(100):
    t0 = time.perf_counter()
    job = runner.execute_async([inp], out_bufs)
    runner.wait(job)
    ts.append((time.perf_counter() - t0) * 1000)

print("p50 %.2f ms  p90 %.2f ms" % (np.median(ts), np.percentile(ts, 90)))
```

At 6.106 GMAC per frame, 79.62 ms corresponds to roughly 153 GOPS, or about 17% of
the B3136 0.92 TOPS peak. That low utilization suggests either cold-start overhead,
memory/ALU stalls, or inefficient layer shapes. Benchmark before assuming a bigger
DPU will scale linearly.

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
waste on the same layers. Effective speedup on YOLO-style models can therefore be
larger than the peak ratio, but this is workload-dependent and must be measured.

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

### 4.6 Expected outcome

| Target | Peak | Expected relative DPU speed | Notes |
| :--- | ---: | :--- | :--- |
| B3136 | 0.92 TOPS | 1.00x | verified, full smartcam overlay |
| B4096 | 1.23 TOPS | ~1.2-1.45x | DPU-only benchmark overlay; measure |

If the B3136 run is limited by memory or fixed per-inference overhead rather than
convolution throughput, the gain will be smaller. If it is limited by convolution
lane utilization, the gain can exceed the 33% peak ratio.

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

1. **Benchmark B3136 properly** (warmup, p50/p90, pre/DPU/post split, end-to-end).
2. **Move post-processing off-board**; measure raw-output bandwidth and compression.
3. **Load B4096 benchmark overlay**, run `xdputil query`, record the fingerprint.
4. **Compile the current INT8 model** with `dpu_workflow/arch_kv260_isa1_b4096.json`
   (or the exact fingerprint from step 3).
5. **A/B benchmark** B3136 vs B4096 with the same input, warmup, and percentile method.
6. Only then invest in model slimming (class count, resolution, head/proto) on
   whichever DPU wins.

Model-level changes still matter more than the DPU choice in many cases: the
detection head is 54.6% of compute, Proto is 20.2%, and the P3 branch alone is 30.7%.
A custom class count and a 512x512 input can outweigh the B4096 gain.

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
