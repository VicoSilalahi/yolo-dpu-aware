#!/usr/bin/env python3
"""Lightweight KV260 runner for a YOLOv8-seg .xmodel.

Prints the DPU input/output layout and the raw output tensors. This is a sanity
check, not the decoded result; use postprocess_b3136.py or the C++ runner
run_kv260_dpu for boxes, NMS, and masks.
"""

import argparse
import sys
import time

import cv2
import numpy as np

try:
    import vart
    import xir
except ImportError:
    print("Error: VART or XIR not found. Run this on the KV260 with Vitis AI runtime installed.")
    sys.exit(1)


_INT_DTYPES = {xir.DataType.INT}
_XINT = getattr(xir.DataType, "XINT", None)
if _XINT is not None:
    _INT_DTYPES.add(_XINT)


def is_integer_tensor(tensor):
    return tensor.dtype in _INT_DTYPES


def preprocess(img_path, input_shape=(640, 640), fix=6):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image {img_path}")
    h, w = img.shape[:2]
    r = min(input_shape[0] / h, input_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (input_shape[0] - nh) // 2
    bottom = input_shape[0] - nh - top
    left = (input_shape[1] - nw) // 2
    right = input_shape[1] - nw - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    if fix is None:
        # Float input: keep normalized RGB float32.
        tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
    else:
        # Integer LUT quantization: ~9x faster than float on the A53 and bit-exact.
        lut = np.clip(
            np.rint(np.arange(256, dtype=np.float32) / 255.0 * (2 ** fix)),
            0,
            127,
        ).astype(np.uint8)
        tensor = cv2.LUT(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), lut).view(np.int8)
    return padded, tensor


def run_dpu(xmodel_path, image_path):
    print(f"Loading XModel: {xmodel_path}...")
    graph = xir.Graph.deserialize(xmodel_path)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraphs = [
        sg for sg in subgraphs
        if sg.has_attr("device") and sg.get_attr("device") == "DPU"
    ]
    if not dpu_subgraphs:
        raise SystemExit("No DPU subgraph found in the xmodel")

    print(f"Found {len(dpu_subgraphs)} DPU subgraph(s).")
    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    print("\n--- DPU Input Tensors ---")
    for tensor in input_tensors:
        print(f"Name: {tensor.name}, Shape: {tensor.dims}, Dtype: {tensor.dtype}")

    print("\n--- DPU Output Tensors ---")
    for tensor in output_tensors:
        print(f"Name: {tensor.name}, Shape: {tensor.dims}, Dtype: {tensor.dtype}")

    input_tensor = input_tensors[0]
    input_shape = input_tensor.dims
    fixpos = None
    if is_integer_tensor(input_tensor) and input_tensor.has_attr("fix_point"):
        fixpos = input_tensor.get_attr("fix_point")

    _, input_data = preprocess(image_path, (input_shape[1], input_shape[2]), fixpos)
    input_buffer = np.expand_dims(input_data, axis=0)
    output_buffers = [
        np.empty(
            tuple(tensor.dims),
            dtype=np.int8 if is_integer_tensor(tensor) else np.float32,
        )
        for tensor in output_tensors
    ]

    print(f"\nExecuting DPU inference on {image_path}...")
    started = time.perf_counter()
    job_id = runner.execute_async([input_buffer], output_buffers)
    status = runner.wait(job_id)
    dpu_time = (time.perf_counter() - started) * 1000.0
    if status != 0:
        raise SystemExit(f"DPU execution failed with status {status}")
    print(f"DPU execution time: {dpu_time:.2f} ms ({1000.0 / dpu_time:.1f} FPS)")

    print("\nRaw output tensors:")
    for index, (tensor, buffer) in enumerate(zip(output_tensors, output_buffers)):
        print(
            f"Output {index} ({tensor.name}): shape={buffer.shape}, "
            f"min={np.min(buffer)}, max={np.max(buffer)}"
        )

    print("\nInference completed successfully on the KV260 DPU.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xmodel", help="compiled .xmodel")
    parser.add_argument("image", help="input image")
    args = parser.parse_args()
    run_dpu(args.xmodel, args.image)


if __name__ == "__main__":
    main()
