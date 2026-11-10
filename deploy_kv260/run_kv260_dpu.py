#!/usr/bin/env python3
"""
KV260 On-Board DPU Inference Runner for YOLOv8-seg (.xmodel)
Runs using VART (Vitis AI Runtime) and python xir/runner.
"""

import os
import sys
import time

import cv2
import numpy as np

try:
    import vart
    import xir
except ImportError:
    print("Error: VART or XIR not found. Ensure you are running on KV260 with Vitis AI runtime installed.")
    sys.exit(1)


def preprocess(img_path, input_shape=(640, 640)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image {img_path}")
    h, w = img.shape[:2]
    r = min(input_shape[0] / h, input_shape[1] / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (input_shape[0] - nh) // 2
    bottom = input_shape[0] - nh - top
    left = (input_shape[1] - nw) // 2
    right = input_shape[1] - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR to RGB
    tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
    return padded, tensor, (h, w, r, left, top)


def run_dpu(xmodel_path, image_path, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading XModel: {xmodel_path}...")
    graph = xir.Graph.deserialize(xmodel_path)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

    # Find DPU subgraphs
    dpu_subgraphs = [s for s in subgraphs if s.has_attr("device") and s.get_attr("device") == "DPU"]
    print(f"Found {len(dpu_subgraphs)} DPU subgraphs.")

    # Create runner for the main DPU subgraph
    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")

    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    print("\n--- DPU Input Tensors ---")
    for t in input_tensors:
        print(f"Name: {t.name}, Shape: {t.dims}, Dtype: {t.dtype}")

    print("\n--- DPU Output Tensors ---")
    for t in output_tensors:
        print(f"Name: {t.name}, Shape: {t.dims}, Dtype: {t.dtype}")

    # Prepare input buffer
    input_shape = input_tensors[0].dims
    input_shape[0]
    _padded_img, input_data, _meta = preprocess(image_path, (input_shape[1], input_shape[2]))

    # Fix point scaling if input is integer
    fixpos = input_tensors[0].get_attr("fix_point") if input_tensors[0].has_attr("fix_point") else 0
    scale = 2**fixpos
    input_quant = (input_data * scale).astype(np.int8 if input_tensors[0].dtype == xir.DataType.INT else np.float32)
    input_buffer = np.expand_dims(input_quant, axis=0)

    # Allocate output buffers
    output_buffers = [
        np.empty(tuple(t.dims), dtype=np.int8 if t.dtype == xir.DataType.INT else np.float32) for t in output_tensors
    ]

    print(f"\nExecuting DPU inference on {image_path}...")
    t0 = time.time()
    job_id = runner.execute_async([input_buffer], output_buffers)
    runner.wait(job_id)
    dpu_time = (time.time() - t0) * 1000
    print(f"✅ DPU Execution Time: {dpu_time:.2f} ms ({1000.0 / dpu_time:.1f} FPS)")

    print("\nOutput shapes obtained from DPU:")
    for i, ob in enumerate(output_buffers):
        print(f"Output {i} ({output_tensors[i].name}): shape={ob.shape}, min={np.min(ob)}, max={np.max(ob)}")

    print("\nInference completed successfully on Kria KV260 DPU!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 run_kv260_dpu.py <path_to_xmodel> <path_to_image>")
        sys.exit(1)
    run_dpu(sys.argv[1], sys.argv[2])
