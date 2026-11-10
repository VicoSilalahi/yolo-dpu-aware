import time

import cv2
import numpy as np
import onnxruntime


def test_inference(onnx_path, img_path):
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    r = min(640 / h, 640 / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (640 - nh) // 2
    bottom = 640 - nh - top
    left = (640 - nw) // 2
    right = 640 - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    input_tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    t0 = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    infer_time = (time.time() - t0) * 1000

    pred_boxes, pred_protos = outputs[0], outputs[1]
    return infer_time, pred_boxes, pred_protos


if __name__ == "__main__":
    fp32_model = "/home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/yolov8n-seg-dpu-base.onnx"
    int8_model = "/home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/yolov8n-seg-dpu-base-int8.onnx"
    test_img = "/home/vicosilalahi/programming/yolo-fpga/datasets/coco/images/val2017/000000000139.jpg"

    print("Running FP32 ONNX Inference...")
    t_fp32, b_fp32, p_fp32 = test_inference(fp32_model, test_img)
    print(f"FP32 latency: {t_fp32:.1f} ms | Box shape: {b_fp32.shape} | Proto shape: {p_fp32.shape}")

    print("Running INT8 ONNX Inference...")
    t_int8, b_int8, p_int8 = test_inference(int8_model, test_img)
    print(f"INT8 latency: {t_int8:.1f} ms | Box shape: {b_int8.shape} | Proto shape: {p_int8.shape}")

    # Compute numerical differences (Mean Absolute Error)
    diff_box = np.mean(np.abs(b_fp32 - b_int8))
    diff_proto = np.mean(np.abs(p_fp32 - p_int8))
    max_box_diff = np.max(np.abs(b_fp32 - b_int8))
    max_proto_diff = np.max(np.abs(p_fp32 - p_int8))

    print("\n--- Numerical Precision Comparison (SIL) ---")
    print(f"Box output Mean Abs Error (MAE): {diff_box:.5f} (Max Diff: {max_box_diff:.5f})")
    print(f"Proto output Mean Abs Error (MAE): {diff_proto:.5f} (Max Diff: {max_proto_diff:.5f})")
