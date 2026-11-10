import glob
import os

import cv2
import numpy as np
import onnxruntime
import onnxruntime.quantization as oq
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static


class YOLOv8SegCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir: str, num_images: int = 100, input_name: str = "images"):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:num_images]
        self.input_name = input_name
        self.enum_data = iter(self.image_paths)
        print(f"[Calibrator] Initialized with {len(self.image_paths)} calibration images from {image_dir}")

    def preprocess(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        # Letterbox/resize to 640x640 with aspect ratio preserved (or direct resize for calibration)
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        nh, nw = round(h * r), round(w * r)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (640 - nh) // 2
        bottom = 640 - nh - top
        left = (640 - nw) // 2
        right = 640 - nw - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        # BGR to RGB, HWC to CHW, uint8 to float32 [0, 1]
        input_data = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(input_data, axis=0)

    def get_next(self):
        try:
            img_path = next(self.enum_data)
        except StopIteration:
            return None
        return {self.input_name: self.preprocess(img_path)}

    def rewind(self):
        self.enum_data = iter(self.image_paths)


if __name__ == "__main__":
    model_fp32 = "/home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/yolov8n-seg-dpu-base.onnx"
    model_int8 = "/home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/yolov8n-seg-dpu-base-int8.onnx"
    val_images_dir = "/home/vicosilalahi/programming/yolo-fpga/datasets/coco/images/val2017"

    print("=== Step 1: Starting ONNX Static INT8 Quantization & Calibration ===")
    session = onnxruntime.InferenceSession(model_fp32, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Model Input Name: {input_name}, Shape: {session.get_inputs()[0].shape}")

    dr = YOLOv8SegCalibrationDataReader(val_images_dir, num_images=100, input_name=input_name)

    # Static calibration using MinMax calibrator
    q_config = oq.CalibrationMethod.MinMax
    quantize_static(
        model_input=model_fp32,
        model_output=model_int8,
        calibration_data_reader=dr,
        quant_format=oq.QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=q_config,
        per_channel=True,
    )
    print(f"=== Successfully generated INT8 Quantized Model: {model_int8} ===")
