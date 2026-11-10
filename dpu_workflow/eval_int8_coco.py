import json
import os
import sys

os.environ["YOLO_OFFLINE"] = "true"
sys.path.insert(0, "/home/vicosilalahi/programming/yolo-fpga/ultralytics")
from ultralytics import YOLO


def main():
    int8_model_path = "/home/vicosilalahi/programming/yolo-fpga/ultralytics/checkpoints/yolov8n-seg-dpu-base-int8.onnx"

    print(f"=== Starting COCO val2017 Evaluation for INT8 Model: {int8_model_path} ===")
    model = YOLO(int8_model_path, task="segment")

    # Run validation across full COCO val2017 (5000 images) or 500 images
    # We can pass data='coco.yaml'
    metrics = model.val(data="coco.yaml", imgsz=640, batch=16, plots=False)

    res = {
        "box_map50_95": float(metrics.box.map),
        "box_map50": float(metrics.box.map50),
        "mask_map50_95": float(metrics.seg.map),
        "mask_map50": float(metrics.seg.map50),
    }
    print("=== INT8 Evaluation Results ===")
    print(json.dumps(res, indent=2))
    with open("/home/vicosilalahi/programming/yolo-fpga/ultralytics/metrics/results_int8_coco.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
