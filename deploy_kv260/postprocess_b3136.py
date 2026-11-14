#!/usr/bin/env python3
"""Run the B3136 YOLOv8-seg xmodel on KV260 and write an annotated JPEG.

The class count is read from the xmodel output tensor. For non-COCO models set
CLASSES to a comma-separated list, e.g. CLASSES=crack,spalling for the 2-class
concrete model.
"""

import os
import sys

import cv2
import numpy as np
import vart
import xir

COCO = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def get_class_names():
    """Return class names from the CLASSES env var (comma-separated), defaulting to COCO."""
    raw = os.environ.get("CLASSES", "").strip()
    return [name.strip() for name in raw.split(",") if name.strip()] if raw else COCO


def letterbox(img, new=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2
    bottom = new - nh - top
    left = (new - nw) // 2
    right = new - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return keep


def color_for(cls_id):
    rng = np.random.default_rng(cls_id + 12345)
    return tuple(int(v) for v in rng.integers(40, 255, size=3))


def main():
    xmodel = sys.argv[1] if len(sys.argv) > 1 else "yolov8n_seg_kv260_b3136.xmodel"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "test.jpg"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "result_b3136.jpg"
    conf_thres = float(os.environ.get("CONF", "0.25"))
    iou_thres = float(os.environ.get("IOU", "0.45"))
    class_names = get_class_names()

    graph = xir.Graph.deserialize(xmodel)
    dpu = [
        sg
        for sg in graph.get_root_subgraph().toposort_child_subgraph()
        if sg.has_attr("device") and sg.get_attr("device") == "DPU"
    ][0]
    runner = vart.Runner.create_runner(dpu, "run")
    input_tensor = runner.get_input_tensors()[0]
    output_tensors = runner.get_output_tensors()
    in_fix = input_tensor.get_attr("fix_point")
    print("input", input_tensor.dims, input_tensor.dtype, "fix", in_fix)

    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"cannot read {image_path}")
    orig_h, orig_w = img.shape[:2]

    padded, ratio, pad_left, pad_top = letterbox(img, 640)
    # Integer LUT quantization: ~9x faster than float on the A53 and bit-exact.
    lut = np.clip(np.rint(np.arange(256, dtype=np.float32) / 255.0 * (2 ** in_fix)), 0, 127).astype(np.uint8)
    inp = cv2.LUT(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), lut).view(np.int8)[None]

    out_bufs = [np.empty(tuple(t.dims), dtype=np.int8) for t in output_tensors]
    job = runner.execute_async([inp], out_bufs)
    status = runner.wait(job)
    print("DPU wait status", status)

    boxes_by_stride = {}
    cls_by_stride = {}
    mask_by_stride = {}
    proto = None

    for tensor, buf in zip(output_tensors, out_bufs):
        name = tensor.name
        fix = tensor.get_attr("fix_point")
        arr = buf.astype(np.float32) / float(2 ** fix)
        if "Proto_proto" in name:
            proto = arr[0]  # (160, 160, 32)
            print("proto", proto.shape, "fix", fix, "range", float(proto.min()), float(proto.max()))
            continue
        try:
            idx = int(name.split("ModuleList_")[1].split("__")[0])
        except Exception:
            print("skip", name)
            continue
        if "Sequential_cv2" in name:
            boxes_by_stride[idx] = arr[0]
        elif "Sequential_cv3" in name:
            cls_by_stride[idx] = arr[0]
        elif "Sequential_cv4" in name:
            mask_by_stride[idx] = arr[0]

    if proto is None:
        raise SystemExit("proto output not found")

    n_cls = next(iter(cls_by_stride.values())).shape[-1]
    names = class_names if len(class_names) == n_cls else [f"class_{i}" for i in range(n_cls)]
    print(f"classes: {n_cls} {names}")

    all_boxes = []
    all_scores = []
    all_classes = []
    all_coeffs = []

    for idx in sorted(boxes_by_stride):
        box_raw = boxes_by_stride[idx]  # (H, W, 64)
        cls_raw = cls_by_stride[idx]  # (H, W, n_cls)
        mask_raw = mask_by_stride[idx]  # (H, W, 32)
        h, w, _ = box_raw.shape
        stride = 640 // h
        print(f"stride {idx}: H={h} W={w} stride={stride}")

        # DFL: 64 channels -> 4 sides x 16 bins -> softmax -> weighted sum
        dfl = box_raw.reshape(h, w, 4, 16)
        dfl = dfl - dfl.max(axis=-1, keepdims=True)
        prob = np.exp(dfl)
        prob = prob / prob.sum(axis=-1, keepdims=True)
        dist = (prob * np.arange(16, dtype=np.float32)).sum(axis=-1)  # (H,W,4)

        ys, xs = np.meshgrid(np.arange(h, dtype=np.float32) + 0.5,
                             np.arange(w, dtype=np.float32) + 0.5, indexing="ij")
        anchor = np.stack([xs, ys], axis=-1)
        lt = dist[..., :2]
        rb = dist[..., 2:]
        boxes = np.concatenate([anchor - lt, anchor + rb], axis=-1) * stride
        boxes = boxes.reshape(-1, 4)

        scores = sigmoid(cls_raw.reshape(-1, n_cls))
        coeffs = mask_raw.reshape(-1, mask_raw.shape[-1])

        cls_ids = scores.argmax(axis=1)
        conf = scores[np.arange(scores.shape[0]), cls_ids]
        keep = np.where(conf >= conf_thres)[0]
        if keep.size:
            all_boxes.append(boxes[keep])
            all_scores.append(conf[keep])
            all_classes.append(cls_ids[keep])
            all_coeffs.append(coeffs[keep])
            print(f"  candidates: {keep.size}, max conf {conf[keep].max():.3f}, best class {names[int(cls_ids[keep][conf[keep].argmax()])]}")

    if not all_boxes:
        print(f"no detections above conf={conf_thres}")
        cv2.imwrite(out_path, img)
        return

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    coeffs = np.concatenate(all_coeffs, axis=0)

    # class-aware NMS
    keep_all = []
    for c in np.unique(classes):
        idxs = np.where(classes == c)[0]
        for k in nms(boxes[idxs], scores[idxs], iou_thres):
            keep_all.append(int(idxs[k]))
    keep_all = keep_all[:100]
    boxes = boxes[keep_all]
    scores = scores[keep_all]
    classes = classes[keep_all]
    coeffs = coeffs[keep_all]

    print(f"detections after NMS: {len(boxes)}")

    proto_flat = proto.reshape(-1, proto.shape[-1]).T  # (32, 160*160)
    overlay = img.copy()

    for box, score, cls_id, coeff in zip(boxes, scores, classes, coeffs):
        # box is in 640x640 letterboxed coordinates
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(639.0, x1))
        y1 = max(0.0, min(639.0, y1))
        x2 = max(0.0, min(639.0, x2))
        y2 = max(0.0, min(639.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # mask in 160x160 proto space
        m = sigmoid(coeff @ proto_flat).reshape(proto.shape[0], proto.shape[1])
        mx1 = int(np.floor(x1 * proto.shape[1] / 640.0))
        my1 = int(np.floor(y1 * proto.shape[0] / 640.0))
        mx2 = int(np.ceil(x2 * proto.shape[1] / 640.0))
        my2 = int(np.ceil(y2 * proto.shape[0] / 640.0))
        mx1 = max(0, min(proto.shape[1] - 1, mx1))
        my1 = max(0, min(proto.shape[0] - 1, my1))
        mx2 = max(mx1 + 1, min(proto.shape[1], mx2))
        my2 = max(my1 + 1, min(proto.shape[0], my2))
        mc = m[my1:my2, mx1:mx2]

        # original image coordinates
        ox1 = int(round((x1 - pad_left) / ratio))
        oy1 = int(round((y1 - pad_top) / ratio))
        ox2 = int(round((x2 - pad_left) / ratio))
        oy2 = int(round((y2 - pad_top) / ratio))
        ox1 = max(0, min(orig_w - 1, ox1))
        oy1 = max(0, min(orig_h - 1, oy1))
        ox2 = max(ox1 + 1, min(orig_w, ox2))
        oy2 = max(oy1 + 1, min(orig_h, oy2))

        if mc.size == 0:
            continue
        mm = cv2.resize(mc, (ox2 - ox1, oy2 - oy1), interpolation=cv2.INTER_LINEAR)
        mask = mm > 0.5
        color = np.array(color_for(int(cls_id)), dtype=np.uint8)
        region = overlay[oy1:oy2, ox1:ox2]
        region[mask] = (region[mask] * 0.45 + color * 0.55).astype(np.uint8)
        overlay[oy1:oy2, ox1:ox2] = region

        cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), color_for(int(cls_id)), 2)
        label = f"{names[int(cls_id)]} {float(score):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (ox1, max(0, oy1 - th - 6)), (ox1 + tw + 4, oy1), color_for(int(cls_id)), -1)
        cv2.putText(overlay, label, (ox1 + 2, max(12, oy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, overlay)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
