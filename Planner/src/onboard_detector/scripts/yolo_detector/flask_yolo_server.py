#!/usr/bin/env python3
"""
Flask YOLO segmentation server — runs in torch conda environment.

Two modes (controlled by DETECT_FIRST flag or --detect-first CLI arg):
  Mode A (DETECT_FIRST=True):
    1. Run detection model (yolo26s.pt) to find objects → get bounding boxes + labels
    2. For each detected box, crop ROI from original image
    3. Run segmentation model (yolo26s-seg.pt) on each cropped ROI → get mask
    4. Return: boxes + masks
  Mode B (DETECT_FIRST=False, default):
    Run segmentation model directly, same as before.

Usage:
    conda activate torch
    python flask_yolo_server.py --port 5000                      # seg only
    python flask_yolo_server.py --port 5000 --detect-first        # detect → seg

POST /detect   JSON body: {"image": "<base64 jpg>", "target_classes": ["car"]}
Response:      [{"mask": "<base64 png>", "label_name": "car",
                 "confidence": 0.85, "label_index": 2,
                 "bbox": [x1, y1, x2, y2]}, ...]
"""

# Mitigate CUDA memory fragmentation (helps when two models share 8 GB VRAM)
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import base64
import io
import sys
import time

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify

from ultralytics import YOLO

app = Flask(__name__)

# --- Globals ---
MODEL = None           # segmentation model (or unified model when DETECT_FIRST=False)
DETECT_MODEL = None    # detection-only model (used when DETECT_FIRST=True)
LABEL_NAMES = []
CONF_THRESH = 0.7
IMG_SIZE = 640
DETECT_FIRST = False   # control flag: True → detect then segment; False → segment directly

path_curr = os.path.dirname(os.path.abspath(__file__))


def _gpu_memory_info():
    """Return (allocated_MiB, reserved_MiB, free_MiB, total_MiB) or None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserv = torch.cuda.memory_reserved() / (1024 ** 2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    free = total - reserv
    return alloc, reserv, free, total


def _log_gpu(label="GPU"):
    """Print GPU memory usage with a label."""
    info = _gpu_memory_info()
    if info:
        alloc, reserv, free, total = info
        print(f"[FlaskYOLO] {label}: allocated={alloc:.0f}MiB reserved={reserv:.0f}MiB "
              f"free={free:.0f}MiB total={total:.0f}MiB")


def load_model(weight_path, detect_weight_path, class_names_path, conf_thresh, img_size, detect_first):
    global MODEL, DETECT_MODEL, LABEL_NAMES, CONF_THRESH, IMG_SIZE, DETECT_FIRST

    DETECT_FIRST = detect_first
    _log_gpu("before loading")

    if DETECT_FIRST:
        # Load both models
        DETECT_MODEL = YOLO(detect_weight_path)
        DETECT_MODEL.model.half()
        torch.cuda.empty_cache()
        _log_gpu("after detect model (FP16)")

        MODEL = YOLO(weight_path)  # segmentation model
        MODEL.model.half()
        torch.cuda.empty_cache()
        _log_gpu("after seg model (FP16)")

        print(f"[FlaskYOLO] Mode: DETECT → SEGMENT (two-stage pipeline, FP16)")
        print(f"[FlaskYOLO] Detection model : {detect_weight_path}")
        print(f"[FlaskYOLO] Segmentation model: {weight_path}")
    else:
        # Load segmentation model only (default)
        MODEL = YOLO(weight_path)
        MODEL.model.half()
        torch.cuda.empty_cache()
        _log_gpu("after seg model (FP16)")

        print(f"[FlaskYOLO] Mode: SEGMENT ONLY (FP16)")
        print(f"[FlaskYOLO] Segmentation model: {weight_path}")

    with open(class_names_path, 'r') as f:
        LABEL_NAMES = [line.strip() for line in f.readlines()]
    CONF_THRESH = conf_thresh
    IMG_SIZE = img_size
    print(f"[FlaskYOLO] Classes: {len(LABEL_NAMES)}, conf_thresh={CONF_THRESH}")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "mode": "detect_then_segment" if DETECT_FIRST else "segment_only",
        "seg_model_loaded": MODEL is not None,
        "detect_model_loaded": DETECT_MODEL is not None,
        "detect_first": DETECT_FIRST,
    })


def encode_mask_to_base64(mask_data, W, H):
    """Resize mask to original image size, binarize, and encode as base64 PNG."""
    mask_resized = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_bool = (mask_resized > 0.5).astype(np.uint8) * 255
    success, png_buf = cv2.imencode('.png', mask_bool)
    if success:
        return base64.b64encode(png_buf.tobytes()).decode('utf-8')
    return None


@app.route('/detect', methods=['POST'])
def detect():
    """Main endpoint: receive RGB image, return detections (with optional masks)."""
    t0 = time.time()
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "missing 'image' field"}), 400

    target_classes = set(data.get('target_classes', []))
    encode_mask = data.get('encode_mask', True)

    # Decode image
    try:
        img_bytes = base64.b64decode(data['image'])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        ori_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if ori_img is None:
            return jsonify({"error": "failed to decode image"}), 400
    except Exception as e:
        return jsonify({"error": f"image decode error: {str(e)}"}), 400

    H, W = ori_img.shape[:2]

    if DETECT_FIRST:
        # ────────────────────────────────────────────────────────
        # Two-stage pipeline: detect objects → segment each ROI
        # ────────────────────────────────────────────────────────
        detections = _detect_then_segment(ori_img, target_classes, encode_mask)
    else:
        # ────────────────────────────────────────────────────────
        # Single-stage: segmentation model directly
        # ────────────────────────────────────────────────────────
        detections = _segment_directly(ori_img, target_classes, encode_mask)

    elapsed = (time.time() - t0) * 1000
    print(f"[FlaskYOLO] {len(detections)} detections in {elapsed:.1f}ms")
    return jsonify(detections)


def _detect_then_segment(ori_img, target_classes, encode_mask):
    """
    Stage 1: Run detection model to find bounding boxes + labels.
    Stage 2: For each detected box, crop the ROI and run segmentation model.
    Returns list of detections with both bbox and mask.
    """
    H, W = ori_img.shape[:2]
    detections = []

    # Stage 1 — Detection
    torch.cuda.empty_cache()
    with torch.no_grad():
        det_results = DETECT_MODEL(ori_img, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)
    torch.cuda.empty_cache()

    for result in det_results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            category_id = int(boxes.cls[i].cpu().numpy())
            category = LABEL_NAMES[category_id]
            confidence = float(boxes.conf[i].cpu().numpy())

            # Filter by target classes
            if target_classes and category not in target_classes:
                continue

            # Bounding box in xyxy format (int)
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            det = {
                "label_name": category,
                "confidence": round(confidence, 4),
                "label_index": category_id,
                "bbox": [x1, y1, x2, y2],
            }

            # Stage 2 — Segmentation on cropped ROI
            if encode_mask and x2 > x1 and y2 > y1:
                roi = ori_img[y1:y2, x1:x2]
                roi_h, roi_w = y2 - y1, x2 - x1
                roi_cx, roi_cy = roi_w // 2, roi_h // 2

                with torch.no_grad():
                    seg_results = MODEL(roi, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)

                # Collect all valid (mask, seg_cls_id, seg_conf) from the ROI
                mask_candidates = []  # each: (mask_np, seg_cls_id, seg_conf, center_dist)
                for seg_r in seg_results:
                    if seg_r.masks is None or len(seg_r.masks) == 0:
                        continue
                    n_masks = len(seg_r.masks)
                    for mi in range(n_masks):
                        seg_cls_id = -1
                        seg_conf = 0.0
                        if seg_r.boxes is not None and mi < len(seg_r.boxes):
                            seg_cls_id = int(seg_r.boxes.cls[mi].cpu().numpy())
                            seg_conf = float(seg_r.boxes.conf[mi].cpu().numpy())
                        mask_np = seg_r.masks.data[mi].cpu().numpy()  # (h_roi, w_roi)

                        # Distance from mask centroid to ROI center (for fallback)
                        moments = cv2.moments((mask_np > 0.5).astype(np.uint8))
                        if moments["m00"] > 0:
                            mask_cx = moments["m10"] / moments["m00"]
                            mask_cy = moments["m01"] / moments["m00"]
                            center_dist = np.sqrt((mask_cx - roi_cx) ** 2 + (mask_cy - roi_cy) ** 2)
                        else:
                            center_dist = float("inf")

                        mask_candidates.append((mask_np, seg_cls_id, seg_conf, center_dist))

                # --- Select the best mask ---
                best_mask = None
                if mask_candidates:
                    # Priority 1: class matches Stage 1 detection category
                    class_matched = [
                        (m, cid, conf, dist)
                        for m, cid, conf, dist in mask_candidates
                        if cid >= 0 and LABEL_NAMES[cid] == category
                    ]
                    if class_matched:
                        # Among class-matched, pick highest confidence
                        best = max(class_matched, key=lambda x: x[2])
                        best_mask = best[0]
                        print(f"[FlaskYOLO] Stage2: class-matched '{category}' "
                              f"conf={best[2]:.3f}")
                    else:
                        # Fallback: pick mask closest to ROI center
                        best = min(mask_candidates, key=lambda x: x[3])
                        best_mask = best[0]
                        seg_label = LABEL_NAMES[best[1]] if best[1] >= 0 else "unknown"
                        print(f"[FlaskYOLO] Stage2: WARNING — no class match for "
                              f"'{category}', fallback to '{seg_label}' "
                              f"(conf={best[2]:.3f}, dist_to_center={best[3]:.1f}px)")

                if best_mask is not None:
                    mask_resized = cv2.resize(best_mask, (roi_w, roi_h),
                                              interpolation=cv2.INTER_NEAREST)
                    mask_full = np.zeros((H, W), dtype=np.uint8)
                    mask_full[y1:y2, x1:x2] = (mask_resized > 0.5).astype(np.uint8) * 255
                    det["mask"] = encode_mask_to_base64(
                        mask_full.astype(np.float32), W, H)
                else:
                    det["mask"] = None

                # Free per-box GPU memory
                del seg_results, mask_candidates, best_mask
                torch.cuda.empty_cache()
            else:
                det["mask"] = None

            detections.append(det)

    # Free detection results memory
    del det_results
    torch.cuda.empty_cache()

    return detections


def _segment_directly(ori_img, target_classes, encode_mask):
    """
    Single-stage: run segmentation model directly.
    Same behavior as the original /detect endpoint.
    """
    H, W = ori_img.shape[:2]
    detections = []

    torch.cuda.empty_cache()
    with torch.no_grad():
        results = MODEL(ori_img, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)
    torch.cuda.empty_cache()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        num_boxes = len(boxes)
        has_masks = result.masks is not None and len(result.masks) == num_boxes

        for i in range(num_boxes):
            category_id = int(boxes.cls[i].cpu().numpy())
            category = LABEL_NAMES[category_id]
            confidence = float(boxes.conf[i].cpu().numpy())

            if target_classes and category not in target_classes:
                continue

            # Bounding box (from seg model)
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()

            det = {
                "label_name": category,
                "confidence": round(confidence, 4),
                "label_index": category_id,
                "bbox": [x1, y1, x2, y2],
            }

            if has_masks and encode_mask:
                mask_data = result.masks.data[i].cpu().numpy()
                det["mask"] = encode_mask_to_base64(mask_data, W, H)
            else:
                det["mask"] = None

            detections.append(det)

    del results
    torch.cuda.empty_cache()

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--weight', default=os.path.join(path_curr, 'weights/yolo26s-seg.pt'),
                        help='Segmentation model weight path')
    parser.add_argument('--detect-weight', default=os.path.join(path_curr, 'weights/yolo26s.pt'),
                        help='Detection model weight path (used when --detect-first)')
    parser.add_argument('--classes', default=os.path.join(path_curr, 'config/coco.names'))
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--detect-first', action='store_true', default=False,
                        help='Enable two-stage pipeline: detect (boxes) → segment (masks)')
    args = parser.parse_args()

    load_model(args.weight, args.detect_weight, args.classes,
               args.conf, args.imgsz, args.detect_first)
    print(f"[FlaskYOLO] Starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)

#  Usage:
#    python flask_yolo_server.py --port 5000                      # segment only
#    python flask_yolo_server.py --port 5000 --detect-first        # detect → segment
