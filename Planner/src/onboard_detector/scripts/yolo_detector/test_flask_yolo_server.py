#!/usr/bin/env python3
"""
Test script for flask_yolo_server.py.

Reads a video file, sends each frame to the Flask YOLO server via HTTP POST,
draws bounding boxes and segmentation masks on the output video.

Usage:
    conda activate torch
    python flask_yolo_server.py --port 5000                      # start server first
    python test_flask_yolo_server.py --video input.mp4 --target car,person,bicycle

    # Or with detect-first mode:
    python flask_yolo_server.py --port 5000 --detect-first
    python test_flask_yolo_server.py --video input.mp4 --target car,person --detect-first
"""

import argparse
import base64
import os
import sys
import time

import cv2
import numpy as np
import requests


def encode_frame_to_base64(frame):
    """Encode an OpenCV BGR frame to base64 JPEG string."""
    success, jpg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    return base64.b64encode(jpg_buf.tobytes()).decode('utf-8')


def decode_mask_from_base64(mask_b64, W, H):
    """Decode a base64 PNG mask back to a (H, W) uint8 numpy array."""
    if mask_b64 is None:
        return None
    try:
        png_bytes = base64.b64decode(mask_b64)
        mask_array = np.frombuffer(png_bytes, dtype=np.uint8)
        mask_img = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return None
        if mask_img.shape[0] != H or mask_img.shape[1] != W:
            mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask_img
    except Exception:
        return None


# Predefined color palette for different classes (BGR format)
COLORS = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (128, 0, 128),    # purple
    (0, 128, 128),    # teal
    (128, 128, 0),    # olive
    (0, 128, 0),      # dark green
    (128, 0, 0),      # maroon
    (255, 128, 0),    # orange
    (0, 0, 128),      # navy
    (255, 192, 203),  # pink
    (255, 255, 255),  # white
]


def get_color(idx):
    return COLORS[idx % len(COLORS)]


def draw_detections(frame, detections, class_colors):
    """
    Draw bounding boxes and segmentation masks onto a frame.

    Args:
        frame: BGR numpy array (H, W, 3)
        detections: list of detection dicts from server
        class_colors: dict mapping label_name -> (B, G, R) tuple

    Returns:
        annotated frame (same array, modified in-place)
    """
    H, W = frame.shape[:2]
    overlay = frame.copy()

    for det in detections:
        bbox = det.get('bbox', [])
        label_name = det.get('label_name', 'unknown')
        confidence = det.get('confidence', 0.0)
        mask_b64 = det.get('mask', None)

        color = class_colors.get(label_name, (0, 255, 0))

        # ---- Draw mask (semi-transparent overlay) ----
        if mask_b64 is not None:
            mask = decode_mask_from_base64(mask_b64, W, H)
            if mask is not None:
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 127] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

        # ---- Draw bounding box ----
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # ---- Draw label ----
            label = f"{label_name} {confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

    return overlay


def print_progress(frame_idx, num_frames, elapsed, detections):
    """Print concise progress information."""
    entries = [f"{d['label_name']} {d['confidence']:.2f}" for d in detections]
    det_str = ", ".join(entries) if entries else "(none)"
    fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"  frame {frame_idx:4d}/{num_frames} | FPS={fps:5.1f} | found: {det_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Test flask_yolo_server.py with a video file")
    parser.add_argument('--video', required=True,
                        help='Path to input video file')
    parser.add_argument('--target', type=str, default='',
                        help='Comma-separated target classes, e.g. car,person,bicycle')
    parser.add_argument('--server', default='http://127.0.0.1:5000',
                        help='Flask YOLO server base URL (default: http://127.0.0.1:5000)')
    parser.add_argument('--output', default='',
                        help='Output video path (default: <input>_annotated.mp4)')
    parser.add_argument('--no-mask', action='store_true',
                        help='Do not request/overlay masks (faster, bbox only)')
    parser.add_argument('--resize', type=str, default='',
                        help='Resize output video e.g. 1280x720')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start processing from this frame index')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Max frames to process (0 = entire video)')
    parser.add_argument('--conf', type=float, default=0.0,
                        help='Post-filter confidence threshold (0 = use server default)')
    args = parser.parse_args()

    # --- Resolve video path ---
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    # --- Parse target classes ---
    target_classes = [t.strip() for t in args.target.split(',') if t.strip()] if args.target else []
    if target_classes:
        print(f"[INFO] Target classes: {target_classes}")
    else:
        print("[INFO] No target_classes filter (server will return all detected objects)")

    encode_mask = not args.no_mask

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    W_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Input video : {total_frames} frames, {fps_in:.1f} fps, {W_in}x{H_in}")

    # --- Determine output path ---
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_annotated.mp4"

    # --- Output dimensions ---
    if args.resize:
        out_w, out_h = map(int, args.resize.split('x'))
    else:
        out_w, out_h = W_in, H_in

    # --- Video writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (out_w, out_h))
    if not writer.isOpened():
        # Fallback for platforms that don't have mp4v
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fallback_path = output_path.replace('.mp4', '.avi')
        print(f"[WARN] mp4v codec unavailable, falling back to {fallback_path}")
        writer = cv2.VideoWriter(fallback_path, fourcc, fps_in, (out_w, out_h))
        output_path = fallback_path

    print(f"[INFO] Output video: {output_path} ({out_w}x{out_h})")

    # --- Assign consistent colors per class ---
    class_colors = {}

    # --- Seek to start frame ---
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        print(f"[INFO] Starting from frame {args.start_frame}")

    # --- Health check ---
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        health = r.json()
        print(f"[INFO] Server mode: {health.get('mode', 'unknown')}")
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {args.server}: {e}")
        print("       Make sure flask_yolo_server.py is running.")
        sys.exit(1)

    # --- Process frames ---
    frame_idx = args.start_frame
    processed = 0
    t_start = time.time()
    server_time_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if out_w != W_in or out_h != H_in:
            frame = cv2.resize(frame, (out_w, out_h))

        H_cur, W_cur = frame.shape[:2]

        # Encode frame to base64 JPEG
        img_b64 = encode_frame_to_base64(frame)

        # Send to server
        payload = {
            "image": img_b64,
            "target_classes": target_classes,
            "encode_mask": encode_mask,
        }

        try:
            t0 = time.time()
            resp = requests.post(f"{args.server}/detect", json=payload, timeout=30)
            server_time_ms += (time.time() - t0) * 1000
            if resp.status_code != 200:
                print(f"  frame {frame_idx}: server error {resp.status_code} — {resp.text}")
                detections = []
            else:
                detections = resp.json()
        except Exception as e:
            print(f"  frame {frame_idx}: request failed — {e}")
            detections = []

        # Assign colors for any new classes seen
        for det in detections:
            name = det.get('label_name', 'unknown')
            if name not in class_colors:
                class_colors[name] = get_color(len(class_colors))

        # Post-filter by confidence if requested
        if args.conf > 0:
            detections = [d for d in detections if d.get('confidence', 0) >= args.conf]

        # Draw annotations
        annotated = draw_detections(frame, detections, class_colors)

        # Write output frame
        writer.write(annotated)

        processed += 1
        frame_idx += 1

        # Progress
        elapsed = time.time() - t_start
        print_progress(frame_idx, total_frames, elapsed, detections)

        if args.max_frames > 0 and processed >= args.max_frames:
            break

    # --- Cleanup ---
    cap.release()
    writer.release()
    elapsed = time.time() - t_start

    avg_server = server_time_ms / processed if processed > 0 else 0
    print(f"\n[DONE] Processed {processed} frames in {elapsed:.1f}s "
          f"({processed / elapsed:.1f} fps overall)")
    print(f"       Avg server latency: {avg_server:.0f} ms/frame")
    print(f"       Output: {output_path}")


if __name__ == '__main__':
    main()


# python test_flask_yolo_server.py --video test.mp4 --target bed,sofa,diningtable,bench --output result.mp4

