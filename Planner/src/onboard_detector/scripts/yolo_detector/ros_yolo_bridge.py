#!/usr/bin/env python3
"""
ROS bridge: subscribes RGB image, calls Flask YOLO server, publishes ROS messages.

Integrates LLM misdetection labels for anti-false-positive:
    1. Lookup LLM cache for target class → confusable labels + fusion_threshold
    2. Send target + confusable labels to YOLO Flask server
    3. Map detections to label_index (0=target, 1/2/3/4=confusable)
    4. Publish fusion_threshold → ObjectMap2D uses it as min_confidence_ gate

Usage (system python with rospy):
    python ros_yolo_bridge.py _target_classes:="['bed']"

Requires Flask server running in torch conda env:
    conda activate torch && python flask_yolo_server.py --detect-first
"""

import base64
import io
import json
import os
import sys

import cv2
import numpy as np
import requests
import rospy
# import ros_numpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

from plan_env.msg import SingleMasksWithConfidence

# ---- Add llm package to path ----
# Find llm/ relative to the source tree (works from both devel symlink and direct run)
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Walk up to find 'src/Planner/src/llm'
_llm_src = None
_d = _script_dir
for _ in range(10):
    _candidate = os.path.join(_d, 'src', 'Planner', 'src', 'llm')
    if os.path.isdir(_candidate):
        _llm_src = _candidate
        break
    _candidate = os.path.join(_d, 'llm')
    if os.path.isdir(_candidate):
        _llm_src = _candidate
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        break
    _d = _parent

if _llm_src is None:
    # Fallback: known workspace path
    _llm_src = os.path.expanduser('~/explorer_ws/src/Planner/src/llm')

_llm_parent = os.path.dirname(_llm_src)  # .../Planner/src/
if _llm_parent not in sys.path:
    sys.path.insert(0, _llm_parent)

from llm.answer_reader.answer_reader import read_answer

path_curr = os.path.dirname(os.path.abspath(__file__))


class LLMClient:
    """Minimal client object matching the interface expected by answer_reader."""
    def __init__(self, client_type='deepseek'):
        self.llm_client = client_type
        self.ollama = 'qwen2.5'


class YOLOBridge:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_received = False
        self.latest_img = None
        self.latest_header = None

        # --- Flask server params ---
        flask_host = rospy.get_param("~flask_host", "127.0.0.1")
        flask_port = rospy.get_param("~flask_port", 5000)
        self.flask_url = f"http://{flask_host}:{flask_port}"

        # --- Target classes ---
        self.target_classes = rospy.get_param("~target_classes", [])  # e.g. ['bed']
        self.use_habitat = rospy.get_param("/use_habitat_mode",True)

        if self.use_habitat:
            img_topic = rospy.get_param("~img_topic","/habitat/camera_rgb")
        else:
            img_topic = rospy.get_param(
                "~img_topic", "/iris_0/realsense/depth_camera/color/image_raw")

        # --- LLM integration: get confusable labels + fusion threshold ---
        llm_answer_path = rospy.get_param(
            "~llm_answer_path",
            os.path.join(_llm_src, "answers", "llm_answer_hm3d.txt"))
        llm_response_path = rospy.get_param(
            "~llm_response_path",
            os.path.join(_llm_src, "answers", "llm_responses.txt"))
        self.use_llm = rospy.get_param("~use_llm", True)

        self.confusable_labels = []
        self.fusion_threshold = 0.5  # default fallback
        self.room = "everywhere"

        if self.use_llm and self.target_classes:
            llm_client = LLMClient(client_type='deepseek')
            for target_label in self.target_classes:
                try:
                    llm_answer, room, fusion_threshold = read_answer(
                        llm_answer_path, llm_response_path, target_label, llm_client)
                    self.confusable_labels = llm_answer  # list of confusable COCO names
                    self.fusion_threshold = fusion_threshold
                    self.room = room
                    rospy.loginfo("[YOLOBridge] LLM for '%s': confusable=%s, threshold=%.2f, room=%s",
                                  target_label, self.confusable_labels,
                                  self.fusion_threshold, self.room)
                    break  # Use first target class result
                except Exception as e:
                    rospy.logwarn("[YOLOBridge] LLM lookup failed for '%s': %s", target_label, str(e))

        # --- Build full detection class list (target + confusable) ---
        if self.confusable_labels:
            self.detect_classes = list(self.target_classes) + [
                c for c in self.confusable_labels if c not in self.target_classes
            ]
        else:
            self.detect_classes = list(self.target_classes)

        rospy.loginfo("[YOLOBridge] Flask URL: %s", self.flask_url)
        rospy.loginfo("[YOLOBridge] Target classes: %s", self.target_classes)
        rospy.loginfo("[YOLOBridge] Confusable labels: %s", self.confusable_labels)
        rospy.loginfo("[YOLOBridge] Detect classes (all): %s", self.detect_classes)
        rospy.loginfo("[YOLOBridge] Fusion threshold: %.2f", self.fusion_threshold)
        rospy.loginfo("[YOLOBridge] Image topic: %s", img_topic)

        # --- Label index mapping ---
        # Build a dict: label_name → label_index
        # 0 = target, 1/2/3/4 = confusable
        self.label_index_map = {}
        for t in self.target_classes:
            self.label_index_map[t] = 0
        for i, c in enumerate(self.confusable_labels):
            if c not in self.label_index_map:
                self.label_index_map[c] = i + 1  # confusable indices start at 1

        # --- Wait for Flask to be ready ---
        self._wait_flask()

        # --- Subscribers ---
        self.img_sub = rospy.Subscriber(img_topic, Image, self.image_callback)

        # --- Publishers ---
        self.mask_pub = rospy.Publisher(
            "/yolo_detector/single_mask", SingleMasksWithConfidence, queue_size=20)
        self.threshold_pub = rospy.Publisher(
            "/detector/confidence_threshold", Float64, queue_size=5, latch=True)

        # --- Publish fusion threshold (latched) ---
        self.threshold_pub.publish(Float64(self.fusion_threshold))
        rospy.loginfo("[YOLOBridge] Published fusion_threshold=%.2f to /detector/confidence_threshold",
                      self.fusion_threshold)

        # --- Timer: process at controlled rate ---
        self.process_rate = rospy.get_param("~process_rate", 5.0)  # Hz
        self.process_timer = rospy.Timer(
            rospy.Duration(1.0 / self.process_rate), self.process_callback)

        rospy.loginfo("[YOLOBridge] Ready, processing at %.1f Hz", self.process_rate)

        self.use_visualize = rospy.get_param("~use_visualize", True)
        if self.use_visualize:
            self.vis_pub = rospy.Publisher(
                "/yolo_detector/visualization", Image, queue_size=5)
            rospy.loginfo("[YOLOBridge] Visualization enabled -> /yolo_detector/visualization")

    def _wait_flask(self, timeout=10.0):
        """Wait for Flask server to be ready."""
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            try:
                r = requests.get(f"{self.flask_url}/health", timeout=1.0)
                if r.status_code == 200:
                    rospy.loginfo("[YOLOBridge] Flask server ready")
                    return
            except Exception:
                pass
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logerr("[YOLOBridge] Flask not available after %.1fs", timeout)
                return
            rospy.sleep(0.5)

    def image_callback(self, msg):
        self.latest_img = msg
        self.latest_header = msg.header
        self.img_received = True

    def _get_label_index(self, label_name):
        """
        Map detection label_name to label_index.
        0 = target object, 1-4 = confusable object.
        Returns -1 if label is not in our detection set.
        """
        return self.label_index_map.get(label_name, -1)

    def process_callback(self, event):
        if not self.img_received:
            return
        self.img_received = False

        try:
            # Convert ROS Image to base64 jpg
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
            _, jpg_buf = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(jpg_buf.tobytes()).decode('utf-8')

            # Call Flask with ALL detection classes (target + confusable)
            payload = {
                "image": img_b64,
                "target_classes": self.detect_classes,  # now includes confusables
                "encode_mask": True,
            }
            resp = requests.post(
                f"{self.flask_url}/detect", json=payload, timeout=3.0)
            H, W = cv_img.shape[:2]
            detections = []
            if resp.status_code == 200:
                detections = resp.json() or []
            else:
                rospy.logerr("[YOLOBridge] Flask error: %s", resp.text)

            # Process each detection
            target_hits = 0
            confusable_hits = 0
            # rospy.loginfo("Get server result")
            for det in detections:
                if det.get("mask") is None:
                    continue

                label_name = det["label_name"]
                label_index = self._get_label_index(label_name)
                if label_index < 0:
                    continue  # not in our detection set

                if label_index == 0:
                    rospy.loginfo("We got the target class")
                    target_hits += 1
                else:
                    confusable_hits += 1

                msg = SingleMasksWithConfidence()
                msg.header.stamp = self.latest_header.stamp
                msg.header.frame_id = self.latest_header.frame_id

                # Decode base64 PNG mask → ROS Image (mono8)
                png_bytes = base64.b64decode(det["mask"])
                mask_array = cv2.imdecode(
                    np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask_array is None:
                    continue
                msg.mask = self.bridge.cv2_to_imgmsg(mask_array, encoding="mono8")

                msg.label_name = label_name
                msg.confidence = det["confidence"]
                msg.label_index = label_index  # KEY: 0=target, >0=confusable
                rospy.logerr("Send the yolo mask")
                self.mask_pub.publish(msg)

            if target_hits > 0 or confusable_hits > 0:
                rospy.logwarn("[YOLOBridge] Frame: %d target + %d confusable detections",
                              target_hits, confusable_hits)

            # 可视化：每帧都发布，有检测画框，无检测发原图
            if self.use_visualize:
                vis_img = cv_img.copy()
                for det in detections:
                    bbox = det.get("bbox")
                    label = det.get("label_name", "?")
                    conf = det.get("confidence", 0.0)
                    label_idx = self._get_label_index(label)
                    if bbox and len(bbox) >= 4:
                        if bbox[2] < 1.0 and bbox[3] < 1.0:
                            x1, y1 = int(bbox[0] * W), int(bbox[1] * H)
                            x2 = int((bbox[0] + bbox[2]) * W)
                            y2 = int((bbox[1] + bbox[3]) * H)
                        else:
                            x1, y1, x2, y2 = map(int, bbox[:4])
                        color = (0, 255, 0) if label_idx == 0 else (0, 0, 255)
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis_img, f"{label} {conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                vis_msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
                vis_msg.header = self.latest_header
                self.vis_pub.publish(vis_msg)

            # Visualize: draw bounding boxes + labels
            if self.use_visualize:
                # vis_img = cv_img.copy()
                for det in detections:
                    bbox = det.get("bbox")  # [x1, y1, x2, y2] or [x, y, w, h]
                    label = det.get("label_name", "?")
                    conf = det.get("confidence", 0.0)
                    label_idx = self._get_label_index(label)

                    if bbox and len(bbox) >= 4:
                        if len(bbox) == 4 and bbox[2] < 1.0 and bbox[3] < 1.0:
                            # Normalized coords → pixels
                            x1 = int(bbox[0] * W); y1 = int(bbox[1] * H)
                            x2 = int((bbox[0] + bbox[2]) * W) if bbox[2] < 1 else int(bbox[2])
                            y2 = int((bbox[1] + bbox[3]) * H) if bbox[3] < 1 else int(bbox[3])
                        else:
                            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        color = (0, 255, 0) if label_idx == 0 else (0, 0, 255)  # green=target, red=confusable
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(cv_img, f"{label} {conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                vis_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
                vis_msg.header = self.latest_header
                vis_msg.header.stamp = rospy.Time.now()
                self.vis_pub.publish(vis_msg)

        except requests.exceptions.Timeout:
            rospy.logwarn_throttle(5.0, "[YOLOBridge] Flask timeout")
        except Exception as e:
            rospy.logerr("[YOLOBridge] Error: %s", str(e))


if __name__ == "__main__":
    rospy.init_node("yolo_bridge")
    node = YOLOBridge()
    rospy.spin()


# rosrun onboard_detector ros_yolo_bridge.py _target_classes:="['toilet']" 
