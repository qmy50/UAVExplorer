#!/usr/bin/env python3
"""
ros_vlm_bridge.py — ApexNav-style VLM pipeline bridge for ROS.

Architecture:
  yolo26s (COCO classes) + GroundingDINO (open-vocabulary) + MobileSAM (segmentation)

Architecture:
  RGB Image (ROS) → YOLO26DetectClient (COCO) / GroundingDINOClient (non-COCO)
                  → MobileSAMClient (per-bbox segmentation)
                  → SingleMasksWithConfidence (ROS msg, same format as before)
                  → C++ ObjectMapManager (unchanged!)

Flask servers must be running BEFORE this node:
  python vlm/detector/yolo26_detect.py --port 12185
  python vlm/detector/grounding_dino.py --port 12181
  python vlm/segmentor/sam.py --port 12183

Usage:
  python ros_vlm_bridge.py _target_classes:="['bed']"
"""

import base64
import io
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np
import rospy
# import ros_numpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, String

from plan_env.msg import SingleMasksWithConfidence

# ── Add VLM modules to path ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Walk up to find vlm/
_vlm_src = None
_d = _script_dir
for _ in range(10):
    _candidate = os.path.join(_d, 'src', 'Planner', 'src', 'vlm')
    if os.path.isdir(_candidate):
        _vlm_src = _candidate
        break
    _candidate = os.path.join(_d, 'vlm')
    if os.path.isdir(_candidate):
        _vlm_src = _candidate
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        break
    _d = _parent

if _vlm_src is None:
    _vlm_src = os.path.expanduser('~/explorer/Planner/src/vlm')

_vlm_parent = os.path.dirname(_vlm_src)  # .../Planner/src/
if _vlm_parent not in sys.path:
    sys.path.insert(0, _vlm_parent)

# ── Add LLM modules to path ──
_llm_src = os.path.join(os.path.dirname(_vlm_src), 'llm')
if os.path.isdir(_llm_src):
    _llm_parent = os.path.dirname(_llm_src)
    if _llm_parent not in sys.path:
        sys.path.insert(0, _llm_parent)

from vlm.coco_classes import COCO_CLASSES
from vlm.detector.yolo26_detect import YOLO26DetectClient
from vlm.detector.grounding_dino import GroundingDINOClient
from vlm.segmentor.sam import MobileSAMClient
from llm.answer_reader.answer_reader import read_answer


# ── Simple config object to match ApexNav's cfg interface ──
class DetectorConfig:
    class yolo:
        agnostic_nms = True
        confidence_threshold_yolo = 0.50
        iou_threshold_yolo = 0.5

    class groundingDINO:
        confidence_threshold_dino = 0.40
        text_threshold = 0.25


class LLMClient:
    """Minimal client for answer_reader compatibility."""
    def __init__(self, client_type='deepseek'):
        self.llm_client = client_type
        self.ollama = 'qwen2.5'


class VLMBridge:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_received = False
        self.latest_img = None
        self.latest_header = None

        # ── Flask server config ──
        yolo_port = rospy.get_param("~yolo_port", 12184)
        dino_port = rospy.get_param("~dino_port", 12181)
        sam_port = rospy.get_param("~sam_port", 12183)

        # ── Target classes ──
        self.target_classes = rospy.get_param("~target_classes", [])
        self.use_habitat = rospy.get_param("/use_habitat_mode", True)

        if self.use_habitat:
            img_topic = rospy.get_param("~img_topic", "/habitat/camera_rgb")
        else:
            img_topic = rospy.get_param(
                "~img_topic", "/iris_0/realsense/depth_camera/color/image_raw")

        # ── LLM integration ──
        llm_answer_path = rospy.get_param(
            "~llm_answer_path",
            os.path.join(_llm_src, "answers", "llm_answer_hm3d.txt"))
        llm_response_path = rospy.get_param(
            "~llm_response_path",
            os.path.join(_llm_src, "answers", "llm_responses.txt"))
        self.use_llm = rospy.get_param("~use_llm", True)

        self.confusable_labels = []
        self.fusion_threshold = 0.5
        self.room = "everywhere"

        if self.use_llm and self.target_classes:
            llm_client = LLMClient(client_type='deepseek')
            for target_label in self.target_classes:
                try:
                    llm_answer, room, fusion_threshold = read_answer(
                        llm_answer_path, llm_response_path, target_label, llm_client)
                    self.confusable_labels = llm_answer
                    self.fusion_threshold = fusion_threshold
                    self.room = room
                    rospy.loginfo(
                        "[VLMBridge] LLM for '%s': confusable=%s, "
                        "threshold=%.2f, room=%s",
                        target_label, self.confusable_labels,
                        self.fusion_threshold, self.room)
                    break
                except Exception as e:
                    rospy.logwarn(
                        "[VLMBridge] LLM lookup failed for '%s': %s",
                        target_label, str(e))

        # ── Build label index map ──
        # 0 = target, 1/2/3/4 = confusable
        self.label_index_map = {}
        for t in self.target_classes:
            self.label_index_map[t] = 0
        for i, c in enumerate(self.confusable_labels):
            if c not in self.label_index_map:
                self.label_index_map[c] = i + 1

        # Split all labels into COCO vs non-COCO (for routing to yolo26s / GroundingDINO)
        all_labels = list(self.target_classes) + [
            c for c in self.confusable_labels if c not in self.target_classes
        ]
        self.coco_labels = [l for l in all_labels if l in COCO_CLASSES]
        self.dino_labels = [l for l in all_labels if l not in COCO_CLASSES]

        # If any target label is non-COCO → GroundingDINO handles ALL labels
        # (same logic as ApexNav's get_object_utils.py lines 82-87)
        if any(item in self.dino_labels for item in self.target_classes):
            self.dino_labels = all_labels
            # COCO target labels still go through yolo26s as supplement
            self.coco_labels = [
                l for l in self.target_classes if l in COCO_CLASSES
            ]

        # ── Init Flask clients ──
        rospy.loginfo("[VLMBridge] Connecting to Flask servers...")
        self.yolo_client = YOLO26DetectClient(port=yolo_port)
        self.dino_client = GroundingDINOClient(port=dino_port)
        self.sam_client = MobileSAMClient(port=sam_port)
        rospy.loginfo("[VLMBridge] All Flask clients initialized")

        # ── Detection config (matches ApexNav's habitat_eval yaml) ──
        self.cfg = DetectorConfig()
        # Allow override from ROS params
        self.cfg.yolo.confidence_threshold_yolo = rospy.get_param(
            "~yolo_conf_thresh", 0.3)
        self.cfg.yolo.iou_threshold_yolo = rospy.get_param(
            "~yolo_iou_thresh", 0.5)
        self.cfg.groundingDINO.confidence_threshold_dino = rospy.get_param(
            "~dino_box_thresh", 0.40)
        self.cfg.groundingDINO.text_threshold = rospy.get_param(
            "~dino_text_thresh", 0.25)

        rospy.loginfo("[VLMBridge] Target classes: %s", self.target_classes)
        rospy.loginfo("[VLMBridge] Confusable labels: %s", self.confusable_labels)
        rospy.loginfo("[VLMBridge] COCO labels → yolo26s: %s", self.coco_labels)
        rospy.loginfo("[VLMBridge] DINO labels → GroundingDINO: %s", self.dino_labels)
        rospy.loginfo("[VLMBridge] YOLO conf=%.2f iou=%.2f | DINO box=%.2f text=%.2f",
                      self.cfg.yolo.confidence_threshold_yolo,
                      self.cfg.yolo.iou_threshold_yolo,
                      self.cfg.groundingDINO.confidence_threshold_dino,
                      self.cfg.groundingDINO.text_threshold)
        rospy.loginfo("[VLMBridge] Image topic: %s", img_topic)

        # ── Subscribers ──
        self.img_sub = rospy.Subscriber(img_topic, Image, self.image_callback)

        # ── Publishers ──
        self.mask_pub = rospy.Publisher(
            "/yolo_detector/single_mask", SingleMasksWithConfidence, queue_size=20)
        self.threshold_pub = rospy.Publisher(
            "/detector/confidence_threshold", Float64, queue_size=5, latch=True)

        # Publish fusion threshold (latched)
        self.threshold_pub.publish(Float64(self.fusion_threshold))
        rospy.loginfo("[VLMBridge] Published fusion_threshold=%.2f",
                      self.fusion_threshold)

        # ── Visualisation ──
        self.use_visualize = rospy.get_param("~use_visualize", True)
        if self.use_visualize:
            self.vis_pub = rospy.Publisher(
                "/yolo_detector/visualization", Image, queue_size=5)

        # ── Timer: process at controlled rate ──
        self.process_rate = rospy.get_param("~process_rate", 5.0)  # Hz
        self.process_timer = rospy.Timer(
            rospy.Duration(1.0 / self.process_rate), self.process_callback)

        # ── Dynamic target reconfiguration ──
        self.target_sub = rospy.Subscriber(
            "/vlm_bridge/target_config", String, self._target_config_callback)

        rospy.loginfo("[VLMBridge] Ready, processing at %.1f Hz",
                      self.process_rate)

    def image_callback(self, msg):
        self.latest_img = msg
        self.latest_header = msg.header
        self.img_received = True

    def _target_config_callback(self, msg: String):
        """Receive dynamic target config from evaluation node."""
        try:
            config = json.loads(msg.data)
            self._reconfigure_targets(config)
        except Exception as e:
            rospy.logerr("[VLMBridge] Failed to parse target_config: %s", e)

    def _reconfigure_targets(self, config: dict):
        """Reconfigure detection targets on the fly."""
        self.target_classes = config.get("target_classes", [])
        self.confusable_labels = config.get("confusable_labels", [])
        self.room = config.get("room", "everywhere")
        self.fusion_threshold = config.get("fusion_threshold", 0.5)

        # Rebuild label index map
        self.label_index_map = {}
        for t in self.target_classes:
            self.label_index_map[t] = 0
        for i, c in enumerate(self.confusable_labels):
            if c not in self.label_index_map:
                self.label_index_map[c] = i + 1

        # Split into COCO vs DINO
        all_labels = list(self.target_classes) + [
            c for c in self.confusable_labels if c not in self.target_classes
        ]
        self.coco_labels = [l for l in all_labels if l in COCO_CLASSES]
        self.dino_labels = [l for l in all_labels if l not in COCO_CLASSES]

        if any(item in self.dino_labels for item in self.target_classes):
            self.dino_labels = list(set(all_labels))
            self.coco_labels = [l for l in self.target_classes if l in COCO_CLASSES]

        # Publish updated threshold
        self.threshold_pub.publish(Float64(self.fusion_threshold))

        rospy.loginfo("[VLMBridge] Reconfigured: target=%s, confusable=%s, room=%s",
                      self.target_classes, self.confusable_labels, self.room)

    def _get_label_index(self, label_name):
        return self.label_index_map.get(label_name, -1)

    def _segment_bbox(self, img, bbox_xyxy):
        """Run MobileSAM on a single bounding box. Returns binary mask (H, W)."""
        try:
            object_mask = self.sam_client.segment_bbox(img, bbox_xyxy)
            return object_mask
        except Exception as e:
            rospy.logwarn("[VLMBridge] MobileSAM segmentation failed: %s", str(e))
            return np.zeros(img.shape[:2], dtype=np.uint8)

    def _encode_mask_to_msg(self, mask_bool, header):
        """Convert bool numpy mask → ROS Image mono8 message."""
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
        return self.bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")

    def process_callback(self, event):
        if not self.img_received:
            return
        self.img_received = False

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
            H, W = cv_img.shape[:2]

            target_hits = 0
            confusable_hits = 0
            vis_detections = []  # (x1,y1,x2,y2, mask_bool, label_name, score, label_index)

            # ═══════════════════════════════════════════════════════════
            # Stage 1: yolo26s for COCO-class targets
            # ═══════════════════════════════════════════════════════════
            if self.coco_labels:
                try:
                    detections = self.yolo_client.predict(
                        cv_img,
                        agnostic_nms=self.cfg.yolo.agnostic_nms,
                        conf_thres=self.cfg.yolo.confidence_threshold_yolo,
                        iou_thres=self.cfg.yolo.iou_threshold_yolo,
                    )
                    for idx in range(detections.num_detections):
                        label_detected = detections.phrases[idx]
                        score = detections.logits[idx].item()
                        label_index = self._get_label_index(label_detected)
                        if label_index < 0:
                            continue

                        # Bbox in normalized xyxy → pixel coords
                        bbox = detections.boxes[idx].tolist()
                        x1, y1, x2, y2 = [
                            int(v * s) for v, s in zip(bbox, [W, H, W, H])
                        ]

                        # Run MobileSAM segmentation
                        mask = self._segment_bbox(cv_img, [x1, y1, x2, y2])

                        # Publish
                        msg = SingleMasksWithConfidence()
                        msg.header.stamp = self.latest_header.stamp
                        msg.header.frame_id = self.latest_header.frame_id
                        msg.mask = self._encode_mask_to_msg(mask > 0, self.latest_header)
                        msg.label_name = label_detected
                        msg.confidence = score
                        msg.label_index = label_index
                        self.mask_pub.publish(msg)

                        # Collect for visualisation
                        vis_detections.append(
                            (x1, y1, x2, y2, mask, label_detected, score, label_index))

                        if label_index == 0:
                            target_hits += 1
                        else:
                            confusable_hits += 1

                except Exception as e:
                    rospy.logerr("[VLMBridge] yolo26s error: %s\n%s",
                                 str(e), traceback.format_exc())

            # ═══════════════════════════════════════════════════════════
            # Stage 2: GroundingDINO for open-vocabulary targets
            # ═══════════════════════════════════════════════════════════
            if self.dino_labels:
                try:
                    caption = ' '.join(f'{item}.  ' for item in self.dino_labels)
                    detections = self.dino_client.predict(
                        cv_img,
                        caption=caption,
                        box_threshold=self.cfg.groundingDINO.confidence_threshold_dino,
                        text_threshold=self.cfg.groundingDINO.text_threshold,
                    )
                    for idx in range(detections.num_detections):
                        label_detected = detections.phrases[idx]
                        score = detections.logits[idx].item()
                        label_index = self._get_label_index(label_detected)
                        if label_index < 0:
                            continue

                        # Bbox in normalized xyxy → pixel coords
                        bbox = detections.boxes[idx].tolist()
                        x1, y1, x2, y2 = [
                            int(v * s) for v, s in zip(bbox, [W, H, W, H])
                        ]
                        # Skip too-large boxes (sampling the whole image is a bug)
                        bbox_area = (x2 - x1) * (y2 - y1)
                        img_area = W * H
                        if bbox_area / img_area >= 0.99:
                            continue

                        # Run MobileSAM segmentation
                        mask = self._segment_bbox(cv_img, [x1, y1, x2, y2])

                        # Publish
                        msg = SingleMasksWithConfidence()
                        msg.header.stamp = self.latest_header.stamp
                        msg.header.frame_id = self.latest_header.frame_id
                        msg.mask = self._encode_mask_to_msg(mask > 0, self.latest_header)
                        msg.label_name = label_detected
                        msg.confidence = score
                        msg.label_index = label_index
                        self.mask_pub.publish(msg)

                        # Collect for visualisation
                        vis_detections.append(
                            (x1, y1, x2, y2, mask, label_detected, score, label_index))

                        if label_index == 0:
                            target_hits += 1
                        else:
                            confusable_hits += 1

                except Exception as e:
                    rospy.logerr("[VLMBridge] GroundingDINO error: %s\n%s",
                                 str(e), traceback.format_exc())

            if target_hits > 0 or confusable_hits > 0:
                rospy.loginfo("[VLMBridge] Frame: %d target + %d confusable",
                              target_hits, confusable_hits)

            # ── Visualisation ──
            if self.use_visualize:
                vis_img = cv_img.copy()
                for (x1, y1, x2, y2, mask, label, score, label_index) in vis_detections:
                    # Color: green(0,255,0) for target, red(0,0,255) for confusable
                    color = (0, 255, 0) if label_index == 0 else (0, 0, 255)

                    # Draw bounding box
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness=2)

                    # Draw label + confidence
                    text = f"{label} {score:.2f}"
                    (tw, th), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis_img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
                    cv2.putText(vis_img, text, (x1 + 1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Draw mask overlay (semi-transparent)
                    if mask is not None and mask.any():
                        overlay = vis_img.copy()
                        overlay[mask > 0] = color
                        vis_img = cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0)

                vis_msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
                vis_msg.header = self.latest_header
                self.vis_pub.publish(vis_msg)

        except Exception as e:
            rospy.logerr("[VLMBridge] process_callback error: %s\n%s",
                         str(e), traceback.format_exc())


if __name__ == "__main__":
    rospy.init_node("vlm_bridge")
    node = VLMBridge()
    rospy.spin()
