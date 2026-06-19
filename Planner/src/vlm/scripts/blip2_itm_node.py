#!/usr/bin/env python3
"""
blip2_itm_node.py — BLIP2 Image-Text Matching ROS Node for ValueMap2D

通过 HTTP 调用独立 Flask 进程中运行的 BLIP2 ITM 模型（端口 12182），
避免 ROS rospy 与 torch/LAVIS 在同一个 Python 进程中的环境冲突。

架构:
  [torch conda env]  blip2itm.py --port 12182   (Flask server, 纯 HTTP)
         ↑ HTTP POST /blip2itm
  [系统 python/rospy] blip2_itm_node.py           (ROS node, 本文件)
         ↓ /blip2/itm_score  +  /blip2/cosine_score
  [C++]               dep_indoor.cpp              (DEP 规划器)

用法:
  1. 先在 torch 环境中启动 Flask 服务器:
     conda run -n torch python -m vlm.itm.blip2itm --port 12182

  2. 再启动本 ROS 节点:
     rosrun global_planner blip2_itm_node.py
"""

import os
import sys
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

# 将 vlm 包的父目录加入 Python path
_VLM_PARENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _VLM_PARENT not in sys.path:
    sys.path.insert(0, _VLM_PARENT)

from vlm.itm.blip2itm import BLIP2ITMClient


class BLIP2ITMNode:
    """ROS 节点：通过 HTTP 调用 BLIP2 Flask 服务，发布 ITM score"""

    def __init__(self):
        rospy.init_node("blip2_itm_node", anonymous=False)

        # ---- 参数 ----
        self.rgb_topic = rospy.get_param("~rgb_topic", "/iris_0/realsense/depth_camera/color/image_raw")
        self.label = rospy.get_param("~label", "bed")
        self.room = rospy.get_param("~room", "everywhere")
        self.publish_rate = rospy.get_param("~publish_rate", 1.0)
        self.server_port = rospy.get_param("~server_port", 12182)

        # ---- HTTP 客户端 (连接 Flask BLIP2 服务器) ----
        rospy.loginfo("[BLIP2] Connecting to BLIP2 Flask server on port %d...", self.server_port)
        self.client = BLIP2ITMClient(port=self.server_port)
        rospy.loginfo("[BLIP2] HTTP client ready (server at localhost:%d)", self.server_port)

        # ---- CV Bridge ----
        self.bridge = CvBridge()

        # ---- 发布者 ----
        self.itm_score_pub = rospy.Publisher("/blip2/itm_score", Float64, queue_size=5)
        self.cosine_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=5)

        # ---- 订阅者 ----
        self.latest_image = None
        self.image_received = False
        self.image_sub = rospy.Subscriber(
            self.rgb_topic, Image, self._image_cb, queue_size=2
        )

        # ---- 定时发布 ----
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._timer_cb
        )

        rospy.loginfo("[BLIP2] Node ready. Subscribing to %s, label='%s', room='%s'",
                       self.rgb_topic, self.label, self.room)

    def _image_cb(self, msg: Image):
        """缓存最新一帧 RGB 图像"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_received = True
        except Exception as e:
            rospy.logwarn("[BLIP2] Failed to convert image: %s", e)

    def _build_prompt(self) -> str:
        """根据 label 和 room 构造 BLIP2 ITM 的文本提示"""
        if self.room.lower() != "everywhere":
            return f"Seems like there is a {self.room} or a {self.label} ahead?"
        else:
            return f"Seems like there is a {self.label} ahead?"

    def _timer_cb(self, event):
        """定时对最新图像做 ITM 推理（通过 HTTP 调用 Flask 服务）并发布"""
        if not self.image_received or self.latest_image is None:
            return

        image = self.latest_image.copy()  # BGR, 保持与 Flask server 兼容
        txt = self._build_prompt()

        try:
            t0 = time.time()
            # 通过 HTTP 调用 Flask 服务器
            cosine_score = self.client.cosine(image, txt)
            itm_score = self.client.itm_score(image, txt)
            elapsed = time.time() - t0

            self.cosine_pub.publish(Float64(cosine_score))
            self.itm_score_pub.publish(Float64(itm_score))

            rospy.loginfo("[BLIP2] cosine=%.3f  itm=%.3f  dt=%.2fs  prompt='%s'",
                           cosine_score, itm_score, elapsed, txt)
        except Exception as e:
            rospy.logerr("[BLIP2] HTTP inference error: %s", e)


if __name__ == "__main__":
    try:
        node = BLIP2ITMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

# python blip2_itm_node.py
