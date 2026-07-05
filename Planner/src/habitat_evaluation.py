#!/usr/bin/env python3
"""
habitat_evaluation.py — Habitat ObjectNav Evaluation with ROS VLM Bridge

VLM/ITM via ROS nodes (ros_vlm_bridge + blip2_itm_node), not direct HTTP.
No torch dependency — evaluation only uses rospy + cv_bridge.

Features:
  - ObjectNav config loading (HM3D-v1/v2, MP3D) via Hydra
  - Full val-set episode iteration with tqdm progress bar
  - VLM detection via ros_vlm_bridge (/yolo_detector/single_mask)
  - ITM scoring via blip2_itm_node (/blip2/cosine_score)
  - LLM integration for confusable labels + fusion threshold
  - ROS FSM handshake: subscribe /habitat/plan_action, publish /habitat/state
  - Evaluation metrics: success, SPL, soft_SPL, distance_to_goal
  - Failure classification (10 categories) + checkpoint/resume

Usage:
    conda activate habitat
    cd /root/gpufree-data/explorer_ws/src/Planner/src

    # Prerequisites:
    #   T1: roscore
    #   T2: Flask VLM servers (yolo26:12184, dino:12181, sam:12183, blip2:12185)
    #   T3: C++ FSM    roslaunch minco_curve run_in_habitat.launch
    #   T3b: ros_vlm_bridge    python onboard_detector/.../ros_vlm_bridge.py
    #   T3c: blip2_itm_node    rosrun global_planner blip2_itm_node.py
    #   T4: python habitat_evaluation.py --dataset hm3dv2

    # Minimal test (no VLM, no ITM):
    python habitat_evaluation.py --dataset hm3dv2 use_vlm=false use_itm=false
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import sys
import time
from copy import deepcopy

import cv2
import numpy as np
import rospy
import tqdm
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion, PoseStamped
from hydra import compose, initialize
from omegaconf import OmegaConf
from nav_msgs.msg import Odometry
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float64, Int32, Int32MultiArray, Float32MultiArray, String, Bool

import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ── Add project src + ROS workspace to path ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# ROS workspace devel path (for plan_env.msg etc.)
_explorer_ws_devel = os.path.expanduser("~/explorer_ws/devel/lib/python3/dist-packages")
if os.path.isdir(_explorer_ws_devel) and _explorer_ws_devel not in sys.path:
    sys.path.insert(0, _explorer_ws_devel)

from plan_env.msg import SingleMasksWithConfidence
from params import HABITAT_STATE, ACTION, RESULT_TYPES, FINAL_RESULT, EXPL_RESULT
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from llm.answer_reader.answer_reader import read_answer
from vlm.Labels import MP3D_ID_TO_NAME, HM3D_ID_TO_NAME

# ═══════════════════════════════════════════════════════════════
# 全局变量 (ROS 回调间共享)
# ═══════════════════════════════════════════════════════════════
global_action: int | None = None
fusion_threshold: float = 0.3
fsm_final_state: int = 0    # 来自 FSM /ros/expl_state
fsm_expl_result: int = 0    # 来自 FSM /ros/expl_result (预留)

# VLM/ITM results from ROS bridge nodes
latest_mask_msgs: list = []   # SingleMasksWithConfidence from ros_vlm_bridge
latest_itm_cosine: float = 0.0  # from blip2_itm_node


# ═══════════════════════════════════════════════════════════════
# ROS 发布辅助
# ═══════════════════════════════════════════════════════════════

def publish_int32(pub, data: int):
    msg = Int32()
    msg.data = data
    pub.publish(msg)


def publish_float64(pub, data: float):
    msg = Float64()
    msg.data = data
    pub.publish(msg)


def publish_int32_array(pub, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    pub.publish(msg)


def publish_float32_array(pub, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    pub.publish(msg)


# ═══════════════════════════════════════════════════════════════
# ROS 回调
# ═══════════════════════════════════════════════════════════════

def ros_action_callback(msg: Int32):
    global global_action
    global_action = msg.data


def ros_final_state_callback(msg: Int32):
    """接收 FSM 的最终状态 (FINAL_RESULT: REACH_OBJECT/STUCKING/NO_FRONTIER/...)"""
    global fsm_final_state
    fsm_final_state = msg.data


def ros_expl_result_callback(msg: Int32):
    """接收 FSM 的探索结果 (EXPL_RESULT: EXPLORATION/SEARCH_BEST_OBJECT/...)"""
    global fsm_expl_result
    fsm_expl_result = msg.data


def ros_mask_callback(msg: SingleMasksWithConfidence):
    """接收 ros_vlm_bridge 发出的检测结果"""
    global latest_mask_msgs
    latest_mask_msgs.append(msg)


def ros_cosine_callback(msg: Float64):
    """接收 blip2_itm_node 发出的 ITM cosine score"""
    global latest_itm_cosine
    latest_itm_cosine = msg.data


# ═══════════════════════════════════════════════════════════════
# 信号处理
# ═══════════════════════════════════════════════════════════════

def signal_handler(sig, frame):
    print("\nCtrl+C detected! Shutting down gracefully...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


# ═══════════════════════════════════════════════════════════════
# 动作映射
# ═══════════════════════════════════════════════════════════════

ACTION_TO_HABITAT = {
    ACTION.STOP:          HabitatSimActions.stop,
    ACTION.MOVE_FORWARD:  HabitatSimActions.move_forward,
    ACTION.TURN_LEFT:     HabitatSimActions.turn_left,
    ACTION.TURN_RIGHT:    HabitatSimActions.turn_right,
    ACTION.TURN_DOWN:     HabitatSimActions.turn_left,   # 禁用俯仰 → 纯旋转
    ACTION.TURN_UP:       HabitatSimActions.turn_left,   # 禁用俯仰 → 纯旋转
}


# ═══════════════════════════════════════════════════════════════
# 编码辅助
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# CLI 解析
# ═══════════════════════════════════════════════════════════════

def _parse_dataset_arg():
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset", type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2 or mp3d (default: hm3dv2)",
    )
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown


# ═══════════════════════════════════════════════════════════════
# 传感器发布辅助函数（与 habitat_bridge.py 一致）
# ═══════════════════════════════════════════════════════════════

def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0; q.y = 0.0
    q.z = np.sin(yaw / 2.0); q.w = np.cos(yaw / 2.0)
    return q


def hab_to_ros_pos(gps: np.ndarray, z_offset: float = 0.5) -> Point:
    return Point(x=-float(gps[2]), y=-float(gps[0]), z=float(gps[1]) + z_offset)


def make_rgb_msg(rgb: np.ndarray, stamp: rospy.Time) -> Image:
    h, w = rgb.shape[:2]
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = "camera_optical"
    msg.height = h; msg.width = w
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = rgb.tobytes()
    return msg


def make_depth_msg(depth: np.ndarray, stamp: rospy.Time) -> Image:
    h, w = depth.shape[:2]
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = "camera_optical"
    msg.height = h; msg.width = w
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = w * 4
    msg.data = depth.astype(np.float32).tobytes()
    return msg


def publish_sensors(pubs: dict, observations: dict, depth_cfg, z_offset: float = 0.5):
    """Publish all sensor data to ROS, matching habitat_bridge.py format."""
    now = rospy.Time.now()
    rgb = observations["rgb"]
    depth = observations["depth"]
    gps = observations["gps"]
    compass = observations["compass"]

    # Depth: de-normalize (matching habitat_bridge)
    depth_m = depth.astype(np.float32) * (depth_cfg.max_depth - depth_cfg.min_depth) + depth_cfg.min_depth
    depth_msg = make_depth_msg(depth_m, now)
    pubs["rgb"].publish(make_rgb_msg(rgb, now))
    pubs["depth"].publish(depth_msg)
    pubs["grid_depth"].publish(depth_msg)

    # Odom
    odom = Odometry()
    odom.header.stamp = now
    odom.header.frame_id = "world"
    odom.child_frame_id = "base_link"
    odom.pose.pose.position = hab_to_ros_pos(gps, z_offset)
    odom.pose.pose.orientation = quat_from_yaw(float(compass))
    pubs["odom"].publish(odom)
    pubs["grid_odom"].publish(odom)

    # Sensor pose
    sensor_pose = Odometry()
    sensor_pose.header.stamp = now
    sensor_pose.header.frame_id = "world"
    sensor_pose.child_frame_id = "camera_optical"
    sensor_pose.pose.pose.position = hab_to_ros_pos(gps, z_offset)
    sensor_pose.pose.pose.orientation = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)
    pubs["sensor_pose"].publish(sensor_pose)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main(cfg: DictConfig) -> None:
    global global_action, fusion_threshold

    # ── 加载 category → name 映射 ──
    # MP3D val data provides category_to_mp3d_category_id mapping
    dataset = cfg.get("_dataset_choice", "hm3dv2")
    val_json_path = cfg.habitat.dataset.data_path.format(split=cfg.habitat.dataset.split)

    # Try absolute path first, then relative
    if not os.path.exists(val_json_path):
        # Try relative to the script directory
        alt_path = os.path.join(_script_dir, val_json_path)
        if os.path.exists(alt_path):
            val_json_path = alt_path

    if os.path.exists(val_json_path):
        with gzip.open(val_json_path, "rt", encoding="utf-8") as f:
            val_data = json.load(f)
        category_to_coco = val_data.get("category_to_mp3d_category_id", {})
        # Choose ID→Name mapping based on dataset
        if "mp3d" in dataset:
            id_to_name_map = MP3D_ID_TO_NAME
        else:
            id_to_name_map = HM3D_ID_TO_NAME
        id_to_name = {
            category_to_coco[cat]: id_to_name_map[idx]
            for idx, cat in enumerate(category_to_coco)
            if category_to_coco[cat] < len(id_to_name_map)
        }
    else:
        print(f"[WARN] val.json.gz not found at {val_json_path}")
        print("[WARN] Category mapping will use HM3D_ID_TO_NAME directly")
        id_to_name = {i: name for i, name in enumerate(HM3D_ID_TO_NAME)}

    start_time = time.time()

    # ── 配置提取 ──
    cfg = patch_config(cfg)

    video_output_path = cfg.get("video_output_path", "videos/eval").format(
        split=cfg.habitat.dataset.split)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    video_output_path = os.path.join(video_output_path, run_tag)
    print(f"[habitat_evaluation] Output dir: {video_output_path}")
    need_video = cfg.get("need_video", False)
    record_file_path = os.path.join(video_output_path, cfg.get("record_file_name", "record.txt"))
    continue_path = os.path.join(video_output_path, cfg.get("continue_file_name", "continue.txt"))
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    use_vlm = cfg.get("use_vlm", True)
    use_itm = cfg.get("use_itm", True)
    use_point_cloud = cfg.get("use_point_cloud", False)  # optional, adds overhead

    llm_cfg = cfg.get("llm", {})
    llm_client_type = llm_cfg.get("llm_client", "deepseek")
    llm_answer_path = llm_cfg.get("llm_answer_path", "llm/answers/llm_answer_hm3d.txt")
    llm_response_path = llm_cfg.get("llm_response_path", "llm/answers/llm_responses.txt")

    # Resolve relative paths against script dir
    if not os.path.isabs(llm_answer_path):
        llm_answer_path = os.path.join(_script_dir, llm_answer_path)
    if not os.path.isabs(llm_response_path):
        llm_response_path = os.path.join(_script_dir, llm_response_path)

    env_num_once = cfg.get("test_epi_num", -1)
    flag_once = env_num_once != -1

    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # ── 添加 top_down_map + collisions 测量 ──
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3, map_resolution=256,
                draw_source=True, draw_border=True, draw_shortest_path=True,
                draw_view_points=True, draw_goal_positions=True, draw_goal_aabbs=False,
                fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=79),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

    env = habitat.Env(cfg)
    print("[habitat_evaluation] Environment created successfully")
    number_of_episodes = env.number_of_episodes

    # ── 断点续评 ──
    (num_total, num_success, spl_all, soft_spl_all,
     distance_to_goal_all, distance_to_goal_reward_all,
     last_time) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError(f"Already finished all {number_of_episodes} episodes.")

    pbar = tqdm.tqdm(total=number_of_episodes)

    # Skip completed episodes
    env_count = num_total if not flag_once else env_num_once
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # ── ROS 初始化 ──
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    # VLM/ITM ROS bridge subscribers
    rospy.Subscriber("/yolo_detector/single_mask", SingleMasksWithConfidence,
                     ros_mask_callback, queue_size=20)
    rospy.Subscriber("/blip2/cosine_score", Float64, ros_cosine_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    # Publish target config to ros_vlm_bridge before each episode
    target_config_pub = rospy.Publisher("/vlm_bridge/target_config", String, queue_size=5)
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)
    obj_point_cloud_pub = rospy.Publisher(
        "/habitat/object_point_cloud", PointCloud2, queue_size=10)
    early_replan_pub = rospy.Publisher("/toggle_early_replan", Bool, queue_size=5)

    # ── Sensor publishers (matching habitat_bridge.py topic set) ──
    pubs = {
        "rgb":        rospy.Publisher("/habitat/camera_rgb", Image, queue_size=5),
        "depth":      rospy.Publisher("/habitat/camera_depth", Image, queue_size=5),
        "grid_depth": rospy.Publisher("/grid_map/depth", Image, queue_size=5),
        "odom":       rospy.Publisher("/drone_0_visual_slam/odom", Odometry, queue_size=10),
        "grid_odom":  rospy.Publisher("/grid_map/odom", Odometry, queue_size=10),
        "sensor_pose": rospy.Publisher("/habitat/sensor_pose", Odometry, queue_size=10),
    }
    depth_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor

    bridge = CvBridge()
    result_list = [0] * len(RESULT_TYPES)

    # ── Episode 迭代 ──
    for epi in range(number_of_episodes - num_total):
        publish_int32_array(progress_pub, [num_total, number_of_episodes])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # ── 初始化 episode 变量 ──
        pass_object = 0.0
        near_object = 0.0
        fsm_final_state = 0
        fsm_expl_result = 0
        # 不重置 global_action —— FSM 的 INIT_ROTATE 可能在过渡期就发了
        count_steps = 0

        observations = env.reset()
        msg_observations = deepcopy(observations)

        # ── 获取目标物体类别（ObjectNav 提供） ──
        label = env.current_episode.object_category

        # 转换 category id → coco name
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        # ── 跳过 tv_monitor（不可达目标） ──
        SKIP_CATEGORIES = {"tv_monitor","plant","sofa","chair","bed"}
        if label in {"bed","sofa"}:
            early_replan_pub.publish(Bool(True))  # 启用 /toggle_early_replan
        else:
            early_replan_pub.publish(Bool(False))  # 禁用 /toggle_early_replan
        if label in SKIP_CATEGORIES:
            print(f"\n{'='*60}")
            print(f"Episode {num_total+1}/{number_of_episodes}  [SKIP]")
            print(f"Target: {label} — skipped")
            print(f"{'='*60}")
            #num_total += 1
            pbar.update()
            env.current_episode = next(env.episode_iterator)
            continue

        # ── LLM 查询 ──
        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label,
            type('LLMClient', (), {'llm_client': llm_client_type})()
        )

        print(f"\n{'='*60}")
        print(f"Episode {num_total+1}/{number_of_episodes}")
        print(f"Target: {label}  |  Room: {room}  |  Threshold: {fusion_threshold:.2f}")
        print(f"LLM confusable: {llm_answer}")
        print(f"Scene: {env.current_episode.scene_id}")
        print(f"{'='*60}")

        # ── 发送 target config 给 ros_vlm_bridge ──
        target_config = json.dumps({
            "target_classes": [label],
            "confusable_labels": llm_answer,
            "room": room,
            "fusion_threshold": fusion_threshold,
        })
        target_config_pub.publish(String(target_config))

        # ── 视频帧 ──
        vis_frames = []
        if need_video:
            info = env.get_metrics()
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # ── 等待 C++ FSM 就绪 ──
        # 发布首帧传感器数据，然后等待第一个 plan_action
        publish_sensors(pubs, msg_observations, depth_cfg)

        # 发送 trigger 通知 FSM 新 episode 开始
        trigger = PoseStamped()
        trigger.header.stamp = rospy.Time.now()
        trigger_pub.publish(trigger)
        publish_int32(state_pub, HABITAT_STATE.READY)

        print("[habitat_evaluation] Waiting for FSM to send first action...")
        rate = rospy.Rate(10)
        while global_action is None and not rospy.is_shutdown():
            # Keep publishing sensor data while waiting
            publish_sensors(pubs, msg_observations, depth_cfg)
            rate.sleep()

        print("[habitat_evaluation] Agent is ready to go!")

        # ═══════════════════════════════════════════════════
        # 步进循环
        # ═══════════════════════════════════════════════════
        episode_timeout = 200  # 5 分钟超时
        episode_start_time = time.time()
        timed_out = False
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not env.episode_over:
            # ── 超时检测 ──
            if time.time() - episode_start_time > episode_timeout:
                print("[habitat_evaluation] Episode TIMEOUT (5 min) — forcing stop")
                timed_out = True
                break
            # ── 检查目标是否可达（同楼层） ──
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode)
            if not is_feasible:
                print("[habitat_evaluation] Target on different floor — skipping")
                break

            # ── 解析 action ──
            action = None
            if global_action is not None:
                if count_steps >= max_episode_steps - 1:
                    global_action = ACTION.STOP

                action = ACTION_TO_HABITAT.get(global_action)
                global_action = None

            if action is None:
                rate.sleep()
                continue

            count_steps += 1
            print(f"\n  --- Step {count_steps}/{max_episode_steps} ---")
            print(f"  Action: {action}  |  Finding: [{label}]")

            # ── 执行 Habitat step ──
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)
            observations = env.step(action)
            global latest_mask_msgs, latest_itm_cosine

            # ── 发布传感器数据（触发 ros_vlm_bridge + blip2_itm_node）──
            msg_observations = deepcopy(observations)
            publish_sensors(pubs, msg_observations, depth_cfg)

            # ── 消费 ros_vlm_bridge 异步到达的 mask ──
            score_list, object_masks_list, label_list = [], [], []
            if use_vlm:
                for mask_msg in latest_mask_msgs:
                    try:
                        mask_uint8 = bridge.imgmsg_to_cv2(mask_msg.mask, "mono8")
                        mask_bool = mask_uint8 > 128
                        score_list.append(mask_msg.confidence)
                        object_masks_list.append(mask_bool)
                        label_list.append(mask_msg.label_index)
                    except Exception as e:
                        print(f"    [WARN] Mask decode failed: {e}")
                target_hits = sum(1 for l in label_list if l == 0)
                if score_list:
                    print(f"  VLM: {len(score_list)} detections ({target_hits} target, {len(score_list)-target_hits} confusable)")
                else:
                    print(f"  VLM: no detections yet")
                latest_mask_msgs = []

            # ── ITM from blip2_itm_node ──
            if use_itm:
                print(f"  ITM cosine: {latest_itm_cosine:.3f} (room: {room})")

            # ── Object point cloud（可选） ──
            if use_point_cloud and object_masks_list:
                try:
                    obj_point_cloud_list = get_object_point_cloud(
                        cfg, observations, object_masks_list)
                    for pc2 in obj_point_cloud_list:
                        obj_point_cloud_pub.publish(pc2)
                except Exception as e:
                    print(f"    [WARN] Point cloud failed: {e}")

            # ── 视频帧 ──
            if need_video:
                info = env.get_metrics()
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # ── 跟踪是否经过目标附近 ──
            info = env.get_metrics()
            distance_to_goal = info.get("distance_to_goal", float("inf"))
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # ═══════════════════════════════════════════════════
        # Episode 结束 — 收集指标
        # ═══════════════════════════════════════════════════
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        info = env.get_metrics()
        spl = info.get("spl", 0.0)
        soft_spl = info.get("soft_spl", 0.0)
        distance_to_goal = info.get("distance_to_goal", float("inf"))
        distance_to_goal_reward = info.get("distance_to_goal_reward", 0.0)
        success = info.get("success", 0.0)

        if distance_to_goal <= success_distance:
            near_object = 1

        # ── 失败分类 ──
        if timed_out:
            result_text = "timeout"
        elif success == 1:
            num_success += 1
            result_text = "success"
        else:
            # 使用 FSM 通过 /ros/expl_state 发布的真实最终状态
            result_text = check_failure(
                env.current_episode,
                final_state=fsm_final_state,
                expl_result=fsm_expl_result,
                count_steps=count_steps,
                max_step=max_episode_steps,
                pass_object=pass_object,
                near_object=near_object,
            )

        # ── 累计统计 ──
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # ── 视频 ──
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)
        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video and vis_frames:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9)
        vis_frames.clear()

        # ── 打印平均指标 ──
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Success Rate", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Dist to Goal", f"{distance_to_goal_all/num_total:.4f}"])
        print(table1)

        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Success", f"{num_success}"])
        table2.add_row(["SPL", f"{spl_all:.2f}"])
        table2.add_row(["Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Dist to Goal", f"{distance_to_goal_all:.4f}"])

        print(f"\nResult: {result_text}  |  Episode {num_total} done\n")

        if flag_once:
            break

        # ── 写入记录 ──
        write_record(scene_id, episode_id, table1, result_text, label,
                     num_total, time_spend, record_file_path)
        write_record(scene_id, episode_id, table2, result_text, label,
                     num_total, time_spend, continue_path)

        # ── 各失败类别文件计数 ──
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]
            folder_path = os.path.join(video_output_path, folder)
            result_list[i] = count_files_in_directory(folder_path)

        # ── 发布记录数据 ──
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)

    env.close()
    pbar.close()
    print(f"\n[habitat_evaluation] Done. {num_total} episodes evaluated.")
    print(f"  Success: {num_success}/{num_total} ({num_success/num_total*100:.1f}%)")
    print(f"  SPL: {spl_all/num_total:.3f}")


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"

        # Compose Hydra config (Hydra requires relative config_path)
        os.chdir(_script_dir)
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)

        # Store dataset choice for later use
        OmegaConf.set_struct(cfg, False)
        cfg["_dataset_choice"] = dataset
        OmegaConf.set_struct(cfg, True)

        main(cfg)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
