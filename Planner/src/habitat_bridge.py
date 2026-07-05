#!/usr/bin/env python3
"""
habitat_bridge.py — Habitat <-> explorer_ws ROS Bridge

用法:
    conda activate habitat
    PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages python habitat_bridge.py
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

# ROS
try:
    import rospy
    from geometry_msgs.msg import Point, Pose, Quaternion
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, PointCloud2, PointField as PF
    from std_msgs.msg import Float64, Int32
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig, CompassSensorConfig,
    FogOfWarConfig, GPSSensorConfig, TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# ── 常量 ──
TARGET_LABEL = "bed"
TARGET_ROOM = "bedroom"
FUSION_THRESHOLD = 0.3

# ── 离散动作映射 ──
ACTION_MAP = {
    HabitatSimActions.move_forward: "MOVE_FORWARD",
    HabitatSimActions.turn_left:    "TURN_LEFT",
    HabitatSimActions.turn_right:   "TURN_RIGHT",
    HabitatSimActions.look_up:      "LOOK_UP",
    HabitatSimActions.look_down:    "LOOK_DOWN",
    HabitatSimActions.stop:         "STOP",
}

# Int32 -> Habitat action (与 FSM_ACTION 枚举对齐)
# NOTE: Habitat PointNav 只有 4 个动作 (STOP/FWD/LEFT/RIGHT)
#       TURN_DOWN(4) / TURN_UP(5) 在 PointNav 中无对应，映射为 TURN_LEFT
INT_TO_HABITAT_ACTION = {
    0: HabitatSimActions.stop,
    1: HabitatSimActions.move_forward,
    2: HabitatSimActions.turn_left,
    3: HabitatSimActions.turn_right,
    4: HabitatSimActions.turn_left,   # ACT_TURN_DOWN -> TURN_LEFT (PointNav 无俯仰)
    5: HabitatSimActions.turn_left,   # ACT_TURN_UP   -> TURN_LEFT (PointNav 无俯仰)
}

# Habitat 状态常量 (与 FSM HABITAT_STATE 枚举对齐)
class HabitatState:
    READY = 0
    ACTION_EXEC = 1
    ACTION_FINISH = 2
    EPISODE_FINISH = 3


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════

def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0; q.y = 0.0
    q.z = np.sin(yaw / 2.0); q.w = np.cos(yaw / 2.0)
    return q


def habitat_to_ros_position(gps: np.ndarray, z_offset: float = 1.3) -> Point:
    return Point(x=-float(gps[2]), y=-float(gps[0]), z=float(gps[1]) + z_offset)


def make_image_msg(rgb: np.ndarray, stamp: rospy.Time) -> Image:
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


# ═══════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════

class HabitatBridge:
    def __init__(self, scene_path: str = "", data_base: str = "data",
                 target_label: str = TARGET_LABEL, target_room: str = TARGET_ROOM,
                 max_steps: int = 2000, spawn_offset: tuple = (0.0, 0.0, 0.0),
                 action_delay: float = 0.3, step_size: float = 0.25, turn_angle: float = 30.0):
        self.target_label = target_label
        self.target_room = target_room
        self.max_steps = max_steps
        self.spawn_offset = spawn_offset
        self.scene_path = scene_path
        self.data_base = data_base
        self.action_delay = action_delay
        self.step_size = step_size
        self.turn_angle = turn_angle

        self.step_count = 0
        self._hab_pos = np.zeros(3)
        self._hab_yaw = 0.0
        self._z_offset = 0.5
        self._next_action: int = -1

        self._create_env()

        if not HAS_ROS:
            print("[HabitatBridge] ERROR: ROS not available."); sys.exit(1)
        self._setup_ros()

        print("[HabitatBridge] Ready.")
        print(f"  Scene:  {self.env.current_episode.scene_id}")

    # ── Habitat 环境 ──────────────────────────────

    def _create_env(self):
        config_paths = [
            "benchmark/nav/pointnav/pointnav_mp3d_example.yaml",
            "benchmark/nav/pointnav/pointnav_mp3d.yaml",
        ]
        cfg = None
        for cp in config_paths:
            try:
                cfg = habitat.get_config(cp)
                break
            except Exception:
                continue
        if cfg is None:
            raise RuntimeError("No PointNav config found.")

        cfg = patch_config(cfg)
        with habitat.config.read_write(cfg):
            cfg.habitat.task.lab_sensors["gps_sensor"] = GPSSensorConfig(type="GPSSensor", dimensionality=3)
            cfg.habitat.task.lab_sensors["compass_sensor"] = CompassSensorConfig(type="CompassSensor")
            for sensor in ["rgb_sensor", "depth_sensor"]:
                cfg.habitat.simulator.agents.main_agent.sim_sensors[sensor].width = 640
                cfg.habitat.simulator.agents.main_agent.sim_sensors[sensor].height = 480
                cfg.habitat.simulator.agents.main_agent.sim_sensors[sensor].hfov = 79
                cfg.habitat.simulator.agents.main_agent.sim_sensors[sensor].position = [0.0, 0.5, 0.0]
            cfg.habitat.simulator.turn_angle = int(self.turn_angle)
            cfg.habitat.simulator.forward_step_size = self.step_size
            cfg.habitat.environment.max_episode_steps = self.max_steps
            cfg.habitat.task.measurements.update({
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3, map_resolution=256,
                    draw_source=True, draw_border=True, draw_shortest_path=True,
                    draw_view_points=True, draw_goal_positions=True, draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=79),
                ),
                "collisions": CollisionsMeasurementConfig(),
            })
        self._cfg = cfg
        self.env = habitat.Env(cfg)

    # ── ROS 接口 ──────────────────────────────────

    def _setup_ros(self):
        rospy.init_node("habitat_bridge", anonymous=True)

        odom_topic = rospy.get_param("~odom_topic", "/drone_0_visual_slam/odom")

        # Publishers
        self.rgb_pub = rospy.Publisher("/habitat/camera_rgb", Image, queue_size=5)
        self.depth_pub = rospy.Publisher("/habitat/camera_depth", Image, queue_size=5)
        self.grid_depth_pub = rospy.Publisher("/grid_map/depth", Image, queue_size=5)
        self.odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=10)
        self.grid_odom_pub = rospy.Publisher("/grid_map/odom", Odometry, queue_size=10)
        self.sensor_pose_pub = rospy.Publisher("/habitat/sensor_pose", Odometry, queue_size=10)
        self.confidence_pub = rospy.Publisher("/detector/confidence_threshold", Float64, queue_size=5)

        # *** 状态反馈: 通知 FSM 动作执行完毕 ***
        self.state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
        rospy.loginfo("[HabitatBridge] state_pub -> /habitat/state")

        # Subscribers
        self._plan_action_sub = rospy.Subscriber(
            "/habitat/plan_action", Int32, self._plan_action_cb, queue_size=10)
        rospy.loginfo("[HabitatBridge] Subscribing /habitat/plan_action")

        # 定时发布传感器数据 @ 10Hz
        self._latest_obs = None
        self._pub_timer = rospy.Timer(rospy.Duration(0.1), self._timer_publish)
        self.confidence_pub.publish(Float64(data=FUSION_THRESHOLD))

    def _timer_publish(self, event):
        if self._latest_obs is not None:
            self.publish_all(self._latest_obs)

    def _plan_action_cb(self, msg: Int32):
        action_names = {0: "STOP", 1: "FWD", 2: "LEFT", 3: "RIGHT", 4: "DOWN", 5: "UP"}
        name = action_names.get(msg.data, str(msg.data))
        rospy.loginfo(f"[HabitatBridge] <- action: {name} ({msg.data})")
        self._next_action = msg.data

    # ── 传感器发布 ────────────────────────────────

    def publish_all(self, observations: dict):
        now = rospy.Time.now()
        rgb = observations["rgb"]
        depth = observations["depth"]
        gps = observations["gps"]
        compass = observations["compass"]

        ds_cfg = self._cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
        depth_m = depth.astype(np.float32) * (ds_cfg.max_depth - ds_cfg.min_depth) + ds_cfg.min_depth
        depth_msg_m = make_depth_msg(depth_m, now)
        self.rgb_pub.publish(make_image_msg(rgb, now))
        self.depth_pub.publish(depth_msg_m)
        self.grid_depth_pub.publish(depth_msg_m)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position = habitat_to_ros_position(gps, self._z_offset)
        odom.pose.pose.orientation = quaternion_from_yaw(float(compass))
        self.odom_pub.publish(odom)
        self.grid_odom_pub.publish(odom)

        sensor_pose = Odometry()
        sensor_pose.header.stamp = now
        sensor_pose.header.frame_id = "world"
        sensor_pose.child_frame_id = "camera_optical"
        sensor_pose.pose.pose.position = habitat_to_ros_position(gps, self._z_offset)
        sensor_pose.pose.pose.orientation = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)
        self.sensor_pose_pub.publish(sensor_pose)

    # ── 主循环 ────────────────────────────────────

    def run(self):
        observations = self.env.reset()

        dx, dy, dz = self.spawn_offset
        if dx != 0 or dy != 0 or dz != 0:
            hab_off = np.array([-dy, dz, -dx])
            agent = self.env.sim.agents[0]
            state = agent.get_state()
            state.position += hab_off
            agent.set_state(state)
            observations = self.env.step(HabitatSimActions.stop)
            print(f"[HabitatBridge] Spawn offset: ROS({dx:.1f},{dy:.1f},{dz:.1f})")

        self._latest_obs = observations
        self._update_agent_state(observations)
        self.step_count = 0

        rate = rospy.Rate(10.0)

        while not rospy.is_shutdown():
            if self.env.episode_over:
                print(f"\n[HabitatBridge] Episode over. Steps={self.step_count}")
                self.state_pub.publish(Int32(data=HabitatState.EPISODE_FINISH))
                break

            if self._next_action < 0:
                rospy.sleep(0.05)
                continue

            action = INT_TO_HABITAT_ACTION.get(self._next_action, None)
            self._next_action = -1

            if action is None or action == HabitatSimActions.stop:
                continue

            self.step_count += 1
            act_name = ACTION_MAP.get(action, str(action))
            hp = self._hab_pos
            print(f"  Step {self.step_count:4d}  {act_name:<14s}  "
                  f"hab_pos=({hp[0]:.2f},{hp[2]:.2f})", end="")

            observations = self.env.step(action)
            self._latest_obs = observations
            self._update_agent_state(observations)

            # 先发布最新 odom (避免 FSM 读到旧位置)
            self.publish_all(observations)

            # 再通知 FSM: 动作执行完毕
            self.state_pub.publish(Int32(data=HabitatState.ACTION_FINISH))

            metrics = self.env.get_metrics()
            coll = metrics.get("collisions", {})
            if isinstance(coll, dict) and coll.get("count", 0) > 0:
                print(" [COLLISION]", end="")
            print()

            # 动作间隔延时 (避免过快的连续动作)
            if self.action_delay > 0:
                rospy.sleep(self.action_delay)

            rate.sleep()

        self.env.close()
        print("[HabitatBridge] Done.")

    def _update_agent_state(self, observations: dict):
        gps = observations["gps"]
        self._hab_pos = np.array([float(gps[0]), float(gps[1]), float(gps[2])])
        self._hab_yaw = float(observations["compass"])


# ═══════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Habitat <-> explorer_ws Bridge")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--data-base", type=str,default="/root/gpufree-data/code")  # "/root/gpufree-data/code"
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--spawn-offset-x", type=float, default=0.0)
    parser.add_argument("--spawn-offset-y", type=float, default=0.0)
    parser.add_argument("--spawn-offset-z", type=float, default=0.0)
    parser.add_argument("--action-delay", type=float, default=0.1,
                        help="Delay between actions in seconds (default: 0.3)")
    parser.add_argument("--step-size", type=float, default=0.125,
                        help="Forward step size in meters (default: 0.25, fine: 0.1)")
    parser.add_argument("--turn-angle", type=float, default=10.0,
                        help="Turn angle in degrees (default: 30, fine: 10)")
    parser.add_argument("--standalone", action="store_true")
    args, _ = parser.parse_known_args()

    if os.path.isdir(args.data_base):
        os.chdir(args.data_base)

    if args.standalone:
        print("[HabitatBridge] Standalone mode")
        cfg = habitat.get_config("benchmark/nav/pointnav/pointnav_mp3d_example.yaml")
        cfg = patch_config(cfg)
        env = habitat.Env(cfg)
        obs = env.reset()
        print(f"Scene: {env.current_episode.scene_id}")
        for i in range(10):
            obs = env.step(HabitatSimActions.move_forward)
            print(f"  Step {i+1}: gps={obs['gps'].round(3)}")
        env.close()
        return

    bridge = HabitatBridge(
        scene_path=args.scene, data_base=args.data_base,
        max_steps=args.max_steps,
        spawn_offset=(args.spawn_offset_x, args.spawn_offset_y, args.spawn_offset_z),
        action_delay=args.action_delay,
        step_size=args.step_size,
        turn_angle=args.turn_angle,
    )
    bridge.run()


if __name__ == "__main__":
    main()


# PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages python /home/tianbot/explorer/src/Planner/src/habitat_bridge.py 
# PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages python /root/gpufree-data/explorer_ws/src/Planner/src/habitat_bridge.py