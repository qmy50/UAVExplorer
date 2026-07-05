#!/usr/bin/env python3
"""
habitat_keyboard_control.py — 键盘手动控制 Habitat agent, 同时发布传感器到 ROS
用于验证 grid_map 建图是否正常

用法:
    conda activate habitat
    unset PYTHONPATH
    PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages python habitat_keyboard_control.py
"""

import argparse, os, sys
import numpy as np
import cv2

import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    GPSSensorConfig, CompassSensorConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# ROS
try:
    import rospy
    from sensor_msgs.msg import Image, PointCloud2, PointField as PF
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64
    from geometry_msgs.msg import Point, Quaternion
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

# 复用 bridge 的工具函数
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from habitat_bridge import (
    make_image_msg, make_depth_msg,
    habitat_to_ros_position, quaternion_from_yaw,
)

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
QUIT_KEY = "q"
LOOK_UP_KEY = "e"
LOOK_DOWN_KEY = "c"

Z_OFFSET = 1.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base", default="/root/gpufree-data/code")
    parser.add_argument("--no-ros", action="store_true", help="Skip ROS, just test env")
    args = parser.parse_args()

    if os.path.isdir(args.data_base):
        os.chdir(args.data_base)

    # ── 创建 Habitat 环境 ──
    cfg = habitat.get_config("benchmark/nav/pointnav/pointnav_mp3d_example.yaml")
    cfg = patch_config(cfg)
    with habitat.config.read_write(cfg):
        cfg.habitat.task.lab_sensors["gps_sensor"] = GPSSensorConfig(
            type="GPSSensor", dimensionality=3)
        cfg.habitat.task.lab_sensors["compass_sensor"] = CompassSensorConfig(
            type="CompassSensor")
        cfg.habitat.environment.max_episode_steps = 10000
        for s in ["rgb_sensor", "depth_sensor"]:
            cfg.habitat.simulator.agents.main_agent.sim_sensors[s].width = 640
            cfg.habitat.simulator.agents.main_agent.sim_sensors[s].height = 480
            cfg.habitat.simulator.agents.main_agent.sim_sensors[s].hfov = 79
            cfg.habitat.simulator.agents.main_agent.sim_sensors[s].position = [0.0, 1.3, 0.0]

    env = habitat.Env(cfg)
    obs = env.reset()

    # ── 相机参数 ──
    hfov = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov
    ds_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    max_d = ds_cfg.max_depth
    min_d = ds_cfg.min_depth

    # ── ROS (定时器发布, 不依赖按键) ──
    _latest_obs = obs
    _publish_enabled = HAS_ROS and not args.no_ros

    if _publish_enabled:
        rospy.init_node("habitat_keyboard", anonymous=True)
        rgb_pub = rospy.Publisher("/habitat/camera_rgb", Image, queue_size=5)
        depth_pub = rospy.Publisher("/habitat/camera_depth", Image, queue_size=5)
        grid_depth_pub = rospy.Publisher("/grid_map/depth", Image, queue_size=5)
        cloud_pub = rospy.Publisher("/grid_map/cloud", PointCloud2, queue_size=5)
        odom_pub = rospy.Publisher("/drone_0_visual_slam/odom", Odometry, queue_size=10)
        grid_odom_pub = rospy.Publisher("/grid_map/odom", Odometry, queue_size=10)
        print("[ROS] Publishers ready, timer @ 4 Hz")

    def _timer_publish(event):
        """定时器回调: 10Hz 持续发布传感器数据, 不依赖按键"""
        obs = _latest_obs
        now = rospy.Time.now()
        rgb_pub.publish(make_image_msg(obs["rgb"], now))

        # 反归一化 → 米制 (grid_map 内部 ×1000 → mm, cv_bridge 自动处理)
        depth_m = obs["depth"].astype(np.float32) * (max_d - min_d) + min_d
        depth_msg_m = make_depth_msg(depth_m, now)
        depth_pub.publish(depth_msg_m)          # /habitat/camera_depth → grid_map 实际监听
        grid_depth_pub.publish(depth_msg_m)     # /grid_map/depth 备用

        cam_pos = habitat_to_ros_position(obs["gps"], Z_OFFSET)
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position = cam_pos
        odom.pose.pose.orientation = quaternion_from_yaw(float(obs["compass"]))
        odom_pub.publish(odom)
        grid_odom_pub.publish(odom)

    # if _publish_enabled:
    #     pub_timer = rospy.Timer(rospy.Duration(0.25), _timer_publish)  # 4 Hz

    # ── 主循环 ──
    print("\n" + "=" * 50)
    print("Habitat Keyboard Control")
    print(f"  Scene: {env.current_episode.scene_id}")
    print(f"  w=forward  a=turn_left  d=turn_right")
    print(f"  e=look_up  c=look_down  q=quit")
    print("=" * 50 + "\n")

    count = 0
    pitch = 0.0
    if _publish_enabled:
        pub_timer = rospy.Timer(rospy.Duration(0.1), _timer_publish)  # 4 Hz

    while True:
        cv2.imshow("Habitat RGB", obs["rgb"][:, :, ::-1])  # RGB→BGR
        key = cv2.waitKey(0) & 0xFF

        if key == ord(QUIT_KEY):
            break
        elif key == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
        elif key == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
        elif key == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
        elif key == ord(LOOK_UP_KEY):
            action = HabitatSimActions.look_up
            pitch += np.pi / 6
        elif key == ord(LOOK_DOWN_KEY):
            action = HabitatSimActions.look_down
            pitch -= np.pi / 6
        else:
            continue

        obs = env.step(action)
        _latest_obs = obs                                    # 更新定时器用的最新帧
        count += 1
        gps = obs["gps"]
        print(f"  Step {count:4d}  gps=({gps[0]:.2f},{gps[1]:.2f},{gps[2]:.2f})  compass={obs['compass'][0]:.2f}")

    env.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

# PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages python /home/tianbot/explorer_ws/src/Planner/src/habitat_keyboard_control.py 
