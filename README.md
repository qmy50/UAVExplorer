
# UAV Autonomous Exploration Based on VLN

# 基于视觉语言导航的无人机自主探索框架 

## 一. 架构概览

```
┌──────────────────────────────────────────────────────────┐
│  高层: DEP 探索规划器 (global_planner)                     │
│    PRM 采样 → Dijkstra 路径 → 信息增益 → 最佳探索路径        │
├──────────────────────────────────────────────────────────┤
│  中层: MINCO 轨迹优化 + FSM 状态机 (minco_curve)            │
│    路径点 → 5阶多项式轨迹(LBFGS) → PolyTraj → 100Hz 控制    │
├──────────────────────────────────────────────────────────┤
│  底层: SO3 几何控制器 + 四旋翼动力学仿真                     │
│    PositionCommand → 推力和姿态 → 电机转速 → 物理仿真        │
├──────────────────────────────────────────────────────────┤
│  感知: 深度相机仿真 → 光线追踪 → 3D占据栅格 → 2D自由地图      │
├──────────────────────────────────────────────────────────┤
│  语义: VLM 检测/分割 + LLM 推理 (Habitat 模式)               │
└──────────────────────────────────────────────────────────┘
```

### 项目测试于UBUNTU20.04 ROS1 noetic

## 二. Quick Start：

在rviz中进行纯数字仿真

克隆项目到本地
```
git clone https://github.com/qmy50/UAVExplorer.git
```

编译
```
catkin_make
```
```
source ./devel/setup.bash
```

打开rviz
```
roslaunch minco_curve rviz_explore.launch
```

运行仿真
```
roslaunch minco_curve explore.launch
```
运行效果如下：


https://github.com/user-attachments/assets/5037e32e-6300-475b-8d69-bd1ed9e3ea27



## 三. 在Gazebo中运行：
结合XTdrone仿真环境，在gazebo中运行。这里需要完成lavis库的安装以使用blp2，并需要安装yolo

在有torch的虚拟环境中运行：
```
python flask_yolo_server.py --port 5000 --detect-first
```

开启PX4仿真环境并启动无人机通讯与键盘控制，操作无人机起飞
```
roslaunch px4 indoor1.launch
python multirotor_communication.py iris 0
python multirotor_keyboard_control.py iris 1 vel
```

开启blip2服务器端
```
python blip2itm.py
```
开启状态机与yolo，blip2客户端
```
roslaunch minco_curve run_in_XTdrone.launch
rosrun onboard_detector ros_yolo_bridge.py _target_classes:="['bed']"
python blip2_itm_node.py
```
运行效果如下：


https://github.com/user-attachments/assets/1d1ec0ef-53ac-46dc-aa82-2588001af1cb



## 四. 在habitat中运行：
项目测试于habitat-sim  0.2.5与 habitat-lab 0.2.5，可以完成objectnav导航任务
为完成以下环节，在配置好habitat环境后，需要下载mp3d或hm3d数据集，并安装号groundingdino与mobilesam库，
具体细节可参考：

确保目录结构为：


```
explorer_ws/
├── src/
│   ├── Planner/                    # 核心探索规划包
│   │   └── src/
│   │       ├── global_planner/     # DEP 探索规划器 (PRM/RRT/RRT*/KDTree)
│   │       ├── minco_curve/        # MINCO 轨迹优化 + FSM 状态机 (核心)
│   │       │   ├── launch/explore/ # 仿真探索 launch 文件
│   │       │   ├── launch/habitat/ # Habitat 模式 launch 文件
│   │       │   └── src/            # FSM, traj_server, plan_manager, traj_optimizer
│   │       ├── path_searching/     # 动态 A* 路径搜索
│   │       ├── plan_env/           # GridMap, ValueMap2D, RayCast, ObjectMap2D
│   │       ├── traj_utils/         # 轨迹工具库 (PolyTraj, MINCOTraj 消息)
│   │       ├── onboard_detector/   # YOLO 机载检测 Flask 服务
│   │       ├── habitat2ros/        # Habitat ↔ ROS 桥接
│   │       ├── vlm/                # VLM: GroundingDINO, YOLO, MobileSAM, BLIP2
│   │       ├── llm/                # LLM: DeepSeek 客户端 + prompt 模板
│   │       ├── basic_utils/        # 失败检测, 记录, 点云工具
│   │       └── config/             # HM3D/MP3D 数据集配置 YAML
│   └── uav_simulator/              # 无人机仿真框架
│       ├── so3_control/            # SO3 几何控制器 (nodelet)
│       ├── so3_quadrotor_simulator/# 四旋翼动力学仿真
│       ├── local_sensing/          # 深度相机渲染 (pcl_render_node)
│       ├── map_generator/          # 随机地图生成 (含边界围栏)
│       └── Utils/                  # 工具: waypoint_generator, odom_viz, rviz_plugins
├── build/                          # catkin 构建产物
├── devel/                          # catkin 开发环境 setup
└── lockfiles/                      # 运行时锁文件
```

运行方式如下：
开启相应的服务器端,在sr/Planner_src目录级别运行
```
python -m vlm.segmentor.sam --port 12183
python -m vlm.detector.yolo26_detect --port 12184
python -m vlm.itm.blip2itm --port 12185
python -m vlm.detector.grounding_dino --port 12181
```

开启habitat仿真环境
```
python habitat_bridge.py
```

开启状态机与客户端
```
roslaunch minco_curve run_in_habitat.launch
rosrun onboard_detector ros_vlm_bridge.py
rosrun onboard_detector ros_vlm_bridge.py
```
运行效果如下：

https://github.com/user-attachments/assets/921d9d12-da3a-46de-9bad-b6d2b8691e30


## 五. Reference：
[1]. VLN部分价值地图VLM,LLM部分参考ApexNav,链接为: https://github.com/Robotics-STAR-Lab/ApexNav

[2]. 自主探索框架参考CERLAB-UAV-Autonomy,链接为: https://github.com/Zhefan-Xu/CERLAB-UAV-Autonomy

[3]. 规划器整体结构及动态环境更新，a*搜索等模块参考/使用 ego planner 链接为：https://github.com/ZJU-FAST-Lab/ego-planner

[4]. PX4仿真部分使用XTdroen，链接为： https://gitee.com/robin_shaun/XTDrone

