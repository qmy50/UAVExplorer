
# UAV Autonomous Exploration Based on VLN

# 基于视觉语言导航的无人机自主探索框架 

## 一. 项目流程
1. 使用HIRE框架，结合信息增益，语义增益与前沿增益构建prm概率路线图，并一次选择最优目标点直到到达终点。
2. blip2提供语义匹配分值，yolo+groundingdino负责目标检测，mobileSAM负责语义分割得到目标物体点云信息
3. 通过LLM推理在目标检测过程中加入易混淆物体检测，将全部检测结果放入object地图中打分，减小少数帧误检导致的任务失败
4. 由fsm负责流程控制。
   
   整体框图如下：

<img width="884" height="363" alt="explorer" src="https://github.com/user-attachments/assets/56c2fbd1-5b54-49fe-823d-d6517e128e5d" />

   
### 项目文件功能如下：
```
explorer_ws/
├── src/
│   ├── Planner/                    # 核心探索规划包
│   │   └── src/
│   │       ├── global_planner/     # DEP 探索规划器 (PRM/KDTree)
│   │       ├── minco_curve/        # MINCO 轨迹优化 + FSM 状态机 
│   │       │   ├── launch/explore/ # 仿真探索 launch 文件
│   │       │   ├── launch/habitat/ # Habitat 模式 launch 文件
│   │       │   └── src/            # FSM, traj_server, plan_manager, traj_optimizer
│   │       ├── path_searching/     # 动态 A* 路径搜索
│   │       ├── plan_env/           # GridMap, ValueMap2D, RayCast, ObjectMap2D
│   │       ├── traj_utils/         # 轨迹工具库 (PolyTraj, MINCOTraj)
│   │       ├── onboard_detector/   # YOLO 机载检测 Flask 服务
│   │       ├── vlm/                # VLM: GroundingDINO, YOLO, MobileSAM, BLIP2
│   │       ├── llm/                # LLM: DeepSeek 客户端 + prompt 模板
│   │       ├── basic_utils/        # 失败检测, 记录, 点云工具
│   │       └── config/             # HM3D/MP3D 数据集配置 YAML
│   └── uav_simulator/              # 无人机仿真框架
```

### 项目测试于UBUNTU20.04 ROS1 noetic

## 二. Quick Start：

在rviz中进行纯数字仿真，语义值为随机值

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
这种情况下语义分值为随机值，相当于纯几何探索，运行效果如下：


https://github.com/user-attachments/assets/7331be54-9a5c-4620-a4c4-2cb46b21d37f



## 三. 在Gazebo中运行：
结合XTdrone仿真环境使用PX4飞控控制无人机在gazebo中运行。这里需要完成lavis库的安装以使用blp2，并需要安装yolo相关库，并完成XTdrone配置

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
运行效果如下，目标物体为床：


https://github.com/user-attachments/assets/f6789046-4f81-4cc4-ad67-cd95cdf484e3





## 四. 在habitat中运行：
项目测试于habitat-sim  0.2.5与 habitat-lab 0.2.5，可以完成objectnav导航任务
为完成以下环节，在配置好habitat环境后，需要下载mp3d或hm3d数据集，并安装号groundingdino与mobilesam库，
具体细节可参考：[ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav)

运行方式如下：
开启相应的服务器端,在sr/Planner_src目录级别运行
```
python -m vlm.segmentor.sam --port 12183
python -m vlm.detector.yolo26_detect --port 12184
python -m vlm.itm.blip2itm --port 12185
python -m vlm.detector.grounding_dino --port 12181
```
开启特定仿真场景
```
python habitat_bridge.py
```

开启状态机与客户端
```
roslaunch minco_curve run_in_habitat.launch
rosrun onboard_detector ros_vlm_bridge.py _target_classes:="[your target class]"
python blip2_itm_node.py
```
运行效果如下，目标物体为toilet：
<div align="center">
   

https://github.com/user-attachments/assets/acd9da62-8161-464c-8186-b99b7f617780


</div>
如果需要运行habitat测试，请执行如下命令

```
python habitat_evaluation.py
roslaunch minco_curve run_in_habitat.launch
rosrun onboard_detector ros_vlm_bridge.py
python blip2_itm_node.py
```

## 五. ToDo
1. 为PRM路线图节点加入所属房间标签，从而更好利用房间间空间关联的先验信息

## 六. Reference：
[1]. VLN部分价值地图VLM,LLM部分参考ApexNav,链接为: https://github.com/Robotics-STAR-Lab/ApexNav

[2]. 自主探索DEP框架参考CERLAB-UAV-Autonomy,链接为: https://github.com/Zhefan-Xu/CERLAB-UAV-Autonomy

[3]. 规划器整体结构及动态环境更新，a*搜索等模块参考/使用 ego planner 链接为：https://github.com/ZJU-FAST-Lab/ego-planner

[4]. PX4仿真部分使用XTdroen，链接为： https://gitee.com/robin_shaun/XTDrone

