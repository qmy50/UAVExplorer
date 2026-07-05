

https://github.com/user-attachments/assets/e25aef3f-002f-4b75-8f6a-35b2f63520f3

# UAV Autonomous Exploration Based on VLN

# 基于视觉语言导航的无人机自主探索框架 

## 一. 工作流程：

on progress 😚

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

在有torch的虚拟环境中运行

## 四. 在habitat中运行：

## 五. Reference：
[1]. VLN部分价值地图VLM,LLM部分参考ApexNav,链接为: https://github.com/Robotics-STAR-Lab/ApexNav

[2]. 自主探索框架参考CERLAB-UAV-Autonomy,链接为: https://github.com/Zhefan-Xu/CERLAB-UAV-Autonomy

[3]. 规划器整体结构及动态环境更新，a*搜索等模块参考/使用 ego planner 链接为：https://github.com/ZJU-FAST-Lab/ego-planner

[4]. PX4仿真部分使用XTdroen，链接为： https://gitee.com/robin_shaun/XTDrone

