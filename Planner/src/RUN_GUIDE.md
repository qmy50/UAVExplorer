# Habitat ObjectNav Evaluation — 全流程启动指南

## 前提条件

```bash
# 确认 conda 环境可用
conda activate habitat
python -c "import habitat; print('habitat:', habitat.__version__)"
python -c "from hydra import initialize, compose; print('hydra OK')"

# 确认 ObjectNav 数据集 JSON 已下载
ls data/datasets/objectnav/hm3d/v2/val/val.json.gz

# 确认 ROS workspace 已编译
source ~/explorer_ws/devel/setup.bash
rospack find minco_curve
rospack find global_planner
```

---

## 终端拓扑（5 个终端，按顺序启动）

```
┌─────────────────────────────────────────────────────────────────┐
│  T1: roscore                                                     │
│  T2: VLM Flask ×3  (yolo26:12185, dino:12181, sam:12183)        │
│  T3: C++ FSM        (fake_explorer_fsm_habitat)                  │
│  T4: habitat_evaluation.py                                       │
│  T5: (可选) RViz 监控                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## T1 — 启动 ROS Master

```bash
roscore
```

---

## T2 — 启动 VLM Flask 服务（3 个进程）

```bash
conda activate habitat
cd /home/tianbot/explorer/Planner/src

# yolo26 检测服务 (port 12185)
python vlm/detector/yolo26_detect.py --port 12185 &

# GroundingDINO 开放词汇检测 (port 12181)
python vlm/detector/grounding_dino.py --port 12181 &

# MobileSAM 分割 (port 12183)
python vlm/segmentor/sam.py --port 12183 &

# 等几秒后验证
sleep 5
curl -s http://localhost:12185/health && echo "✅ yolo26 OK"
curl -s http://localhost:12181/health && echo "✅ dino OK"
curl -s http://localhost:12183/health && echo "✅ sam OK"
```

> **注意**: 如果 T4 用 `use_vlm=false`，可以跳过 T2。

---

## T3 — 启动 C++ FSM（探索规划器）

```bash
source ~/explorer_ws/devel/setup.bash
roslaunch minco_curve run_in_habitat.launch
```

这个 launch 文件做了：
- 启动 `fake_explorer_fsm_habitat_node`（核心 FSM，Habitat 模式）
- 发布 `/habitat/plan_action`（离散动作） → Python 端订阅
- 订阅 `/habitat/state`、`/yolo_detector/single_mask`、`/detector/confidence_threshold` ← Python 端发布
- 加载 DEP 参数 + odom 可视化 + RViz

---

## T4 — 启动 Habitat Evaluation

### 最小模式（不走 VLM，单 episode，验证联通性）

推荐先跑这个：

```bash
conda activate habitat
source ~/explorer_ws/devel/setup.bash
cd /home/tianbot/explorer/Planner/src

PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$HOME/explorer_ws/devel/lib/python3/dist-packages \
python habitat_evaluation.py --dataset hm3dv2 test_epi_num=0 use_vlm=false use_itm=false
```

### 全流程模式

```bash
conda activate habitat
source ~/explorer_ws/devel/setup.bash
cd /home/tianbot/explorer/Planner/src

PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$HOME/explorer_ws/devel/lib/python3/dist-packages \
python habitat_evaluation.py --dataset hm3dv2
```

---

## T5 — (可选) 单独 RViz 监控

```bash
source ~/explorer_ws/devel/setup.bash
rviz -d /home/tianbot/explorer/Planner/src/minco_curve/rviz/display_habitat.rviz
```

---

## Topic 对接速查

```
habitat_evaluation.py (Python)           C++ FSM
─────────────────────────────────        ──────────────────────────
→ /habitat/camera_depth      ────────→   ~grid_map/depth
→ /grid_map/depth            ────────→   ~grid_map/depth
→ /drone_0_visual_slam/odom  ────────→   odom_world
→ /grid_map/odom             ────────→   /grid_map/odom
→ /habitat/state             ────────→   /habitat/state
→ /yolo_detector/single_mask ────────→   ObjectMapManager
→ /detector/confidence_threshold ─────→   ObjectMapManager
← /habitat/plan_action       ←────────   discrete_action_pub_
```

---

## 常用参数组合

| 场景 | 额外参数 |
|---|---|
| 单集测试（无 VLM） | `test_epi_num=0 use_vlm=false use_itm=false` |
| 单集测试（有 VLM） | `test_epi_num=0` |
| 开视频录制 | `need_video=true` |
| 关 ITM | `use_itm=false` |
| 开 3D 点云 | `use_point_cloud=true` |
| HM3D v1 | `--dataset hm3dv1` |
| MP3D | `--dataset mp3d` |
| 断点续评 | 不加参数，自动读 `continue.txt` |

---

## 常见问题排查

| 症状 | 检查 |
|---|---|
| 卡在 "Waiting for FSM..." | `rostopic echo /habitat/plan_action` 看 C++ 是否发 action |
| VLM 连不上 | `curl http://localhost:12185/health` 确认 Flask 服务存活 |
| 找不到 dataset JSON | `ls data/datasets/objectnav/hm3d/v2/val/val.json.gz` |
| odom 不更新 | `rostopic echo /drone_0_visual_slam/odom` |
| Topic 不通 | `rostopic list \| grep habitat` 检查 topic 列表 |
| C++ FSM 崩溃 | T3 终端看报错，通常是参数或 topic remap 问题 |
