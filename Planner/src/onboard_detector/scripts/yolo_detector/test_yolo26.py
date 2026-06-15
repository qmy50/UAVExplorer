from ultralytics import YOLO

# 自动下载 yolo26n.pt 权重并加载模型
model = YOLO(r'/home/tianbot/explorer_ws/src/Planner/src/onboard_detector/scripts/yolo_detector/weights/yolo26n-seg.pt')

# 对单张图片进行推理
results = model(r'/home/tianbot/explorer_ws/src/Planner/src/onboard_detector/scripts/yolo_detector/test_data/1.jpg')

# 显示结果
results[0].show()