#!/usr/bin/env python3
"""
blip2_itm_test.py — BLIP2 ITM 本地测试脚本

不依赖 ROS / cv_bridge，直接读取本地图片，测试 BLIP2 模型能否正常工作。
用法:  python blip2_itm_test.py --image /path/to/img.jpg --label "chair"
"""

import os
import sys
import time
import argparse

# 将 conda torch 环境的 site-packages 提到最前面，避免被 ROS PYTHONPATH 中的系统旧包（numpy/cv2）覆盖
_CONDA_TORCH_SITE = os.path.expanduser("~/anaconda3/envs/torch/lib/python3.9/site-packages")
if os.path.isdir(_CONDA_TORCH_SITE) and _CONDA_TORCH_SITE not in sys.path:
    sys.path.insert(0, _CONDA_TORCH_SITE)

# 将当前脚本所在目录的上级目录（即 vlm 包的父目录）加入 Python path
# blip2_itm_test.py 位于 src/Planner/src/vlm/scripts/
# vlm 包位于 src/Planner/src/vlm/
# 所以我们需要将 src/Planner/src/ 加入 sys.path
_VLM_PARENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _VLM_PARENT not in sys.path:
    sys.path.insert(0, _VLM_PARENT)

# BLIP2 离线模式环境变量 (可选，避免下载/联网)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_LOCAL_FILES_CHECK", "1")
os.environ.setdefault("HF_HUB_DISABLE_RECOMMENDER", "1")

import cv2
import numpy as np

from vlm.itm.blip2itm import BLIP2ITM


def main():
    parser = argparse.ArgumentParser(description="BLIP2 ITM 本地测试")
    parser.add_argument("--image", type=str, required=True, help="图片路径")
    parser.add_argument("--label", type=str, default="chair", help="目标物体标签 (默认: chair)")
    parser.add_argument("--room", type=str, default="everywhere", help="房间类型 (默认: everywhere)")
    parser.add_argument("--model_type", type=str, default="coco", help="模型类型: coco 或 pretrain (默认: coco)")
    parser.add_argument("--warmup", action="store_true", help="是否先做一次 warmup 推理")
    args = parser.parse_args()

    # 读图片
    if not os.path.exists(args.image):
        print(f"[ERROR] 图片不存在: {args.image}")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] 无法读取图片: {args.image}")
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"[INFO] 图片尺寸: {image.shape}")

    # 加载模型
    print(f"[INFO] 加载 BLIP2 ITM 模型 (model_type={args.model_type})...")
    t0 = time.time()
    matcher = BLIP2ITM(model_type=args.model_type)
    print(f"[INFO] 模型加载完成, 耗时 {time.time() - t0:.2f}s")

    # warmup
    if args.warmup:
        print("[INFO] Warmup 推理...")
        _ = matcher.cosine(image, "warmup")

    # 测试多个 prompt
    prompts = [
        f"Is there a {args.label} in the image?",
        f"Seems like there is a {args.label} ahead?",
    ]
    if args.room.lower() != "everywhere":
        prompts.append(f"Seems like there is a {args.room} or a {args.label} ahead?")

    print(f"\n{'='*60}")
    print(f"图片: {args.image}")
    print(f"标签: {args.label}, 房间: {args.room}")
    print(f"{'='*60}")

    for prompt in prompts:
        print(f"\n[Prompt] \"{prompt}\"")

        t0 = time.time()
        cosine = matcher.cosine(image, prompt)
        t1 = time.time()

        itm = matcher.itm_scores(image, prompt)
        t2 = time.time()

        print(f"  cosine_score = {cosine:.4f}  ({t1-t0:.3f}s)")
        print(f"  itm_score    = {itm:.4f}  ({t2-t1:.3f}s)")

    print(f"\n{'='*60}")
    print("测试完成!")


if __name__ == "__main__":
    main()


# python blip2_itm_test.py --image /home/tianbot/explorer_ws/src/Planner/src/vlm/scripts/image/test1.jpg --label "bed"
