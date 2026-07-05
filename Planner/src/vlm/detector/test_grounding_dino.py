#!/usr/bin/env python3
"""
测试 Grounding DINO 检测 COCO 数据集以外的物体。

用法:
    # 检测单个图片中的任意物体
    python test_grounding_dino.py  --prompt "tank"

    # 检测目录下所有图片
    python test_grounding_dino.py --image_dir /path/to/images/ --prompt "monkey . banana . sunglasses ."

    # 调低阈值以获取更多检测结果
    python test_grounding_dino.py --image /path/to/image.jpg --prompt "cat . dog ." --box_threshold 0.2 --text_threshold 0.15

    # 保存标注后的图片
    python test_grounding_dino.py --image /path/to/image.jpg --prompt "laptop . coffee mug ." --save

    # 逐个交互式测试
    python test_grounding_dino.py --image_dir /path/to/images/ --interactive
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 将 vlm 包的父目录加入 sys.path，确保能正确导入
SCRIPT_DIR = Path(__file__).resolve().parent
VLM_DIR = SCRIPT_DIR.parent  # vlm/
PLANNER_SRC_DIR = VLM_DIR.parent  # Planner/src/
if str(PLANNER_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNER_SRC_DIR))

from vlm.detector.grounding_dino import GroundingDINO


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(path: str) -> list[Path]:
    """收集待检测的图片路径。可以是单张图片或目录。"""
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [p]
        else:
            print(f"⚠ 不支持的文件格式: {p.suffix}, 跳过")
            return []
    elif p.is_dir():
        images = sorted(
            [
                f
                for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )
        if not images:
            print(f"⚠ 目录 {path} 中未找到支持的图片文件")
        return images
    else:
        print(f"⚠ 路径不存在: {path}")
        return []


def run_detection(
    detector: GroundingDINO,
    image_path: Path,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    save: bool,
    output_dir: Path | None = None,
) -> None:
    """对单张图片进行检测并打印/保存结果。"""
    print(f"\n{'=' * 70}")
    print(f"📷 图片: {image_path}")
    print(f"🔍 检测目标: {prompt}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = detector.predict(
        image=image_rgb,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if detections.num_detections == 0:
        print("⚠ 未检测到任何物体。可以尝试降低阈值或更换 prompt。")
    else:
        print(f"✅ 检测到 {detections.num_detections} 个物体:\n{detections}")

    # 显示/保存标注图像
    annotated = detections.annotated_frame
    if annotated is not None:
        if save and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"detected_{image_path.name}"
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), annotated_bgr)
            print(f"💾 标注图片已保存至: {out_path}")
        else:
            # 在窗口中显示
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imshow("Grounding DINO Detection", annotated_bgr)
            print("按任意键继续... (在图片窗口上按键)")
            cv2.waitKey(0)


def interactive_mode(
    detector: GroundingDINO,
    images: list[Path],
    default_prompt: str,
    box_threshold: float,
    text_threshold: float,
    save: bool,
    output_dir: Path | None,
) -> None:
    """交互式逐个测试，允许每张图使用不同的 prompt 和阈值。"""
    print("\n🎮 进入交互模式。可用命令:")
    print("   <输入 prompt>  - 使用新的 prompt 检测当前图片")
    print("   回车 (空输入)  - 使用上一次的 prompt 检测当前图片")
    print("   s              - 跳过当前图片")
    print("   t box_thresh text_thresh  - 修改阈值，例如: t 0.3 0.2")
    print("   q              - 退出\n")

    current_prompt = default_prompt
    current_box_thresh = box_threshold
    current_text_thresh = text_threshold

    for img_path in images:
        print(f"\n--- 当前图片: {img_path.name} ---")
        print(f"当前 prompt: \"{current_prompt}\" | box_threshold={current_box_thresh} | text_threshold={current_text_thresh}")

        user_input = input("👉 ").strip()

        if user_input.lower() == "q":
            print("👋 退出")
            break
        elif user_input.lower() == "s":
            print("⏭ 跳过")
            continue
        elif user_input.lower().startswith("t "):
            parts = user_input.split()
            if len(parts) >= 3:
                try:
                    current_box_thresh = float(parts[1])
                    current_text_thresh = float(parts[2])
                    print(f"阈值已更新: box_threshold={current_box_thresh}, text_threshold={current_text_thresh}")
                except ValueError:
                    print("⚠ 阈值格式错误，使用默认值")
            # 使用当前 prompt 检测
            run_detection(
                detector, img_path, current_prompt,
                current_box_thresh, current_text_thresh, save, output_dir,
            )
        elif user_input:
            current_prompt = user_input
            run_detection(
                detector, img_path, current_prompt,
                current_box_thresh, current_text_thresh, save, output_dir,
            )
        else:
            # 空输入，使用当前 prompt
            run_detection(
                detector, img_path, current_prompt,
                current_box_thresh, current_text_thresh, save, output_dir,
            )

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="测试 Grounding DINO 检测 COCO 数据集以外的物体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 检测自定义物体
    python test_grounding_dino.py --image cat.jpg --prompt "Persian cat . British Shorthair ."

    # 批量检测并保存结果
    python test_grounding_dino.py --image_dir ./my_photos/ --prompt "bicycle . motorcycle . helmet ." --save

    # 交互式逐步检测
    python test_grounding_dino.py --image_dir ./my_photos/ --interactive --save
        """,
    )

    parser.add_argument(
        "--image", type=str, default=r"/root/gpufree-data/explorer_ws/src/Planner/src/vlm/tank.jpeg",
        help="单张图片的路径",
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="包含图片的目录路径",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help='要检测的物体描述，用 " . " 分隔多个类别。例如: "red car . blue bicycle . coffee cup ." '
             "留空则在交互模式下手动输入。",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.35,
        help="边界框置信度阈值 (默认: 0.35)",
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25,
        help="文本匹配置信度阈值 (默认: 0.25)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="保存标注后的图片到 ./test_output/ 目录",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./test_output",
        help="标注图片的输出目录 (默认: ./test_output)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="交互模式：逐张图片检测，支持动态切换 prompt 和阈值",
    )

    args = parser.parse_args()

    # 检查输入来源
    if args.image is None and args.image_dir is None:
        parser.error("必须指定 --image 或 --image_dir")

    # 收集图片
    images = []
    if args.image:
        images = collect_images(args.image)
    if args.image_dir:
        images = collect_images(args.image_dir)

    if not images:
        print("❌ 未找到任何可检测的图片，退出。")
        sys.exit(1)

    print(f"📁 共找到 {len(images)} 张图片待检测")
    print(f"🚀 加载 Grounding DINO 模型...")

    # 加载模型
    detector = GroundingDINO()
    print("✅ 模型加载完成！")

    output_dir = Path(args.output_dir) if args.save else None

    if args.interactive:
        interactive_mode(
            detector=detector,
            images=images,
            default_prompt=args.prompt or "a cat . a dog .",
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            save=args.save,
            output_dir=output_dir,
        )
    else:
        if args.prompt is None:
            parser.error("非交互模式必须指定 --prompt")

        for img_path in images:
            run_detection(
                detector=detector,
                image_path=img_path,
                prompt=args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                save=args.save,
                output_dir=output_dir,
            )

        cv2.destroyAllWindows()

    print("\n✨ 全部检测完成！")


if __name__ == "__main__":
    main()