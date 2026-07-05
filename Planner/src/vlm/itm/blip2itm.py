from typing import Any, Optional

import numpy as np
import cv2
import os, sys

# server_wrapper 导入兼容两种运行方式:
#   1. python -m vlm.itm.blip2itm  (模块模式, 相对导入)
#   2. python blip2itm.py          (直接运行, 自动修复 sys.path)
try:
    from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image
except ImportError:
    _SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    from vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_LOCAL_FILES_CHECK", "1")
os.environ.setdefault("HF_HUB_DISABLE_RECOMMENDER", "1")

class BLIP2ITM:
    """BLIP 2 Image-Text Matching model.

    Torch/LAVIS/PIL are imported lazily so that BLIP2ITMClient can be used
    from a ROS environment without torch installed.
    """

    def __init__(
        self,
        name: str = "blip2_image_text_matching",
        model_type: str = "coco",
        device: Optional[Any] = None,
        use_fp16: bool = True,
    ) -> None:
        """
        Args:
            name: LAVIS model name.
            model_type: LAVIS model type (e.g. "coco", "pretrain").
            device: torch device. None → auto-detect CUDA/CPU.
            use_fp16: Convert model to FP16 on GPU to halve memory usage
                      (~6.8 GB → ~3.4 GB). Set False to keep FP32 precision.
        """
        import torch
        from PIL import Image
        from lavis.models import load_model_and_preprocess

        self._torch = torch
        self._Image = Image
        self._use_fp16 = use_fp16

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                name=name,
                model_type=model_type,
                is_eval=True,
                device=device,
            )
        )
        self.device = device

        if device.type == "cuda" and use_fp16:
            # Only convert ViT encoder to FP16 (~2 GB savings).
            # Q-Former must stay FP32 because Blip2ITM.forward() calls
            # image_embeds.float() before passing to Q-Former → would
            # cause Float/Half dtype mismatch in cross-attention linear layers.
            self.model.visual_encoder = self.model.visual_encoder.half()
            self.model.ln_vision = self.model.ln_vision.half()
            self._torch.cuda.empty_cache()
            alloc = self._torch.cuda.memory_allocated() / (1024 ** 2)
            total = self._torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            print(f"[BLIP2ITM] FP16-ViT mode: allocated={alloc:.0f}MiB / total={total:.0f}MiB")
        elif device.type == "cuda":
            self._torch.cuda.empty_cache()
            alloc = self._torch.cuda.memory_allocated() / (1024 ** 2)
            total = self._torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            print(f"[BLIP2ITM] FP32 mode: allocated={alloc:.0f}MiB / total={total:.0f}MiB")

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        pil_img = self._Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with self._torch.inference_mode():
            cosine = self.model(
                {"image": img, "text_input": txt}, match_head="itc"
            ).item()
        return cosine

    def itm_scores(self, image: np.ndarray, txt: str) -> np.ndarray:
        pil_img = self._Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with self._torch.inference_mode():
            itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
            itm_scores = self._torch.nn.functional.softmax(itm_output, dim=1)

        itm_score = itm_scores[:, 1].item()
        return itm_score


class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        # print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])

    def itm_score(self, image: np.ndarray, txt: str) -> np.ndarray:
        print(f"Question of blip2 is:{txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["itm score"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use FP16 to reduce GPU memory (default: ON, ~3.4 GB)")
    parser.add_argument("--fp32", action="store_true", default=True,
                        help="Use FP32 full precision (overrides --fp16, ~6.8 GB)")
    args = parser.parse_args()

    use_fp16 = not args.fp32

    print(f"Loading model (FP16={use_fp16})...")

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {
                "response": self.cosine(image, payload["txt"]),
                "itm score": self.itm_scores(image, payload["txt"]),
            }

    blip = BLIP2ITMServer(use_fp16=use_fp16)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port)
