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
    ) -> None:
        import torch
        from PIL import Image
        from lavis.models import load_model_and_preprocess

        self._torch = torch
        self._Image = Image

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
    args = parser.parse_args()

    print("Loading model...")

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {
                "response": self.cosine(image, payload["txt"]),
                "itm score": self.itm_scores(image, payload["txt"]),
            }

    blip = BLIP2ITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port)
