import sys
import os
import time
# Add project root to sys.path so absolute imports work when running this script directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_LOCAL_FILES_CHECK"] = "1"
os.environ["HF_HUB_DISABLE_RECOMMENDER"] = "1"

from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
import numpy as np
from vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image



class BLIP2Matcher:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            name="blip2_image_text_matching",
            model_type="coco",   # 可改成 "pretrain"
            is_eval=True,
            device=self.device      
        )

    def cosine_from_path(self, image_path: str, txt: str) -> float:
        raw_image = Image.open(image_path).convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)

        text = self.text_processors["eval"](txt)

        with torch.inference_mode():
            score = self.model(
                {"image": image, "text_input": text},
                match_head="itc"
            ).item()

        return float(score)

    def cosine_from_numpy(self, image: np.ndarray, txt: str) -> float:
        pil_img = Image.fromarray(image).convert("RGB")
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)

        text = self.text_processors["eval"](txt)

        with torch.inference_mode():
            score = self.model(
                {"image": img, "text_input": text},
                match_head="itc"
            ).item()

        return float(score)

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """Compute cosine similarity between image and text (same as cosine_from_numpy)."""
        return self.cosine_from_numpy(image, txt)

    def itm_scores(self, image: np.ndarray, txt: str) -> float:
        """Compute ITM (Image-Text Matching) score between image and text."""
        pil_img = Image.fromarray(image).convert("RGB")
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)

        text = self.text_processors["eval"](txt)

        with torch.inference_mode():
            itm_output = self.model({"image": img, "text_input": text}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)

        itm_score = itm_scores[:, 1].item()
        return itm_score


class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        # print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])

    def itm_score(self, image: np.ndarray, txt: str) -> float:
        print(f"Question of blip2 is:{txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["itm score"])


if __name__ == "__main__":
    matcher = BLIP2Matcher()
    image_path = os.path.join(os.path.dirname(__file__), "img2.png")
    txt = "Is there a car in the image?"

    # warm up
    print("Warming up...")
    _ = matcher.cosine_from_path(image_path, txt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # real timing
    for i in range(5):
        time_start = time.time()
        cosine_score = matcher.cosine_from_path(image_path, txt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_end = time.time()

        print(f"[{i}] Time: {time_end - time_start:.4f}s, score: {cosine_score}")

    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", type=int, default=12182)
    # args = parser.parse_args()

    # print("Loading model...")

    # class BLIP2ITMServer(ServerMixin, BLIP2Matcher):
    #     def process_payload(self, payload: dict) -> dict:
    #         image = str_to_image(payload["image"])
    #         return {
    #             "response": self.cosine(image, payload["txt"]),
    #             "itm score": self.itm_scores(image, payload["txt"]),
    #         }

    # blip = BLIP2ITMServer()
    # print("Model loaded!")
    # print(f"Hosting on port {args.port}...")
    # host_model(blip, name="blip2itm", port=args.port)