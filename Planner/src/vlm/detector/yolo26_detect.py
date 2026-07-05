from typing import List, Optional

import cv2
import numpy as np

from vlm.coco_classes import COCO_CLASSES
from vlm.detector.detections import ObjectDetections

from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image


class YOLO26Detect:
    """yolo26s detection-only model (no segmentation)."""

    def __init__(
        self,
        weights: str,
        image_size: int = 640,
    ):
        import torch
        from ultralytics import YOLO

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._torch = torch
        self.model = YOLO(weights)
        self.image_size = image_size

        # Warm-up
        if self.device.type != "cpu":
            dummy_img = np.random.randint(
                0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8
            )
            for _ in range(3):
                self.model(dummy_img, imgsz=self.image_size, verbose=False)

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic_nms: bool = False,
    ) -> ObjectDetections:
        """
        Run detection on an image and return ObjectDetections.

        Args:
            image: BGR image as numpy array (H, W, 3).
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold for NMS.
            classes: List of class names to filter (not used here, kept for API compat).
            agnostic_nms: Not used by ultralytics, kept for API compat.

        Returns:
            ObjectDetections with normalized xyxy boxes.
        """
        orig_h, orig_w = image.shape[:2]

        results = self.model(
            image,
            conf=conf_thres,
            iou=iou_thres,
            imgsz=self.image_size,
            verbose=False,
        )

        all_boxes = []
        all_logits = []
        all_phrases = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                xyxy = boxes.xyxy[i].cpu().numpy()  # pixel coords [x1, y1, x2, y2]

                # Normalize to [0, 1]
                xyxy_norm = [
                    xyxy[0] / orig_w,
                    xyxy[1] / orig_h,
                    xyxy[2] / orig_w,
                    xyxy[3] / orig_h,
                ]
                all_boxes.append(xyxy_norm)
                all_logits.append(conf)
                all_phrases.append(COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}")

        boxes_t = self._torch.tensor(all_boxes) if all_boxes else self._torch.empty((0, 4))
        logits_t = self._torch.tensor(all_logits) if all_logits else self._torch.empty((0,))

        return ObjectDetections(
            boxes_t, logits_t, all_phrases, image_source=image, fmt="xyxy"
        )


class YOLO26DetectClient:
    """HTTP client for yolo26s detection Flask server."""

    def __init__(self, port: int = 12185):
        self.url = f"http://localhost:{port}/yolo26"

    def predict(
        self,
        image_numpy: np.ndarray,
        agnostic_nms: bool = True,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> ObjectDetections:
        response = send_request(
            self.url,
            image=image_numpy,
            agnostic_nms=agnostic_nms,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12185)
    parser.add_argument(
        "--weights",
        type=str,
        default=r"/root/gpufree-data/explorer_ws/src/Planner/src/onboard_detector/scripts/yolo_detector/weights/yolo26s.pt",
    )
    args = parser.parse_args()

    print("Loading yolo26m detection model...")

    class YOLO26DetectServer(ServerMixin, YOLO26Detect):
        def process_payload(self, payload: dict) -> dict:
            agnostic_nms = payload.get("agnostic_nms", True)
            conf_thres = payload.get("conf_thres", 0.25)
            iou_thres = payload.get("iou_thres", 0.45)
            image = str_to_image(payload["image"])
            return self.predict(
                image,
                agnostic_nms=agnostic_nms,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            ).to_json()

    yolo26 = YOLO26DetectServer(weights=args.weights)
    print("Model loaded!")
    print(f"Hosting yolo26s detection on port {args.port}...")
    host_model(yolo26, name="yolo26", port=args.port)