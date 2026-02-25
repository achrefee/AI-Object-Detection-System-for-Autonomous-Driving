"""
detector.py — YOLO object detection wrapper.

Wraps Ultralytics YOLO for inference with a clean API.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ── 11-class names (BDD100K) ─────────────────────────────────────────
CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorcycle",
    4: "bicycle",
    5: "pedestrian",
    6: "cyclist",
    7: "traffic_light_red",
    8: "traffic_light_green",
    9: "traffic_light_yellow",
    10: "traffic_sign",
}


@dataclass
class Detection:
    """A single detected object."""

    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    track_id: int = -1
    distance: float = 0.0             # Estimated distance in meters
    risk_level: str = "SAFE"
    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) pixels/frame

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class ObjectDetector:
    """
    YOLO-based object detector for autonomous driving.

    Usage:
        detector = ObjectDetector("weights/best.pt")
        detections = detector.detect(frame)
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8s.pt",
        device: str = "cuda",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        classes: list[int] | None = None,
    ):
        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. Install with: pip install ultralytics"
            )

        self.model_path = str(model_path)
        self.device = device
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.classes = classes

        # Load model
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # Class names — use model's names if available, else defaults
        self._class_names = CLASS_NAMES.copy()
        if hasattr(self.model, "names") and self.model.names:
            self._class_names = {int(k): v for k, v in self.model.names.items()}

    def warmup(self, imgsz: int = 640):
        """Run a dummy inference to warm up the model."""
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model.predict(
            dummy,
            device=self.device,
            verbose=False,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of Detection objects.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    det = Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        class_id=cls_id,
                        class_name=self._class_names.get(cls_id, f"class_{cls_id}"),
                        confidence=conf,
                    )
                    detections.append(det)

        return detections

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection with built-in tracking (Bot-SORT).

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of Detection objects with track IDs.
        """
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            classes=self.classes,
            device=self.device,
            persist=True,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    track_id = -1
                    if box.id is not None:
                        track_id = int(box.id[0].item())

                    det = Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        class_id=cls_id,
                        class_name=self._class_names.get(cls_id, f"class_{cls_id}"),
                        confidence=conf,
                        track_id=track_id,
                    )
                    detections.append(det)

        return detections

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names
