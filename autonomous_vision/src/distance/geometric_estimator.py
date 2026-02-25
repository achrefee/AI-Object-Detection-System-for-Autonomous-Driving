"""
geometric_estimator.py — Pinhole camera model distance estimation.

Uses the formula: D = (H_real × f_y) / h_bbox
where:
    D       = Distance to object (meters)
    H_real  = Known real-world height of the object (meters)
    f_y     = Focal length in pixels (vertical)
    h_bbox  = Height of bounding box in pixels
"""

import numpy as np

from ..utils.config import Config


# Default known heights (meters) per class
DEFAULT_HEIGHTS = {
    "car": 1.50,
    "truck": 3.50,
    "bus": 3.20,
    "motorcycle": 1.10,
    "bicycle": 1.00,
    "pedestrian": 1.70,
    "cyclist": 1.70,
    "traffic_light_red": 0.40,
    "traffic_light_green": 0.40,
    "traffic_light_yellow": 0.40,
    "traffic_sign": 0.75,
}


class GeometricEstimator:
    """
    Estimate object distance using the pinhole camera model.

    Fast and reliable for objects with known real-world dimensions,
    especially at close to medium range (5–30 meters).

    Usage:
        estimator = GeometricEstimator(focal_length=800.0)
        distance = estimator.estimate(detection)
    """

    def __init__(
        self,
        focal_length: float = 800.0,
        known_heights: dict[str, float] | None = None,
        min_bbox_height: int = 5,
    ):
        """
        Args:
            focal_length: Camera focal length in pixels (f_y).
            known_heights: Dict mapping class_name → real height in meters.
            min_bbox_height: Minimum bbox height to compute distance (avoid div by zero).
        """
        self.focal_length = focal_length
        self.known_heights = known_heights or DEFAULT_HEIGHTS.copy()
        self.min_bbox_height = min_bbox_height

    @classmethod
    def from_config(cls, config: Config) -> "GeometricEstimator":
        """Create from a Config object (camera_params.yaml)."""
        focal_length = config.camera.fy if hasattr(config, "camera") else 800.0

        # Load known heights from config
        known_heights = DEFAULT_HEIGHTS.copy()
        if hasattr(config, "known_dimensions"):
            dims = config.known_dimensions
            for cls_name in DEFAULT_HEIGHTS:
                if hasattr(dims, cls_name):
                    cls_dims = getattr(dims, cls_name)
                    known_heights[cls_name] = cls_dims.height

        return cls(focal_length=focal_length, known_heights=known_heights)

    def estimate(self, detection) -> float:
        """
        Estimate distance to a single detection.

        Args:
            detection: Detection object with bbox and class_name.

        Returns:
            Estimated distance in meters. Returns -1 if cannot estimate.
        """
        bbox_height = detection.height

        if bbox_height < self.min_bbox_height:
            return -1.0

        class_name = detection.class_name
        real_height = self.known_heights.get(class_name)

        if real_height is None:
            return -1.0

        distance = (real_height * self.focal_length) / bbox_height

        # Clamp to reasonable range
        distance = max(0.5, min(distance, 200.0))

        return round(distance, 2)

    def estimate_batch(self, detections: list) -> list[float]:
        """Estimate distances for a batch of detections."""
        return [self.estimate(det) for det in detections]
