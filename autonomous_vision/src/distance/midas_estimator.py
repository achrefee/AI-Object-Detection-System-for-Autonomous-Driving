"""
midas_estimator.py — MiDaS monocular depth estimation.

Uses Intel MiDaS v3 to produce relative depth maps from single images,
then scales them to approximate absolute distances.
"""

import numpy as np

try:
    import torch
    import cv2
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MiDaSEstimator:
    """
    Estimate depth using MiDaS monocular depth estimation.

    MiDaS produces relative (inverse) depth maps. We scale them
    using reference objects or calibration data.

    Usage:
        estimator = MiDaSEstimator()
        depth_map = estimator.get_depth_map(frame)
        distance = estimator.estimate(detection, frame)
    """

    # Available MiDaS model types (speed vs accuracy tradeoff)
    MODEL_TYPES = {
        "small": "MiDaS_small",       # Fastest, lower accuracy
        "hybrid": "DPT_Hybrid",       # Good balance
        "large": "DPT_Large",         # Best accuracy, slowest
    }

    def __init__(
        self,
        model_type: str = "small",
        device: str = "cuda",
        scale_factor: float = 1.0,
    ):
        """
        Args:
            model_type: "small", "hybrid", or "large".
            device: "cuda" or "cpu".
            scale_factor: Multiplier to convert relative depth to meters.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.scale_factor = scale_factor
        self._depth_cache = None
        self._cache_frame_id = None

        # Load MiDaS model
        model_name = self.MODEL_TYPES.get(model_type, "MiDaS_small")
        self.model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        # Load appropriate transform
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if model_type == "large":
            self.transform = midas_transforms.dpt_transform
        elif model_type == "hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    @torch.no_grad()
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate a relative depth map for the entire frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            2D numpy array of relative depth values (higher = closer).
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform
        input_batch = self.transform(rgb).to(self.device)

        # Inference
        prediction = self.model(input_batch)

        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def estimate(self, detection, frame: np.ndarray) -> float:
        """
        Estimate distance to a detection using MiDaS depth map.

        Args:
            detection: Detection object with bbox.
            frame: BGR image.

        Returns:
            Estimated distance in meters (approximate).
        """
        depth_map = self.get_depth_map(frame)

        x1, y1, x2, y2 = detection.bbox
        h, w = depth_map.shape[:2]

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return -1.0

        # Get center region of bbox (more reliable than edges)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        pad_x = max(1, (x2 - x1) // 6)
        pad_y = max(1, (y2 - y1) // 6)

        roi = depth_map[
            max(0, cy - pad_y):min(h, cy + pad_y),
            max(0, cx - pad_x):min(w, cx + pad_x),
        ]

        if roi.size == 0:
            return -1.0

        # MiDaS outputs inverse depth (higher value = closer)
        # Convert to distance: distance ∝ 1 / depth
        median_depth = np.median(roi)

        if median_depth <= 0:
            return -1.0

        distance = self.scale_factor / median_depth

        # Clamp to reasonable range
        distance = max(0.5, min(distance, 200.0))

        return round(distance, 2)

    def calibrate_scale(self, known_distance: float, depth_value: float):
        """
        Calibrate the scale factor using a known reference.

        Args:
            known_distance: True distance to reference object (meters).
            depth_value: MiDaS depth value at that object.
        """
        if depth_value > 0:
            self.scale_factor = known_distance * depth_value
