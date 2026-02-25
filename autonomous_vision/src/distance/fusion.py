"""
fusion.py — Hybrid distance estimation combining geometric and MiDaS.

Stage 1 (fast): Geometric pinhole model estimate.
Stage 2 (accurate): MiDaS monocular depth refinement.
Output: Weighted fusion + Kalman smoothing.
"""

import numpy as np

from .geometric_estimator import GeometricEstimator
from .kalman_filter import KalmanFilterManager


class DistanceFusion:
    """
    Fuse geometric and MiDaS distance estimates.

    - Close range (< 15m): Trust geometric more (alpha → 1.0)
    - Far range (> 30m): Trust MiDaS more (alpha → 0.0)
    - Medium range: Weighted blend

    Usage:
        fusion = DistanceFusion(geometric_estimator, midas_estimator)
        distance = fusion.estimate(detection, frame)
    """

    def __init__(
        self,
        geometric: GeometricEstimator,
        midas=None,  # Optional MiDaSEstimator
        alpha_close: float = 0.8,
        alpha_far: float = 0.3,
        close_threshold: float = 15.0,
        far_threshold: float = 30.0,
        use_kalman: bool = True,
    ):
        """
        Args:
            geometric: GeometricEstimator instance.
            midas: MiDaSEstimator instance (optional, can be None).
            alpha_close: Weight for geometric at close range.
            alpha_far: Weight for geometric at far range.
            close_threshold: Distance below which geometric is heavily favored.
            far_threshold: Distance above which MiDaS is heavily favored.
            use_kalman: Whether to apply Kalman smoothing.
        """
        self.geometric = geometric
        self.midas = midas
        self.alpha_close = alpha_close
        self.alpha_far = alpha_far
        self.close_threshold = close_threshold
        self.far_threshold = far_threshold

        self.kalman = KalmanFilterManager() if use_kalman else None

    def estimate(self, detection, frame: np.ndarray | None = None) -> float:
        """
        Estimate distance using the best available method.

        Args:
            detection: Detection object.
            frame: BGR image (required for MiDaS).

        Returns:
            Fused distance in meters.
        """
        # Stage 1: Geometric estimate (always available)
        d_geo = self.geometric.estimate(detection)

        # Stage 2: MiDaS estimate (if available)
        d_midas = -1.0
        if self.midas is not None and frame is not None:
            try:
                d_midas = self.midas.estimate(detection, frame)
            except Exception:
                d_midas = -1.0

        # Fusion logic
        if d_geo > 0 and d_midas > 0:
            # Both available — weighted fusion
            alpha = self._compute_alpha(d_geo)
            distance = alpha * d_geo + (1 - alpha) * d_midas
        elif d_geo > 0:
            distance = d_geo
        elif d_midas > 0:
            distance = d_midas
        else:
            distance = -1.0

        # Kalman smoothing
        if distance > 0 and self.kalman is not None and detection.track_id >= 0:
            distance = self.kalman.update(detection.track_id, distance)

        return round(distance, 2) if distance > 0 else -1.0

    def estimate_batch(
        self, detections: list, frame: np.ndarray | None = None
    ) -> list:
        """
        Estimate distances for all detections and update their distance field.

        Args:
            detections: List of Detection objects.
            frame: BGR image.

        Returns:
            Same detections with distance field populated.
        """
        for det in detections:
            det.distance = self.estimate(det, frame)
        return detections

    def _compute_alpha(self, d_geo: float) -> float:
        """
        Compute the geometric weight based on estimated distance.

        Close → alpha ≈ alpha_close (trust geometric)
        Far   → alpha ≈ alpha_far (trust MiDaS)
        """
        if d_geo <= self.close_threshold:
            return self.alpha_close
        elif d_geo >= self.far_threshold:
            return self.alpha_far
        else:
            # Linear interpolation
            t = (d_geo - self.close_threshold) / (self.far_threshold - self.close_threshold)
            return self.alpha_close + t * (self.alpha_far - self.alpha_close)

    def cleanup_tracks(self, active_ids: set[int]):
        """Remove Kalman filters for stale tracks."""
        if self.kalman:
            self.kalman.cleanup(active_ids)
