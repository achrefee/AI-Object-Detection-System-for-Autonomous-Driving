"""
tracker.py — Multi-object tracker with velocity estimation.

Uses Ultralytics built-in Bot-SORT/ByteTrack for persistent tracking,
and estimates per-object velocity from track history.
"""

from collections import defaultdict, deque

import numpy as np

from .detector import Detection


class ObjectTracker:
    """
    Track objects across frames and estimate their velocity.

    Works with detections that already have track_id assigned
    (via ObjectDetector.detect_and_track()).

    Usage:
        tracker = ObjectTracker()
        detections = detector.detect_and_track(frame)
        detections = tracker.update(detections)
        vx, vy = tracker.get_velocity(track_id=5)
    """

    def __init__(self, history_length: int = 10):
        """
        Args:
            history_length: Number of past positions to keep per track.
        """
        self.history_length = history_length
        # track_id → deque of (center_x, center_y) positions
        self._tracks: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        # track_id → (vx, vy) in pixels/frame
        self._velocities: dict[int, tuple[float, float]] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        """
        Update tracker with new detections and compute velocities.

        Args:
            detections: List of Detection objects with track_id set.

        Returns:
            Same detections with velocity field populated.
        """
        seen_ids = set()

        for det in detections:
            if det.track_id < 0:
                continue

            seen_ids.add(det.track_id)
            cx, cy = det.center
            self._tracks[det.track_id].append((cx, cy))

            # Compute velocity from position history
            vx, vy = self._compute_velocity(det.track_id)
            self._velocities[det.track_id] = (vx, vy)
            det.velocity = (vx, vy)

        # Clean up stale tracks
        stale_ids = [tid for tid in self._tracks if tid not in seen_ids]
        for tid in stale_ids:
            # Keep for a while in case the object re-appears
            if len(self._tracks[tid]) > 0:
                self._tracks[tid].append(self._tracks[tid][-1])

        return detections

    def get_velocity(self, track_id: int) -> tuple[float, float]:
        """
        Get the velocity of a tracked object.

        Args:
            track_id: The track ID.

        Returns:
            (vx, vy) in pixels per frame. (0, 0) if unknown.
        """
        return self._velocities.get(track_id, (0.0, 0.0))

    def get_speed(self, track_id: int) -> float:
        """
        Get the scalar speed of a tracked object.

        Returns:
            Speed in pixels per frame. 0 if unknown.
        """
        vx, vy = self.get_velocity(track_id)
        return float(np.sqrt(vx ** 2 + vy ** 2))

    def _compute_velocity(self, track_id: int) -> tuple[float, float]:
        """
        Compute average velocity from recent position history.
        Uses weighted average favoring recent positions.
        """
        positions = self._tracks[track_id]
        n = len(positions)

        if n < 2:
            return (0.0, 0.0)

        # Compute velocity from last few positions (weighted)
        total_vx = 0.0
        total_vy = 0.0
        total_weight = 0.0

        for i in range(1, min(n, 5)):  # Use last 5 frames max
            dx = positions[-i][0] - positions[-i - 1 if -i - 1 >= -n else 0][0]
            dy = positions[-i][1] - positions[-i - 1 if -i - 1 >= -n else 0][1]
            weight = 1.0 / i  # Recent frames weighted more
            total_vx += dx * weight
            total_vy += dy * weight
            total_weight += weight

        if total_weight > 0:
            return (total_vx / total_weight, total_vy / total_weight)
        return (0.0, 0.0)

    def reset(self):
        """Clear all tracking data."""
        self._tracks.clear()
        self._velocities.clear()
