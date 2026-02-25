"""
kalman_filter.py — Kalman filter for temporal distance smoothing.

Reduces jitter in distance estimates by smoothing over time.
One filter instance per tracked object.
"""

import numpy as np


class DistanceKalmanFilter:
    """
    1D Kalman filter for smoothing distance estimates over time.

    State: [distance, velocity]
    Observation: [distance]

    Usage:
        kf = DistanceKalmanFilter()
        smoothed = kf.update(raw_distance)
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
        initial_distance: float = 10.0,
    ):
        """
        Args:
            process_noise: How much we expect the distance to change per step.
            measurement_noise: How noisy the measurements are.
            initial_distance: Initial distance estimate.
        """
        # State vector: [distance, velocity]
        self.x = np.array([initial_distance, 0.0])

        # State covariance
        self.P = np.array([
            [10.0, 0.0],
            [0.0, 5.0],
        ])

        # State transition (constant velocity model)
        # dt = 1 frame
        self.F = np.array([
            [1.0, 1.0],
            [0.0, 1.0],
        ])

        # Observation matrix (we only observe distance)
        self.H = np.array([[1.0, 0.0]])

        # Process noise
        self.Q = np.array([
            [process_noise, 0.0],
            [0.0, process_noise * 0.5],
        ])

        # Measurement noise
        self.R = np.array([[measurement_noise]])

        self._initialized = False

    def predict(self) -> float:
        """
        Predict the next state.

        Returns:
            Predicted distance.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0])

    def update(self, measurement: float) -> float:
        """
        Update the filter with a new distance measurement.

        Args:
            measurement: Raw distance estimate (meters).

        Returns:
            Smoothed distance estimate (meters).
        """
        if not self._initialized:
            self.x[0] = measurement
            self._initialized = True
            return measurement

        # Predict
        self.predict()

        # Update
        z = np.array([measurement])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        # Ensure distance is positive
        self.x[0] = max(0.1, self.x[0])

        return float(self.x[0])

    @property
    def distance(self) -> float:
        """Current smoothed distance estimate."""
        return float(self.x[0])

    @property
    def velocity(self) -> float:
        """Current estimated velocity (m/frame)."""
        return float(self.x[1])

    def reset(self, distance: float = 10.0):
        """Reset the filter state."""
        self.x = np.array([distance, 0.0])
        self.P = np.array([[10.0, 0.0], [0.0, 5.0]])
        self._initialized = False


class KalmanFilterManager:
    """
    Manage multiple Kalman filters, one per tracked object.

    Usage:
        manager = KalmanFilterManager()
        smoothed = manager.update(track_id=5, raw_distance=12.3)
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._filters: dict[int, DistanceKalmanFilter] = {}

    def update(self, track_id: int, raw_distance: float) -> float:
        """
        Update the Kalman filter for a specific track.

        Args:
            track_id: Object track ID.
            raw_distance: Raw distance measurement.

        Returns:
            Smoothed distance.
        """
        if track_id not in self._filters:
            self._filters[track_id] = DistanceKalmanFilter(
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
                initial_distance=raw_distance,
            )

        return self._filters[track_id].update(raw_distance)

    def remove_track(self, track_id: int):
        """Remove filter for a track that no longer exists."""
        self._filters.pop(track_id, None)

    def cleanup(self, active_ids: set[int]):
        """Remove filters for tracks not in the active set."""
        stale = [tid for tid in self._filters if tid not in active_ids]
        for tid in stale:
            del self._filters[tid]

    def reset(self):
        """Clear all filters."""
        self._filters.clear()
