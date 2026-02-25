"""
ttc_calculator.py — Time-to-Collision calculation.

TTC = Distance / Relative_Velocity

If relative velocity ≤ 0 → TTC = ∞ (object moving away or same speed)
"""

import math


class TTCCalculator:
    """
    Calculate Time-to-Collision for detected objects.

    Usage:
        ttc_calc = TTCCalculator()
        ttc = ttc_calc.compute(distance=15.0, relative_velocity=5.0)
        level = ttc_calc.classify(ttc)
    """

    def __init__(
        self,
        emergency_threshold: float = 1.5,
        warning_threshold: float = 3.0,
        caution_threshold: float = 5.0,
        ego_speed: float = 0.0,
    ):
        """
        Args:
            emergency_threshold: TTC below this = emergency (seconds).
            warning_threshold: TTC below this = warning (seconds).
            caution_threshold: TTC below this = caution (seconds).
            ego_speed: Speed of our vehicle in m/s (can be updated per frame).
        """
        self.emergency_threshold = emergency_threshold
        self.warning_threshold = warning_threshold
        self.caution_threshold = caution_threshold
        self.ego_speed = ego_speed

    def compute(
        self,
        distance: float,
        relative_velocity: float | None = None,
        object_velocity: float | None = None,
    ) -> float:
        """
        Compute Time-to-Collision.

        Args:
            distance: Distance to object in meters.
            relative_velocity: Relative velocity (m/s). Positive = approaching.
            object_velocity: Object's velocity (m/s). Used with ego_speed
                             if relative_velocity is not provided.

        Returns:
            TTC in seconds. math.inf if not approaching.
        """
        if distance <= 0:
            return 0.0  # Already colliding

        # Determine relative velocity
        if relative_velocity is None:
            if object_velocity is not None:
                relative_velocity = self.ego_speed - object_velocity
            else:
                # Estimate from distance change rate (requires history)
                return math.inf

        # If not approaching (moving away or same speed)
        if relative_velocity <= 0:
            return math.inf

        ttc = distance / relative_velocity
        return round(ttc, 3)

    def classify(self, ttc: float) -> str:
        """
        Classify TTC into urgency levels.

        Returns:
            "EMERGENCY", "WARNING", "CAUTION", or "SAFE".
        """
        if ttc <= self.emergency_threshold:
            return "EMERGENCY"
        elif ttc <= self.warning_threshold:
            return "WARNING"
        elif ttc <= self.caution_threshold:
            return "CAUTION"
        else:
            return "SAFE"

    def compute_for_detection(self, detection, fps: float = 30.0) -> float:
        """
        Compute TTC for a detection using its velocity.

        Args:
            detection: Detection object with distance and velocity.
            fps: Camera framerate for pixel→speed conversion.

        Returns:
            TTC in seconds.
        """
        if detection.distance <= 0:
            return math.inf

        # Velocity in pixels/frame → estimate speed change
        # Positive vy = object moving down in frame = getting closer
        _, vy = detection.velocity

        # Rough approximation: if bbox is growing (vy > 0), object is approaching
        # This is a simplification — real velocity needs calibration
        if vy <= 0:
            return math.inf

        # Use ego speed if available
        relative_velocity = max(self.ego_speed, 1.0)  # Assume at least 1 m/s
        return self.compute(detection.distance, relative_velocity)

    def update_ego_speed(self, speed_mps: float):
        """Update the ego vehicle speed in m/s."""
        self.ego_speed = max(0.0, speed_mps)
