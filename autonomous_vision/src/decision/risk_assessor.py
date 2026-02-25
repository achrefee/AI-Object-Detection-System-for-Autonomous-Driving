"""
risk_assessor.py — Zone-based risk assessment for detected objects.

Classifies objects into risk zones based on their distance:
    CRITICAL: 0–5m   → Emergency brake
    DANGER:   5–15m  → Hard brake
    WARNING:  15–30m → Slow down
    SAFE:     30m+   → Maintain speed
"""

from enum import Enum
from dataclasses import dataclass


class RiskLevel(Enum):
    """Risk classification levels."""
    CRITICAL = "CRITICAL"
    DANGER = "DANGER"
    WARNING = "WARNING"
    SAFE = "SAFE"
    UNKNOWN = "UNKNOWN"

    @property
    def priority(self) -> int:
        """Higher number = more urgent."""
        return {
            RiskLevel.CRITICAL: 4,
            RiskLevel.DANGER: 3,
            RiskLevel.WARNING: 2,
            RiskLevel.SAFE: 1,
            RiskLevel.UNKNOWN: 0,
        }[self]

    @property
    def color_bgr(self) -> tuple[int, int, int]:
        """Color for visualization (BGR)."""
        return {
            RiskLevel.CRITICAL: (0, 0, 255),      # Red
            RiskLevel.DANGER: (0, 100, 255),       # Orange
            RiskLevel.WARNING: (0, 255, 255),      # Yellow
            RiskLevel.SAFE: (0, 255, 0),           # Green
            RiskLevel.UNKNOWN: (128, 128, 128),    # Gray
        }[self]


# Default zone boundaries (meters)
DEFAULT_ZONES = {
    RiskLevel.CRITICAL: (0.0, 5.0),
    RiskLevel.DANGER: (5.0, 15.0),
    RiskLevel.WARNING: (15.0, 30.0),
    RiskLevel.SAFE: (30.0, 999.0),
}

# Object class priority weights (higher = more dangerous)
DEFAULT_PRIORITIES = {
    "pedestrian": 1.0,
    "cyclist": 0.95,
    "motorcycle": 0.8,
    "bicycle": 0.85,
    "car": 0.7,
    "truck": 0.75,
    "bus": 0.75,
    "traffic_light_red": 0.9,
    "traffic_light_green": 0.3,
    "traffic_light_yellow": 0.7,
    "traffic_sign": 0.5,
}


@dataclass
class RiskAssessment:
    """Risk assessment result for a single detection."""
    risk_level: RiskLevel
    distance: float
    priority_score: float    # Combined risk score (0–1)
    class_name: str


class RiskAssessor:
    """
    Assess risk level for each detection based on distance and class.

    Usage:
        assessor = RiskAssessor()
        risk = assessor.assess(detection)
        print(risk.risk_level, risk.priority_score)
    """

    def __init__(
        self,
        zones: dict[RiskLevel, tuple[float, float]] | None = None,
        priorities: dict[str, float] | None = None,
    ):
        self.zones = zones or DEFAULT_ZONES.copy()
        self.priorities = priorities or DEFAULT_PRIORITIES.copy()

    def assess(self, detection) -> RiskAssessment:
        """
        Assess the risk level of a single detection.

        Args:
            detection: Detection object with distance and class_name.

        Returns:
            RiskAssessment with level and scores.
        """
        distance = detection.distance

        if distance <= 0:
            return RiskAssessment(
                risk_level=RiskLevel.UNKNOWN,
                distance=distance,
                priority_score=0.0,
                class_name=detection.class_name,
            )

        # Determine zone
        risk_level = RiskLevel.SAFE
        for level, (min_d, max_d) in self.zones.items():
            if min_d <= distance < max_d:
                risk_level = level
                break

        # Compute priority score
        class_weight = self.priorities.get(detection.class_name, 0.5)
        zone_weight = risk_level.priority / 4.0  # Normalize to 0–1
        priority_score = class_weight * zone_weight

        return RiskAssessment(
            risk_level=risk_level,
            distance=distance,
            priority_score=priority_score,
            class_name=detection.class_name,
        )

    def assess_batch(self, detections: list) -> list[RiskAssessment]:
        """Assess risk for all detections."""
        return [self.assess(det) for det in detections]

    def get_highest_risk(self, assessments: list[RiskAssessment]) -> RiskAssessment | None:
        """
        Get the most critical risk assessment from a list.

        Returns:
            The RiskAssessment with the highest priority, or None.
        """
        if not assessments:
            return None

        return max(assessments, key=lambda a: a.priority_score)
