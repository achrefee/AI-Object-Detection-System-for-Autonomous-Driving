"""
action_planner.py — Driving action decisions based on risk assessment.

Maps risk levels and traffic signals to driving actions:
    - EMERGENCY_BRAKE: Critical zone or TTC < 1.5s
    - HARD_BRAKE: Danger zone
    - SLOW_DOWN: Warning zone
    - MAINTAIN: Safe zone
    - STOP: Red traffic light
    - LANE_CHANGE: If safe lane available
"""

from enum import Enum
from dataclasses import dataclass

from .risk_assessor import RiskAssessor, RiskLevel, RiskAssessment
from .ttc_calculator import TTCCalculator


class ActionType(Enum):
    """Available driving actions."""
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
    HARD_BRAKE = "HARD_BRAKE"
    SLOW_DOWN = "SLOW_DOWN"
    MAINTAIN = "MAINTAIN"
    STOP = "STOP"
    LANE_CHANGE = "LANE_CHANGE"
    PROCEED_CAUTION = "PROCEED_CAUTION"

    @property
    def priority(self) -> int:
        """Higher = more urgent."""
        return {
            ActionType.EMERGENCY_BRAKE: 7,
            ActionType.STOP: 6,
            ActionType.HARD_BRAKE: 5,
            ActionType.LANE_CHANGE: 4,
            ActionType.SLOW_DOWN: 3,
            ActionType.PROCEED_CAUTION: 2,
            ActionType.MAINTAIN: 1,
        }[self]


@dataclass
class Action:
    """A planned driving action."""
    type: ActionType
    intensity: float = 0.0      # 0.0–1.0 (e.g., brake force)
    direction: str = "straight"  # "straight", "left", "right"
    priority: int = 0
    reason: str = ""            # Why this action was chosen
    source_detection: str = ""  # Class name that triggered it

    def __post_init__(self):
        self.priority = self.type.priority


class ActionPlanner:
    """
    Plan driving actions based on detections, risk assessments, and TTC.

    Usage:
        planner = ActionPlanner()
        action = planner.plan(detections)
        print(action.type, action.reason)
    """

    def __init__(
        self,
        risk_assessor: RiskAssessor | None = None,
        ttc_calculator: TTCCalculator | None = None,
    ):
        self.risk_assessor = risk_assessor or RiskAssessor()
        self.ttc_calculator = ttc_calculator or TTCCalculator()

    def plan(self, detections: list) -> Action:
        """
        Plan the best action given all current detections.

        Args:
            detections: List of Detection objects with distance populated.

        Returns:
            The highest-priority Action to take.
        """
        if not detections:
            return Action(
                type=ActionType.MAINTAIN,
                reason="No objects detected",
            )

        actions = []

        for det in detections:
            # Check traffic lights first
            tl_action = self._check_traffic_light(det)
            if tl_action:
                actions.append(tl_action)
                continue

            # Assess risk for non-traffic-light objects
            if det.distance <= 0:
                continue

            risk = self.risk_assessor.assess(det)
            action = self._risk_to_action(risk, det)
            actions.append(action)

        if not actions:
            return Action(type=ActionType.MAINTAIN, reason="No actionable detections")

        # Return highest priority action
        return self.get_priority_action(actions)

    def get_priority_action(self, actions: list[Action]) -> Action:
        """Get the most urgent action from a list."""
        if not actions:
            return Action(type=ActionType.MAINTAIN)
        return max(actions, key=lambda a: a.priority)

    def _check_traffic_light(self, detection) -> Action | None:
        """Handle traffic light detection logic."""
        class_name = detection.class_name

        if class_name == "traffic_light_red":
            return Action(
                type=ActionType.STOP,
                intensity=1.0,
                reason=f"Red traffic light at {detection.distance:.1f}m",
                source_detection=class_name,
            )
        elif class_name == "traffic_light_yellow":
            # If close, proceed cautiously; if far, slow down
            if detection.distance > 0 and detection.distance < 10:
                return Action(
                    type=ActionType.PROCEED_CAUTION,
                    intensity=0.5,
                    reason=f"Yellow light close ({detection.distance:.1f}m) — proceed",
                    source_detection=class_name,
                )
            else:
                return Action(
                    type=ActionType.SLOW_DOWN,
                    intensity=0.6,
                    reason=f"Yellow light at {detection.distance:.1f}m — slowing",
                    source_detection=class_name,
                )
        elif class_name == "traffic_light_green":
            return Action(
                type=ActionType.MAINTAIN,
                reason="Green light — proceed",
                source_detection=class_name,
            )

        return None

    def _risk_to_action(self, risk: RiskAssessment, detection) -> Action:
        """Convert a risk assessment to an action."""
        if risk.risk_level == RiskLevel.CRITICAL:
            return Action(
                type=ActionType.EMERGENCY_BRAKE,
                intensity=1.0,
                reason=f"{risk.class_name} at {risk.distance:.1f}m — CRITICAL",
                source_detection=risk.class_name,
            )
        elif risk.risk_level == RiskLevel.DANGER:
            # Check TTC for more nuanced response
            ttc = self.ttc_calculator.compute_for_detection(detection)
            if ttc < self.ttc_calculator.emergency_threshold:
                return Action(
                    type=ActionType.EMERGENCY_BRAKE,
                    intensity=1.0,
                    reason=f"{risk.class_name} at {risk.distance:.1f}m, TTC={ttc:.1f}s",
                    source_detection=risk.class_name,
                )
            return Action(
                type=ActionType.HARD_BRAKE,
                intensity=0.7,
                reason=f"{risk.class_name} at {risk.distance:.1f}m — DANGER",
                source_detection=risk.class_name,
            )
        elif risk.risk_level == RiskLevel.WARNING:
            return Action(
                type=ActionType.SLOW_DOWN,
                intensity=0.4,
                reason=f"{risk.class_name} at {risk.distance:.1f}m — WARNING",
                source_detection=risk.class_name,
            )
        else:
            return Action(
                type=ActionType.MAINTAIN,
                reason=f"{risk.class_name} at {risk.distance:.1f}m — SAFE",
                source_detection=risk.class_name,
            )
