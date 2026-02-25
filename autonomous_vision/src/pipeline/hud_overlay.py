"""
hud_overlay.py — Heads-Up Display overlay for the driving view.

Draws bounding boxes, class labels, distances, risk indicators,
FPS counter, and action alerts on the video frame.
"""

import cv2
import numpy as np


class HUDOverlay:
    """
    Draw a HUD overlay on detection frames.

    Usage:
        hud = HUDOverlay()
        frame = hud.draw(frame, detections, action, fps=30)
    """

    # Risk level → color (BGR)
    RISK_COLORS = {
        "CRITICAL": (0, 0, 255),       # Red
        "DANGER":   (0, 100, 255),     # Orange
        "WARNING":  (0, 255, 255),     # Yellow
        "SAFE":     (0, 255, 0),       # Green
        "UNKNOWN":  (128, 128, 128),   # Gray
    }

    # Action → color (BGR)
    ACTION_COLORS = {
        "EMERGENCY_BRAKE": (0, 0, 255),
        "STOP":            (0, 0, 200),
        "HARD_BRAKE":      (0, 100, 255),
        "SLOW_DOWN":       (0, 255, 255),
        "MAINTAIN":        (0, 255, 0),
        "LANE_CHANGE":     (255, 200, 0),
        "PROCEED_CAUTION": (0, 200, 255),
    }

    def __init__(
        self,
        font_scale: float = 0.5,
        thickness: int = 2,
        show_distance: bool = True,
        show_confidence: bool = True,
        show_track_id: bool = True,
        show_action_bar: bool = True,
    ):
        self.font_scale = font_scale
        self.thickness = thickness
        self.show_distance = show_distance
        self.show_confidence = show_confidence
        self.show_track_id = show_track_id
        self.show_action_bar = show_action_bar
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        detections: list,
        action=None,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draw the full HUD on a frame.

        Args:
            frame: BGR input image.
            detections: List of Detection objects.
            action: Current Action object.
            fps: Current FPS.

        Returns:
            Frame with HUD overlays.
        """
        output = frame.copy()

        # Draw detections
        for det in detections:
            self._draw_detection(output, det)

        # Draw FPS counter
        self._draw_fps(output, fps)

        # Draw action bar
        if self.show_action_bar and action is not None:
            self._draw_action_bar(output, action)

        # Draw detection count
        self._draw_stats(output, len(detections))

        return output

    def _draw_detection(self, frame: np.ndarray, det):
        """Draw a single detection with bbox, label, and distance."""
        x1, y1, x2, y2 = det.bbox
        risk = getattr(det, "risk_level", "UNKNOWN")
        color = self.RISK_COLORS.get(risk, (128, 128, 128))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        # Build label text
        parts = [det.class_name]
        if self.show_track_id and det.track_id >= 0:
            parts.append(f"#{det.track_id}")
        if self.show_confidence:
            parts.append(f"{det.confidence:.0%}")
        if self.show_distance and det.distance > 0:
            parts.append(f"{det.distance:.1f}m")

        label = " | ".join(parts)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
        label_y = max(y1 - 8, th + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + 2),
            color,
            -1,
        )

        # Label text
        text_color = (0, 0, 0) if risk in ("WARNING", "SAFE") else (255, 255, 255)
        cv2.putText(
            frame, label, (x1 + 2, label_y - 2),
            self.font, self.font_scale, text_color, 1, cv2.LINE_AA,
        )

        # Distance line (from bbox bottom-center downward)
        if self.show_distance and det.distance > 0:
            bcx = (x1 + x2) // 2
            cv2.circle(frame, (bcx, y2), 3, color, -1)

    def _draw_fps(self, frame: np.ndarray, fps: float):
        """Draw FPS counter in top-right corner."""
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.0f}"
        color = (0, 255, 0) if fps >= 25 else (0, 255, 255) if fps >= 15 else (0, 0, 255)

        (tw, th), _ = cv2.getTextSize(text, self.font, 0.6, 2)
        x = w - tw - 15
        cv2.rectangle(frame, (x - 5, 5), (w - 5, th + 15), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, th + 10), self.font, 0.6, color, 2, cv2.LINE_AA)

    def _draw_action_bar(self, frame: np.ndarray, action):
        """Draw the action indicator bar at the bottom of the frame."""
        h, w = frame.shape[:2]
        bar_height = 45

        action_type = action.type.value if hasattr(action.type, "value") else str(action.type)
        color = self.ACTION_COLORS.get(action_type, (128, 128, 128))

        # Semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Action text
        action_text = action_type.replace("_", " ")
        reason = getattr(action, "reason", "")
        text = f"  ▶ {action_text}"
        if reason:
            text += f"  —  {reason}"

        text_color = (0, 0, 0) if action_type in ("SLOW_DOWN", "MAINTAIN") else (255, 255, 255)
        cv2.putText(
            frame, text, (10, h - 15),
            self.font, 0.55, text_color, 1, cv2.LINE_AA,
        )

    def _draw_stats(self, frame: np.ndarray, count: int):
        """Draw detection count in top-left."""
        text = f"Objects: {count}"
        cv2.rectangle(frame, (5, 5), (130, 30), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 23), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
