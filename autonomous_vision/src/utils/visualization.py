"""
visualization.py — Debug visualization helpers.
"""

import cv2
import numpy as np


def draw_depth_map(depth_map: np.ndarray, colormap: int = cv2.COLORMAP_MAGMA) -> np.ndarray:
    """
    Convert a depth map to a colorized visualization.

    Args:
        depth_map: 2D array of depth values.
        colormap: OpenCV colormap constant.

    Returns:
        Colorized depth image (BGR, uint8).
    """
    # Normalize to 0-255
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, colormap)


def draw_detection_grid(
    frame: np.ndarray,
    grid_size: tuple[int, int] = (8, 8),
    color: tuple[int, int, int] = (50, 50, 50),
) -> np.ndarray:
    """
    Draw a grid overlay on the frame for debugging spatial awareness.

    Args:
        frame: Input image.
        grid_size: (rows, cols) of the grid.
        color: Grid line color (BGR).

    Returns:
        Frame with grid overlay.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    rows, cols = grid_size
    for i in range(1, rows):
        y = int(h * i / rows)
        cv2.line(overlay, (0, y), (w, y), color, 1)
    for j in range(1, cols):
        x = int(w * j / cols)
        cv2.line(overlay, (x, 0), (x, h), color, 1)

    return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)


def draw_risk_zones(
    frame: np.ndarray,
    zones: dict,
    alpha: float = 0.15,
) -> np.ndarray:
    """
    Draw horizontal risk zone bands on the frame.

    The bottom of the frame = near (critical), top = far (safe).

    Args:
        frame: Input image.
        zones: Dict with zone configs (from decision_thresholds.yaml).
        alpha: Overlay transparency.

    Returns:
        Frame with risk zone overlay.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Draw zones from bottom (near) to top (far)
    zone_order = ["critical", "danger", "warning", "safe"]
    zone_heights = [0.15, 0.20, 0.25, 0.40]  # Proportions of frame

    y_start = h
    for zone_name, zone_h in zip(zone_order, zone_heights):
        zone = zones.get(zone_name, {})
        color = tuple(zone.get("color", [128, 128, 128]))
        y_end = y_start - int(h * zone_h)
        y_end = max(y_end, 0)

        cv2.rectangle(overlay, (0, y_end), (w, y_start), color, -1)
        y_start = y_end

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def create_dashboard(
    frame: np.ndarray,
    detections: list,
    fps: float = 0.0,
    action: str = "MAINTAIN",
) -> np.ndarray:
    """
    Create a side dashboard panel showing detection stats.

    Args:
        frame: Main video frame.
        detections: List of Detection objects.
        fps: Current FPS.
        action: Current action string.

    Returns:
        Combined frame with dashboard panel.
    """
    h, w = frame.shape[:2]
    panel_w = 250
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark background

    # Title
    cv2.putText(panel, "AUTONOMOUS VISION", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    # FPS
    cv2.putText(panel, f"FPS: {fps:.1f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Detections count
    cv2.putText(panel, f"Objects: {len(detections)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Current action
    action_colors = {
        "MAINTAIN": (0, 255, 0),
        "SLOW_DOWN": (0, 255, 255),
        "HARD_BRAKE": (0, 100, 255),
        "EMERGENCY_BRAKE": (0, 0, 255),
        "LANE_CHANGE": (255, 200, 0),
        "STOP": (0, 0, 255),
    }
    color = action_colors.get(action, (255, 255, 255))
    cv2.putText(panel, f"Action: {action}", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Separator
    cv2.line(panel, (10, 145), (panel_w - 10, 145), (80, 80, 80), 1)

    # Detection list
    y = 170
    for det in detections[:10]:  # Show max 10
        name = getattr(det, "class_name", "unknown")
        dist = getattr(det, "distance", 0.0)
        conf = getattr(det, "confidence", 0.0)
        text = f"{name}: {dist:.1f}m ({conf:.0%})"
        cv2.putText(panel, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y += 20

    return np.hstack([frame, panel])
