"""
realtime_pipeline.py — Main real-time detection pipeline.

Orchestrates: Capture → Detection → Tracking → Distance → Risk → Action → HUD

Can be run as a standalone script:
    python -m src.pipeline.realtime_pipeline --source video.mp4
"""

import time
import argparse
from pathlib import Path

import cv2
import numpy as np

from ..detection.detector import ObjectDetector
from ..detection.tracker import ObjectTracker
from ..distance.geometric_estimator import GeometricEstimator
from ..distance.fusion import DistanceFusion
from ..decision.risk_assessor import RiskAssessor
from ..decision.ttc_calculator import TTCCalculator
from ..decision.action_planner import ActionPlanner, ActionType
from ..utils.config import load_config
from ..utils.logger import setup_logger
from .video_capture import VideoCapture
from .hud_overlay import HUDOverlay


logger = setup_logger("pipeline")


class RealTimePipeline:
    """
    End-to-end real-time detection pipeline.

    Usage:
        pipeline = RealTimePipeline(model_path="weights/best.pt")
        pipeline.run(source="video.mp4")
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        config_dir: str = "configs",
        device: str = "cuda",
        confidence: float = 0.35,
        use_midas: bool = False,
        show_display: bool = True,
    ):
        """
        Args:
            model_path: Path to YOLO weights.
            config_dir: Path to configs/ directory.
            device: "cuda" or "cpu".
            confidence: Detection confidence threshold.
            use_midas: Whether to use MiDaS depth (slower but more accurate).
            show_display: Whether to show the OpenCV window.
        """
        logger.info(f"Initializing pipeline (device={device})")

        # Load configuration
        config_path = Path(config_dir)
        if config_path.exists():
            self.config = load_config(config_dir)
            logger.info("Loaded configuration files")
        else:
            self.config = None
            logger.warning("No configs/ directory found, using defaults")

        # Initialize modules
        self.detector = ObjectDetector(
            model_path=model_path,
            device=device,
            confidence=confidence,
        )
        logger.info(f"Loaded model: {model_path}")

        self.tracker = ObjectTracker()

        # Distance estimation
        if self.config and hasattr(self.config, "camera"):
            self.geometric = GeometricEstimator.from_config(self.config)
        else:
            self.geometric = GeometricEstimator()

        # MiDaS (optional — significantly slower)
        midas = None
        if use_midas:
            try:
                from ..distance.midas_estimator import MiDaSEstimator
                midas = MiDaSEstimator(model_type="small", device=device)
                logger.info("MiDaS depth estimation enabled")
            except Exception as e:
                logger.warning(f"MiDaS unavailable: {e}")

        self.distance_fusion = DistanceFusion(
            geometric=self.geometric,
            midas=midas,
        )

        # Decision making
        self.risk_assessor = RiskAssessor()
        self.ttc_calculator = TTCCalculator()
        self.action_planner = ActionPlanner(
            risk_assessor=self.risk_assessor,
            ttc_calculator=self.ttc_calculator,
        )

        # Display
        self.hud = HUDOverlay()
        self.show_display = show_display

        # Performance tracking
        self._fps = 0.0
        self._frame_times = []

        logger.info("Pipeline initialized successfully")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR image.

        Returns:
            Dict with keys: detections, action, fps, annotated_frame
        """
        t0 = time.time()

        # 1. Detection + Tracking
        detections = self.detector.detect_and_track(frame)

        # 2. Update tracker (velocity estimation)
        detections = self.tracker.update(detections)

        # 3. Distance estimation
        detections = self.distance_fusion.estimate_batch(detections, frame)

        # 4. Risk assessment (update each detection's risk_level)
        for det in detections:
            risk = self.risk_assessor.assess(det)
            det.risk_level = risk.risk_level.value

        # 5. Action planning
        action = self.action_planner.plan(detections)

        # 6. FPS calculation
        elapsed = time.time() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg_time = sum(self._frame_times) / len(self._frame_times)
        self._fps = 1.0 / avg_time if avg_time > 0 else 0.0

        # 7. Draw HUD
        annotated_frame = self.hud.draw(frame, detections, action, self._fps)

        return {
            "detections": detections,
            "action": action,
            "fps": self._fps,
            "annotated_frame": annotated_frame,
        }

    def run(
        self,
        source: str | int = 0,
        output_path: str | None = None,
        max_frames: int | None = None,
    ):
        """
        Run the pipeline on a video source.

        Args:
            source: Video file path or camera index (0).
            output_path: Optional path to save output video.
            max_frames: Max frames to process (None = all).
        """
        logger.info(f"Starting pipeline on source: {source}")

        # Warm up model
        self.detector.warmup()

        # Open video source
        with VideoCapture(source) as cap:
            logger.info(
                f"Video: {cap.width}x{cap.height} @ {cap.fps:.1f} FPS"
                f" ({cap.total_frames} frames)"
            )

            # Video writer
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_w = cap.width
                out_h = cap.height
                writer = cv2.VideoWriter(output_path, fourcc, cap.fps, (out_w, out_h))
                logger.info(f"Saving output to: {output_path}")

            frame_idx = 0

            while cap.is_running:
                frame = cap.read()
                if frame is None:
                    time.sleep(0.001)
                    continue

                # Process frame
                result = self.process_frame(frame)
                annotated = result["annotated_frame"]
                action = result["action"]
                frame_idx += 1

                # Log important actions
                if action.type not in (ActionType.MAINTAIN,):
                    logger.info(
                        f"Frame {frame_idx}: {action.type.value} — {action.reason}"
                    )

                # Display
                if self.show_display:
                    cv2.imshow("Autonomous Vision", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # q or ESC
                        logger.info("User requested stop")
                        break

                # Write output
                if writer:
                    writer.write(annotated)

                # Frame limit
                if max_frames and frame_idx >= max_frames:
                    logger.info(f"Reached max frames: {max_frames}")
                    break

        # Cleanup
        if writer:
            writer.release()
        if self.show_display:
            cv2.destroyAllWindows()

        logger.info(
            f"Pipeline finished. Processed {frame_idx} frames "
            f"at {self._fps:.1f} avg FPS"
        )

    def stop(self):
        """Release resources."""
        if self.show_display:
            cv2.destroyAllWindows()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Vision — Real-Time Object Detection Pipeline"
    )
    parser.add_argument(
        "--source", "-s",
        default="0",
        help="Video file path or camera index (default: 0 for webcam)",
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8s.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs",
        help="Path to configs/ directory",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output video file path",
    )
    parser.add_argument(
        "--midas",
        action="store_true",
        help="Enable MiDaS depth estimation (slower)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process",
    )

    args = parser.parse_args()

    # Parse source (int for camera, string for file)
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    pipeline = RealTimePipeline(
        model_path=args.model,
        config_dir=args.config,
        device=args.device,
        confidence=args.confidence,
        use_midas=args.midas,
        show_display=not args.no_display,
    )

    pipeline.run(
        source=source,
        output_path=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
