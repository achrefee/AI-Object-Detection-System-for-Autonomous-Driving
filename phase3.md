Phase 3: Model Development
Create the full src/ codebase for the autonomous driving detection system as specified in the report (Section 9, 14, 15). This includes YOLO detection, object tracking, distance estimation, decision-making, and the real-time pipeline.

Proposed Changes
Project Setup
[NEW] 
requirements.txt
All Python dependencies: ultralytics, torch, opencv, timm (MiDaS), numpy, pyyaml, etc.

[NEW] configs/
model_config.yaml — model hyperparameters, confidence thresholds
camera_params.yaml — focal length, principal point, known object heights
decision_thresholds.yaml — risk zone distances, TTC thresholds
Detection Module (src/detection/)
[NEW] 
detector.py
ObjectDetector class wrapping Ultralytics YOLO
Methods: detect(frame), warmup(), configurable confidence/IoU thresholds
Returns list of Detection dataclass objects
[NEW] 
tracker.py
ObjectTracker class using Ultralytics built-in Bot-SORT/ByteTrack
Persistent track IDs, velocity estimation per track
Distance Estimation (src/distance/)
[NEW] 
geometric_estimator.py
Pinhole camera model: D = (H_real × f_y) / h_bbox
Known heights per class from report Section 5.3
[NEW] 
midas_estimator.py
MiDaS v3 monocular depth estimation
Relative depth → absolute distance scaling
[NEW] 
fusion.py
Weighted fusion of geometric + MiDaS estimates
Geometric for close range, MiDaS for far range
[NEW] 
kalman_filter.py
Per-track Kalman filter for temporal distance smoothing
Reduces jitter in distance estimates
Decision Module (src/decision/)
[NEW] 
risk_assessor.py
4 risk zones: CRITICAL (0–5m), DANGER (5–15m), WARNING (15–30m), SAFE (30m+)
Per-detection risk scoring
[NEW] 
ttc_calculator.py
Time-to-collision: TTC = Distance / Relative_Velocity
Handles approaching vs receding objects
[NEW] 
action_planner.py
Maps risk levels → driving actions (brake, steer, maintain)
Traffic light decision logic (red=stop, yellow=assess, green=go)
Priority-based action selection
Pipeline (src/pipeline/)
[NEW] 
realtime_pipeline.py
Main RealTimePipeline class orchestrating all modules
process_frame() returns full pipeline result
Support for video file or camera input
[NEW] 
video_capture.py
Multi-threaded video capture with ring buffer
Handles camera/file sources, auto-reconnect
[NEW] 
hud_overlay.py
Draw bounding boxes, class labels, distances, risk zones
Color-coded by risk level, FPS counter
Utilities (src/utils/)
[NEW] 
config.py
YAML config loader with dot-notation access
[NEW] 
logger.py
Structured logging with colored console output
[NEW] 
visualization.py
Debug visualization helpers (depth maps, detection grids)
Training & Calibration
[NEW] 
notebooks/kaggle_training.py
Ready-to-paste Kaggle notebook code
Phase 1: frozen backbone (10 epochs) → Phase 2: fine-tune (100 epochs)
Resume support, checkpoint saving
[NEW] 
scripts/camera_calibration.py
Checkerboard calibration to determine camera intrinsics
Verification Plan
Automated
Run python -c "from src.detection.detector import ObjectDetector" — import checks
Run python -c "from src.pipeline.realtime_pipeline import RealTimePipeline" — verify full import chain
Manual
User uploads dataset to Kaggle, runs training notebook
User tests pipeline on a sample video: python -m src.pipeline.realtime_pipeline --source video.mp4