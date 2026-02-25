"""
============================================================
KAGGLE NOTEBOOK — YOLOv8 Training for Autonomous Driving
============================================================
Platform:  Kaggle Notebooks (free GPU)
GPU:       NVIDIA Tesla T4 × 2 or P100
Dataset:   BDD100K (11 classes, YOLO format)
Model:     YOLOv8s (pretrained COCO → fine-tuned)

HOW TO USE:
  1. Upload your YOLO-format dataset as a Kaggle Dataset
  2. Create a new Kaggle Notebook
  3. Settings → Accelerator → GPU T4 x2 (or P100)
  4. Settings → Internet → ON
  5. Copy-paste this script into a code cell and run
============================================================
"""

# ── Step 0: Install / Upgrade ─────────────────────────────────
# !pip install -q ultralytics --upgrade

import os
import torch
from pathlib import Path
from ultralytics import YOLO


# ── Step 1: Verify GPU ────────────────────────────────────────
print("=" * 60)
print("GPU Check")
print("=" * 60)
print(f"  PyTorch version : {torch.__version__}")
print(f"  CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU name        : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU memory      : {props.total_mem / 1e9:.1f} GB")
print("=" * 60)


# ── Step 2: Configuration ────────────────────────────────────
# Update this path to match your Kaggle Dataset name
DATASET_PATH = "/kaggle/input/driving-object-detection"
OUTPUT_PATH = "/kaggle/working"
DATA_YAML = f"{DATASET_PATH}/dataset.yaml"

# Verify dataset exists
assert Path(DATA_YAML).exists(), f"dataset.yaml not found at {DATA_YAML}"

# Count images
for split in ["train", "val", "test"]:
    img_dir = Path(DATASET_PATH) / split / "images"
    if img_dir.exists():
        count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"  {split}: {count} images")


# ── Step 3: Phase 1 — Transfer Learning (Frozen Backbone) ────
print("\n" + "=" * 60)
print("PHASE 1: Transfer Learning (Frozen Backbone)")
print("=" * 60)

model = YOLO("yolov8s.pt")  # Auto-downloads pretrained COCO weights

# Freeze the backbone (first 10 layers)
model.model.model[:10].requires_grad_(False)

results_p1 = model.train(
    data=DATA_YAML,
    epochs=10,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,

    # Hardware
    device=0,
    workers=2,

    # Saving
    project=f"{OUTPUT_PATH}/runs/train",
    name="phase1_transfer_learning",
    save=True,
    save_period=5,
    plots=True,
)

phase1_best = f"{OUTPUT_PATH}/runs/train/phase1_transfer_learning/weights/best.pt"
print(f"\nPhase 1 best weights: {phase1_best}")


# ── Step 4: Phase 2 — Fine-Tuning (All Layers) ───────────────
print("\n" + "=" * 60)
print("PHASE 2: Fine-Tuning (All Layers Unfrozen)")
print("=" * 60)

# Load Phase 1 best weights and unfreeze all layers
model = YOLO(phase1_best)

results_p2 = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,             # Lower learning rate for fine-tuning
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=1,
    patience=20,           # Early stopping

    # Same augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,

    # Hardware
    device=0,
    workers=2,

    # Saving — checkpoint every 10 epochs for session recovery
    project=f"{OUTPUT_PATH}/runs/train",
    name="phase2_fine_tuning",
    save=True,
    save_period=10,
    plots=True,
)

phase2_best = f"{OUTPUT_PATH}/runs/train/phase2_fine_tuning/weights/best.pt"
print(f"\nPhase 2 best weights: {phase2_best}")


# ── Step 5: Evaluate on Test Set ──────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

model = YOLO(phase2_best)
metrics = model.val(
    data=DATA_YAML,
    split="test",
    device=0,
    plots=True,
    project=f"{OUTPUT_PATH}/runs/val",
    name="test_evaluation",
)

print(f"\n  mAP@0.5        : {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
print(f"  Precision       : {metrics.box.mp:.4f}")
print(f"  Recall          : {metrics.box.mr:.4f}")


# ── Step 6: Export Model ──────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL EXPORT")
print("=" * 60)

# Export to ONNX
model.export(format="onnx", imgsz=640, simplify=True)
print(f"  ONNX exported ✅")

# The best model is at:
print(f"\n  📦 Best weights: {phase2_best}")
print(f"  📦 ONNX model:   {phase2_best.replace('.pt', '.onnx')}")
print("\n  → Click 'Save Version' → 'Save & Run All' to persist outputs")


# ── Bonus: Resume Training (if session expired) ──────────────
# Uncomment below if you need to resume a previous training session:
#
# model = YOLO("/kaggle/input/previous-run/last.pt")
# results = model.train(resume=True)
