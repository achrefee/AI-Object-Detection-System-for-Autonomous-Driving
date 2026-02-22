# 🚗 Autonomous Vision — AI Object Detection for Driving

Real-time object detection, distance estimation, and driving decision system using **Ultralytics YOLO**.

## 📁 Project Structure

```
autonomous_vision/
│
├── data/
│   ├── raw/              ← Original downloaded dataset (BDD100K)
│   │   ├── images/
│   │   └── labels/
│   ├── processed/        ← After filtering & cleaning
│   │   ├── images/
│   │   └── labels/
│   ├── train/            ← 70% — Training set
│   ├── val/              ← 20% — Validation set
│   └── test/             ← 10% — Test set
│
├── scripts/
│   ├── filter_classes.py    ← Step 4: Keep only needed classes
│   ├── clean_dataset.py     ← Step 5: Remove invalid data
│   ├── balance_dataset.py   ← Step 6: Fix class imbalance
│   └── split_dataset.py     ← Step 7: Train/Val/Test split
│
├── dataset.yaml          ← YOLO training config
└── README.md             ← This file
```

## 🎯 Target Classes (6)

| ID | Class |
|----|-------|
| 0 | Car |
| 1 | Truck |
| 2 | Motorcycle |
| 3 | Person |
| 4 | Traffic Light |
| 5 | Stop Sign |

## 🧩 Data Pipeline (Step by Step)

### Step 1 — Download BDD100K Dataset

1. Go to [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/)
2. Download **Images** and **Labels** (detection format)
3. Place them in:
   ```
   data/raw/images/
   data/raw/labels/
   ```

### Step 2 — Filter Classes

Keep only the 6 target classes and remap IDs:

```bash
cd autonomous_vision
python scripts/filter_classes.py --raw-dir data/raw --out-dir data/processed
```

### Step 3 — Clean Dataset

Remove empty labels, corrupted images, and orphan files:

```bash
python scripts/clean_dataset.py --data-dir data/processed
```

> 💡 Use `--dry-run` first to preview what will be removed.

### Step 4 — Balance Dataset

Check class distribution and oversample rare classes:

```bash
# Analyze only (no changes)
python scripts/balance_dataset.py --data-dir data/processed --analyze-only

# Balance to minimum 1000 objects per class
python scripts/balance_dataset.py --data-dir data/processed --min-objects 1000
```

### Step 5 — Split Dataset

Split into train (70%) / val (20%) / test (10%):

```bash
python scripts/split_dataset.py --src-dir data/processed --out-dir data --copy
```

### Step 6 — Train with YOLO

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(data="dataset.yaml", epochs=100, imgsz=640)
```

## 📦 Requirements

```bash
pip install ultralytics pillow
```

## 📄 License

PFE Project — AI Object Detection for Autonomous Driving
