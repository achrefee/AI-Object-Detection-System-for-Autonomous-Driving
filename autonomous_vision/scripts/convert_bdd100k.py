"""
convert_bdd100k.py — Convert BDD100K labels to YOLO format.

Reads BDD100K JSON label files and images, converts bounding boxes
to YOLO format, and maps classes to the project's 11-class target set.

Expected BDD100K directory structure (after extracting ZIPs):
    data/bdd100k/
    ├── images/
    │   └── 100k/
    │       ├── train/
    │       └── val/
    └── labels/
        ├── det_20/
        │   ├── det_train.json
        │   └── det_val.json
        └── (or older format: bdd100k_labels_images_train.json, etc.)

Output goes to data/raw/images/ and data/raw/labels/.

Usage:
    python scripts/convert_bdd100k.py
    python scripts/convert_bdd100k.py --max-images 500    # Quick test
    python scripts/convert_bdd100k.py --split train       # Train only
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from collections import Counter


# ── 11-class target set ───────────────────────────────────────────────
TARGET_CLASSES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorcycle",
    4: "bicycle",
    5: "pedestrian",
    6: "cyclist",
    7: "traffic_light_red",
    8: "traffic_light_green",
    9: "traffic_light_yellow",
    10: "traffic_sign",
}

# BDD100K category name → our target class ID
# Traffic lights are handled specially via attributes
BDD100K_CATEGORY_MAP = {
    "car":            0,
    "truck":          1,
    "bus":            2,
    "motorcycle":     3,
    "bicycle":        4,
    "pedestrian":     5,
    "rider":          6,   # cyclist
    "traffic light":  None,  # handled via trafficLightColor attribute
    "traffic sign":   10,
    # "train" → skipped (rail trains, not relevant)
}


def find_bdd100k_files(bdd_dir: Path):
    """
    Auto-detect BDD100K file structure.
    Returns dict: {split_name: (images_dir, labels_json_path)}
    """
    found = {}

    # ── Try det_20 format first (newer) ───────────────────────────────
    det20_dir = bdd_dir / "labels" / "det_20"
    if det20_dir.exists():
        for split, json_name in [("train", "det_train.json"), ("val", "det_val.json")]:
            json_path = det20_dir / json_name
            img_dir = bdd_dir / "images" / "100k" / split
            if json_path.exists():
                found[split] = (img_dir, json_path)

    # ── Try older label format ────────────────────────────────────────
    if not found:
        labels_dir = bdd_dir / "labels"
        for split in ["train", "val"]:
            for pattern in [
                f"bdd100k_labels_images_{split}.json",
                f"det_{split}.json",
            ]:
                json_path = labels_dir / pattern
                if json_path.exists():
                    img_dir = bdd_dir / "images" / "100k" / split
                    found[split] = (img_dir, json_path)
                    break

    # ── Try flat image directory ──────────────────────────────────────
    if found:
        for split in list(found.keys()):
            img_dir, json_path = found[split]
            if not img_dir.exists():
                # Try without 100k subdirectory
                alt_img_dir = bdd_dir / "images" / split
                if alt_img_dir.exists():
                    found[split] = (alt_img_dir, json_path)

    return found


def parse_det20_format(json_path: Path):
    """
    Parse BDD100K det_20 format.
    Returns list of (image_name, labels_list) tuples.

    det_20 format:
    [{
        "name": "image.jpg",
        "labels": [{
            "id": "...",
            "category": "car",
            "box2d": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
        }]
    }]
    """
    print(f"   Loading {json_path.name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for frame in data:
        image_name = frame.get("name", "")
        labels = frame.get("labels", [])
        if labels is None:
            labels = []
        results.append((image_name, labels))

    return results


def parse_old_format(json_path: Path):
    """
    Parse older BDD100K label format.
    Returns list of (image_name, labels_list) tuples.

    Old format:
    [{
        "name": "image.jpg",
        "frames": [{
            "objects": [{
                "category": "car",
                "box2d": {"x1": ..., "y1": ..., "x2": ..., "y2": ...},
                "attributes": {"trafficLightColor": "red"}
            }]
        }]
    }]
    """
    print(f"   Loading {json_path.name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        image_name = item.get("name", "")
        labels = []

        # Old format nests objects inside frames
        frames = item.get("frames", [])
        if frames:
            objects = frames[0].get("objects", [])
            for obj in objects:
                label = {
                    "category": obj.get("category", ""),
                    "box2d": obj.get("box2d"),
                }
                attrs = obj.get("attributes", {})
                if attrs:
                    label["attributes"] = attrs
                labels.append(label)
        else:
            # Some versions put labels directly
            labels = item.get("labels", [])
            if labels is None:
                labels = []

        results.append((image_name, labels))

    return results


def get_class_id(label: dict) -> int | None:
    """
    Map a BDD100K label to our target class ID.
    Handles traffic light color via attributes.
    """
    category = label.get("category", "").lower().strip()

    # Handle traffic lights specially
    if category == "traffic light":
        attrs = label.get("attributes", {})
        if attrs is None:
            attrs = {}
        color = attrs.get("trafficLightColor", "").lower().strip()

        if color == "red" or color == "r":
            return 7   # traffic_light_red
        elif color == "green" or color == "g":
            return 8   # traffic_light_green
        elif color == "yellow" or color == "y":
            return 9   # traffic_light_yellow
        else:
            # Unknown color or "none" — skip this label
            return None

    return BDD100K_CATEGORY_MAP.get(category)


def convert_bbox_to_yolo(box2d: dict, img_width: int, img_height: int) -> tuple | None:
    """
    Convert BDD100K box2d (x1, y1, x2, y2 in pixels) to YOLO format
    (x_center, y_center, width, height) normalized to [0, 1].
    """
    if box2d is None:
        return None

    x1 = float(box2d.get("x1", 0))
    y1 = float(box2d.get("y1", 0))
    x2 = float(box2d.get("x2", 0))
    y2 = float(box2d.get("y2", 0))

    # Clip to image bounds
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    # Skip invalid boxes
    if x2 <= x1 or y2 <= y1:
        return None

    # Convert to YOLO format
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # Skip tiny boxes (< 0.1% of image)
    if width < 0.001 or height < 0.001:
        return None

    return (
        round(x_center, 6),
        round(y_center, 6),
        round(width, 6),
        round(height, 6),
    )


def convert_split(
    split_name: str,
    images_dir: Path,
    json_path: Path,
    raw_dir: Path,
    max_images: int | None,
    img_width: int = 1280,
    img_height: int = 720,
) -> Counter:
    """Convert one split of BDD100K data to YOLO format."""

    print(f"\n{'='*60}")
    print(f"📥 Converting BDD100K — {split_name} split")
    print(f"{'='*60}")

    # Auto-detect format
    with open(json_path, "r", encoding="utf-8") as f:
        # Peek at the first item to detect format
        data_peek = json.load(f)

    if data_peek and isinstance(data_peek, list):
        first_item = data_peek[0]
        if "frames" in first_item:
            frames = parse_old_format(json_path)
        else:
            frames = parse_det20_format(json_path)
    else:
        print(f"   ❌ Unknown label format in {json_path}")
        return Counter()

    if max_images:
        frames = frames[:max_images]

    print(f"   Found {len(frames)} labeled images")

    raw_images = raw_dir / "images"
    raw_labels = raw_dir / "labels"
    raw_images.mkdir(parents=True, exist_ok=True)
    raw_labels.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    converted = 0
    skipped_no_image = 0
    skipped_no_labels = 0

    for image_name, labels in frames:
        if not image_name:
            continue

        # Find source image
        src_image = images_dir / image_name
        if not src_image.exists():
            skipped_no_image += 1
            continue

        # Convert labels for this image
        yolo_lines = []
        for label in labels:
            class_id = get_class_id(label)
            if class_id is None:
                continue

            box2d = label.get("box2d")
            yolo_bbox = convert_bbox_to_yolo(box2d, img_width, img_height)
            if yolo_bbox is None:
                continue

            cx, cy, w, h = yolo_bbox
            yolo_lines.append(f"{class_id} {cx} {cy} {w} {h}")
            stats[TARGET_CLASSES[class_id]] += 1

        # Skip images with no valid detections
        if not yolo_lines:
            skipped_no_labels += 1
            continue

        # Save label file
        stem = Path(image_name).stem
        label_path = raw_labels / f"{stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + "\n")

        # Copy image
        dst_image = raw_images / image_name
        if not dst_image.exists():
            shutil.copy2(src_image, dst_image)

        converted += 1

    print(f"\n   ✅ Converted: {converted} images")
    if skipped_no_image:
        print(f"   ⚠️  Skipped (image not found): {skipped_no_image}")
    if skipped_no_labels:
        print(f"   ⏭️  Skipped (no valid labels): {skipped_no_labels}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert BDD100K labels to YOLO format for autonomous driving."
    )
    parser.add_argument(
        "--bdd-dir",
        type=Path,
        default=Path("data/bdd100k"),
        help="Path to extracted BDD100K data (default: data/bdd100k)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for YOLO data (default: data/raw)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max images to convert per split (default: all)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which split to convert (default: both)",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=1280,
        help="BDD100K image width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=720,
        help="BDD100K image height in pixels (default: 720)",
    )
    args = parser.parse_args()

    # ── Detect BDD100K files ──────────────────────────────────────────
    print("🔍 Searching for BDD100K files...")
    found_splits = find_bdd100k_files(args.bdd_dir)

    if not found_splits:
        print(f"\n❌ No BDD100K data found in {args.bdd_dir}")
        print(f"\nExpected structure:")
        print(f"   {args.bdd_dir}/")
        print(f"   ├── images/")
        print(f"   │   └── 100k/")
        print(f"   │       ├── train/   (*.jpg)")
        print(f"   │       └── val/     (*.jpg)")
        print(f"   └── labels/")
        print(f"       └── det_20/")
        print(f"           ├── det_train.json")
        print(f"           └── det_val.json")
        print(f"\nPlease extract the BDD100K ZIP files into {args.bdd_dir}/")
        return

    for split, (img_dir, json_path) in found_splits.items():
        print(f"   ✅ {split}: images={img_dir}, labels={json_path.name}")

    # ── Filter splits ─────────────────────────────────────────────────
    if args.split != "both":
        if args.split in found_splits:
            found_splits = {args.split: found_splits[args.split]}
        else:
            print(f"\n❌ Split '{args.split}' not found. Available: {list(found_splits.keys())}")
            return

    # ── Convert each split ────────────────────────────────────────────
    total_stats = Counter()
    for split_name, (img_dir, json_path) in found_splits.items():
        stats = convert_split(
            split_name, img_dir, json_path, args.raw_dir,
            args.max_images, args.img_width, args.img_height,
        )
        total_stats += stats

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🎉 BDD100K Conversion Complete!")
    print(f"{'='*60}")
    print(f"\n📊 Class distribution:")
    for class_name in TARGET_CLASSES.values():
        count = total_stats.get(class_name, 0)
        print(f"   {class_name:<24s}: {count:>7d}")
    print(f"   {'─'*40}")
    print(f"   {'TOTAL':<24s}: {sum(total_stats.values()):>7d}")

    raw_labels = args.raw_dir / "labels"
    raw_images = args.raw_dir / "images"
    n_labels = len(list(raw_labels.glob("*.txt"))) if raw_labels.exists() else 0
    n_images = len([f for f in raw_images.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if raw_images.exists() else 0

    print(f"\n📁 Output: {args.raw_dir}")
    print(f"   Images: {n_images}")
    print(f"   Labels: {n_labels}")

    print(f"\n📋 Next steps:")
    print(f"   1. python scripts/filter_classes.py")
    print(f"   2. python scripts/clean_dataset.py")
    print(f"   3. python scripts/balance_dataset.py --analyze-only")
    print(f"   4. python scripts/balance_dataset.py --min-objects 1000")
    print(f"   5. python scripts/split_dataset.py --copy")


if __name__ == "__main__":
    main()
