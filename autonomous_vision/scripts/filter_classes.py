"""
filter_classes.py — Filter YOLO labels to keep only needed classes.

Reads labels from data/raw/labels/, remaps class IDs to the target
class set, copies matching images, and saves results to data/processed/.

Target classes (18 classes for autonomous driving):
    0: car                  7: traffic_light_red    14: road_barrier
    1: truck                8: traffic_light_green   15: cone
    2: bus                  9: traffic_light_yellow  16: pothole
    3: motorcycle          10: stop_sign             17: crosswalk
    4: bicycle             11: speed_limit_sign
    5: pedestrian          12: yield_sign
    6: cyclist             13: no_entry_sign
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import Counter


# ── BDD100K original class IDs → our target class IDs ──────────────────
# Adjust this mapping based on the actual class IDs in your raw dataset.
# Key = original class ID in raw labels, Value = new class ID (or None to drop)
CLASS_MAPPING = {
    # BDD100K detection classes (typical order):
    #  0: pedestrian, 1: rider, 2: car, 3: truck, 4: bus,
    #  5: train, 6: motorcycle, 7: bicycle, 8: traffic light, 9: traffic sign
    0: 5,       # pedestrian → pedestrian (5)
    1: 6,       # rider → cyclist (6)
    2: 0,       # car → car (0)
    3: 1,       # truck → truck (1)
    4: 2,       # bus → bus (2)
    5: None,    # train → drop
    6: 3,       # motorcycle → motorcycle (3)
    7: 4,       # bicycle → bicycle (4)
    8: 7,       # traffic light → traffic_light_red (7) — refine by color later
    9: 10,      # traffic sign → stop_sign (10) — refine by subclass later
}

# Full 18-class target set from the AI Object Detection System Report
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
    10: "stop_sign",
    11: "speed_limit_sign",
    12: "yield_sign",
    13: "no_entry_sign",
    14: "road_barrier",
    15: "cone",
    16: "pothole",
    17: "crosswalk",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def filter_label_file(src_label: Path, dst_label: Path) -> int:
    """
    Read a YOLO label file, remap classes, drop unwanted ones.
    Returns the number of valid objects kept.
    """
    kept_lines = []

    with open(src_label, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            original_class = int(parts[0])
            new_class = CLASS_MAPPING.get(original_class)

            if new_class is not None:
                parts[0] = str(new_class)
                kept_lines.append(" ".join(parts))

    if kept_lines:
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label, "w") as f:
            f.write("\n".join(kept_lines) + "\n")

    return len(kept_lines)


def find_image(images_dir: Path, stem: str) -> Path | None:
    """Find an image file matching the label stem (any extension)."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter YOLO labels to keep only target classes."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to output directory (default: data/processed)",
    )
    args = parser.parse_args()

    raw_labels = args.raw_dir / "labels"
    raw_images = args.raw_dir / "images"
    out_labels = args.out_dir / "labels"
    out_images = args.out_dir / "images"

    if not raw_labels.exists():
        print(f"❌ Labels directory not found: {raw_labels}")
        return

    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    label_files = sorted(raw_labels.glob("*.txt"))
    print(f"📂 Found {len(label_files)} label files in {raw_labels}")

    stats = Counter()
    kept = 0
    dropped = 0

    for label_path in label_files:
        dst_label = out_labels / label_path.name
        n_objects = filter_label_file(label_path, dst_label)

        if n_objects > 0:
            kept += 1
            # Count per-class stats
            with open(dst_label, "r") as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    stats[TARGET_CLASSES.get(cls_id, f"unknown_{cls_id}")] += 1

            # Copy matching image
            img = find_image(raw_images, label_path.stem)
            if img:
                shutil.copy2(img, out_images / img.name)
        else:
            dropped += 1

    print(f"\n✅ Filtering complete!")
    print(f"   Kept:    {kept} images")
    print(f"   Dropped: {dropped} images (no valid objects)")
    print(f"\n📊 Class distribution:")
    for cls_name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"   {cls_name:<20s}: {count:>6d}")
    print(f"\n📁 Output saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
