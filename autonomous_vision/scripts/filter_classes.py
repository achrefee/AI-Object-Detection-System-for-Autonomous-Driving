"""
filter_classes.py — Filter YOLO labels to keep only needed classes.

Reads labels from data/raw/labels/, validates class IDs against the
target class set, copies matching images, and saves results to data/processed/.

Target classes (11 classes for autonomous driving — BDD100K):
    0: car                  6: cyclist
    1: truck                7: traffic_light_red
    2: bus                  8: traffic_light_green
    3: motorcycle           9: traffic_light_yellow
    4: bicycle             10: traffic_sign
    5: pedestrian
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import Counter


# ── 11-class target set (from BDD100K) ────────────────────────────────
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

# Valid class IDs (convert_bdd100k.py already outputs correct IDs)
VALID_CLASS_IDS = set(TARGET_CLASSES.keys())

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def filter_label_file(src_label: Path, dst_label: Path) -> int:
    """
    Read a YOLO label file, keep only valid class IDs, drop the rest.
    Returns the number of valid objects kept.
    """
    kept_lines = []

    with open(src_label, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id in VALID_CLASS_IDS:
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
        print(f"   {cls_name:<24s}: {count:>6d}")
    print(f"\n📁 Output saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
