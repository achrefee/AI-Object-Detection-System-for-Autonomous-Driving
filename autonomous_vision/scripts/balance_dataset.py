"""
balance_dataset.py — Analyze and balance class distribution.

1. Counts objects per class across all label files.
2. Identifies under-represented classes.
3. Oversamples (duplicates) images containing rare classes.
4. Optionally applies basic augmentations to oversampled images.

Operates on data/processed/ by default.
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from collections import Counter

try:
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
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


def count_class_distribution(labels_dir: Path) -> tuple[Counter, dict]:
    """
    Count total objects per class and build index of which images contain
    each class.

    Returns:
        class_counts: Counter {class_id: total_objects}
        class_images: dict {class_id: [label_filenames]}
    """
    class_counts = Counter()
    class_images = {cid: [] for cid in CLASS_NAMES}

    for label_path in sorted(labels_dir.glob("*.txt")):
        with open(label_path, "r") as f:
            classes_in_file = set()
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    class_counts[cls_id] += 1
                    classes_in_file.add(cls_id)

            for cid in classes_in_file:
                if cid in class_images:
                    class_images[cid].append(label_path.name)

    return class_counts, class_images


def augment_image(img_path: Path, dst_path: Path, aug_type: str):
    """Apply a simple augmentation and save."""
    if not HAS_PIL:
        shutil.copy2(img_path, dst_path)
        return

    with Image.open(img_path) as img:
        if aug_type == "brightness":
            factor = random.uniform(0.6, 1.4)
            img = ImageEnhance.Brightness(img).enhance(factor)
        elif aug_type == "contrast":
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Contrast(img).enhance(factor)
        elif aug_type == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        elif aug_type == "flip":
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img.save(dst_path)


def flip_labels(label_lines: list[str]) -> list[str]:
    """Horizontally flip YOLO bounding boxes (mirror x_center)."""
    flipped = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            parts[1] = str(round(1.0 - float(parts[1]), 6))
            flipped.append(" ".join(parts))
    return flipped


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze and balance class distribution.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to dataset directory (default: data/processed)",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=1000,
        help="Minimum objects per class target (default: 1000)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only print class distribution, don't balance",
    )
    args = parser.parse_args()

    labels_dir = args.data_dir / "labels"
    images_dir = args.data_dir / "images"

    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return

    # ── Step 1: Analyze ────────────────────────────────────────────────
    class_counts, class_images = count_class_distribution(labels_dir)

    print("📊 Current class distribution:")
    print(f"   {'Class':<20s} {'Objects':>8s}  {'Images':>8s}  Status")
    print("   " + "─" * 55)
    for cid in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[cid]
        count = class_counts.get(cid, 0)
        n_imgs = len(class_images.get(cid, []))
        status = "✅" if count >= args.min_objects else "⚠️  UNDER-REPRESENTED"
        print(f"   {name:<20s} {count:>8d}  {n_imgs:>8d}  {status}")

    total = sum(class_counts.values())
    print(f"\n   Total objects: {total}")
    print(f"   Total label files: {len(list(labels_dir.glob('*.txt')))}")

    if args.analyze_only:
        return

    # ── Step 2: Oversample under-represented classes ───────────────────
    augmentation_types = ["brightness", "contrast", "blur", "flip"]
    oversampled = 0

    for cid in sorted(CLASS_NAMES.keys()):
        count = class_counts.get(cid, 0)
        imgs = class_images.get(cid, [])

        if count >= args.min_objects or not imgs:
            continue

        needed = args.min_objects - count
        name = CLASS_NAMES[cid]
        print(f"\n🔄 Oversampling '{name}': need ~{needed} more objects...")

        copy_idx = 0
        while class_counts[cid] < args.min_objects:
            # Pick a random image containing this class
            src_label_name = random.choice(imgs)
            src_label = labels_dir / src_label_name
            stem = src_label.stem
            src_image = find_image(images_dir, stem)

            if src_image is None:
                continue

            aug_type = augmentation_types[copy_idx % len(augmentation_types)]
            new_stem = f"{stem}_aug{cid}_{copy_idx}"
            new_label = labels_dir / f"{new_stem}.txt"
            new_image = images_dir / f"{new_stem}{src_image.suffix}"

            # Read and potentially transform labels
            with open(src_label, "r") as f:
                lines = f.readlines()

            if aug_type == "flip":
                lines = flip_labels(lines)

            # Save augmented image and label
            augment_image(src_image, new_image, aug_type)
            with open(new_label, "w") as f:
                f.write("\n".join(l.strip() for l in lines) + "\n")

            # Update counts
            for line in lines:
                parts = line.strip().split()
                if parts:
                    c = int(parts[0])
                    class_counts[c] += 1

            oversampled += 1
            copy_idx += 1

    print(f"\n✅ Balancing complete! Created {oversampled} augmented copies.")

    # ── Final stats ────────────────────────────────────────────────────
    class_counts, _ = count_class_distribution(labels_dir)
    print("\n📊 Updated class distribution:")
    for cid in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[cid]
        count = class_counts.get(cid, 0)
        print(f"   {name:<20s}: {count:>6d}")


if __name__ == "__main__":
    main()
