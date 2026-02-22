"""
clean_dataset.py — Clean the processed dataset.

Removes:
  1. Empty label files (0 bytes or only whitespace)
  2. Labels with no matching image
  3. Images with no matching label
  4. Corrupted / unreadable images
  5. Labels with invalid format

Operates on data/processed/ by default.
"""

import os
import argparse
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  Pillow not installed — skipping image corruption checks.")
    print("   Install with: pip install Pillow\n")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_label(label_path: Path) -> bool:
    """Check if a label file contains valid YOLO-format lines."""
    try:
        text = label_path.read_text().strip()
        if not text:
            return False

        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                return False
            # class_id must be int, coords must be float in [0, 1]
            int(parts[0])
            coords = [float(x) for x in parts[1:5]]
            if not all(0.0 <= c <= 1.0 for c in coords):
                return False
        return True
    except (ValueError, UnicodeDecodeError):
        return False


def is_valid_image(image_path: Path) -> bool:
    """Check if an image file is readable and not corrupted."""
    if not HAS_PIL:
        return image_path.stat().st_size > 0

    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open to ensure full decode works
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception:
        return False


def find_image(images_dir: Path, stem: str) -> Path | None:
    """Find an image matching the label stem."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Clean dataset by removing invalid entries.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to dataset directory (default: data/processed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything",
    )
    args = parser.parse_args()

    labels_dir = args.data_dir / "labels"
    images_dir = args.data_dir / "images"

    if not labels_dir.exists() or not images_dir.exists():
        print(f"❌ Directory not found: {args.data_dir}")
        return

    removed_labels = 0
    removed_images = 0
    corrupted_images = 0
    orphan_images = 0

    # ── Pass 1: Check labels ───────────────────────────────────────────
    print("🔍 Pass 1: Checking labels...")
    label_files = sorted(labels_dir.glob("*.txt"))

    for label_path in label_files:
        remove = False
        reason = ""

        if not is_valid_label(label_path):
            remove = True
            reason = "invalid/empty label"

        img = find_image(images_dir, label_path.stem)
        if img is None:
            remove = True
            reason = "no matching image"

        if remove:
            if args.dry_run:
                print(f"  [DRY RUN] Would remove: {label_path.name} ({reason})")
            else:
                label_path.unlink()
                removed_labels += 1

    # ── Pass 2: Check images ───────────────────────────────────────────
    print("🔍 Pass 2: Checking images...")
    image_files = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_files:
        label = labels_dir / f"{img_path.stem}.txt"

        if not label.exists():
            if args.dry_run:
                print(f"  [DRY RUN] Would remove: {img_path.name} (orphan image)")
            else:
                img_path.unlink()
                orphan_images += 1
            continue

        if not is_valid_image(img_path):
            if args.dry_run:
                print(f"  [DRY RUN] Would remove: {img_path.name} (corrupted)")
            else:
                img_path.unlink()
                label.unlink(missing_ok=True)
                corrupted_images += 1

    # ── Summary ────────────────────────────────────────────────────────
    prefix = "[DRY RUN] " if args.dry_run else ""
    remaining_labels = len(list(labels_dir.glob("*.txt")))
    remaining_images = len(
        [f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    )

    print(f"\n✅ {prefix}Cleaning complete!")
    print(f"   Removed invalid labels:  {removed_labels}")
    print(f"   Removed orphan images:   {orphan_images}")
    print(f"   Removed corrupted imgs:  {corrupted_images}")
    print(f"\n📊 Remaining dataset:")
    print(f"   Labels: {remaining_labels}")
    print(f"   Images: {remaining_images}")


if __name__ == "__main__":
    main()
