"""
split_dataset.py — Split dataset into train / val / test sets.

Reads from data/processed/ and splits into:
  - data/train/  (70%)
  - data/val/    (20%)
  - data/test/   (10%)

Images and labels are moved together. Uses deterministic shuffling
with a fixed seed for reproducibility.
"""

import os
import random
import shutil
import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10,
}


def find_image(images_dir: Path, stem: str) -> Path | None:
    """Find an image matching the label stem."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test.")
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/processed"),
        help="Source directory with images/ and labels/ (default: data/processed)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output root directory (default: data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them",
    )
    args = parser.parse_args()

    src_labels = args.src_dir / "labels"
    src_images = args.src_dir / "images"

    if not src_labels.exists():
        print(f"❌ Source labels not found: {src_labels}")
        return

    # ── Gather all valid pairs ─────────────────────────────────────────
    label_files = sorted(src_labels.glob("*.txt"))
    pairs = []

    for label_path in label_files:
        img = find_image(src_images, label_path.stem)
        if img:
            pairs.append((img, label_path))

    print(f"📂 Found {len(pairs)} valid image-label pairs")

    if not pairs:
        print("❌ No valid pairs found. Exiting.")
        return

    # ── Shuffle and split ──────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    # test gets the remainder

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    # ── Copy / Move files ──────────────────────────────────────────────
    transfer = shutil.copy2 if args.copy else shutil.move

    for split_name, split_pairs in splits.items():
        dst_images = args.out_dir / split_name / "images"
        dst_labels = args.out_dir / split_name / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_pairs:
            transfer(str(img_path), str(dst_images / img_path.name))
            transfer(str(label_path), str(dst_labels / label_path.name))

    # ── Summary ────────────────────────────────────────────────────────
    action = "Copied" if args.copy else "Moved"
    print(f"\n✅ Split complete! ({action} files)")
    print(f"   Train: {len(splits['train']):>6d}  ({SPLIT_RATIOS['train']*100:.0f}%)")
    print(f"   Val:   {len(splits['val']):>6d}  ({SPLIT_RATIOS['val']*100:.0f}%)")
    print(f"   Test:  {len(splits['test']):>6d}  ({SPLIT_RATIOS['test']*100:.0f}%)")
    print(f"\n📁 Output:")
    print(f"   {args.out_dir / 'train'}")
    print(f"   {args.out_dir / 'val'}")
    print(f"   {args.out_dir / 'test'}")


if __name__ == "__main__":
    main()
