"""
download_coco.py — Download COCO 2017 and export relevant classes in YOLO format.

Uses FiftyOne to download COCO 2017 (train + validation splits),
filters only the classes relevant to autonomous driving, and exports
them in YOLO format to data/raw/.

Requirements:
    pip install fiftyone ultralytics Pillow

Usage:
    python scripts/download_coco.py
    python scripts/download_coco.py --max-samples 5000    # Quick test
    python scripts/download_coco.py --split train          # Train only
"""

import os
import argparse
from pathlib import Path

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    print("❌ FiftyOne is not installed. Install it with:")
    print("   pip install fiftyone")
    exit(1)


# ── COCO classes relevant to autonomous driving ───────────────────────
# These are the COCO class NAMES that map to our 18-class target set.
COCO_RELEVANT_CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",          # → pedestrian
    "traffic light",
    "stop sign",
]

# Mapping from COCO class names → our target class IDs
COCO_TO_TARGET = {
    "car":            0,
    "truck":          1,
    "bus":            2,
    "motorcycle":     3,
    "bicycle":        4,
    "person":         5,   # pedestrian
    "traffic light":  7,   # traffic_light_red (refine later)
    "stop sign":     10,   # stop_sign
}


def download_and_export(split: str, max_samples: int | None, output_dir: Path):
    """Download a COCO split and export in YOLO format."""

    print(f"\n{'='*60}")
    print(f"📥 Downloading COCO 2017 — {split} split")
    if max_samples:
        print(f"   (Limited to {max_samples} samples)")
    print(f"{'='*60}\n")

    # ── Step 1: Download from FiftyOne Zoo ─────────────────────────────
    kwargs = {
        "split": split,
        "label_types": ["detections"],
        "classes": COCO_RELEVANT_CLASSES,
    }
    if max_samples:
        kwargs["max_samples"] = max_samples

    dataset = foz.load_zoo_dataset("coco-2017", **kwargs)

    print(f"\n✅ Downloaded {len(dataset)} samples")

    # ── Step 2: Filter to keep only relevant classes ───────────────────
    from fiftyone import ViewField as F

    # Filter detections to only keep our target classes
    filtered = dataset.filter_labels(
        "ground_truth",
        F("label").is_in(COCO_RELEVANT_CLASSES),
    )

    # Remove samples with no remaining detections
    filtered = filtered.match(
        F("ground_truth.detections").length() > 0
    )

    print(f"   After filtering: {len(filtered)} samples with target classes")

    # ── Step 3: Export in YOLO format ──────────────────────────────────
    export_dir = str(output_dir)

    print(f"\n📦 Exporting to YOLO format → {export_dir}")

    filtered.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=COCO_RELEVANT_CLASSES,
        split=split,
    )

    print(f"✅ Export complete!")
    return len(filtered)


def remap_labels(output_dir: Path):
    """
    Remap the YOLO labels from FiftyOne's class ordering
    (alphabetical) to our target class IDs.

    FiftyOne exports classes in alphabetical order:
        0: bicycle, 1: bus, 2: car, 3: motorcycle,
        4: person, 5: stop sign, 6: traffic light, 7: truck

    We need to remap to our target IDs:
        0: car, 1: truck, 2: bus, 3: motorcycle,
        4: bicycle, 5: pedestrian, 7: traffic_light_red, 10: stop_sign
    """
    # FiftyOne alphabetical order → our target class ID
    FIFTYONE_TO_TARGET = {
        0: 4,    # bicycle → bicycle (4)
        1: 2,    # bus → bus (2)
        2: 0,    # car → car (0)
        3: 3,    # motorcycle → motorcycle (3)
        4: 5,    # person → pedestrian (5)
        5: 10,   # stop sign → stop_sign (10)
        6: 7,    # traffic light → traffic_light_red (7)
        7: 1,    # truck → truck (1)
    }

    labels_dirs = [
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
        output_dir / "labels" / "validation",
        output_dir / "labels",
    ]

    remapped = 0
    for labels_dir in labels_dirs:
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            lines = label_file.read_text().strip().splitlines()
            new_lines = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_id = int(parts[0])
                    new_id = FIFTYONE_TO_TARGET.get(old_id)
                    if new_id is not None:
                        parts[0] = str(new_id)
                        new_lines.append(" ".join(parts))

            if new_lines:
                label_file.write_text("\n".join(new_lines) + "\n")
                remapped += 1
            else:
                label_file.unlink()  # Remove empty files

    print(f"\n🔄 Remapped {remapped} label files to target class IDs")


def reorganize_to_raw(output_dir: Path, raw_dir: Path):
    """
    Move all images and labels into data/raw/images and data/raw/labels
    for compatibility with the pipeline.
    """
    import shutil

    raw_images = raw_dir / "images"
    raw_labels = raw_dir / "labels"
    raw_images.mkdir(parents=True, exist_ok=True)
    raw_labels.mkdir(parents=True, exist_ok=True)

    img_count = 0
    lbl_count = 0

    # Find all images and labels from the export
    for subdir in ["train", "val", "validation", ""]:
        img_dir = output_dir / "images" / subdir if subdir else output_dir / "images"
        lbl_dir = output_dir / "labels" / subdir if subdir else output_dir / "labels"

        if img_dir.exists():
            for f in img_dir.glob("*"):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    dest = raw_images / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
                        img_count += 1

        if lbl_dir.exists():
            for f in lbl_dir.glob("*.txt"):
                if f.name == "classes.txt":
                    continue
                dest = raw_labels / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    lbl_count += 1

    print(f"\n📁 Organized into {raw_dir}:")
    print(f"   Images: {img_count}")
    print(f"   Labels: {lbl_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Download COCO 2017 and export relevant classes in YOLO format."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/coco_export"),
        help="Temporary export directory (default: data/coco_export)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Final output directory (default: data/raw)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to download per split (default: all)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "both"],
        default="both",
        help="Which split to download (default: both)",
    )
    args = parser.parse_args()

    splits = ["train", "validation"] if args.split == "both" else [args.split]
    total = 0

    for split in splits:
        n = download_and_export(split, args.max_samples, args.output_dir)
        total += n

    # Remap FiftyOne's alphabetical class IDs to our target IDs
    remap_labels(args.output_dir)

    # Move everything into data/raw/ for the pipeline
    reorganize_to_raw(args.output_dir, args.raw_dir)

    print(f"\n{'='*60}")
    print(f"🎉 COCO Download Complete!")
    print(f"   Total samples: {total}")
    print(f"   Data ready at: {args.raw_dir}")
    print(f"\n📋 Next steps:")
    print(f"   1. python scripts/filter_classes.py")
    print(f"   2. python scripts/clean_dataset.py")
    print(f"   3. python scripts/balance_dataset.py --analyze-only")
    print(f"   4. python scripts/balance_dataset.py --min-objects 1000")
    print(f"   5. python scripts/split_dataset.py --copy")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
