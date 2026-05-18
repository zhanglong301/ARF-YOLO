"""
VisDrone2019 to YOLO Format Converter.

Converts VisDrone2019-DET annotation format to YOLO format.

VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
                 <score>,<object_category>,<truncation>,<occlusion>

YOLO format: <class_id> <x_center> <y_center> <width> <height>
             (all values normalized to [0, 1])

Usage:
    python utils/visdrone2yolo.py --source path/to/VisDrone2019-DET-train
"""

import argparse
import os
from pathlib import Path
from PIL import Image


# VisDrone category mapping (0-indexed for YOLO)
# VisDrone categories: 0=ignored, 1=pedestrian, 2=people, 3=bicycle,
# 4=car, 5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor
# 11=others (ignored)
# We map VisDrone categories 1-10 to YOLO classes 0-9
VISDRONE_TO_YOLO = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}


def convert_visdrone(source_dir, output_dir=None):
    """
    Convert VisDrone annotations to YOLO format.

    Args:
        source_dir: Path to VisDrone split directory (e.g., VisDrone2019-DET-train)
        output_dir: Output directory for YOLO labels. If None, creates 'labels'
                    directory alongside 'annotations'.
    """
    source_dir = Path(source_dir)
    ann_dir = source_dir / "annotations"
    img_dir = source_dir / "images"

    if output_dir is None:
        output_dir = source_dir / "labels"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ann_dir.exists():
        print(f"Annotation directory not found: {ann_dir}")
        return

    ann_files = sorted(ann_dir.glob("*.txt"))
    print(f"Found {len(ann_files)} annotation files in {ann_dir}")

    converted = 0
    skipped = 0

    for ann_file in ann_files:
        img_name = ann_file.stem + ".jpg"
        img_path = img_dir / img_name

        if not img_path.exists():
            # Try png
            img_name = ann_file.stem + ".png"
            img_path = img_dir / img_name

        if not img_path.exists():
            skipped += 1
            continue

        # Get image dimensions
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # Parse VisDrone annotation
        yolo_lines = []
        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue

                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_w = float(parts[2])
                bbox_h = float(parts[3])
                score = int(parts[4])
                category = int(parts[5])
                truncation = int(parts[6])
                occlusion = int(parts[7])

                # Skip ignored and 'others' categories
                if category not in VISDRONE_TO_YOLO:
                    continue

                # Skip score=0 (ignored region)
                if score == 0:
                    continue

                # Convert to YOLO format (normalized x_center, y_center, w, h)
                yolo_cls = VISDRONE_TO_YOLO[category]
                x_center = (bbox_left + bbox_w / 2) / img_w
                y_center = (bbox_top + bbox_h / 2) / img_h
                w_norm = bbox_w / img_w
                h_norm = bbox_h / img_h

                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                # Skip tiny or invalid boxes
                if w_norm < 1e-6 or h_norm < 1e-6:
                    continue

                yolo_lines.append(
                    f"{yolo_cls} {x_center:.6f} {y_center:.6f} "
                    f"{w_norm:.6f} {h_norm:.6f}\n"
                )

        # Write YOLO annotation
        out_file = output_dir / (ann_file.stem + ".txt")
        with open(out_file, "w") as f:
            f.writelines(yolo_lines)

        converted += 1

    print(f"Converted {converted} files, skipped {skipped} files")
    print(f"YOLO labels saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="VisDrone to YOLO Converter")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to VisDrone split directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for YOLO labels")
    args = parser.parse_args()

    convert_visdrone(args.source, args.output)


if __name__ == "__main__":
    main()
