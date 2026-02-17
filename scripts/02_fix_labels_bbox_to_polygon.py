"""
Fix mixed label formats in YOLOv8 segmentation dataset.

Some label files contain detection-format lines (class cx cy w h) mixed with
segmentation polygon lines. This script converts bbox lines to rectangular
polygon format so they're compatible with segmentation training.

Detection format:  class cx cy w h          (5 fields)
Segment format:    class x1 y1 x2 y2 ...   (>5 fields, polygon vertices)

Bbox -> Polygon conversion:
  cx, cy, w, h -> 4-corner rectangle polygon
"""

import os
import glob
import argparse


def bbox_to_polygon(cls, cx, cy, w, h):
    """Convert a bounding box to a rectangular polygon (4 corners)."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    # 4 corners: top-left, top-right, bottom-right, bottom-left
    return f"{cls} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"


def fix_label_file(filepath, dry_run=False):
    """Fix a single label file. Returns (converted_count, total_lines)."""
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    converted = 0
    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            # Detection format -> convert to polygon
            cls = parts[0]
            cx, cy, w, h = map(float, parts[1:])
            new_line = bbox_to_polygon(cls, cx, cy, w, h)
            new_lines.append(new_line)
            converted += 1
        elif len(parts) > 5:
            # Already segmentation format
            new_lines.append(line)
        else:
            # Malformed line, skip
            print(f"  WARNING: Skipping malformed line in {filepath}: {line}")

    if converted > 0 and not dry_run:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')

    return converted, len(lines)


def main():
    parser = argparse.ArgumentParser(description="Fix bbox labels in segmentation dataset")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset root")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't modify files")
    args = parser.parse_args()

    splits = ["train", "valid", "test"]
    total_files_fixed = 0
    total_lines_converted = 0

    for split in splits:
        label_dir = os.path.join(args.dataset_dir, split, "labels")
        if not os.path.isdir(label_dir):
            print(f"Skipping {split}/ (not found)")
            continue

        files = glob.glob(os.path.join(label_dir, "*.txt"))
        split_fixed = 0
        split_converted = 0

        for filepath in files:
            converted, total = fix_label_file(filepath, dry_run=args.dry_run)
            if converted > 0:
                split_fixed += 1
                split_converted += converted

        print(f"{split}: {split_fixed} files fixed, {split_converted} lines converted")
        total_files_fixed += split_fixed
        total_lines_converted += split_converted

    action = "Would fix" if args.dry_run else "Fixed"
    print(f"\n{action} {total_lines_converted} bbox lines in {total_files_fixed} files total.")


if __name__ == "__main__":
    main()
