"""
Download datasets from Roboflow and convert to YOLO format.

Roboflow download formats (as confirmed working):
  - Segmentation : coco-segmentation  → converted to YOLO polygon txt
  - Detection    : coco               → converted to YOLO bbox txt

Usage:
    python scripts/download_datasets.py --api-key <key>
    python scripts/download_datasets.py --api-key <key> --dataset seg
    python scripts/download_datasets.py --api-key <key> --dataset det
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # model-dev-pipeline/
DATA = ROOT / "data"

DATASETS = {
    "seg": {
        "workspace": "stelar",
        "project":   "rdd-mining-road-seg",
        "version":   1,
        "format":    "coco-segmentation",
        "dest":      DATA / "segmentation",
    },
    "det": {
        "workspace": "stelar",
        "project":   "rdd-mining-road-det",
        "version":   2,
        "format":    "coco",
        "dest":      DATA / "detection",
    },
}


# --------------------------------------------------------------------------- #
# COCO → YOLO converters                                                       #
# --------------------------------------------------------------------------- #

def coco_seg_to_yolo(coco_json: Path, images_dir: Path, labels_dir: Path) -> None:
    """Convert COCO segmentation JSON to YOLO polygon txt files."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(coco_json) as f:
        data = json.load(f)

    # Build lookup maps
    cat_ids   = {c["id"]: i for i, c in enumerate(data["categories"])}
    img_info  = {img["id"]: img for img in data["images"]}
    # Group annotations by image
    ann_by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, img in img_info.items():
        W, H  = img["width"], img["height"]
        fname = Path(img["file_name"]).stem
        lines = []
        for ann in ann_by_img.get(img_id, []):
            cls_idx = cat_ids[ann["category_id"]]
            for seg in ann.get("segmentation", []):
                # seg is a flat list [x1,y1,x2,y2,...] — normalize to 0-1
                coords = [
                    f"{seg[i]/W:.6f} {seg[i+1]/H:.6f}"
                    for i in range(0, len(seg), 2)
                ]
                lines.append(f"{cls_idx} " + " ".join(coords))
        (labels_dir / f"{fname}.txt").write_text("\n".join(lines))


def coco_det_to_yolo(coco_json: Path, images_dir: Path, labels_dir: Path) -> None:
    """Convert COCO detection JSON to YOLO bbox txt files."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(coco_json) as f:
        data = json.load(f)

    cat_ids  = {c["id"]: i for i, c in enumerate(data["categories"])}
    img_info = {img["id"]: img for img in data["images"]}
    ann_by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, img in img_info.items():
        W, H  = img["width"], img["height"]
        fname = Path(img["file_name"]).stem
        lines = []
        for ann in ann_by_img.get(img_id, []):
            cls_idx = cat_ids[ann["category_id"]]
            x, y, w, h = ann["bbox"]           # COCO: top-left x,y + w,h
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (labels_dir / f"{fname}.txt").write_text("\n".join(lines))


def convert_split(split_dir: Path, fmt: str) -> None:
    """Find the COCO JSON in a split dir and convert annotations to YOLO txt."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # Roboflow names the JSON differently per format
    coco_candidates = list(split_dir.glob("*.json")) + list(split_dir.glob("**/*.json"))
    if not coco_candidates:
        print(f"  No COCO JSON found in {split_dir}, skipping conversion")
        return

    coco_json = coco_candidates[0]
    print(f"  Converting {coco_json.name} → labels/")

    if fmt == "coco-segmentation":
        coco_seg_to_yolo(coco_json, images_dir, labels_dir)
    else:
        coco_det_to_yolo(coco_json, images_dir, labels_dir)


# --------------------------------------------------------------------------- #
# Download + organize                                                           #
# --------------------------------------------------------------------------- #

def download(api_key: str, dataset_key: str) -> None:
    cfg  = DATASETS[dataset_key]
    dest = cfg["dest"]
    fmt  = cfg["format"]

    print(f"\n{'='*60}")
    print(f"Downloading: {cfg['project']} v{cfg['version']} ({fmt}) → {dest}")
    print(f"{'='*60}")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    rf      = Roboflow(api_key=api_key)
    version = rf.workspace(cfg["workspace"]).project(cfg["project"]).version(cfg["version"])

    # Roboflow downloads to CWD — chdir to DATA so the folder appears there
    DATA.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(DATA)
    try:
        dataset = version.download(fmt)
    finally:
        os.chdir(original_cwd)

    # Locate the downloaded folder (returned by dataset.location or find it)
    if hasattr(dataset, "location") and dataset.location:
        extracted = Path(dataset.location)
    else:
        # Fallback: find the newest directory under DATA that has train/ or valid/
        candidates = sorted(
            [d for d in DATA.iterdir() if d.is_dir() and d != dest],
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        extracted = next(
            (d for d in candidates if (d / "train").exists() or (d / "valid").exists()),
            None,
        )
        if extracted is None:
            print(f"Error: could not find downloaded dataset under {DATA}")
            print("Contents:", [str(d) for d in DATA.iterdir()])
            sys.exit(1)

    print(f"  Downloaded to: {extracted}")

    # Move splits into dest, rename valid → val
    dest.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [("train", "train"), ("valid", "val"), ("test", "test")]:
        src = extracted / src_name
        dst = dest / dst_name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"  Moved: {src_name} → {dst}")

    # Clean up Roboflow's folder if it's not dest itself
    if extracted != dest and extracted.exists():
        shutil.rmtree(extracted, ignore_errors=True)

    # Convert COCO JSON annotations → YOLO txt labels
    print("\nConverting COCO annotations → YOLO format...")
    for split in ["train", "val", "test"]:
        split_dir = dest / split
        if split_dir.exists():
            convert_split(split_dir, fmt)

    # Create per-condition test dir placeholders
    for condition in ["day", "wet", "night"]:
        for sub in ["images", "labels"]:
            (dest / "test" / condition / sub).mkdir(parents=True, exist_ok=True)

    print(f"\nDone. Data ready at: {dest}")
    print(
        "\n[ACTION REQUIRED] Move test images into condition subdirs:\n"
        f"  {dest}/test/day/images/\n"
        f"  {dest}/test/wet/images/\n"
        f"  {dest}/test/night/images/\n"
        "  (and matching labels/) — needed for per-condition evaluation."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Roboflow datasets")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument(
        "--dataset", choices=["seg", "det", "all"], default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    keys = ["seg", "det"] if args.dataset == "all" else [args.dataset]
    for key in keys:
        download(args.api_key, key)


if __name__ == "__main__":
    main()
