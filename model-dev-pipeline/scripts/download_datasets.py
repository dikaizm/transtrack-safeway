"""
Download training datasets from Roboflow.

Usage:
    python scripts/download_datasets.py --api-key <key>
    python scripts/download_datasets.py --api-key <key> --dataset seg
    python scripts/download_datasets.py --api-key <key> --dataset det
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DATASETS = {
    "seg": {
        "workspace": "stelar",
        "project":   "rdd-mining-road-seg",
        "version":   1,
        "format":    "yolov8",
        "dest":      DATA / "segmentation",
    },
    "det": {
        "workspace": "stelar",
        "project":   "rdd-mining-road-det",
        "version":   2,
        "format":    "yolov8",
        "dest":      DATA / "detection",
    },
}


def download(api_key: str, dataset_key: str) -> None:
    cfg  = DATASETS[dataset_key]
    dest = cfg["dest"]

    print(f"\n{'='*60}")
    print(f"Downloading: {cfg['project']} v{cfg['version']} → {dest}")
    print(f"{'='*60}")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    rf      = Roboflow(api_key=api_key)
    version = rf.workspace(cfg["workspace"]).project(cfg["project"]).version(cfg["version"])

    # Download directly into dest — Roboflow will create a project subfolder inside it
    dest.mkdir(parents=True, exist_ok=True)
    version.download(cfg["format"], location=str(dest))

    # Find where Roboflow actually put the files
    # It may be: dest/train/ directly, or dest/{project-version}/train/
    if (dest / "train").exists() or (dest / "valid").exists():
        extracted = dest
    else:
        subdirs = sorted(dest.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        subdirs = [d for d in subdirs if d.is_dir()]
        if not subdirs:
            print(f"Error: nothing was downloaded into {dest}")
            print("Contents:", list(dest.iterdir()))
            sys.exit(1)
        extracted = subdirs[0]
        print(f"  Found dataset at: {extracted}")

        # Move contents up to dest
        for item in list(extracted.iterdir()):
            target = dest / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(dest))
        extracted.rmdir()

    # Rename valid → val
    if (dest / "valid").exists() and not (dest / "val").exists():
        (dest / "valid").rename(dest / "val")
        print("  Renamed: valid → val")

    # Create per-condition test placeholders
    for condition in ["day", "wet", "night"]:
        for sub in ["images", "labels"]:
            (dest / "test" / condition / sub).mkdir(parents=True, exist_ok=True)

    print(f"\nDone. Data at: {dest}")
    print(
        "\n[ACTION REQUIRED] Move test images into condition subdirs:\n"
        f"  {dest}/test/day/images/\n"
        f"  {dest}/test/wet/images/\n"
        f"  {dest}/test/night/images/\n"
        "  (and matching labels/) — used by evaluate.py for per-condition mAP."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Roboflow datasets")
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
