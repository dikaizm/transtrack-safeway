"""
Download datasets from Roboflow in YOLOv8 format.

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
        "version":   3,
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

    # chdir to DATA so Roboflow's CWD-relative download lands there
    DATA.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(DATA)
    try:
        dataset = version.download(cfg["format"])
    finally:
        os.chdir(original_cwd)

    # Find where Roboflow put the files
    if hasattr(dataset, "location") and dataset.location:
        extracted = Path(dataset.location)
    else:
        candidates = sorted(
            [d for d in DATA.iterdir() if d.is_dir() and d != dest],
            key=lambda p: p.stat().st_mtime, reverse=True,
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

    if extracted != dest and extracted.exists():
        shutil.rmtree(extracted, ignore_errors=True)

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
