"""
Download training datasets from Roboflow and organize them for the pipeline.

Datasets:
  - Segmentation : stelar/rdd-mining-road-seg  v1  (yolov8 format → data/segmentation/)
  - Detection    : stelar/rdd-mining-road-det  v2  (yolov8 format → data/detection/)

Usage:
    python scripts/download_datasets.py --api-key <key>
    python scripts/download_datasets.py --api-key <key> --dataset seg   # seg only
    python scripts/download_datasets.py --api-key <key> --dataset det   # det only

Output structure (per dataset):
    data/{seg|detection}/
      train/images/  train/labels/
      val/images/    val/labels/
      test/images/   test/labels/      ← aggregate test
      test/day/      test/wet/         ← manual split required (see note below)
      test/night/

NOTE: Per-condition test splits (day/wet/night) must be arranged manually.
      After download, move test images into data/{task}/test/day|wet|night/
      The evaluate.py pipeline reads from those subdirs for condition-specific mAP.
"""

import argparse
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
        "format":    "yolov8",          # polygon txt — required by YOLOv8 seg pipeline
        "dest":      DATA / "segmentation",
    },
    "det": {
        "workspace": "stelar",
        "project":   "rdd-mining-road-det",
        "version":   2,
        "format":    "yolov8",          # bbox txt — required by YOLOv8 detect pipeline
        "dest":      DATA / "detection",
    },
}


def download(api_key: str, dataset_key: str) -> None:
    cfg = DATASETS[dataset_key]
    dest: Path = cfg["dest"]

    print(f"\n{'='*60}")
    print(f"Downloading: {cfg['project']} v{cfg['version']} → {dest}")
    print(f"{'='*60}")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(cfg["workspace"]).project(cfg["project"])
    version = project.version(cfg["version"])

    # Download to a temp location, then move into our data/ structure
    tmp_dir = DATA / f"_tmp_{dataset_key}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    version.download(cfg["format"], location=str(tmp_dir))

    # Roboflow extracts into a subdirectory — find it
    subdirs = [d for d in tmp_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"Error: no subdirectory found inside {tmp_dir}")
        sys.exit(1)
    extracted = subdirs[0]

    # Move splits into dest, renaming "valid" → "val"
    dest.mkdir(parents=True, exist_ok=True)
    for split_src_name, split_dst_name in [("train", "train"), ("valid", "val"), ("test", "test")]:
        src = extracted / split_src_name
        dst = dest / split_dst_name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"  {split_src_name:6s} → {dst}")

    # Create per-condition test placeholders
    for condition in ["day", "wet", "night"]:
        cond_dir = dest / "test" / condition
        (cond_dir / "images").mkdir(parents=True, exist_ok=True)
        (cond_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Clean up temp dir
    shutil.rmtree(tmp_dir)

    print(f"\nDone. Data at: {dest}")
    print(
        "\n[ACTION REQUIRED] Distribute test images into per-condition dirs:\n"
        f"  {dest}/test/day/images/\n"
        f"  {dest}/test/wet/images/\n"
        f"  {dest}/test/night/images/\n"
        "  (and matching labels/ for each)\n"
        "  These are used by evaluate.py for per-condition mAP reporting."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Roboflow datasets")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument(
        "--dataset",
        choices=["seg", "det", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    keys = ["seg", "det"] if args.dataset == "all" else [args.dataset]
    for key in keys:
        download(args.api_key, key)


if __name__ == "__main__":
    main()
