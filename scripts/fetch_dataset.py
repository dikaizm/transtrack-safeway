"""
Fetch and merge road damage detection datasets from Roboflow.

Requirements:
    pip install roboflow pyyaml
"""

import os
import shutil
import yaml
from pathlib import Path
from roboflow import Roboflow

# Set paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Dataset paths
SRC1 = Path(DATA_DIR) / "road-damage-detection-yolov8-int"
SRC2 = Path(DATA_DIR) / "road-damage-detection-yolov8-ext"
DST = Path(DATA_DIR) / "road-damage-detection-yolov8-merged"

SPLITS = ["train", "valid", "test"]


def download_datasets():
    """Download datasets from Roboflow."""
    print("Downloading internal dataset...")
    rf1 = Roboflow(api_key="MYuIstjO07DOJx3bmDZW")
    project1 = rf1.workspace("farrellhrs").project("road-damage-detection-uutxu")
    version1 = project1.version(3)
    version1.download("yolov8", location=str(SRC1))
    
    print("Downloading external dataset...")
    rf2 = Roboflow(api_key="0CEdpB53poTT1jUXgvm7")
    project2 = rf2.workspace("rdd-heop9").project("road-damage-detection-ytvlw")
    version2 = project2.version(2)
    version2.download("yolov8", location=str(SRC2))
    
    print("Dataset downloads complete.")


def ensure_structure():
    """Create directory structure for merged dataset."""
    for split in SPLITS:
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_split(src, name_prefix):
    """Copy dataset split with prefix to avoid filename collisions."""
    for split in SPLITS:
        img_src = src / split / "images"
        lbl_src = src / split / "labels"

        if not img_src.exists():
            continue

        for img in img_src.glob("*"):
            new_name = f"{name_prefix}_{img.name}"
            shutil.copy2(img, DST / split / "images" / new_name)

        for lbl in lbl_src.glob("*"):
            new_name = f"{name_prefix}_{lbl.name}"
            shutil.copy2(lbl, DST / split / "labels" / new_name)


def merge_yaml():
    """Merge data.yaml files from both datasets."""
    yaml1 = yaml.safe_load(open(SRC1 / "data.yaml"))
    yaml2 = yaml.safe_load(open(SRC2 / "data.yaml"))

    # Verify class names match
    if yaml1["names"] != yaml2["names"]:
        raise ValueError("Class mismatch between datasets")

    merged = {
        "path": str(DST.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": yaml1["names"],
        "nc": len(yaml1["names"]),
    }

    with open(DST / "data.yaml", "w") as f:
        yaml.dump(merged, f)


def merge_datasets():
    """Merge the downloaded datasets."""
    print("Merging datasets...")
    ensure_structure()

    # Prefix prevents filename collision
    print("Copying internal dataset with 'int' prefix...")
    copy_split(SRC1, "int")
    
    print("Copying external dataset with 'ext' prefix...")
    copy_split(SRC2, "ext")

    print("Creating merged data.yaml...")
    merge_yaml()

    print(f"Merged dataset ready at: {DST}")


def main():
    """Main execution function."""
    # Download datasets from Roboflow
    download_datasets()
    
    # Merge the datasets
    merge_datasets()
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
