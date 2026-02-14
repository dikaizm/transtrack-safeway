import os
import subprocess
import sys
import re
from datetime import datetime

# Set paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# --- Configuration ---
YOLO_DIR = os.path.join(PROJECT_ROOT, 'yolov5')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'road-damage-detection-yolov8-merged')  # YOLOv5/v8 share same format
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

EPOCHS = 5
BATCH_SIZE = 8
IMG_SIZE = 640
MODEL_WEIGHTS = 'yolov5s-seg.pt'  # Pretrained weights

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


def patch_polygons2masks_overlap():
    """
    Patch YOLOv5's polygons2masks_overlap to fix OverflowError when there are 256+ instances.
    The original code uses uint8 which can only hold 0-255, but mosaic augmentation can create
    samples with 256+ instances.
    """
    dataloaders_path = os.path.join(YOLO_DIR, 'utils', 'segment', 'dataloaders.py')
    
    if not os.path.exists(dataloaders_path):
        print(f"Warning: Could not find {dataloaders_path}, skipping patch")
        return False
    
    with open(dataloaders_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'uint16 if len(segments) > 255' in content:
        print("YOLOv5 polygons2masks_overlap already patched")
        return True
    
    # Try multiple patterns to handle different YOLOv5 versions
    patterns = [
        # Pattern 1: Original format
        r'dtype=np\.int32 if len\(segments\) > 255 else np\.uint8',
        # Pattern 2: With extra spaces
        r'dtype\s*=\s*np\.int32\s+if\s+len\(segments\)\s*>\s*255\s+else\s+np\.uint8',
        # Pattern 3: Just uint8 dtype assignment in polygons2masks_overlap context
        r'(def polygons2masks_overlap[^}]+?)dtype\s*=\s*np\.uint8',
    ]
    
    replacement = 'dtype=np.int32 if len(segments) > 65535 else (np.uint16 if len(segments) > 255 else np.uint8)'
    
    patched = False
    for pattern in patterns[:2]:  # Try first two patterns
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            patched = True
            break
    
    if patched:
        try:
            with open(dataloaders_path, 'w') as f:
                f.write(content)
            print(f"✓ Patched {dataloaders_path} to fix uint8 overflow")
            return True
        except IOError as e:
            print(f"Error writing patch: {e}")
            return False
    else:
        print(f"Warning: Could not find target pattern in {dataloaders_path}")
        print("The YOLOv5 version may already handle this case or has different code structure.")
        return False


def setup_yolov5():
    """Clone YOLOv5 repository and apply necessary patches."""
    if not os.path.exists(YOLO_DIR):
        print(f"Cloning YOLOv5 repo to {YOLO_DIR}...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5", YOLO_DIR], check=True)
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-qr", os.path.join(YOLO_DIR, "requirements.txt")], check=True)
    else:
        print(f"YOLOv5 repo already exists at {YOLO_DIR}")
    
    # Always try to apply patch (idempotent - checks if already applied)
    patch_polygons2masks_overlap()

def main():
    setup_yolov5()
    
    print(f"Checking for dataset at: {DATASET_DIR}")
    
    if not os.path.exists(YAML_PATH):
        print(f"Error: data.yaml not found at {YAML_PATH}")
        print("Please ensure the dataset is in YOLO format and contains data.yaml.")
        return

    print(f"Found dataset config: {YAML_PATH}")

    # Train Model
    print("Starting Training...")
    train_script = os.path.join(YOLO_DIR, 'segment', 'train.py')
    
    # Generate timestamped folder name: {modelname}-yyyymmdd-hhmm
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    model_name = MODEL_WEIGHTS.replace('.pt', '')  # e.g., 'yolov5s-seg'
    folder_name = f"{model_name}-{timestamp}"
    
    print(f"Model will be saved to: {os.path.join(MODELS_DIR, folder_name)}")
    
    # Construct command args
    args = [
        sys.executable, train_script,
        "--img", str(IMG_SIZE),
        "--batch", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--data", YAML_PATH,
        "--weights", MODEL_WEIGHTS,
        "--name", folder_name,
        "--project", MODELS_DIR,
        "--workers", "0"  # Disable multiprocessing to avoid DataLoader errors
    ]

    print(f"Running command: {' '.join(args)}")
    
    # Run training
    subprocess.run(args, check=True)

if __name__ == "__main__":
    main()
