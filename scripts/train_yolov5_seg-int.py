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
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'road-damage-detection-yolov5-int')  # YOLOv5/v8 share same format
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 640
MODEL_WEIGHTS = 'yolov5s-seg.pt'  # Pretrained weights

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


def patch_polygons2masks_overlap():
    """
    Patch YOLOv5's polygons2masks_overlap to fix OverflowError when there are 256+ instances.
    The original code uses uint8 which can only hold 0-255, but mosaic augmentation can create
    samples with 256+ instances. The overflow happens at `mask = ms[i] * (i + 1)` because each
    individual mask (ms[i]) is uint8, so the multiplication overflows before being stored.
    """
    dataloaders_path = os.path.join(YOLO_DIR, 'utils', 'segment', 'dataloaders.py')
    
    if not os.path.exists(dataloaders_path):
        print(f"Warning: Could not find {dataloaders_path}, skipping patch")
        return False
    
    with open(dataloaders_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if '# PATCHED: uint8 overflow fix' in content:
        print("YOLOv5 polygons2masks_overlap already patched")
        return True
    
    patched = False
    
    # Fix 1: Cast ms[i] to int32 before multiplication to prevent uint8 overflow.
    # This is the actual line that causes OverflowError when i+1 >= 256.
    old_mult = 'mask = ms[i] * (i + 1)'
    new_mult = 'mask = ms[i].astype(np.int32) * (i + 1)  # PATCHED: uint8 overflow fix'
    
    if old_mult in content:
        content = content.replace(old_mult, new_mult)
        patched = True
    
    # Fix 2: Also upgrade the accumulator masks array dtype from uint8 to int32
    # so it can store the larger index values.
    old_dtype_pattern = re.compile(
        r'(def polygons2masks_overlap.*?masks\s*=\s*np\.zeros\([^)]+,\s*dtype\s*=\s*np\.)uint8',
        re.DOTALL
    )
    if old_dtype_pattern.search(content):
        content = old_dtype_pattern.sub(r'\g<1>int32', content)
        patched = True
    
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


def patch_random_perspective():
    """
    Patch random_perspective in augmentations.py to fix IndexError when len(targets) != len(segments).
    """
    aug_path = os.path.join(YOLO_DIR, 'utils', 'segment', 'augmentations.py')
    
    if not os.path.exists(aug_path):
        print(f"Warning: Could not find {aug_path}")
        return

    with open(aug_path, 'r') as f:
        content = f.read()

    if 'targets = targets[:n]' in content:
        print("YOLOv5 random_perspective already patched")
        return

    # Pattern to match
    old_code = '    if n := len(targets):\n        new = np.zeros((n, 4))'
    
    # New code with synchronization
    new_code = ('    if n := len(targets):\n'
                '        if len(segments) != n:\n'
                '            n = min(n, len(segments))\n'
                '            targets = targets[:n]\n'
                '            segments = segments[:n]\n'
                '        new = np.zeros((n, 4))')

    if old_code in content:
        content = content.replace(old_code, new_code)
        try:
            with open(aug_path, 'w') as f:
                f.write(content)
            print(f"✓ Patched {aug_path} to fix segment/label mismatch")
        except IOError as e:
            print(f"Error patching random_perspective: {e}")
    else:
        print(f"Warning: Could not find target pattern in {aug_path}")

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
    patch_random_perspective()

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
