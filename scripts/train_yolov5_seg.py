import os
import shutil
import json
import yaml
import numpy as np
import subprocess
import sys
from datetime import datetime

# Set paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# --- Configuration ---
YOLO_DIR = os.path.join(PROJECT_ROOT, 'yolov5')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'yolo_seg_dataset')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
MODEL_WEIGHTS = 'yolov5s-seg.pt'  # Pretrained weights

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Clone YOLOv5 Repository
def setup_yolov5():
    if not os.path.exists(YOLO_DIR):
        print(f"Cloning YOLOv5 repo to {YOLO_DIR}...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5", YOLO_DIR], check=True)
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-qr", os.path.join(YOLO_DIR, "requirements.txt")], check=True)
    else:
        print(f"YOLOv5 repo already exists at {YOLO_DIR}")

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
        "--project", MODELS_DIR
    ]

    print(f"Running command: {' '.join(args)}")
    
    # Run training
    subprocess.run(args, check=True)

if __name__ == "__main__":
    main()
