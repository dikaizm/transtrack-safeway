import os
import sys
import yaml
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import torch

# Set paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# --- Configuration ---
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'road-damage-detection-yolov8-merged')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
MODEL_WEIGHTS = 'yolov8l-seg.pt'  # Pretrained weights

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print(f"Checking for dataset at: {DATASET_DIR}")
    
    if not os.path.exists(YAML_PATH):
        print(f"Error: data.yaml not found at {YAML_PATH}")
        print("Please ensure the dataset is in YOLO format and contains data.yaml.")
        return

    print(f"Found dataset config: {YAML_PATH}")

    # Train Model
    print("Starting YOLOv8 Training...")
    
    # Initialize YOLOv8 Model
    model = YOLO(MODEL_WEIGHTS)
    
    # Check device
    device = '0' if torch.cuda.is_available() else 'cpu'
        
    print(f"Using device: {device}")

    # Generate timestamped folder name: {modelname}-yyyymmdd-hhmm
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    model_name = MODEL_WEIGHTS.replace('.pt', '')  # e.g., 'yolov8s-seg'
    folder_name = f"{model_name}-{timestamp}"
    
    print(f"Model will be saved to: {os.path.join(MODELS_DIR, folder_name)}")

    # Train
    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=MODELS_DIR,
        name=folder_name,
        device=device,
        plots=True,
        workers=0,  # Disable multiprocessing to avoid DataLoader errors
        cache=False,  # Disable caching to avoid corrupted data
        patience=40
    )
    
    print(f"Training complete. Results saved to {results.save_dir}")

if __name__ == "__main__":
    main()
