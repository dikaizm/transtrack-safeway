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
BATCH_SIZE = 8
IMG_SIZE = 640
MODEL_WEIGHTS = 'yolov8s-seg.pt'  # Pretrained weights

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


def _patch_format_call():
    """
    Monkey-patch ultralytics Format.__call__ to fix IndexError when
    augmentations produce samples with zero valid instances but non-empty
    segments. The bug is at augment.py line ~2067:
      sem_masks = cls_tensor[masks[0].long() - 1]
    which fails when cls_tensor is empty.
    """
    from ultralytics.data.augment import Format

    _original_call = Format.__call__

    def _patched_call(self, labels):
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
                cls_tensor = torch.from_numpy(cls.squeeze(1))
                # FIX: guard against empty cls_tensor after segment formatting
                if cls_tensor.numel() == 0:
                    masks = torch.zeros(
                        1 if self.mask_overlap else 0,
                        img.shape[0] // self.mask_ratio,
                        img.shape[1] // self.mask_ratio,
                    )
                    sem_masks = torch.zeros(
                        img.shape[0] // self.mask_ratio,
                        img.shape[1] // self.mask_ratio,
                    )
                    nl = 0
                elif self.mask_overlap:
                    # Safe indexing: clamp indices so background (0-1=-1) maps to 0
                    indices = masks[0].long() - 1
                    # Pad cls_tensor with a background class (0) at index 0
                    padded = torch.cat([torch.zeros(1, dtype=cls_tensor.dtype), cls_tensor])
                    sem_masks = padded[masks[0].long()]  # 0 stays 0 (bg), 1+ map to real classes
                else:
                    sem_masks = (masks * cls_tensor[:, None, None]).max(0).values
                    overlap = masks.sum(dim=0) > 1
                    if overlap.any():
                        weights = masks.sum(axis=(1, 2))
                        weighted_masks = masks * weights[:, None, None]
                        weighted_masks[masks == 0] = weights.max() + 1
                        smallest_idx = weighted_masks.argmin(dim=0)
                        sem_masks[overlap] = cls_tensor[smallest_idx[overlap]]
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl,
                    img.shape[0] // self.mask_ratio,
                    img.shape[1] // self.mask_ratio,
                )
                sem_masks = torch.zeros(
                    img.shape[0] // self.mask_ratio,
                    img.shape[1] // self.mask_ratio,
                )
            labels["masks"] = masks
            labels["sem_masks"] = sem_masks.float()

        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl, 1)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))

        if self.return_keypoint:
            labels["keypoints"] = (
                torch.empty(0, 3) if instances.keypoints is None else torch.from_numpy(instances.keypoints)
            )
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            from ultralytics.utils.ops import xyxyxyxy2xywhr
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments))
                if len(instances.segments)
                else torch.zeros((0, 5))
            )
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    Format.__call__ = _patched_call
    print("Applied monkey-patch for ultralytics Format.__call__ (sem_masks IndexError fix)")


def main():
    print(f"Checking for dataset at: {DATASET_DIR}")
    
    if not os.path.exists(YAML_PATH):
        print(f"Error: data.yaml not found at {YAML_PATH}")
        print("Please ensure the dataset is in YOLO format and contains data.yaml.")
        return

    print(f"Found dataset config: {YAML_PATH}")

    # Train Model
    print("Starting YOLOv8 Training...")
    
    # Apply bug fix for sem_masks IndexError in ultralytics augmentation
    _patch_format_call()
    
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
        overlap_mask=True,  # Allow overlapping masks
        mask_ratio=4  # Downsample ratio for masks
    )
    
    print(f"Training complete. Results saved to {results.save_dir}")

if __name__ == "__main__":
    main()
