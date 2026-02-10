import os
import sys
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# Add parent directory to path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.dataset import RoadDamageDataset
from src.training import setup_experiment, train_semantic_model

def main():
    # Config
    MODEL_NAME = "DeepLabV3Plus"
    DATA_ROOT = os.path.join(parent_dir, 'data/road-damage-detection-coco')
    MODELS_DIR = os.path.join(parent_dir, 'models')
    BATCH_SIZE = 4
    LR = 0.001
    EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        DEVICE = 'mps'

    # Setup Experiment
    exp_dir, logger = setup_experiment(MODEL_NAME, MODELS_DIR)
    logger.info(f"Using device: {DEVICE}")

    # Transforms
    train_transform = A.Compose([
        A.Resize(320, 320),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(320, 320),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Datasets
    logger.info("Loading datasets...")
    train_dataset = RoadDamageDataset(DATA_ROOT, split='train', mode='semantic', transform=train_transform)
    valid_dataset = RoadDamageDataset(DATA_ROOT, split='valid', mode='semantic', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    NUM_CLASSES = len(train_dataset.cat_names) + 1 # +1 for background
    logger.info(f"Classes: {train_dataset.cat_names}")
    logger.info(f"Num classes (with background): {NUM_CLASSES}")

    # Model
    logger.info("Initializing DeepLabV3+ (ResNet34)...")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=NUM_CLASSES
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    logger.info("Starting training...")
    train_semantic_model(model, train_loader, valid_loader, criterion, optimizer, DEVICE, EPOCHS, exp_dir, logger)
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
