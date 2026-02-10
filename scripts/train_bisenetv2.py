import os
import sys
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# Add parent directory to path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.dataset import RoadDamageDataset
from src.bisenet_v2 import BiSeNetV2
from src.training import setup_experiment

def train_bisenet_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs, exp_dir, logger):
    """
    Custom training loop for BiSeNetV2 with auxiliary loss.
    """
    best_iou = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # BiSeNet V2 returns 5 outputs in training: logits, aux2, aux3, aux4, aux5_4
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
                aux_logits = outputs[1:]
                
                loss_main = criterion(logits, masks)
                loss_aux = sum([criterion(aux, masks) for aux in aux_logits])
                
                # BiSeNet paper recommends weighting aux losses. 
                # Typical weights are 1.0 for main and 1.0 for aux, or variations.
                loss = loss_main + loss_aux
            else:
                loss = criterion(outputs, masks)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        valid_loss = 0
        intersection = 0
        union = 0
        
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # In eval mode, BiSeNetV2 returns only logits
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                valid_loss += loss.item()
                
                # IOU Calculation
                preds = torch.argmax(outputs, dim=1)
                intersection += (preds & masks).float().sum().item()
                union += (preds | masks).float().sum().item()
                
        avg_valid_loss = valid_loss / len(valid_loader)
        iou = intersection / (union + 1e-6)
        logger.info(f"Valid Loss: {avg_valid_loss:.4f}, IOU: {iou:.4f}")
        
        # Save Last Model
        torch.save(model.state_dict(), os.path.join(exp_dir, "last_model.pth"))
        
        # Save Best Model
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            logger.info("Saved Best Model!")

def main():
    # Config
    MODEL_NAME = "BiSeNetV2"
    DATA_ROOT = os.path.join(parent_dir, 'data/road-damage-detection-coco')
    if not os.path.exists(DATA_ROOT):
         DATA_ROOT = os.path.join(parent_dir, 'data/road-damage-detection-coco')

    MODELS_DIR = os.path.join(parent_dir, 'models')
    BATCH_SIZE = 4 # BiSeNet might be able to handle larger batch sizes
    LR = 0.01 # SGD usually used for BiSeNet
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
    logger.info(f"Loading datasets from {DATA_ROOT}...")
    train_dataset = RoadDamageDataset(DATA_ROOT, split='train', mode='semantic', transform=train_transform)
    valid_dataset = RoadDamageDataset(DATA_ROOT, split='valid', mode='semantic', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    NUM_CLASSES = len(train_dataset.cat_names) + 1 # +1 for background
    logger.info(f"Classes: {train_dataset.cat_names}")
    logger.info(f"Num classes (with background): {NUM_CLASSES}")

    # Model
    logger.info("Initializing BiSeNetV2...")
    model = BiSeNetV2(num_classes=NUM_CLASSES).to(DEVICE)
    
    # BiSeNet usually uses SGD with momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR) # Option
    
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    logger.info("Starting training...")
    train_bisenet_model(model, train_loader, valid_loader, criterion, optimizer, DEVICE, EPOCHS, exp_dir, logger)
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
