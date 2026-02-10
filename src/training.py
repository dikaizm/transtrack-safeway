import os
import torch
import logging
import numpy as np
from datetime import datetime

def setup_experiment(model_name, models_dir='models'):
    """
    Sets up the experiment directory and logger.
    Returns:
        exp_dir (str): Path to the experiment directory.
        logger (logging.Logger): Configured logger.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    exp_dir = os.path.join(models_dir, f"{model_name}-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Logger
    log_file = os.path.join(exp_dir, 'training_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info(f"Experiment initialized for {model_name}")
    logger.info(f"Saving outputs to {exp_dir}")
    return exp_dir, logger

def train_semantic_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs, exp_dir, logger):
    """
    Training loop for semantic segmentation models (FastSCNN, UNet, DeepLab).
    """
    best_iou = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            # Handle models that return tuples (e.g. FastSCNN with aux output)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
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
                
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
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

def train_instance_model(model, train_loader, valid_loader, optimizer, device, epochs, exp_dir, logger, lr_scheduler=None):
    """
    Training loop for instance segmentation models (Mask R-CNN).
    Note: Validation for Mask R-CNN is complex (COCO eval), so we hereby track training loss 
    and save checkpoints.
    """
    min_train_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        steps = 0
        
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            steps += 1
            
        avg_train_loss = train_loss / steps if steps > 0 else 0
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        if lr_scheduler:
            lr_scheduler.step()
        
        # Save Last Model
        torch.save(model.state_dict(), os.path.join(exp_dir, "last_model.pth"))
        
        # Save Best Training Loss Model (as proxy for best)
        if avg_train_loss < min_train_loss:
            min_train_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            logger.info("Saved Best Model (Lowest Train Loss)!")
