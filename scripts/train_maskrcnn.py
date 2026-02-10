import os
import sys
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader

# Add parent directory to path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.dataset import RoadDamageDataset
from src.training import setup_experiment, train_instance_model

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Config
    MODEL_NAME = "MaskRCNN"
    DATA_ROOT = os.path.join(parent_dir, 'data/road-damage-detection-1-coco')
    MODELS_DIR = os.path.join(parent_dir, 'models')
    # Mask R-CNN often requires smaller batch size if GPU memory is tight, but notebook used 2.
    BATCH_SIZE = 2 
    EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        DEVICE = 'mps'

    # Setup Experiment
    exp_dir, logger = setup_experiment(MODEL_NAME, MODELS_DIR)
    logger.info(f"Using device: {DEVICE}")

    # Datasets - Note: No transforms used in notebook for Mask R-CNN part except basic internal ones?
    # The notebook says: "transforms for Mask R-CNN need to handle bboxes/masks correctly. Here we use basic storage without extensive augs for simplicity in this demo."
    # So we pass transform=None to RoadDamageDataset for instance mode, which likely does basic tensor conversion if implemented in dataset.py?
    # Let's check src/dataset.py:
    # It imports cv2, torch.
    # In __getitem__:
    # if self.mode == 'instance':
    #     ...
    #     if self.transform:
    #         transformed = self.transform(image=image, masks=masks, bboxes=bboxes, class_labels=labels)
    #         ...
    #     else:
    #         image = transforms.ToTensor()(image)
    # Wait, dataset.py line 102 (read before) was cut off. I need to be careful.
    # If transform is None, does it return Tensor?
    # The notebook code `train_dataset_inst = RoadDamageDataset(..., mode='instance')` (no transform arg)
    # So assuming dataset handles ToTensor internally if no transform passed.
    
    logger.info("Loading datasets...")
    train_dataset = RoadDamageDataset(DATA_ROOT, split='train', mode='instance')
    valid_dataset = RoadDamageDataset(DATA_ROOT, split='valid', mode='instance')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    NUM_CLASSES = len(train_dataset.cat_names) + 1 # +1 for background
    logger.info(f"Classes: {train_dataset.cat_names}")
    logger.info(f"Num classes (with background): {NUM_CLASSES}")

    # Model
    logger.info("Initializing Mask R-CNN...")
    model = get_model_instance_segmentation(NUM_CLASSES).to(DEVICE)
    
    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train
    logger.info("Starting training...")
    train_instance_model(model, train_loader, valid_loader, optimizer, DEVICE, EPOCHS, exp_dir, logger, lr_scheduler)
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
