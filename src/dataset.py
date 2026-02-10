
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class RoadDamageDataset(Dataset):
    def __init__(self, root, split='train', mode='semantic', transform=None, classes=None):
        """
        Args:
            root (str): Metrics to the dataset root (e.g. data/road-damage-detection-1-coco).
            split (str): 'train', 'valid', or 'test'.
            mode (str): 'semantic' for UNet/FastSCNN/DeepLab, 'instance' for Mask R-CNN.
            transform (albumentations.Compose): Augmentations.
            classes (list): List of class names to include. If None, use all.
        """
        self.root = root
        self.split = split
        self.mode = mode
        self.transform = transform
        
        self.img_dir = os.path.join(root, split)
        self.ann_file = os.path.join(self.img_dir, '_annotations.coco.json')
        
        self.coco = COCO(self.ann_file)
        
        # Load categories
        self.cat_ids = self.coco.getCatIds()
        self.cats = self.coco.loadCats(self.cat_ids)
        self.cat_names = [cat['name'] for cat in self.cats]
        self.cat_id_map = {cat['id']: i+1 for i, cat in enumerate(self.cats)} # 0 is background
        
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_anns = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.img_dir, path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == 'semantic':
            # Create a single mask for semantic segmentation
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            for ann in coco_anns:
                cat_id = ann['category_id']
                pixel_value = self.cat_id_map[cat_id]
                
                # annToMask returns binary mask
                ann_mask = coco.annToMask(ann)
                mask[ann_mask > 0] = pixel_value
                
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                
            # Convert to Tensor (C, H, W) for image, (H, W) or (1, H, W) for mask
            # Typically SMP expects image as float and mask as long
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
            return image, mask
            
        elif self.mode == 'instance':
            # For Mask R-CNN
            boxes = []
            labels = []
            masks = []
            
            for ann in coco_anns:
                xmin, ymin, w, h = ann['bbox']
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                labels.append(ann['category_id'])
                masks.append(coco.annToMask(ann))
                
            if len(boxes) > 0:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            else:
                # Handle images with no annotations
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                masks = torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.uint8)
            
            image_id = torch.tensor([img_id])
            
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            
            # Note: Mask R-CNN transforms are tricky with boxes/masks. 
            # Usually handled by specific libraries or custom code.
            # Here we just convert image to tensor.
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
            return image, target

    def __len__(self):
        return len(self.ids)
