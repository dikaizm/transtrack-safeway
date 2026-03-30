"""
RT-DETR pipeline — Apache 2.0 licensed.

Architecture: Real-Time Detection Transformer with ResNet backbone.
Paper: "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024)
Source: https://github.com/lyuwenyu/RT-DETR
HuggingFace: https://huggingface.co/PekingU/rtdetr_r50vd

Dependencies:
    pip install transformers torch torchvision albumentations pycocotools torchmetrics
"""

import json
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from pipelines.base import BasePipeline
from pipelines.registry import register_pipeline
from utils.mlflow_helper import (
    log_artifacts_from_dir,
    log_config,
    log_metrics_per_condition,
    setup_mlflow,
)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #

class COCODetectionDataset(Dataset):
    """
    COCO-format dataset compatible with RT-DETR image processor.
    Expects:
        data_dir/
          images/
          annotations.json   (COCO format)
    """

    def __init__(
        self,
        data_dir: str,
        processor: RTDetrImageProcessor,
        aug_cfg: dict | None = None,
        is_train: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.is_train = is_train

        ann_file = self.data_dir / "annotations.json"
        with open(ann_file) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = [img["id"] for img in coco["images"]]

        # Group annotations by image id
        self.annotations: dict[int, list] = {id_: [] for id_ in self.image_ids}
        for ann in coco.get("annotations", []):
            self.annotations[ann["image_id"]].append(ann)

        self.transform = self._build_transform(aug_cfg or {}) if is_train else None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = self.data_dir / "images" / image_info["file_name"]

        if HAS_CV2:
            import cv2
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        else:
            from PIL import Image as PILImage
            image = np.array(PILImage.open(image_path).convert("RGB"))

        anns = self.annotations[image_id]
        boxes = [ann["bbox"] for ann in anns]           # [x, y, w, h] COCO format
        labels = [ann["category_id"] for ann in anns]

        if self.transform and HAS_ALBUMENTATIONS and boxes:
            h, w = image.shape[:2]
            # Albumentations expects [x_min, y_min, x_max, y_max]
            bboxes_xyxy = [
                [b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes
            ]
            transformed = self.transform(
                image=image,
                bboxes=bboxes_xyxy,
                class_labels=labels,
            )
            image = transformed["image"]
            boxes = [
                [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                for b in transformed["bboxes"]
            ]
            labels = list(transformed["class_labels"])

        target = {
            "image_id": image_id,
            "annotations": [
                {"bbox": b, "category_id": l, "iscrowd": 0, "area": b[2] * b[3]}
                for b, l in zip(boxes, labels)
            ],
        }

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels_enc = encoding["labels"][0]

        return pixel_values, labels_enc

    def _build_transform(self, aug_cfg: dict):
        if not HAS_ALBUMENTATIONS:
            return None
        transforms = [
            A.HorizontalFlip(p=aug_cfg.get("horizontal_flip", 0.5)),
            A.ColorJitter(
                brightness=aug_cfg.get("color_jitter", {}).get("brightness", 0.4),
                contrast=aug_cfg.get("color_jitter", {}).get("contrast", 0.4),
                saturation=aug_cfg.get("color_jitter", {}).get("saturation", 0.4),
                hue=aug_cfg.get("color_jitter", {}).get("hue", 0.1),
                p=0.5,
            ),
            A.Blur(blur_limit=aug_cfg.get("blur_limit", 3), p=0.1),
        ]
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.3,
            ),
        )


def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return pixel_values, labels


# ------------------------------------------------------------------ #
#  Pipeline                                                            #
# ------------------------------------------------------------------ #

@register_pipeline("rtdetr")
class RTDETRPipeline(BasePipeline):
    """
    Training and evaluation pipeline for RT-DETR (Apache 2.0).
    Uses HuggingFace Transformers implementation.
    """

    def train(self, run_name: str | None = None) -> dict:
        setup_mlflow(
            self.mlflow_cfg["tracking_uri"],
            self.mlflow_cfg["experiment_name"],
        )

        tags = {
            "model_type": "rtdetr",
            "task": "detect",
            "checkpoint": self.model_cfg.get("checkpoint", ""),
            "license": "Apache-2.0",
        }

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            log_config(self.config)

            classes = self.data_cfg["classes"]
            id2label = {i: c for i, c in enumerate(classes)}
            label2id = {c: i for i, c in enumerate(classes)}

            processor = RTDetrImageProcessor.from_pretrained(
                self.model_cfg["checkpoint"]
            )
            model = RTDetrForObjectDetection.from_pretrained(
                self.model_cfg["checkpoint"],
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

            train_ds = COCODetectionDataset(
                self.data_cfg["train"], processor,
                aug_cfg=self.aug_cfg, is_train=True,
            )
            val_ds = COCODetectionDataset(
                self.data_cfg["val"], processor, is_train=False,
            )

            train_loader = DataLoader(
                train_ds, batch_size=self.train_cfg.get("batch", 8),
                shuffle=True, collate_fn=collate_fn,
                num_workers=self.train_cfg.get("workers", 4),
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.train_cfg.get("batch", 8),
                shuffle=False, collate_fn=collate_fn,
                num_workers=self.train_cfg.get("workers", 4),
            )

            device = torch.device(
                self.train_cfg.get("device", "cuda")
                if torch.cuda.is_available() else "cpu"
            )
            model.to(device)

            optimizer = AdamW(
                [
                    {"params": model.model.backbone.parameters(),
                     "lr": self.train_cfg.get("lr_backbone", 1e-5)},
                    {"params": [p for n, p in model.named_parameters()
                                if "backbone" not in n],
                     "lr": self.train_cfg.get("lr", 1e-4)},
                ],
                weight_decay=self.train_cfg.get("weight_decay", 1e-4),
            )

            epochs = self.train_cfg.get("epochs", 100)
            scheduler = OneCycleLR(
                optimizer, max_lr=self.train_cfg.get("lr", 1e-4),
                steps_per_epoch=len(train_loader), epochs=epochs,
            )

            output_dir = Path(self.train_cfg.get("output_dir", "runs/rtdetr"))
            output_dir.mkdir(parents=True, exist_ok=True)

            best_map = 0.0
            patience = self.train_cfg.get("patience", 20)
            patience_counter = 0
            final_metrics = {}

            for epoch in range(epochs):
                # --- Train ---
                model.train()
                train_loss = 0.0
                for pixel_values, labels in train_loader:
                    pixel_values = pixel_values.to(device)
                    labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in labels]

                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.train_cfg.get("gradient_clip", 0.1),
                    )
                    optimizer.step()
                    scheduler.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # --- Validate ---
                val_metrics = self._run_val(model, val_loader, processor, device)
                current_map = val_metrics.get("mAP50", 0.0)

                epoch_metrics = {
                    "train/loss": avg_train_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }
                mlflow.log_metrics(epoch_metrics, step=epoch)

                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"loss={avg_train_loss:.4f} | "
                    f"mAP50={current_map:.4f}"
                )

                # --- Save best ---
                if current_map > best_map:
                    best_map = current_map
                    patience_counter = 0
                    model.save_pretrained(str(output_dir / "best"))
                    processor.save_pretrained(str(output_dir / "best"))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

                final_metrics = epoch_metrics

            mlflow.log_metrics({"best/mAP50": best_map})
            log_artifacts_from_dir(output_dir / "best", artifact_path="weights")

            print(f"\nTraining complete. Run ID: {run.info.run_id}")
            print(f"Best mAP50: {best_map:.4f}")
            return {**final_metrics, "best/mAP50": best_map}

    def evaluate(
        self,
        model_path: str | Path,
        conditions: list[str] | None = None,
        run_name: str | None = None,
    ) -> dict:
        setup_mlflow(
            self.mlflow_cfg["tracking_uri"],
            self.mlflow_cfg["experiment_name"],
        )

        model_path = Path(model_path)
        test_cfg = self.data_cfg.get("test", {})

        if conditions is None:
            conditions = [k for k, v in test_cfg.items() if Path(v).exists()]
        if not conditions:
            conditions = ["all"]

        all_metrics: dict[str, dict] = {}
        tags = {
            "model_type": "rtdetr",
            "task": "detect",
            "license": "Apache-2.0",
            "model_path": str(model_path),
        }

        with mlflow.start_run(run_name=run_name or f"eval-rtdetr-{model_path.name}", tags=tags):
            mlflow.log_param("model_path", str(model_path))
            model, processor = self.load_model(model_path)

            device = torch.device(
                self.train_cfg.get("device", "cuda")
                if torch.cuda.is_available() else "cpu"
            )
            model.to(device)

            for condition in conditions:
                data_path = test_cfg.get(condition)
                if not data_path or not Path(data_path).exists():
                    print(f"Skipping '{condition}' — path not found: {data_path}")
                    continue

                print(f"\nEvaluating on condition: {condition}")
                ds = COCODetectionDataset(data_path, processor, is_train=False)
                loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn,
                                    num_workers=self.train_cfg.get("workers", 4))

                metrics = self._run_val(model, loader, processor, device)
                all_metrics[condition] = metrics
                log_metrics_per_condition(metrics, condition)

        return all_metrics

    def load_model(self, model_path: str | Path):
        model_path = Path(model_path)
        processor = RTDetrImageProcessor.from_pretrained(str(model_path))
        model = RTDetrForObjectDetection.from_pretrained(str(model_path))
        return model, processor

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _run_val(self, model, loader, processor, device) -> dict[str, float]:
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")
        classes = self.data_cfg["classes"]

        with torch.no_grad():
            for pixel_values, labels in loader:
                pixel_values = pixel_values.to(device)

                outputs = model(pixel_values=pixel_values)

                # Post-process predictions
                orig_sizes = torch.stack(
                    [torch.tensor([pixel_values.shape[-2], pixel_values.shape[-1]])]
                    * pixel_values.shape[0]
                )
                results = processor.post_process_object_detection(
                    outputs,
                    threshold=0.0,   # use all predictions; torchmetrics handles threshold
                    target_sizes=orig_sizes.to(device),
                )

                preds = [
                    {
                        "boxes": r["boxes"].cpu(),
                        "scores": r["scores"].cpu(),
                        "labels": r["labels"].cpu(),
                    }
                    for r in results
                ]
                targets = [
                    {
                        "boxes": lbl["boxes"].cpu(),
                        "labels": lbl["class_labels"].cpu(),
                    }
                    for lbl in labels
                ]
                metric.update(preds, targets)

        result = metric.compute()

        metrics: dict[str, float] = {
            "mAP50":    float(result.get("map_50", 0.0)),
            "mAP50-95": float(result.get("map", 0.0)),
            "mAP_small": float(result.get("map_small", 0.0)),
        }

        # Per-class mAP
        per_class = result.get("map_per_class", [])
        if per_class is not None and len(per_class) == len(classes):
            for i, cls_name in enumerate(classes):
                metrics[f"mAP50-95/{cls_name}"] = float(per_class[i])

        return metrics
