"""
SegFormer-B0 segmentation pipeline — Apache 2.0 licensed.

Architecture : SegFormer with Mix Transformer (MiT-B0) encoder
Model        : nvidia/segformer-b0-finetuned-ade-512-512 (pretrained)
Source       : https://huggingface.co/nvidia/mit-b0
License      : Apache 2.0

Task         : Semantic segmentation — drive_area / off_road (2 classes)
Input format : YOLO polygon txt labels → converted to pixel masks on-the-fly
Metrics      : mIoU per class, mean pixel accuracy

Dependencies:
    pip install transformers torch torchvision
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from pipelines.base import BasePipeline
from pipelines.registry import register_pipeline
from utils.mlflow_helper import (
    log_config,
    log_metrics_per_condition,
    setup_mlflow,
    tag_gdrive_link,
)

IGNORE_INDEX = 255   # unlabeled pixels — excluded from loss and mIoU

# RGB overlay colors per class index (drive_area=green, off_road=red, fallback=blue)
CLASS_COLORS = [(0, 200, 0), (200, 0, 0), (0, 0, 200)]


# ------------------------------------------------------------------ #
#  Dataset — YOLO polygon txt → pixel semantic mask                   #
# ------------------------------------------------------------------ #

class YOLOSegDataset(Dataset):
    """
    Reads YOLO-format segmentation labels (polygon txt) and converts them
    to pixel-wise semantic masks for SegFormer training.

    Label format per txt line:
        class_idx x1 y1 x2 y2 ... xn yn   (coords normalized 0-1)

    Mask values:
        0..N-1  → class index
        255     → unlabeled background (ignored in loss)
    """

    def __init__(
        self,
        data_dir: str | Path,
        processor: SegformerImageProcessor,
        img_size: int = 640,
        is_train: bool = True,
        aug_cfg: dict | None = None,
    ):
        self.images_dir = Path(data_dir) / "images"
        self.labels_dir = Path(data_dir) / "labels"
        self.processor  = processor
        self.img_size   = img_size
        self.is_train   = is_train
        self.aug_cfg    = aug_cfg or {}

        self.image_paths = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png"))
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path   = self.image_paths[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H, W  = image.shape[:2]

        # Build semantic mask — start with IGNORE_INDEX for unlabeled pixels
        mask = np.full((H, W), IGNORE_INDEX, dtype=np.int32)
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                cls_idx = int(parts[0])
                coords  = list(map(float, parts[1:]))
                pts = np.array(
                    [[int(coords[i] * W), int(coords[i + 1] * H)]
                     for i in range(0, len(coords) - 1, 2)],
                    dtype=np.int32,
                )
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], cls_idx)

        # Resize to model input size (nearest for mask to preserve labels)
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask  = cv2.resize(mask, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)

        # Training augmentations
        if self.is_train:
            image, mask = self._augment(image, mask)

        encoding     = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels       = torch.tensor(mask, dtype=torch.long)
        return pixel_values, labels

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        # Horizontal flip
        if np.random.rand() < self.aug_cfg.get("fliplr", 0.5):
            image = cv2.flip(image, 1)
            mask  = cv2.flip(mask, 1)

        # Color jitter (image only, not mask)
        if np.random.rand() < 0.5:
            factor = 1.0 + (np.random.rand() - 0.5) * 0.4
            image  = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return image, mask


def seg_collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels       = torch.stack([item[1] for item in batch])
    return pixel_values, labels


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #

def compute_seg_metrics(
    preds: np.ndarray,    # (N, H, W) int — predicted class per pixel
    targets: np.ndarray,  # (N, H, W) int — ground truth
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, float]:
    """Compute mIoU and mean pixel accuracy, ignoring unlabeled pixels."""
    iou_per_class = []
    acc_per_class = []

    for cls in range(num_classes):
        valid   = targets != ignore_index
        pred_c  = (preds == cls) & valid
        true_c  = (targets == cls) & valid

        intersection = (pred_c & true_c).sum()
        union        = (pred_c | true_c).sum()
        iou = intersection / (union + 1e-6) if union > 0 else float("nan")

        correct = (pred_c & true_c).sum()
        total   = true_c.sum()
        acc = correct / (total + 1e-6) if total > 0 else float("nan")

        iou_per_class.append(float(iou))
        acc_per_class.append(float(acc))

    valid_iou = [v for v in iou_per_class if not np.isnan(v)]
    valid_acc = [v for v in acc_per_class if not np.isnan(v)]

    metrics = {
        "mIoU":     float(np.mean(valid_iou)) if valid_iou else 0.0,
        "pixel_acc": float(np.mean(valid_acc)) if valid_acc else 0.0,
    }
    return metrics


# ------------------------------------------------------------------ #
#  Pipeline                                                            #
# ------------------------------------------------------------------ #

@register_pipeline("segformer")
class SegFormerPipeline(BasePipeline):
    """
    Training and evaluation pipeline for SegFormer-B0 (Apache 2.0).
    Semantic segmentation: drive_area / off_road.
    """

    CHECKPOINT = "nvidia/mit-b0"

    def train(self, run_name: str | None = None) -> dict:
        setup_mlflow(
            self.mlflow_cfg["tracking_uri"],
            self.mlflow_cfg["experiment_name"],
        )

        classes   = self.data_cfg["classes"]
        nc        = len(classes)
        id2label  = {i: c for i, c in enumerate(classes)}
        label2id  = {c: i for i, c in enumerate(classes)}
        img_size  = self.train_cfg.get("imgsz", 640)
        epochs    = self.train_cfg.get("epochs", 100)
        patience  = self.train_cfg.get("patience", 20)
        batch     = self.train_cfg.get("batch", 8)
        checkpoint = self.model_cfg.get("checkpoint", self.CHECKPOINT)

        tags = {
            "model_type": "segformer",
            "task": "segment",
            "checkpoint": checkpoint,
            "license": "Apache-2.0",
        }

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            log_config(self.config)

            print(f"Run ID    : {run.info.run_id}")
            print(f"Model     : {checkpoint}")
            print(f"Classes   : {classes}")
            print(f"Epochs    : {epochs}  Batch: {batch}  ImgSz: {img_size}")

            processor = SegformerImageProcessor.from_pretrained(checkpoint)
            model     = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                num_labels=nc,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

            train_ds = YOLOSegDataset(
                self.data_cfg["train"], processor,
                img_size=img_size, is_train=True, aug_cfg=self.aug_cfg,
            )
            val_ds = YOLOSegDataset(
                self.data_cfg["val"], processor,
                img_size=img_size, is_train=False,
            )

            train_loader = DataLoader(
                train_ds, batch_size=batch, shuffle=True,
                collate_fn=seg_collate_fn,
                num_workers=self.train_cfg.get("workers", 4),
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch, shuffle=False,
                collate_fn=seg_collate_fn,
                num_workers=self.train_cfg.get("workers", 4),
            )

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model.to(device)

            weight_decay = float(self.train_cfg.get("weight_decay", 0.01))
            lr_backbone  = float(self.train_cfg.get("lr_backbone", 6e-5))
            lr_head      = float(self.train_cfg.get("lr", 6e-4))

            # Two separate optimizers avoids a PyTorch bug where lr is passed
            # as a list to _single_tensor_adam when param groups share device/dtype.
            optimizer_enc = AdamW(model.segformer.encoder.parameters(),
                                  lr=lr_backbone, weight_decay=weight_decay)
            optimizer_dec = AdamW(model.decode_head.parameters(),
                                  lr=lr_head, weight_decay=weight_decay)
            scheduler_enc = PolynomialLR(optimizer_enc, total_iters=epochs, power=0.9)
            scheduler_dec = PolynomialLR(optimizer_dec, total_iters=epochs, power=0.9)

            output_dir = Path(self.train_cfg.get("output_dir", "runs/segformer"))
            output_dir.mkdir(parents=True, exist_ok=True)

            best_miou     = 0.0
            patience_ctr  = 0
            final_metrics: dict = {}
            history: dict[str, list] = {"loss": [], "mIoU": [], "pixel_acc": []}

            for epoch in range(epochs):
                # ── Train ──────────────────────────────────────────────
                model.train()
                train_loss = 0.0
                for pixel_values, labels in train_loader:
                    pixel_values = pixel_values.to(device)
                    labels       = labels.to(device)

                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss    = outputs.loss

                    optimizer_enc.zero_grad()
                    optimizer_dec.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.train_cfg.get("gradient_clip", 1.0),
                    )
                    optimizer_enc.step()
                    optimizer_dec.step()
                    train_loss += loss.item()

                scheduler_enc.step()
                scheduler_dec.step()
                avg_loss = train_loss / len(train_loader)

                # ── Validate ───────────────────────────────────────────
                val_metrics, _ = self._run_val(model, val_loader, device, nc, img_size)
                current_miou = val_metrics.get("mIoU", 0.0)

                epoch_metrics = {
                    "train/loss": avg_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }
                mlflow.log_metrics(epoch_metrics, step=epoch)
                history["loss"].append(avg_loss)
                history["mIoU"].append(current_miou)
                history["pixel_acc"].append(val_metrics.get("pixel_acc", 0.0))

                print(
                    f"[Epoch {epoch+1:>3}/{epochs}] "
                    f"loss={avg_loss:.4f}  "
                    f"mIoU={current_miou:.4f}  "
                    f"pixel_acc={val_metrics.get('pixel_acc', 0):.4f}"
                )

                # ── Save best ──────────────────────────────────────────
                if current_miou > best_miou:
                    best_miou    = current_miou
                    patience_ctr = 0
                    model.save_pretrained(str(output_dir / "best"))
                    processor.save_pretrained(str(output_dir / "best"))
                    print(f"  ✓ New best mIoU: {best_miou:.4f}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                final_metrics = epoch_metrics

            mlflow.log_metrics({"best/mIoU": best_miou})
            mlflow.log_artifacts(str(output_dir / "best"), artifact_path="weights")
            mlflow.set_tag("best_weights", str(output_dir / "best"))

            # Training curves
            curve_path = output_dir / "training_curves.png"
            self._plot_training_curves(history, curve_path)
            mlflow.log_artifact(str(curve_path), artifact_path="training_results")

            # Val sample visualizations from best model
            best_model, _ = self.load_model(output_dir / "best")
            best_model.to(device)
            self._log_val_samples(
                best_model, val_ds, device, nc, img_size,
                classes, output_dir / "val_samples",
            )

            # GDrive upload
            best_dir = output_dir / "best"
            if best_dir.exists():
                try:
                    from utils.gdrive import build_service, get_run_weights_folder, upload_and_share
                    gdrive_run = run_name or run.info.run_id
                    svc        = build_service()
                    folder_id  = get_run_weights_folder(svc, gdrive_run, "seg-segformer")
                    # Zip the directory for upload
                    zip_path = Path(tempfile.mktemp(suffix=".zip"))
                    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(best_dir))
                    link = upload_and_share(svc, zip_path, folder_id)
                    tag_gdrive_link(run.info.run_id, "gdrive_seg_segformer_weights", link)
                    zip_path.unlink(missing_ok=True)
                    print(f"GDrive weights: {link}")
                except Exception as e:
                    print(f"GDrive upload failed (non-fatal): {e}")

            print(f"\nTraining complete. Run ID: {run.info.run_id}")
            print(f"Best mIoU: {best_miou:.4f}")
            print(f"Weights  : {output_dir / 'best'}")

            final_metrics["best/mIoU"] = best_miou
            final_metrics["_best_weights"] = str(output_dir / "best")
            return final_metrics

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
        test_cfg   = self.data_cfg.get("test", {})
        classes    = self.data_cfg["classes"]
        nc         = len(classes)
        img_size   = self.train_cfg.get("imgsz", 640)

        def _has_images(path: str) -> bool:
            p = Path(path) / "images" if path else None
            if p is None or not p.exists():
                return False
            return any(p.glob("*.*"))

        if conditions is None:
            conditions = [
                k for k in ["day", "wet", "night"]
                if _has_images(test_cfg.get(k, ""))
            ]
        if not conditions:
            conditions = ["all"]

        tags = {
            "model_type": "segformer",
            "task": "segment",
            "license": "Apache-2.0",
            "model_path": str(model_path),
        }

        all_metrics: dict[str, dict] = {}

        with mlflow.start_run(run_name=run_name or f"eval-segformer", tags=tags):
            mlflow.log_param("model_path", str(model_path))
            mlflow.log_param("conditions", ",".join(conditions))

            model, processor = self.load_model(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            for condition in conditions:
                data_path = test_cfg.get(condition)
                if not data_path or not _has_images(data_path):
                    fallback = test_cfg.get("all")
                    if fallback and _has_images(fallback):
                        print(
                            f"Warning: condition '{condition}' dir is empty — "
                            f"falling back to default test set: {fallback}"
                        )
                        data_path = fallback
                    else:
                        print(f"Skipping '{condition}' — no images found and no fallback available.")
                        continue

                print(f"\nEvaluating on condition: {condition}")
                ds = YOLOSegDataset(
                    data_path, processor, img_size=img_size, is_train=False,
                )
                loader = DataLoader(
                    ds, batch_size=4, shuffle=False,
                    collate_fn=seg_collate_fn,
                    num_workers=self.train_cfg.get("workers", 4),
                )

                metrics, cm = self._run_val(model, loader, device, nc, img_size)
                all_metrics[condition] = metrics
                log_metrics_per_condition(metrics, condition)

                cm_path = Path(tempfile.mkdtemp()) / f"confusion_matrix_{condition}.png"
                self._plot_confusion_matrix(cm, classes, condition, cm_path)
                mlflow.log_artifact(str(cm_path), artifact_path=f"confusion_matrix/{condition}")

                print(f"  mIoU={metrics['mIoU']:.4f}  pixel_acc={metrics['pixel_acc']:.4f}")

            # Per-class mIoU summary
            if len(all_metrics) > 1:
                self._log_condition_summary(all_metrics, classes)

            print("\nEvaluation complete.")
        return all_metrics

    def load_model(self, model_path: str | Path):
        model_path = Path(model_path)
        processor  = SegformerImageProcessor.from_pretrained(str(model_path))
        model      = SegformerForSemanticSegmentation.from_pretrained(str(model_path))
        return model, processor

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _run_val(
        self, model, loader, device, num_classes: int, img_size: int,
    ) -> tuple[dict[str, float], np.ndarray]:
        model.eval()
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for pixel_values, labels in loader:
                pixel_values = pixel_values.to(device)

                outputs = model(pixel_values=pixel_values)
                logits  = F.interpolate(
                    outputs.logits, size=(img_size, img_size),
                    mode="bilinear", align_corners=False,
                )
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(labels.numpy())

        all_preds   = np.concatenate(all_preds,   axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Pixel-level confusion matrix (rows=GT, cols=pred), ignoring IGNORE_INDEX
        valid   = all_targets != IGNORE_INDEX
        gt_flat = all_targets[valid].astype(np.int64)
        pd_flat = all_preds[valid].astype(np.int64)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        np.add.at(cm, (gt_flat, pd_flat), 1)

        return compute_seg_metrics(all_preds, all_targets, num_classes), cm

    def _plot_training_curves(self, history: dict, out_path: Path) -> None:
        epochs = range(1, len(history["loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, history["loss"], color="tab:blue")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["mIoU"],     label="mIoU",      color="tab:green")
        ax2.plot(epochs, history["pixel_acc"], label="Pixel Acc", color="tab:orange", linestyle="--")
        ax2.set_title("Validation Metrics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

    def _log_val_samples(
        self,
        model,
        val_ds: Dataset,
        device,
        num_classes: int,
        img_size: int,
        classes: list[str],
        out_dir: Path,
        n: int = 8,
    ) -> None:
        """Save side-by-side panels: original | GT mask | predicted mask."""
        out_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        indices = np.linspace(0, len(val_ds) - 1, min(n, len(val_ds)), dtype=int)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                pixel_values, gt_mask = val_ds[idx]
                outputs = model(pixel_values=pixel_values.unsqueeze(0).to(device))
                logits  = F.interpolate(
                    outputs.logits, size=(img_size, img_size),
                    mode="bilinear", align_corners=False,
                )
                pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                gt_np     = gt_mask.numpy()

                # Reconstruct RGB image from normalized tensor
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                img  = pixel_values.permute(1, 2, 0).numpy()
                img  = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
                # SegformerImageProcessor resizes to 512 internally; upsample to img_size
                if img.shape[0] != img_size or img.shape[1] != img_size:
                    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

                def _overlay(base, mask):
                    out = base.copy()
                    for cls_idx, color in enumerate(CLASS_COLORS[:num_classes]):
                        out[mask == cls_idx] = (
                            out[mask == cls_idx] * 0.5
                            + np.array(color, dtype=np.float32) * 0.5
                        ).astype(np.uint8)
                    return out

                gt_overlay   = _overlay(img, gt_np)
                pred_overlay = _overlay(img, pred_mask)
                panel = np.concatenate([img, gt_overlay, pred_overlay], axis=1)

                out_path = out_dir / f"sample_{i:02d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        mlflow.log_artifacts(str(out_dir), artifact_path="val_samples")

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        classes: list[str],
        condition: str,
        out_path: Path,
    ) -> None:
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"Confusion Matrix — {condition}")
        for r in range(len(classes)):
            for c in range(len(classes)):
                ax.text(c, r, f"{cm_norm[r, c]:.2f}",
                        ha="center", va="center",
                        color="white" if cm_norm[r, c] > 0.5 else "black",
                        fontsize=9)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

    def _log_condition_summary(
        self, all_metrics: dict[str, dict], classes: list[str],
    ) -> None:
        conditions = list(all_metrics.keys())
        col_w = 12
        header  = f"{'Metric':<22}" + "".join(f"{c:>{col_w}}" for c in conditions)
        divider = "-" * len(header)
        rows    = [header, divider]

        for metric_key in ["mIoU", "pixel_acc"]:
            row = f"{metric_key:<22}"
            for cond in conditions:
                val = all_metrics[cond].get(metric_key, 0.0)
                row += f"{val:>{col_w}.4f}"
            rows.append(row)

        summary = "\n".join(rows)
        print(f"\n{'='*len(divider)}")
        print("CONDITION SUMMARY — SegFormer")
        print(summary)
        print(f"{'='*len(divider)}\n")

        import tempfile as _tmp
        with _tmp.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="segformer_summary_"
        ) as f:
            f.write("CONDITION SUMMARY — SegFormer\n\n")
            f.write(summary)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="evaluation")

        for cond in conditions:
            mlflow.log_metric(f"summary/mIoU_{cond}", all_metrics[cond].get("mIoU", 0.0))
