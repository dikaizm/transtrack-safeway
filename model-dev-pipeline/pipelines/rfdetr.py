"""
RF-DETR pipeline — Apache 2.0 licensed.

Architecture: Roboflow Detection Transformer with DINOv2 backbone.
Source: https://github.com/roboflow/rf-detr
Paper: Real-time Object Detection with RF-DETR (Roboflow, 2024)

Dependencies:
    pip install rfdetr supervision
"""

from pathlib import Path

import mlflow
import torch

from pipelines.base import BasePipeline
from pipelines.registry import register_pipeline
from utils.mlflow_helper import (
    log_artifacts_from_dir,
    log_config,
    log_metrics_per_condition,
    setup_mlflow,
)

try:
    from rfdetr import RFDETRBase, RFDETRLarge
    from rfdetr.util.coco_utils import get_coco_api_from_dataset
    HAS_RFDETR = True
except ImportError:
    HAS_RFDETR = False


@register_pipeline("rfdetr")
class RFDETRPipeline(BasePipeline):
    """
    Training and evaluation pipeline for RF-DETR (Apache 2.0).
    Uses Roboflow's rfdetr package.

    Expects dataset in COCO format:
        data_dir/
          train/  images/ + annotations.json
          valid/  images/ + annotations.json
          test/   images/ + annotations.json   (or day/ wet/ night/ subfolders)
    """

    def train(self, run_name: str | None = None) -> dict:
        if not HAS_RFDETR:
            raise ImportError("rfdetr not installed. Run: pip install rfdetr")

        setup_mlflow(
            self.mlflow_cfg["tracking_uri"],
            self.mlflow_cfg["experiment_name"],
        )

        tags = {
            "model_type": "rfdetr",
            "task": "detect",
            "size": self.model_cfg.get("size", "base"),
            "license": "Apache-2.0",
        }

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            log_config(self.config)

            model = self._build_model()
            output_dir = Path(self.train_cfg.get("output_dir", "runs/rfdetr"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # RF-DETR callback for per-epoch MLflow logging
            best_map = {"value": 0.0}

            def on_epoch_end(metrics: dict, epoch: int):
                mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                    step=epoch,
                )
                current_map = float(metrics.get("mAP50", 0.0))
                if current_map > best_map["value"]:
                    best_map["value"] = current_map
                    mlflow.set_tag("best_epoch", epoch)

            model.train(
                dataset_dir=str(Path(self.data_cfg["train"]).parent),
                epochs=self.train_cfg.get("epochs", 100),
                batch_size=self.train_cfg.get("batch", 8),
                lr=self.train_cfg.get("lr", 1e-4),
                weight_decay=self.train_cfg.get("weight_decay", 1e-4),
                warmup_steps=self.train_cfg.get("warmup_steps", 500),
                grad_accum_steps=1,
                output_dir=str(output_dir),
                callbacks=[on_epoch_end],
            )

            best_weights = output_dir / "checkpoint_best_total.pth"
            if best_weights.exists():
                mlflow.log_artifact(str(best_weights), artifact_path="weights")

            mlflow.log_metric("best/mAP50", best_map["value"])
            log_artifacts_from_dir(output_dir, artifact_path="training_results")

            print(f"\nTraining complete. Run ID: {run.info.run_id}")
            print(f"Best mAP50: {best_map['value']:.4f}")

            return {"best/mAP50": best_map["value"]}

    def evaluate(
        self,
        model_path: str | Path,
        conditions: list[str] | None = None,
        run_name: str | None = None,
    ) -> dict:
        if not HAS_RFDETR:
            raise ImportError("rfdetr not installed. Run: pip install rfdetr")

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
            "model_type": "rfdetr",
            "task": "detect",
            "license": "Apache-2.0",
            "model_path": str(model_path),
        }

        with mlflow.start_run(run_name=run_name or f"eval-rfdetr-{model_path.stem}", tags=tags):
            mlflow.log_param("model_path", str(model_path))

            model = self.load_model(model_path)

            for condition in conditions:
                data_path = test_cfg.get(condition)
                if not data_path or not Path(data_path).exists():
                    print(f"Skipping '{condition}' — path not found: {data_path}")
                    continue

                print(f"\nEvaluating on condition: {condition}")
                metrics = model.val(dataset_dir=data_path)

                condition_metrics = self._extract_metrics(metrics)
                all_metrics[condition] = condition_metrics
                log_metrics_per_condition(condition_metrics, condition)

        return all_metrics

    def load_model(self, model_path: str | Path):
        model = self._build_model()
        checkpoint = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(checkpoint.get("model", checkpoint))
        return model

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_model(self):
        size = self.model_cfg.get("size", "base")
        nc = self.data_cfg.get("nc", 4)
        if size == "large":
            return RFDETRLarge(num_classes=nc)
        return RFDETRBase(num_classes=nc)

    def _extract_metrics(self, metrics) -> dict[str, float]:
        classes = self.data_cfg.get("classes", [])
        result: dict[str, float] = {}

        if isinstance(metrics, dict):
            result["mAP50"] = float(metrics.get("mAP50", 0.0))
            result["mAP50-95"] = float(metrics.get("mAP50-95", 0.0))
            result["precision"] = float(metrics.get("precision", 0.0))
            result["recall"] = float(metrics.get("recall", 0.0))

            for cls_name in classes:
                key = f"mAP50/{cls_name}"
                if key in metrics:
                    result[key] = float(metrics[key])

        return result
