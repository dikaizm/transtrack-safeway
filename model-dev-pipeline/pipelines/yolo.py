import shutil
import tempfile
from pathlib import Path

import cv2
import mlflow
import yaml
from ultralytics import YOLO

from pipelines.base import BasePipeline
from pipelines.registry import register_pipeline
from utils.mlflow_helper import (
    log_artifacts_from_dir,
    log_config,
    log_metrics_per_condition,
    setup_mlflow,
)
from utils.preprocess import preprocess_frame


@register_pipeline("yolo")
class YOLOPipeline(BasePipeline):
    """
    Training and evaluation pipeline for Ultralytics YOLO models.
    Supports both detection (yolov8m) and segmentation (yolov8n-seg) tasks.
    """

    # ------------------------------------------------------------------ #
    #  Train                                                               #
    # ------------------------------------------------------------------ #

    def train(self, run_name: str | None = None) -> dict:
        setup_mlflow(
            self.mlflow_cfg["tracking_uri"],
            self.mlflow_cfg["experiment_name"],
        )

        tags = {
            "model_type": "yolo",
            "task": self.model_cfg.get("task", "detect"),
            "weights": self.model_cfg.get("weights", ""),
        }

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            log_config(self.config)

            # Apply condition-aware preprocessing to train/val images
            # Night → CLAHE clip=4.0, Dusty → CLAHE clip=2.0, Day → no-op
            print("Preprocessing training images...")
            self._preprocess_dataset_images(self.data_cfg["train"])
            self._preprocess_dataset_images(self.data_cfg["val"])

            dataset_yaml = self._write_dataset_yaml()
            model = YOLO(self.model_cfg["weights"])

            # Register MLflow callbacks for per-epoch metric logging
            self._register_callbacks(model)

            results = model.train(
                data=dataset_yaml,
                epochs=self.train_cfg.get("epochs", 100),
                batch=self.train_cfg.get("batch", 16),
                imgsz=self.train_cfg.get("imgsz", 640),
                lr0=self.train_cfg.get("lr0", 0.01),
                lrf=self.train_cfg.get("lrf", 0.01),
                momentum=self.train_cfg.get("momentum", 0.937),
                weight_decay=self.train_cfg.get("weight_decay", 0.0005),
                warmup_epochs=self.train_cfg.get("warmup_epochs", 3),
                patience=self.train_cfg.get("patience", 20),
                save_period=self.train_cfg.get("save_period", 10),
                workers=self.train_cfg.get("workers", 8),
                device=self.train_cfg.get("device", 0),
                # Augmentation
                hsv_h=self.aug_cfg.get("hsv_h", 0.015),
                hsv_s=self.aug_cfg.get("hsv_s", 0.7),
                hsv_v=self.aug_cfg.get("hsv_v", 0.4),
                degrees=self.aug_cfg.get("degrees", 0.0),
                translate=self.aug_cfg.get("translate", 0.1),
                scale=self.aug_cfg.get("scale", 0.5),
                fliplr=self.aug_cfg.get("fliplr", 0.5),
                mosaic=self.aug_cfg.get("mosaic", 1.0),
                blur=self.aug_cfg.get("blur", 0.01),
                project="runs/train",
                exist_ok=True,
            )

            save_dir = Path(results.save_dir)
            final_metrics = self._extract_final_metrics(results)

            mlflow.log_metrics(final_metrics)
            log_artifacts_from_dir(save_dir, artifact_path="training_results")

            best_weights = save_dir / "weights" / "best.pt"
            if best_weights.exists():
                mlflow.log_artifact(str(best_weights), artifact_path="weights")

            mlflow.set_tag("best_weights", str(best_weights))
            mlflow.set_tag("run_id", run.info.run_id)

            print(f"\nTraining complete. Run ID: {run.info.run_id}")
            print(f"Best weights: {best_weights}")
            return final_metrics

    # ------------------------------------------------------------------ #
    #  Evaluate                                                            #
    # ------------------------------------------------------------------ #

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

        # Default: always evaluate per condition, never just aggregate
        if conditions is None:
            conditions = [
                k for k in ["day", "wet", "night"]
                if Path(test_cfg.get(k, "")).exists()
            ]
        if not conditions:
            # Fallback to "all" only if no condition subdirs exist yet
            print(
                "Warning: no per-condition test dirs found "
                "(data/test/day, data/test/wet, data/test/night). "
                "Falling back to aggregate 'all'. "
                "Split your test set by condition for meaningful evaluation."
            )
            conditions = ["all"]

        all_metrics: dict[str, dict] = {}
        classes = self.data_cfg.get("classes", [])
        tags = {
            "model_type": "yolo",
            "task": self.model_cfg.get("task", "detect"),
            "model_path": str(model_path),
            "eval_conditions": ",".join(conditions),
        }

        with mlflow.start_run(run_name=run_name or f"eval-{model_path.stem}", tags=tags):
            mlflow.log_param("model_path", str(model_path))
            mlflow.log_param("conditions", ",".join(conditions))

            model = self.load_model(model_path)

            for condition in conditions:
                data_path = test_cfg.get(condition)
                if not data_path or not Path(data_path).exists():
                    print(f"Skipping condition '{condition}' — path not found: {data_path}")
                    continue

                print(f"\nEvaluating on condition: {condition}")

                # Apply same preprocessing used at training/inference
                # (CLAHE for night, light CLAHE for dusty, no-op for day)
                self._preprocess_dataset_images(data_path)

                dataset_yaml = self._write_dataset_yaml(split_override=data_path)

                metrics = model.val(
                    data=dataset_yaml,
                    split="test",
                    imgsz=self.train_cfg.get("imgsz", 640),
                    device=self.train_cfg.get("device", 0),
                )

                condition_metrics = self._extract_eval_metrics(metrics, condition)
                all_metrics[condition] = condition_metrics
                log_metrics_per_condition(condition_metrics, condition)

                if hasattr(metrics, "confusion_matrix"):
                    self._log_confusion_matrix(metrics, condition)

            # Log and print per-condition comparison summary
            if len(all_metrics) > 1:
                self._log_condition_summary(all_metrics, classes)

            print("\nEvaluation complete.")
            return all_metrics

    # ------------------------------------------------------------------ #
    #  Load model                                                          #
    # ------------------------------------------------------------------ #

    def load_model(self, model_path: str | Path) -> YOLO:
        return YOLO(str(model_path))

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _preprocess_dataset_images(self, data_dir: str) -> None:
        """
        Apply condition-aware preprocessing (CLAHE for night/dusty) to all
        images in data_dir in-place before training.

        Skips images that are already in normal daytime condition.
        The same preprocess_frame() function is used at inference time,
        ensuring train/inference consistency.
        """
        images_dir = Path(data_dir) / "images"
        if not images_dir.exists():
            return

        image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        counts = {"night": 0, "dusty": 0, "day": 0}

        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            processed, condition = preprocess_frame(frame)
            counts[condition] += 1
            if condition != "day":
                cv2.imwrite(str(path), processed)

        print(
            f"  Preprocessing {data_dir}: "
            f"{counts['day']} day | {counts['night']} night | {counts['dusty']} dusty"
        )

    def _write_dataset_yaml(self, split_override: str | None = None) -> str:
        """Write a temporary YOLO-format dataset.yaml and return its path.
        Uses absolute paths to avoid cwd-dependent resolution issues.
        """
        classes = self.data_cfg.get("classes", [])
        test_path = split_override or self.data_cfg.get("test", {}).get("all", "")

        dataset = {
            "path": "/",   # absolute anchor — all splits below are absolute too
            "train": str(Path(self.data_cfg.get("train", "data/train")).resolve()),
            "val":   str(Path(self.data_cfg.get("val",   "data/val")).resolve()),
            "test":  str(Path(test_path).resolve()) if test_path else "",
            "nc": len(classes),
            "names": classes,
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(dataset, tmp)
        tmp.close()
        return tmp.name

    def _register_callbacks(self, model: YOLO) -> None:
        """Register MLflow logging callbacks on the YOLO trainer."""

        def on_train_epoch_end(trainer):
            metrics = {
                k: float(v)
                for k, v in trainer.metrics.items()
                if isinstance(v, (int, float))
            }
            mlflow.log_metrics(metrics, step=trainer.epoch)

        def on_val_end(trainer):
            metrics = {
                f"val/{k}": float(v)
                for k, v in trainer.metrics.items()
                if isinstance(v, (int, float))
            }
            mlflow.log_metrics(metrics, step=trainer.epoch)

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_val_end", on_val_end)

    def _extract_final_metrics(self, results) -> dict[str, float]:
        """Extract scalar metrics from YOLO training results."""
        metrics = {}
        task = self.model_cfg.get("task", "detect")

        if task == "detect":
            box = results.results_dict
            metrics.update({
                "final/precision":    float(box.get("metrics/precision(B)", 0)),
                "final/recall":       float(box.get("metrics/recall(B)", 0)),
                "final/mAP50":        float(box.get("metrics/mAP50(B)", 0)),
                "final/mAP50-95":     float(box.get("metrics/mAP50-95(B)", 0)),
            })
        elif task == "segment":
            metrics.update({
                "final/precision_box":  float(results.results_dict.get("metrics/precision(B)", 0)),
                "final/recall_box":     float(results.results_dict.get("metrics/recall(B)", 0)),
                "final/mAP50_box":      float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "final/mAP50_mask":     float(results.results_dict.get("metrics/mAP50(M)", 0)),
                "final/mAP50-95_mask":  float(results.results_dict.get("metrics/mAP50-95(M)", 0)),
            })

        return metrics

    def _extract_eval_metrics(self, metrics, condition: str) -> dict[str, float]:
        """Extract per-class and aggregate metrics from YOLO val results."""
        classes = self.data_cfg.get("classes", [])
        result = {}
        task = self.model_cfg.get("task", "detect")

        if task == "detect":
            box = metrics.box
            result["mAP50"] = float(box.map50)
            result["mAP50-95"] = float(box.map)
            result["precision"] = float(box.mp)
            result["recall"] = float(box.mr)

            # Per-class mAP50
            if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
                for idx, class_idx in enumerate(box.ap_class_index):
                    if class_idx < len(classes):
                        class_name = classes[class_idx]
                        result[f"mAP50/{class_name}"] = float(box.ap50[idx])
                        result[f"mAP50-95/{class_name}"] = float(box.ap[idx])

        elif task == "segment":
            seg = metrics.seg
            result["mAP50_mask"] = float(seg.map50)
            result["mAP50-95_mask"] = float(seg.map)

        return result

    def _log_condition_summary(
        self,
        all_metrics: dict[str, dict],
        classes: list[str],
    ) -> None:
        """
        Print and log a side-by-side mAP50 comparison table across conditions.
        Logged to MLflow as a text artifact: evaluation/condition_summary.txt
        Helps quickly spot which condition the model underperforms on.
        """
        conditions = list(all_metrics.keys())
        col_w = 12

        header  = f"{'Class':<22}" + "".join(f"{c:>{col_w}}" for c in conditions)
        divider = "-" * len(header)
        rows = [header, divider]

        # Aggregate mAP50 row
        agg_row = f"{'mAP50 (all)':<22}"
        for cond in conditions:
            val = all_metrics[cond].get("mAP50", 0.0)
            agg_row += f"{val:>{col_w}.4f}"
        rows.append(agg_row)
        rows.append(divider)

        # Per-class mAP50 rows
        for cls in classes:
            row = f"{cls:<22}"
            for cond in conditions:
                val = all_metrics[cond].get(f"mAP50/{cls}", 0.0)
                row += f"{val:>{col_w}.4f}"
            rows.append(row)

        summary = "\n".join(rows)
        print(f"\n{'='*len(divider)}")
        print("CONDITION SUMMARY (mAP50)")
        print(summary)
        print(f"{'='*len(divider)}\n")

        # Log as MLflow artifact
        import tempfile as _tmp
        with _tmp.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="condition_summary_"
        ) as f:
            f.write("CONDITION SUMMARY (mAP50)\n\n")
            f.write(summary)
            tmp_path = f.name

        mlflow.log_artifact(tmp_path, artifact_path="evaluation")

        # Also log per-condition aggregate as flat metrics for MLflow chart view
        for cond in conditions:
            mlflow.log_metric(f"summary/mAP50_{cond}", all_metrics[cond].get("mAP50", 0.0))

    def _log_confusion_matrix(self, metrics, condition: str) -> None:
        try:
            import matplotlib.pyplot as plt
            cm = metrics.confusion_matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            cm.plot(ax=ax)
            ax.set_title(f"Confusion Matrix — {condition}")
            path = f"/tmp/confusion_matrix_{condition}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(path, artifact_path=f"confusion_matrix/{condition}")
        except Exception as e:
            print(f"Could not log confusion matrix for {condition}: {e}")
