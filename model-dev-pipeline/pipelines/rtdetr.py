import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import mlflow
import yaml
from ultralytics import RTDETR

from pipelines.base import BasePipeline
from pipelines.registry import register_pipeline
from utils.gdrive import (
    build_service,
    get_run_weights_folder,
    get_run_visuals_folder,
    upload_and_share,
)
from utils.mlflow_helper import (
    log_artifacts_from_dir,
    log_config,
    log_metrics_per_condition,
    setup_mlflow,
    tag_gdrive_link,
)
from utils.preprocess import preprocess_frame


@register_pipeline("rtdetr")
class RTDETRPipeline(BasePipeline):
    """
    Training and evaluation pipeline for Ultralytics RT-DETR models.
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
            "model_type": "rtdetr",
            "task": "detect",
            "weights": self.model_cfg.get("weights", ""),
        }

        try:
            from ultralytics import settings as yolo_settings
            yolo_settings.update({"mlflow": False})
        except Exception:
            pass

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            log_config(self.config)

            log_file = Path(tempfile.mktemp(suffix=".log", prefix="train_rtdetr_"))

            with self._tee_log(log_file):
                print(f"Run ID   : {run.info.run_id}")
                print(f"Weights  : {self.model_cfg.get('weights')}")
                print(f"Epochs   : {self.train_cfg.get('epochs', 100)}")
                print(f"Batch    : {self.train_cfg.get('batch', 8)}")
                print(f"Imgsz    : {self.train_cfg.get('imgsz', 640)}")
                print(f"Classes  : {self.data_cfg.get('classes', [])}")

                print("\nPreprocessing training images...")
                self._preprocess_dataset_images(self.data_cfg["train"])
                self._preprocess_dataset_images(self.data_cfg["val"])

                dataset_yaml = self._write_dataset_yaml()
                model = RTDETR(self.model_cfg["weights"])

                self._register_callbacks(model)

                results = model.train(
                    data=dataset_yaml,
                    epochs=self.train_cfg.get("epochs", 100),
                    batch=self.train_cfg.get("batch", 8),
                    imgsz=self.train_cfg.get("imgsz", 640),
                    lr0=self.train_cfg.get("lr0", 0.0001),
                    lrf=self.train_cfg.get("lrf", 0.01),
                    weight_decay=self.train_cfg.get("weight_decay", 0.0001),
                    warmup_epochs=self.train_cfg.get("warmup_epochs", 3),
                    patience=self.train_cfg.get("patience", 20),
                    save_period=self.train_cfg.get("save_period", 10),
                    workers=self.train_cfg.get("workers", 4),
                    device=self.train_cfg.get("device", 0),
                    verbose=True,
                    # Augmentation
                    hsv_h=self.aug_cfg.get("hsv_h", 0.015),
                    hsv_s=self.aug_cfg.get("hsv_s", 0.7),
                    hsv_v=self.aug_cfg.get("hsv_v", 0.4),
                    translate=self.aug_cfg.get("translate", 0.1),
                    scale=self.aug_cfg.get("scale", 0.5),
                    fliplr=self.aug_cfg.get("fliplr", 0.5),
                    mosaic=self.aug_cfg.get("mosaic", 0.0),  # RT-DETR: mosaic off by default
                    project="runs/rtdetr",
                    exist_ok=True,
                )

                save_dir = Path(results.save_dir)
                final_metrics = self._extract_final_metrics(results)

                mlflow.log_metrics(final_metrics)
                # Log training artifacts (plots, curves) — skip weights folder;
                # model weights are distributed via GDrive, not stored in MLflow.
                for item in save_dir.iterdir():
                    if item.name != "weights":
                        if item.is_file():
                            mlflow.log_artifact(str(item), artifact_path="training_results")
                        elif item.is_dir():
                            log_artifacts_from_dir(item, artifact_path=f"training_results/{item.name}")

                best_weights = save_dir / "weights" / "best.pt"
                mlflow.set_tag("best_weights", str(best_weights))
                mlflow.set_tag("run_id", run.info.run_id)

                print(f"\nTraining complete. Run ID: {run.info.run_id}")
                print(f"Best weights: {best_weights}")

                if best_weights.exists():
                    try:
                        gdrive_run = run_name or run.info.run_id
                        svc        = build_service()
                        folder_id  = get_run_weights_folder(svc, gdrive_run, "det")
                        link       = upload_and_share(svc, best_weights, folder_id)
                        tag_gdrive_link(run.info.run_id, "gdrive_det_weights", link)
                        print(f"GDrive weights: {link}")
                    except Exception as e:
                        print(f"GDrive weights upload failed (non-fatal): {e}")

            if log_file.exists():
                mlflow.log_artifact(str(log_file), artifact_path="logs")
                log_file.unlink(missing_ok=True)

            final_metrics["_best_weights"] = str(best_weights)
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
            print(
                "Warning: no per-condition test dirs found. "
                "Falling back to aggregate 'all'."
            )
            conditions = ["all"]

        all_metrics: dict[str, dict] = {}
        classes = self.data_cfg.get("classes", [])
        tags = {
            "model_type": "rtdetr",
            "task": "detect",
            "model_path": str(model_path),
            "eval_conditions": ",".join(conditions),
        }

        with mlflow.start_run(run_name=run_name or f"eval-rtdetr-{model_path.stem}", tags=tags):
            mlflow.log_param("model_path", str(model_path))
            mlflow.log_param("conditions", ",".join(conditions))

            model = self.load_model(model_path)

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
                        print(f"Skipping condition '{condition}' — no images found.")
                        continue

                print(f"\nEvaluating on condition: {condition}")

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

                self._log_visual_samples(model, condition, data_path)

            if len(all_metrics) > 1:
                self._log_condition_summary(all_metrics, classes)

            print("\nEvaluation complete.")
            return all_metrics

    # ------------------------------------------------------------------ #
    #  Load model                                                          #
    # ------------------------------------------------------------------ #

    def load_model(self, model_path: str | Path) -> RTDETR:
        return RTDETR(str(model_path))

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _preprocess_dataset_images(self, data_dir: str) -> None:
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
        classes = self.data_cfg.get("classes", [])
        test_path = split_override or self.data_cfg.get("test", {}).get("all", "")

        dataset = {
            "path": "/",
            "train": str(Path(self.data_cfg.get("train", "data/train")).resolve()),
            "val":   str(Path(self.data_cfg.get("val",   "data/val")).resolve()),
            "test":  str(Path(test_path).resolve()) if test_path else "",
            "nc": len(classes),
            "names": classes,
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(dataset, tmp)
        tmp.close()
        return tmp.name

    @contextmanager
    def _tee_log(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", buffering=1)

        class _Tee:
            def __init__(self, *streams):
                self._streams = streams

            def write(self, data):
                for s in self._streams:
                    s.write(data)

            def flush(self):
                for s in self._streams:
                    s.flush()

            def isatty(self):
                return False

        orig_stdout = sys.stdout
        sys.stdout = _Tee(orig_stdout, log_fh)
        try:
            yield
        finally:
            sys.stdout = orig_stdout
            log_fh.close()

    def _register_callbacks(self, model: RTDETR) -> None:
        def on_train_epoch_end(trainer):
            epoch = trainer.epoch
            metrics: dict[str, float] = {}

            try:
                loss_items = trainer.label_loss_items(trainer.tloss, prefix="train")
                metrics.update({
                    k: float(v) for k, v in loss_items.items()
                    if isinstance(v, (int, float))
                })
            except Exception:
                pass

            try:
                for i, pg in enumerate(trainer.optimizer.param_groups):
                    metrics[f"lr/pg{i}"] = float(pg["lr"])
            except Exception:
                pass

            if metrics:
                mlflow.log_metrics(metrics, step=epoch)
                summary = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[Epoch {epoch+1}] {summary}")

        def on_val_end(validator):
            try:
                raw = validator.metrics.results_dict
            except AttributeError:
                return
            val_metrics = {
                f"val/{k}".replace("(", "").replace(")", ""): float(v)
                for k, v in raw.items()
                if isinstance(v, (int, float))
            }
            step = getattr(validator, "epoch", None)
            mlflow.log_metrics(val_metrics, step=step)
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"[Val] {metrics_str}")

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_val_end", on_val_end)

    def _extract_final_metrics(self, results) -> dict[str, float]:
        box = results.results_dict
        return {
            "final/precision": float(box.get("metrics/precision(B)", 0)),
            "final/recall":    float(box.get("metrics/recall(B)", 0)),
            "final/mAP50":     float(box.get("metrics/mAP50(B)", 0)),
            "final/mAP50-95":  float(box.get("metrics/mAP50-95(B)", 0)),
        }

    def _extract_eval_metrics(self, metrics, condition: str) -> dict[str, float]:
        classes = self.data_cfg.get("classes", [])
        box = metrics.box
        result = {
            "mAP50":     float(box.map50),
            "mAP50-95":  float(box.map),
            "precision": float(box.mp),
            "recall":    float(box.mr),
        }

        if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
            for idx, class_idx in enumerate(box.ap_class_index):
                if class_idx < len(classes):
                    class_name = classes[class_idx]
                    result[f"mAP50/{class_name}"]     = float(box.ap50[idx])
                    result[f"mAP50-95/{class_name}"]  = float(box.ap[idx])

        return result

    def _log_condition_summary(
        self,
        all_metrics: dict[str, dict],
        classes: list[str],
    ) -> None:
        conditions = list(all_metrics.keys())
        col_w = 12

        header  = f"{'Class':<22}" + "".join(f"{c:>{col_w}}" for c in conditions)
        divider = "-" * len(header)
        rows = [header, divider]

        agg_row = f"{'mAP50 (all)':<22}"
        for cond in conditions:
            val = all_metrics[cond].get("mAP50", 0.0)
            agg_row += f"{val:>{col_w}.4f}"
        rows.append(agg_row)
        rows.append(divider)

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

        import tempfile as _tmp
        with _tmp.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="condition_summary_"
        ) as f:
            f.write("CONDITION SUMMARY (mAP50)\n\n")
            f.write(summary)
            tmp_path = f.name

        mlflow.log_artifact(tmp_path, artifact_path="evaluation")

        for cond in conditions:
            mlflow.log_metric(f"summary/mAP50_{cond}", all_metrics[cond].get("mAP50", 0.0))

    def _log_visual_samples(
        self,
        model: RTDETR,
        condition: str,
        data_path: str,
        n_samples: int = 30,
        fps: int = 3,
    ) -> None:
        images_dir = Path(data_path) / "images"
        if not images_dir.exists():
            return

        image_paths = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        )
        if not image_paths:
            return

        step = max(1, len(image_paths) // n_samples)
        samples = image_paths[::step][:n_samples]

        print(f"  Generating visual video for condition '{condition}': {len(samples)} frames @ {fps}fps")

        imgsz  = self.train_cfg.get("imgsz", 640)
        device = self.train_cfg.get("device", 0)

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"visuals_{condition}_"))
        try:
            results = model.predict(
                source=[str(p) for p in samples],
                imgsz=imgsz,
                device=device,
                verbose=False,
                conf=0.25,
            )

            frames = []
            for result in results:
                annotated = result.plot(labels=True, conf=True, boxes=True, line_width=2)
                label = f"RTDETR | {condition}"
                cv2.rectangle(annotated, (0, 0), (len(label) * 11 + 10, 30), (0, 0, 0), -1)
                cv2.putText(annotated, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
                frames.append(annotated)

            if frames:
                h, w = frames[0].shape[:2]
                out_path = tmp_dir / f"rtdetr_{condition}.mp4"
                for fourcc_str in ("avc1", "mp4v"):
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                    if writer.isOpened():
                        break
                for frame in frames:
                    writer.write(frame)
                writer.release()

                mlflow.log_artifact(str(out_path), artifact_path=f"visuals/{condition}")
                duration = len(frames) / fps
                print(f"  Video logged to MLflow: visuals/{condition}/{out_path.name} (~{duration:.1f}s)")

                try:
                    active = mlflow.active_run()
                    if active:
                        svc       = build_service()
                        run_label = active.info.run_name or active.info.run_id
                        folder_id = get_run_visuals_folder(svc, run_label)
                        link      = upload_and_share(svc, out_path, folder_id)
                        tag_gdrive_link(active.info.run_id, f"gdrive_vis_det_{condition}", link)
                        print(f"  GDrive video: {link}")
                except Exception as e:
                    print(f"  GDrive video upload failed (non-fatal): {e}")

        except Exception as e:
            print(f"  Could not generate visual video for {condition}: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
