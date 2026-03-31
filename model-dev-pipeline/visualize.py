"""
Visualization entry point — generate annotated sample images from a trained model.

Runs model.predict() on test images per condition and saves annotated results:
  - Segmentation : colored mask overlay (drive_area / off_road) + label
  - Detection    : bounding boxes + class label + confidence score

Results are saved to runs/visualize/{run_name}/{condition}/ and logged to MLflow.

Usage:
    python visualize.py --config config/yolo_segmentation.yaml --model runs/train/weights/best.pt
    python visualize.py --config config/yolo_detection.yaml   --model best.pt --n-samples 20
    python visualize.py --config config/yolo_detection.yaml   --model best.pt --conditions day night
    python visualize.py --config config/yolo_detection.yaml   --model best.pt --source path/to/images/
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import mlflow
import yaml

import pipelines  # noqa: F401
from pipelines.registry import get_pipeline
from utils.mlflow_helper import setup_mlflow
from utils.preprocess import preprocess_frame


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_path = path.parent / "base.yaml"
    if base_path.exists() and str(base_path) != str(path):
        with open(base_path) as f:
            base = yaml.safe_load(f)
        merged = {**base, **cfg}
        for key in base:
            if isinstance(base.get(key), dict) and isinstance(cfg.get(key), dict):
                merged[key] = {**base[key], **cfg[key]}
        return merged
    return cfg


def collect_images(source: str, n_samples: int) -> list[Path]:
    src = Path(source)
    all_imgs = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")))
    if not all_imgs:
        return []
    step = max(1, len(all_imgs) // n_samples)
    return all_imgs[::step][:n_samples]


def run_visualization(
    config: dict,
    model_path: Path,
    conditions: list[str],
    n_samples: int,
    source: str | None,
    run_name: str,
    out_dir: Path,
) -> None:
    from ultralytics import YOLO

    task = config.get("model", {}).get("task", "detect")
    classes = config.get("data", {}).get("classes", [])
    test_cfg = config.get("data", {}).get("test", {})
    imgsz = config.get("train", {}).get("imgsz", 640)
    device = config.get("train", {}).get("device", 0)

    setup_mlflow(
        config["mlflow"]["tracking_uri"],
        config["mlflow"]["experiment_name"],
    )

    tags = {
        "model_type": "yolo",
        "task": task,
        "model_path": str(model_path),
        "type": "visualization",
    }

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("conditions", ",".join(conditions))

        model = YOLO(str(model_path))

        for condition in conditions:
            if source:
                img_dir = Path(source)
            else:
                data_path = test_cfg.get(condition)
                if not data_path:
                    print(f"Skipping '{condition}' — not in config test paths")
                    continue
                img_dir = Path(data_path) / "images"

            if not img_dir.exists():
                print(f"Skipping '{condition}' — images dir not found: {img_dir}")
                continue

            images = collect_images(str(img_dir), n_samples)
            if not images:
                print(f"Skipping '{condition}' — no images found in {img_dir}")
                continue

            print(f"\n[{condition}] Running inference on {len(images)} images...")

            cond_out = out_dir / condition
            cond_out.mkdir(parents=True, exist_ok=True)

            results = model.predict(
                source=[str(p) for p in images],
                imgsz=imgsz,
                device=device,
                verbose=False,
                conf=0.25,
            )

            for i, (result, img_path) in enumerate(zip(results, images)):
                annotated = result.plot(
                    labels=True,
                    conf=True,
                    masks=(task == "segment"),
                    boxes=True,
                    line_width=2,
                    font_size=12,
                )
                out_name = f"{condition}_{i+1:02d}_{img_path.stem}.jpg"
                cv2.imwrite(str(cond_out / out_name), annotated)
                print(f"  Saved: {out_name}")

            mlflow.log_artifacts(str(cond_out), artifact_path=f"visuals/{condition}")
            print(f"  Logged to MLflow: visuals/{condition}/")

        print(f"\nAll visuals saved to: {out_dir}")
        print(f"MLflow run: {config['mlflow']['tracking_uri']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated visual results from a trained YOLO model"
    )
    parser.add_argument("--config",  required=True, help="Config YAML (same one used for training)")
    parser.add_argument("--model",   required=True, help="Path to trained weights (best.pt)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=["all", "day", "wet", "night"],
                        help="Conditions to visualize (default: day wet night)")
    parser.add_argument("--source",  default=None,
                        help="Custom image directory (overrides per-condition test paths)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of images to sample per condition (default: 10)")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--out-dir",  default="runs/visualize",
                        help="Output directory for annotated images (default: runs/visualize)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    config = load_config(args.config)
    task = config.get("model", {}).get("task", "detect")
    test_cfg = config.get("data", {}).get("test", {})

    conditions = args.conditions or [
        k for k in ["day", "wet", "night"] if Path(test_cfg.get(k, "")).exists()
    ] or ["all"]

    run_name = args.run_name or f"vis-{task}-{model_path.stem}"
    out_dir = Path(args.out_dir) / run_name

    print(f"Task       : {task}")
    print(f"Model      : {model_path}")
    print(f"Conditions : {conditions}")
    print(f"Samples    : {args.n_samples} per condition")
    print(f"Output     : {out_dir}")
    print()

    run_visualization(
        config=config,
        model_path=model_path,
        conditions=conditions,
        n_samples=args.n_samples,
        source=args.source,
        run_name=run_name,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
