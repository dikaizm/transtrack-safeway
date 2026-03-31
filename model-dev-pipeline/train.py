"""
Training entry point.

Usage:
    python train.py --config config/yolo_detection.yaml
    python train.py --config config/yolo_detection.yaml --run-name "yolov8m-baseline"
    python train.py --config config/yolo_segmentation.yaml --run-name "yolov8n-seg-v1"
"""

import argparse
import sys
from pathlib import Path

import yaml

import pipelines  # noqa: F401 — triggers all @register_pipeline decorators
from pipelines.registry import get_pipeline, list_pipelines


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Merge base.yaml if not already merged
    base_path = path.parent / "base.yaml"
    if base_path.exists() and str(base_path) != str(path):
        with open(base_path) as f:
            base = yaml.safe_load(f)
        # Config-specific values override base
        merged = {**base, **cfg}
        # Deep merge nested dicts
        for key in base:
            if isinstance(base.get(key), dict) and isinstance(cfg.get(key), dict):
                merged[key] = {**base[key], **cfg[key]}
        return merged

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train a road safety detection model")
    parser.add_argument(
        "--config", required=True,
        help="Path to model config YAML (e.g. config/yolo_detection.yaml)"
    )
    parser.add_argument(
        "--run-name", default=None,
        help="MLflow run name (optional)"
    )
    parser.add_argument(
        "--list-pipelines", action="store_true",
        help="List all available pipeline types and exit"
    )
    args = parser.parse_args()

    if args.list_pipelines:
        print("Available pipelines:", list_pipelines())
        sys.exit(0)

    config = load_config(args.config)
    pipeline_type = config.get("model", {}).get("type")

    if not pipeline_type:
        print("Error: config must specify model.type (e.g. 'yolo')")
        sys.exit(1)

    print(f"Pipeline : {pipeline_type}")
    print(f"Task     : {config['model'].get('task', 'detect')}")
    print(f"Weights  : {config['model'].get('weights')}")
    print(f"Epochs   : {config['train'].get('epochs')}")
    print(f"MLflow   : {config['mlflow']['tracking_uri']}")
    print()

    pipeline = get_pipeline(pipeline_type, config)
    metrics = pipeline.train(run_name=args.run_name)

    print("\nFinal metrics:")
    for k, v in metrics.items():
        if not k.startswith("_"):
            print(f"  {k}: {v:.4f}")

    # Machine-readable marker — lets shell scripts capture the weights path
    best = metrics.get("_best_weights")
    if best:
        print(f"\nBEST_WEIGHTS={best}")


if __name__ == "__main__":
    main()
