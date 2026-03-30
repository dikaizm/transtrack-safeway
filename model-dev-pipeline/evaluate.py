"""
Evaluation entry point.

Always evaluates per condition (day / wet / night) separately.
Never use only aggregate mAP — it hides poor performance on specific conditions.

Usage:
    # Evaluate on all available conditions (recommended)
    python evaluate.py --config config/yolo_detection.yaml --model runs/train/weights/best.pt

    # Evaluate on specific conditions only
    python evaluate.py --config config/yolo_detection.yaml --model best.pt --conditions day night

    # Custom MLflow run name
    python evaluate.py --config config/yolo_detection.yaml --model best.pt --run-name "eval-v1"

Expected test dataset structure:
    data/test/day/    images/ + labels/
    data/test/wet/    images/ + labels/
    data/test/night/  images/ + labels/
"""

import argparse
import sys
from pathlib import Path

import yaml

import pipelines  # noqa: F401 — triggers all @register_pipeline decorators
from pipelines.registry import get_pipeline


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained road safety model")
    parser.add_argument(
        "--config", required=True,
        help="Path to model config YAML (must match the config used during training)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained model weights (e.g. runs/train/weights/best.pt)"
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        choices=["all", "day", "wet", "night"],
        help=(
            "Conditions to evaluate on. Defaults to day+wet+night if dirs exist. "
            "Avoid using 'all' alone — aggregate mAP hides per-condition failures."
        )
    )
    parser.add_argument(
        "--run-name", default=None,
        help="MLflow run name (optional)"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model weights not found: {args.model}")
        sys.exit(1)

    config = load_config(args.config)
    pipeline_type = config.get("model", {}).get("type")

    if not pipeline_type:
        print("Error: config must specify model.type")
        sys.exit(1)

    # Warn if user explicitly requests aggregate-only evaluation
    if args.conditions == ["all"]:
        print(
            "Warning: evaluating on 'all' only. "
            "This hides per-condition failures (night/wet). "
            "Use --conditions day wet night for a complete picture."
        )

    print(f"Pipeline   : {pipeline_type}")
    print(f"Task       : {config['model'].get('task', 'detect')}")
    print(f"Model      : {model_path}")
    print(f"Conditions : {args.conditions or 'day + wet + night (auto)'}")
    print(f"MLflow     : {config['mlflow']['tracking_uri']}")
    print()

    pipeline = get_pipeline(pipeline_type, config)
    results = pipeline.evaluate(
        model_path=model_path,
        conditions=args.conditions,
        run_name=args.run_name,
    )

    print("\nEvaluation results:")
    for condition, metrics in results.items():
        print(f"\n  [{condition}]")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
