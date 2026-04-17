"""
Evaluation entry point.

Usage:
    # Per-condition evaluation (default — auto-detects day/wet/night dirs)
    python evaluate.py --config config/yolo_detection.yaml --model runs/train/weights/best.pt

    # Aggregate evaluation on the full test set
    python evaluate.py --config config/yolo_detection.yaml --model best.pt --eval-mode all

    # Specific conditions only
    python evaluate.py --config config/yolo_detection.yaml --model best.pt --eval-mode day night

    # Custom MLflow run name
    python evaluate.py --config config/yolo_detection.yaml --model best.pt --run-name "eval-v1"

Expected test dataset structure:
    data/test/           images/ + labels/   (used by --eval-mode all)
    data/test/day/       images/ + labels/
    data/test/wet/       images/ + labels/
    data/test/night/     images/ + labels/
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
        "--eval-mode", nargs="+", default=None,
        choices=["all", "condition", "day", "wet", "night"],
        metavar="MODE",
        help=(
            "Evaluation mode. 'condition' (default) auto-detects day/wet/night dirs. "
            "'all' evaluates on the aggregate test set. "
            "Or pass specific conditions: day wet night."
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

    # Translate --eval-mode to conditions list for the pipeline
    eval_mode = args.eval_mode or ["condition"]
    if "condition" in eval_mode:
        # None triggers auto-detection of day/wet/night dirs inside the pipeline
        conditions = None
    else:
        conditions = eval_mode  # e.g. ["all"] or ["day", "wet"]

    print(f"Pipeline   : {pipeline_type}")
    print(f"Task       : {config['model'].get('task', 'detect')}")
    print(f"Model      : {model_path}")
    print(f"Eval mode  : {' + '.join(eval_mode)}")
    print(f"MLflow     : {config['mlflow']['tracking_uri']}")
    print()

    pipeline = get_pipeline(pipeline_type, config)
    results = pipeline.evaluate(
        model_path=model_path,
        conditions=conditions,
        run_name=args.run_name,
    )

    print("\nEvaluation results:")
    for condition, metrics in results.items():
        print(f"\n  [{condition}]")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
