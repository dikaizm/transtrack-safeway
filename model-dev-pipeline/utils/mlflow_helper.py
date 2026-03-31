import os
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """
    Configure MLflow tracking URI and ensure experiment exists.

    Returns:
        experiment_id
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_config(config: dict, prefix: str = "") -> None:
    """Flatten and log a nested config dict as MLflow params."""
    flat = _flatten(config, prefix)
    # MLflow limits param values to 500 chars
    mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})


def log_metrics_per_condition(
    metrics: dict[str, float],
    condition: str,
    step: int | None = None,
) -> None:
    """
    Log metrics namespaced by condition.

    Example key: "day/mAP50/road_depression"
    """
    prefixed = {f"{condition}/{k}": v for k, v in metrics.items()}
    mlflow.log_metrics(prefixed, step=step)


def log_artifacts_from_dir(directory: str | Path, artifact_path: str = "") -> None:
    """Log all files in a directory as MLflow artifacts."""
    directory = Path(directory)
    if directory.exists():
        mlflow.log_artifacts(str(directory), artifact_path=artifact_path)


def tag_gdrive_link(run_id: str, tag_key: str, link: str) -> None:
    """
    Set a GDrive shareable link as a tag on an existing MLflow run.
    Uses MlflowClient so it works both inside and outside an active run context.

    Args:
        run_id:  MLflow run ID (from mlflow.active_run().info.run_id)
        tag_key: e.g. "gdrive_seg_weights", "gdrive_vis_det_night"
        link:    https://drive.google.com/file/d/{id}/view
    """
    client = MlflowClient()
    client.set_tag(run_id, tag_key, link)


def _flatten(d: dict, prefix: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, key, sep))
        else:
            items[key] = v
    return items
