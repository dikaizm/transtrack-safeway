from abc import ABC, abstractmethod
from pathlib import Path


class BasePipeline(ABC):
    """
    Abstract base for all model training/evaluation pipelines.

    To add a new model type (e.g. RT-DETR, FCOS):
      1. Subclass BasePipeline
      2. Implement train(), evaluate(), load_model()
      3. Register with @register_pipeline("your_model_name")
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config.get("model", {})
        self.train_cfg = config.get("train", {})
        self.data_cfg = config.get("data", {})
        self.mlflow_cfg = config.get("mlflow", {})
        self.aug_cfg = config.get("augmentation", {})

    @abstractmethod
    def train(self, run_name: str | None = None) -> dict:
        """
        Train the model.

        Args:
            run_name: Optional MLflow run name.

        Returns:
            dict of final training metrics logged to MLflow.
        """

    @abstractmethod
    def evaluate(
        self,
        model_path: str | Path,
        conditions: list[str] | None = None,
        run_name: str | None = None,
    ) -> dict:
        """
        Evaluate a trained model on the test set.

        Args:
            model_path: Path to trained weights.
            conditions: Subset of ["all", "day", "wet", "night"].
                        Defaults to all available.
            run_name: Optional MLflow run name.

        Returns:
            dict of evaluation metrics per condition per class.
        """

    @abstractmethod
    def load_model(self, model_path: str | Path):
        """Load trained model from path and return model object."""
