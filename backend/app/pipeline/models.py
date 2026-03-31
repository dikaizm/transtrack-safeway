"""
Model loader — loads both YOLO models once per worker process and caches them.

Models are loaded lazily on first use. Each Celery worker process holds its own
copy in memory — intentional (no shared GPU memory across processes).
"""

from functools import lru_cache

from ultralytics import YOLO

from app.core.config import settings


@lru_cache(maxsize=1)
def get_segmentation_model() -> YOLO:
    """Custom YOLOv8n-seg — binary road area segmentation (drivable_area / off_road)."""
    return YOLO(settings.model_segmentation_weights)


@lru_cache(maxsize=1)
def get_detection_model() -> YOLO:
    """Custom YOLOv8m — 4-class safety hazard detection."""
    return YOLO(settings.model_detection_weights)
