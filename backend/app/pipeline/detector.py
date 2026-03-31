"""
Safety hazard detection stage.

Runs the 4-class YOLOv8m model on a preprocessed frame.
Filters results using the road segmentation mask only.
Vehicle exclusion zones removed — handled by road mask + temporal smoothing.

Detection zones per class:
    road_depression  → road area only
    mud_patch        → road area only
                       suppressed if road surface variance is uniform (fully wet road)
    soil_mound       → road area only
    traffic_sign     → full frame, no mask applied
"""

import cv2
import numpy as np

from app.core.config import settings
from app.pipeline.models import get_detection_model
from app.pipeline.segmentation import is_in_road_area

ROAD_AREA_CLASSES = {"road_depression", "mud_patch", "soil_mound"}

# Below this variance threshold the road is uniformly wet — suppress mud_patch
MUD_PATCH_VARIANCE_THRESHOLD = 15.0


def detect_hazards(
    frame: np.ndarray,
    road_mask: np.ndarray,
    class_names: list[str],
) -> list[dict]:
    """
    Run safety hazard detection and filter by road mask.

    Returns:
        List of detection dicts:
        {
            "class_name": str,
            "confidence": float,
            "bbox": [x, y, w, h],
            "zone": "road_area" | "full_frame",
        }
    """
    model = get_detection_model()
    results = model(
        frame,
        conf=settings.detection_confidence,
        device=settings.device,
        verbose=False,
    )[0]

    uniform_surface = _road_surface_variance(frame, road_mask) < MUD_PATCH_VARIANCE_THRESHOLD

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id >= len(class_names):
            continue

        class_name = class_names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = [x1, y1, x2 - x1, y2 - y1]

        if class_name in ROAD_AREA_CLASSES:
            if not is_in_road_area(bbox, road_mask):
                continue
            if class_name == "mud_patch" and uniform_surface:
                continue
            zone = "road_area"
        else:
            # traffic_sign — full frame, no mask
            zone = "full_frame"

        detections.append({
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox,
            "zone": zone,
        })

    return detections


def _road_surface_variance(frame: np.ndarray, road_mask: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    road_pixels = gray[road_mask > 0]
    if len(road_pixels) == 0:
        return 100.0
    return float(road_pixels.std())
