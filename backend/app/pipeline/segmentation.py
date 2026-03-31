"""
Road segmentation stage.

Produces a binary road area mask using a custom YOLOv8n-seg model.
The mask is used downstream to filter detections: only road_depression,
mud_patch, and soil_mound inside the road area are reported.

traffic_sign is NOT filtered by this mask — signs are on road-side.
"""

import numpy as np

from app.core.config import settings
from app.pipeline.models import get_segmentation_model


def get_road_mask(frame: np.ndarray) -> np.ndarray:
    """
    Run road segmentation and return a binary mask (uint8, 0 or 255)
    with the same H×W as the input frame.

    Falls back to a center-frame polygon mask if the model returns
    no mask or very low confidence — common in night frames where
    road boundaries are invisible.
    """
    model = get_segmentation_model()
    results = model(
        frame,
        conf=settings.segmentation_confidence,
        device=settings.device,
        verbose=False,
    )[0]

    h, w = frame.shape[:2]

    if results.masks is None or len(results.masks) == 0:
        return _fallback_mask(h, w)

    # Merge all road-class masks into one binary mask
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask_tensor in results.masks.data:
        mask = mask_tensor.cpu().numpy()
        # Resize mask to original frame size if needed
        if mask.shape != (h, w):
            import cv2
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = np.maximum(combined, (mask > 0.5).astype(np.uint8) * 255)

    # If combined mask is implausibly small, use fallback
    if combined.sum() < (h * w * 0.05 * 255):
        return _fallback_mask(h, w)

    return combined


def is_in_road_area(bbox: list[float], mask: np.ndarray, threshold: float = 0.4) -> bool:
    """
    Returns True if at least `threshold` fraction of the bbox area
    falls within the road mask.

    bbox: [x, y, w, h] in pixel coordinates.
    """
    x, y, bw, bh = [int(v) for v in bbox]
    h, w = mask.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)

    if x2 <= x1 or y2 <= y1:
        return False

    roi = mask[y1:y2, x1:x2]
    road_pixels = (roi > 0).sum()
    total_pixels = (x2 - x1) * (y2 - y1)

    return (road_pixels / total_pixels) >= threshold


def _fallback_mask(h: int, w: int) -> np.ndarray:
    """
    Center-lower polygon approximating the road area when
    the segmentation model fails (e.g. night with no visible boundary).
    Covers the bottom 60% of the frame, trapezoid shape.
    """
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([
        [w * 0.1,  h * 0.4],
        [w * 0.9,  h * 0.4],
        [w * 1.0,  h * 1.0],
        [w * 0.0,  h * 1.0],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask
