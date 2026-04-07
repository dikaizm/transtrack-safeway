"""
Annotated video renderer.

Draws bounding boxes, labels, and confidence scores on extracted frames,
then encodes them into an MP4 file using the original video's FPS.
"""

import os
from pathlib import Path

import cv2
import numpy as np

# BGR colors per class
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "road_depression": (0, 0, 220),    # red
    "mud_patch":       (0, 140, 255),  # orange
    "soil_mound":      (0, 220, 220),  # yellow
    "traffic_sign":    (220, 200, 0),  # cyan-blue
}
DEFAULT_COLOR = (200, 200, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS = 2


def render_video(
    frame_paths: list[str],
    detections_per_frame: list[list[dict]],
    output_path: str,
    source_fps: float = 3.0,
) -> str:
    """
    Draw detection overlays on frames and encode to MP4.

    Args:
        frame_paths: ordered list of frame image paths (same order as pipeline)
        detections_per_frame: per-frame detection dicts AFTER temporal smoothing,
                              keyed by frame_index
        output_path: destination .mp4 path
        source_fps: FPS to use for the output video

    Returns:
        output_path (convenience)
    """
    if not frame_paths:
        raise ValueError("No frames to render")

    # Build frame_index → detections lookup
    det_by_frame: dict[int, list[dict]] = {}
    for det in detections_per_frame:
        idx = det["frame_index"]
        det_by_frame.setdefault(idx, []).append(det)

    # Read first frame to get dimensions
    sample = cv2.imread(frame_paths[0])
    if sample is None:
        raise RuntimeError(f"Cannot read frame: {frame_paths[0]}")
    h, w = sample.shape[:2]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, source_fps, (w, h))

    try:
        for idx, fpath in enumerate(frame_paths):
            frame = cv2.imread(fpath)
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)

            for det in det_by_frame.get(idx, []):
                _draw_detection(frame, det)

            writer.write(frame)
    finally:
        writer.release()

    return output_path


def _draw_detection(frame: np.ndarray, det: dict) -> None:
    bbox = det["bbox"]
    cls = det["class_name"]
    conf = det["confidence"]
    color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

    x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x2, y2 = x + bw, y + bh

    cv2.rectangle(frame, (x, y), (x2, y2), color, THICKNESS)

    label = f"{cls} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
    label_y = max(y - 6, th + 4)

    # Background pill for readability
    cv2.rectangle(frame, (x, label_y - th - 4), (x + tw + 4, label_y + baseline), color, -1)
    cv2.putText(frame, label, (x + 2, label_y - 2), FONT, FONT_SCALE, (0, 0, 0), THICKNESS - 1)
