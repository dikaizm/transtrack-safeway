"""
Post-processing stage.

Temporal smoothing: suppress single-frame noise by requiring a detection
to appear in at least N of M consecutive frames before reporting it.

Rules:
    road_depression, mud_patch, soil_mound: ≥ 2 of 3 consecutive frames
    traffic_sign:                           ≥ 1 frame (static object)
"""

TEMPORAL_RULES = {
    "road_depression": {"min_frames": 2, "window": 3},
    "mud_patch":       {"min_frames": 2, "window": 3},
    "soil_mound":      {"min_frames": 2, "window": 3},
    "traffic_sign":    {"min_frames": 1, "window": 1},
}

IOU_MATCH_THRESHOLD = 0.3


def temporal_smooth(
    frame_detections: list[list[dict]],
    sample_fps: int,
) -> list[dict]:
    """
    Apply temporal smoothing across all frames.

    Args:
        frame_detections: List of per-frame detection lists.
                          frame_detections[i] = list of dicts from detector.py
        sample_fps:       Frames per second used during extraction.

    Returns:
        Flat list of confirmed detections, each with frame_index and frame_sec added.
    """
    confirmed: list[dict] = []
    n_frames = len(frame_detections)

    for frame_idx, frame_dets in enumerate(frame_detections):
        for det in frame_dets:
            class_name = det["class_name"]
            rule = TEMPORAL_RULES.get(class_name, {"min_frames": 1, "window": 1})
            min_frames = rule["min_frames"]
            window = rule["window"]

            if min_frames == 1:
                confirmed.append({
                    **det,
                    "frame_index": frame_idx,
                    "frame_sec": round(frame_idx / sample_fps, 2),
                })
                continue

            # Count how many frames in the window contain a matching detection
            start = max(0, frame_idx - window + 1)
            end = min(n_frames, frame_idx + 1)
            match_count = 0

            for w_idx in range(start, end):
                for w_det in frame_detections[w_idx]:
                    if w_det["class_name"] == class_name and _iou(det["bbox"], w_det["bbox"]) >= IOU_MATCH_THRESHOLD:
                        match_count += 1
                        break  # one match per frame is enough

            if match_count >= min_frames:
                confirmed.append({
                    **det,
                    "frame_index": frame_idx,
                    "frame_sec": round(frame_idx / sample_fps, 2),
                })

    return _deduplicate(confirmed)


def _deduplicate(detections: list[dict]) -> list[dict]:
    """
    Remove duplicate detections of the same class at nearly the same
    location and frame. Keeps the highest-confidence one.
    """
    unique: list[dict] = []
    for det in sorted(detections, key=lambda d: -d["confidence"]):
        is_dup = any(
            d["class_name"] == det["class_name"]
            and d["frame_index"] == det["frame_index"]
            and _iou(d["bbox"], det["bbox"]) >= IOU_MATCH_THRESHOLD
            for d in unique
        )
        if not is_dup:
            unique.append(det)
    return sorted(unique, key=lambda d: d["frame_index"])


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    return inter_area / (a[2] * a[3] + b[2] * b[3] - inter_area)
