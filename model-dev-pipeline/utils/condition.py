import cv2
import numpy as np


NIGHT_BRIGHTNESS_THRESHOLD = 60
DUSTY_VARIANCE_THRESHOLD = 30


def detect_condition(frame: np.ndarray) -> str:
    """
    Detect lighting/surface condition of a single frame.

    Returns:
        "night"  — avg brightness below threshold
        "dusty"  — low pixel variance (hazy/dusty)
        "day"    — normal daylight
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    variance = float(gray.std())

    if brightness < NIGHT_BRIGHTNESS_THRESHOLD:
        return "night"
    if variance < DUSTY_VARIANCE_THRESHOLD:
        return "dusty"
    return "day"


def detect_condition_from_path(image_path: str) -> str:
    frame = cv2.imread(image_path)
    if frame is None:
        return "day"
    return detect_condition(frame)


def group_images_by_condition(image_paths: list[str]) -> dict[str, list[str]]:
    """
    Group a flat list of image paths by detected condition.

    Returns:
        { "day": [...], "night": [...], "dusty": [...] }
    """
    groups: dict[str, list[str]] = {"day": [], "night": [], "dusty": []}
    for path in image_paths:
        condition = detect_condition_from_path(path)
        groups[condition].append(path)
    return groups
