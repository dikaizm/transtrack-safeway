"""
Condition-aware frame preprocessing.

Applies CLAHE (on LAB colorspace) to night and dusty frames.
Normal daytime frames are returned unchanged.

IMPORTANT: These thresholds and parameters must match what was used
during model training (model-dev-pipeline/utils/preprocess.py).
"""

import cv2
import numpy as np

NIGHT_BRIGHTNESS_THRESHOLD = 60
DUSTY_VARIANCE_THRESHOLD = 30


def detect_condition(frame: np.ndarray) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < NIGHT_BRIGHTNESS_THRESHOLD:
        return "night"
    if gray.std() < DUSTY_VARIANCE_THRESHOLD:
        return "dusty"
    return "day"


def _apply_clahe(frame: np.ndarray, clip: float) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Returns (processed_frame, condition).
    condition: "night" | "dusty" | "day"
    """
    condition = detect_condition(frame)
    if condition == "night":
        return _apply_clahe(frame, clip=4.0), condition
    if condition == "dusty":
        return _apply_clahe(frame, clip=2.0), condition
    return frame, condition
