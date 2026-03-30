"""
Frame preprocessing for night and dusty/hazy conditions.

Applied consistently at BOTH training time (on each frame before augmentation)
and inference time — same transform, same parameters.

Rules:
    night (avg brightness < 60)   → CLAHE LAB clip=4.0
    dusty (pixel std < 30)        → CLAHE LAB clip=2.0
    normal day                    → no preprocessing

IMPORTANT: If you change these thresholds or clip values,
update them in both training and the inference pipeline.
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


def apply_clahe(frame: np.ndarray, clip: float) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB colorspace."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_frame(frame: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Preprocess a single BGR frame based on its detected condition.

    Returns:
        (preprocessed_frame, condition)
        condition is one of: "night", "dusty", "day"
    """
    condition = detect_condition(frame)
    if condition == "night":
        return apply_clahe(frame, clip=4.0), condition
    if condition == "dusty":
        return apply_clahe(frame, clip=2.0), condition
    return frame, condition


def preprocess_batch(frames: list[np.ndarray]) -> list[tuple[np.ndarray, str]]:
    """Preprocess a list of frames. Returns list of (frame, condition) tuples."""
    return [preprocess_frame(f) for f in frames]
