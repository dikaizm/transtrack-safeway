"""
Celery task: full detection pipeline.

Flow:
  1. Frame extraction     (ffmpeg @ 3fps)
  2. Condition detection  (night / dusty / day)
  3. Preprocessing        (CLAHE for night/dusty)
  4. Road segmentation    (YOLOv8n-seg) → drivable_area mask
  5. Hazard detection     (YOLOv8m)     → filtered by road mask
  6. Temporal smoothing   (≥2 of 3 frames for road classes, ≥1 for traffic_sign)
  7. Persist to DB, delete temp video
"""

import os
import shutil
import tempfile
import time
from collections import Counter

import cv2
from sqlalchemy.orm import Session

from app.core.celery import celery_app
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.task import Detection, DetectionTask
from app.pipeline.detector import detect_hazards
from app.pipeline.extractor import extract_frames
from app.pipeline.postprocessor import temporal_smooth
from app.pipeline.preprocessor import preprocess
from app.pipeline.renderer import render_video
from app.pipeline.segmentation import get_road_mask

CLASS_NAMES = ["road_depression", "mud_patch", "soil_mound", "traffic_sign"]


@celery_app.task(bind=True, name="detect.run_pipeline")
def run_detection_pipeline(
    self, task_id: str, video_path: str, render: bool = False
) -> None:
    db: Session = SessionLocal()
    frame_dir = tempfile.mkdtemp(prefix=f"frames_{task_id}_")

    try:
        _update_status(db, task_id, "processing")
        t_start = time.monotonic()

        # 1. Extract frames at 3fps (time-based)
        frame_paths = extract_frames(
            video_path,
            frame_dir,
            sample_fps=settings.frame_sample_fps,
        )
        _update_field(db, task_id, frames_to_process=len(frame_paths))

        # 2-5. Per-frame pipeline
        frame_detections: list[list[dict]] = []
        condition_counts: Counter = Counter()

        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                frame_detections.append([])
                continue

            # Preprocess: CLAHE for night/dusty, no-op for day
            frame, condition = preprocess(frame)
            condition_counts[condition] += 1

            # Road segmentation → drivable_area mask
            road_mask = get_road_mask(frame)

            # Safety hazard detection, filtered by road mask
            dets = detect_hazards(frame, road_mask, CLASS_NAMES)
            frame_detections.append(dets)

        # 6. Temporal smoothing
        confirmed = temporal_smooth(frame_detections, settings.frame_sample_fps)

        # Dominant condition across all frames
        dominant_condition = (
            condition_counts.most_common(1)[0][0] if condition_counts else "day"
        )

        # 7. Optionally render annotated MP4
        rendered_path = None
        if render:
            out_path = os.path.join(settings.rendered_video_dir, f"{task_id}.mp4")
            render_video(
                frame_paths,
                confirmed,
                out_path,
                source_fps=float(settings.frame_sample_fps),
            )
            rendered_path = out_path

        # 8. Persist and clean up
        processing_time = round(time.monotonic() - t_start, 2)
        _save_results(db, task_id, confirmed, dominant_condition, len(frame_paths), rendered_path, processing_time)

    except Exception as exc:
        _update_status(db, task_id, "failed", error=str(exc))
        raise

    finally:
        db.close()
        shutil.rmtree(frame_dir, ignore_errors=True)
        if os.path.exists(video_path):
            os.remove(video_path)


# ------------------------------------------------------------------ #
#  DB helpers                                                          #
# ------------------------------------------------------------------ #

def _update_status(
    db: Session,
    task_id: str,
    status: str,
    error: str | None = None,
) -> None:
    task = db.query(DetectionTask).filter_by(id=task_id).first()
    if task:
        task.status = status
        if error:
            task.error_message = error
        db.commit()


def _update_field(db: Session, task_id: str, **kwargs) -> None:
    task = db.query(DetectionTask).filter_by(id=task_id).first()
    if task:
        for k, v in kwargs.items():
            setattr(task, k, v)
        db.commit()


def _save_results(
    db: Session,
    task_id: str,
    confirmed: list[dict],
    condition: str,
    frames_analyzed: int,
    rendered_path: str | None = None,
    processing_time_sec: float | None = None,
) -> None:
    task = db.query(DetectionTask).filter_by(id=task_id).first()
    if not task:
        return

    task.status = "done"
    task.conditions = condition
    task.frames_analyzed = frames_analyzed
    if rendered_path:
        task.rendered_video_path = rendered_path
    if processing_time_sec is not None:
        task.processing_time_sec = processing_time_sec

    for det in confirmed:
        bbox = det["bbox"]
        db.add(Detection(
            task_id=task_id,
            frame_index=det["frame_index"],
            frame_sec=det["frame_sec"],
            class_name=det["class_name"],
            confidence=det["confidence"],
            bbox_x=bbox[0],
            bbox_y=bbox[1],
            bbox_w=bbox[2],
            bbox_h=bbox[3],
            zone=det["zone"],
        ))

    db.commit()
