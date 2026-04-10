"""
Celery task: full detection pipeline.

Flow:
  1. Frame extraction     (ffmpeg @ 3fps)
  2. Condition detection  (night / dusty / day)
  3. Preprocessing        (CLAHE for night/dusty)
  4. Road segmentation    (YOLOv8n-seg) → drivable_area mask
  5. Hazard detection     (YOLOv8m)     → filtered by road mask
  6. Temporal smoothing   (≥2 of 3 frames for road classes, ≥1 for traffic_sign)
  7. Save per-frame snapshots with all bboxes drawn
  8. Render annotated MP4
  9. Persist to DB, delete temp video
"""

import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from app.core.celery import celery_app
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.storage import upload_file
from app.models.task import Detection, DetectionTask
from app.pipeline.detector import detect_hazards
from app.pipeline.extractor import extract_frames
from app.pipeline.postprocessor import temporal_smooth
from app.pipeline.preprocessor import preprocess
from app.pipeline.renderer import render_video
from app.pipeline.segmentation import get_road_mask
from app.pipeline.webhook import fire_webhook

CLASS_NAMES = ["road_depression", "mud_patch", "soil_mound", "traffic_sign"]

_BBOX_COLORS = {
    "road_depression": (0, 0, 255),
    "mud_patch":       (0, 165, 255),
    "soil_mound":      (0, 255, 255),
    "traffic_sign":    (0, 255, 0),
}


@celery_app.task(bind=True, name="detect.run_pipeline")
def run_detection_pipeline(
    self, task_id: str, video_path: str, webhook_url: str | None = None
) -> None:
    db: Session = SessionLocal()
    frame_dir = tempfile.mkdtemp(prefix=f"frames_{task_id}_")

    try:
        _update_status(db, task_id, "processing")
        t_start = time.monotonic()

        if webhook_url:
            fire_webhook(webhook_url, _webhook_payload(
                task_id=task_id, event="started", status="processing",
                frames_processed=0, frames_total=0, progress_pct=0.0,
                condition_summary={},
            ))

        # 1. Extract frames at configured fps
        frame_paths = extract_frames(
            video_path,
            frame_dir,
            sample_fps=settings.frame_sample_fps,
        )
        total_frames = len(frame_paths)
        _update_field(db, task_id, frames_to_process=total_frames)

        # 2-5. Per-frame pipeline
        frame_detections: list[list[dict]] = []
        condition_counts: Counter = Counter()

        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                frame_detections.append([])
                continue

            frame, condition = preprocess(frame)
            condition_counts[condition] += 1

            road_mask = get_road_mask(frame)
            dets = detect_hazards(frame, road_mask, CLASS_NAMES)

            for d in dets:
                d["camera_condition"] = condition

            frame_detections.append(dets)

            if (i + 1) % 10 == 0 or i == total_frames - 1:
                _update_field(db, task_id, frames_processed=i + 1)

                if webhook_url and total_frames > 0:
                    pct = (i + 1) / total_frames * 100
                    prev_pct = i / total_frames * 100
                    milestones = [25, 50, 75]
                    if any(prev_pct < m <= pct for m in milestones):
                        fire_webhook(webhook_url, _webhook_payload(
                            task_id=task_id, event="progress", status="processing",
                            frames_processed=i + 1, frames_total=total_frames,
                            progress_pct=round(pct, 1),
                            condition_summary=dict(condition_counts),
                        ))

        # 6. Temporal smoothing
        confirmed = temporal_smooth(frame_detections, settings.frame_sample_fps)

        # Dominant condition across all frames
        dominant_condition = (
            condition_counts.most_common(1)[0][0] if condition_counts else "day"
        )

        # 7. Save per-frame snapshots with all confirmed bboxes drawn, upload to S3
        _save_snapshots(task_id, frame_paths, confirmed)

        # 8. Render annotated MP4, upload to S3
        local_video_path = os.path.join(settings.rendered_video_dir, f"{task_id}.mp4")
        render_video(
            frame_paths,
            confirmed,
            local_video_path,
            source_fps=float(settings.frame_sample_fps),
        )
        rendered_video_url = upload_file(local_video_path, f"videos/{task_id}.mp4")

        # 9. Persist and clean up
        processing_time = round(time.monotonic() - t_start, 2)
        _save_results(db, task_id, confirmed, dominant_condition, total_frames, rendered_video_url, processing_time)

        if webhook_url:
            fire_webhook(webhook_url, _webhook_payload(
                task_id=task_id, event="done", status="done",
                frames_processed=total_frames, frames_total=total_frames,
                progress_pct=100.0,
                condition_summary=dict(condition_counts),
                detections_count=len(confirmed),
                dominant_condition=dominant_condition,
                processing_time_sec=processing_time,
            ))

    except Exception as exc:
        _update_status(db, task_id, "failed", error=str(exc))
        if webhook_url:
            fire_webhook(webhook_url, _webhook_payload(
                task_id=task_id, event="failed", status="failed",
                frames_processed=0, frames_total=0, progress_pct=0.0,
                condition_summary={},
                error=str(exc),
            ))
        raise

    finally:
        db.close()
        shutil.rmtree(frame_dir, ignore_errors=True)
        if os.path.exists(video_path):
            os.remove(video_path)


# ------------------------------------------------------------------ #
#  Snapshot rendering                                                  #
# ------------------------------------------------------------------ #

def _save_snapshots(task_id: str, frame_paths: list[str], confirmed: list[dict]) -> None:
    """Render one snapshot per frame that has confirmed detections, with all bboxes drawn."""
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for det in confirmed:
        by_frame[det["frame_index"]].append(det)

    if not by_frame:
        return

    snap_dir = Path(settings.snapshot_dir) / task_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, dets in by_frame.items():
        frame = cv2.imread(frame_paths[frame_idx])
        if frame is None:
            continue

        # Re-apply preprocessing so snapshot matches what the model saw
        frame, _ = preprocess(frame)

        for det in dets:
            bbox = det["bbox"]
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = _BBOX_COLORS.get(det["class_name"], (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x, max(y - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        snap_path = str(snap_dir / f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(snap_path, frame)

        s3_url = upload_file(snap_path, f"snapshots/{task_id}/frame_{frame_idx:05d}.jpg")
        for det in dets:
            det["snapshot_url"] = s3_url


# ------------------------------------------------------------------ #
#  Webhook payload builder                                             #
# ------------------------------------------------------------------ #

def _webhook_payload(
    task_id: str,
    event: str,
    status: str,
    frames_processed: int,
    frames_total: int,
    progress_pct: float,
    condition_summary: dict,
    detections_count: int | None = None,
    dominant_condition: str | None = None,
    processing_time_sec: float | None = None,
    error: str | None = None,
) -> dict:
    payload: dict = {
        "task_id": task_id,
        "event": event,
        "status": status,
        "frames_processed": frames_processed,
        "frames_total": frames_total,
        "progress_pct": progress_pct,
        "condition_summary": condition_summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if detections_count is not None:
        payload["detections_count"] = detections_count
    if dominant_condition is not None:
        payload["dominant_condition"] = dominant_condition
    if processing_time_sec is not None:
        payload["processing_time_sec"] = processing_time_sec
    if error is not None:
        payload["error"] = error
    return payload


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
    rendered_video_url: str,
    processing_time_sec: float,
) -> None:
    task = db.query(DetectionTask).filter_by(id=task_id).first()
    if not task:
        return

    task.status = "done"
    task.conditions = condition
    task.frames_analyzed = frames_analyzed
    task.rendered_video_url = rendered_video_url
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
            camera_condition=det.get("camera_condition"),
            snapshot_url=det.get("snapshot_url"),
        ))

    db.commit()
