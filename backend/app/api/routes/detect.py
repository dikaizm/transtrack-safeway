import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.task import Detection, DetectionTask
from app.pipeline.extractor import VideoValidationError, validate_video
from app.schemas.detect import (
    BBox,
    DetectionItem,
    TaskCreatedResponse,
    TaskResultResponse,
    TaskStatusResponse,
)
from app.tasks.detect import run_detection_pipeline

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("", response_model=TaskCreatedResponse, status_code=202)
async def submit_detection(
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Basic file type check before saving
    if not video.filename or not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=422, detail="Only .mp4 files are accepted.")

    # Save to temp storage
    Path(settings.temp_video_dir).mkdir(parents=True, exist_ok=True)
    task_id = str(uuid.uuid4())
    video_path = os.path.join(settings.temp_video_dir, f"{task_id}.mp4")

    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    # Validate size and duration via ffprobe (fail fast before queuing)
    try:
        meta = validate_video(
            video_path,
            max_size_mb=settings.max_video_size_mb,
            max_duration_sec=settings.max_video_duration_sec,
        )
    except VideoValidationError as e:
        os.remove(video_path)
        raise HTTPException(status_code=422, detail=str(e))

    frames_to_process = int(meta["duration"] * settings.frame_sample_fps)

    # Persist task record
    task = DetectionTask(
        id=task_id,
        status="pending",
        video_filename=video.filename,
        frames_to_process=frames_to_process,
    )
    db.add(task)
    db.commit()

    # Enqueue Celery task
    run_detection_pipeline.delay(task_id, video_path)

    return TaskCreatedResponse(task_id=task_id, frames_to_process=frames_to_process)


@router.get("/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str, db: Session = Depends(get_db)):
    task = db.query(DetectionTask).filter_by(id=task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatusResponse(task_id=task.id, status=task.status)


@router.get("/{task_id}/result", response_model=TaskResultResponse)
def get_task_result(task_id: str, db: Session = Depends(get_db)):
    task = db.query(DetectionTask).filter_by(id=task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Task is not done yet. Current status: {task.status}",
        )

    detections = (
        db.query(Detection)
        .filter_by(task_id=task_id)
        .order_by(Detection.frame_index)
        .all()
    )

    return TaskResultResponse(
        task_id=task.id,
        status=task.status,
        conditions=task.conditions,
        frames_analyzed=task.frames_analyzed,
        detections=[
            DetectionItem(
                frame_index=d.frame_index,
                frame_sec=d.frame_sec,
                class_name=d.class_name,
                confidence=round(d.confidence, 4),
                bbox=BBox(x=d.bbox_x, y=d.bbox_y, w=d.bbox_w, h=d.bbox_h),
                zone=d.zone,
            )
            for d in detections
        ],
    )
