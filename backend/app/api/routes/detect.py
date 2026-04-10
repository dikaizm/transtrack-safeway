import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.task import Detection, DetectionTask
from app.pipeline.extractor import VideoValidationError, validate_video
from app.schemas.detect import (
    BBox,
    DetectionItem,
    FrameSnapshot,
    TaskCreatedResponse,
    TaskResultResponse,
    TaskStatusResponse,
)
from app.schemas.response import SuccessResponse
from app.tasks.detect import run_detection_pipeline

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("", response_model=SuccessResponse, status_code=202)
async def submit_detection(
    video: UploadFile = File(...),
    task_id: str = Form(...),
    webhook_url: str | None = Form(None),
    db: Session = Depends(get_db),
):
    if not video.filename or not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=422, detail="Only .mp4 files are accepted.")

    Path(settings.temp_video_dir).mkdir(parents=True, exist_ok=True)
    video_path = os.path.join(settings.temp_video_dir, f"{task_id}.mp4")

    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

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

    task = DetectionTask(
        id=task_id,
        status="pending",
        video_filename=video.filename,
        webhook_url=webhook_url,
        frames_to_process=frames_to_process,
    )
    db.add(task)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        os.remove(video_path)
        raise HTTPException(status_code=409, detail=f"Task ID '{task_id}' already exists.")

    run_detection_pipeline.delay(task_id, video_path, webhook_url=webhook_url)

    return SuccessResponse(
        message="Task submitted successfully.",
        data=TaskCreatedResponse(
            task_id=task_id,
            frames_to_process=frames_to_process,
            created_at=task.created_at,
        ),
    )


@router.get("/{task_id}", response_model=SuccessResponse)
def get_task_status(task_id: str, db: Session = Depends(get_db)):
    task = db.query(DetectionTask).filter_by(id=task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    progress_pct = None
    if task.frames_processed is not None and task.frames_to_process:
        progress_pct = round(task.frames_processed / task.frames_to_process * 100, 1)

    processing_time_sec = (
        task.processing_time_sec
        if task.status == "done"
        else round((datetime.now(timezone.utc) - task.created_at).total_seconds(), 2)
    )

    return SuccessResponse(
        message="Task status retrieved.",
        data=TaskStatusResponse(
            task_id=task.id,
            status=task.status,
            frames_processed=task.frames_processed,
            frames_to_process=task.frames_to_process,
            progress_pct=progress_pct,
            created_at=task.created_at,
            processing_time_sec=processing_time_sec,
        ),
    )


@router.get("/{task_id}/result", response_model=SuccessResponse)
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

    condition_summary = dict(
        Counter(d.camera_condition for d in detections if d.camera_condition)
    ) or None

    # Deduplicated snapshots — one entry per frame
    seen: dict[int, FrameSnapshot] = {}
    for d in detections:
        if d.snapshot_url and d.frame_index not in seen:
            seen[d.frame_index] = FrameSnapshot(
                frame_index=d.frame_index,
                frame_sec=d.frame_sec,
                snapshot_url=d.snapshot_url,
            )

    return SuccessResponse(
        message="Task result retrieved.",
        data=TaskResultResponse(
            task_id=task.id,
            status=task.status,
            video_url=task.rendered_video_url or "",
            conditions=task.conditions,
            condition_summary=condition_summary,
            frames_analyzed=task.frames_analyzed,
            processing_time_sec=task.processing_time_sec,
            detections=[
                DetectionItem(
                    frame_index=d.frame_index,
                    frame_sec=d.frame_sec,
                    class_name=d.class_name,
                    confidence=round(d.confidence, 4),
                    bbox=BBox(x=d.bbox_x, y=d.bbox_y, w=d.bbox_w, h=d.bbox_h),
                    zone=d.zone,
                    camera_condition=d.camera_condition,
                )
                for d in detections
            ],
            snapshots=list(seen.values()),
        ),
    )
