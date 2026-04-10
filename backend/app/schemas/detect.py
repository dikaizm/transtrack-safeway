from datetime import datetime

from pydantic import BaseModel


class TaskCreatedResponse(BaseModel):
    task_id: str
    frames_to_process: int | None = None
    created_at: datetime


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # pending | processing | done | failed
    frames_processed: int | None = None
    frames_to_process: int | None = None
    progress_pct: float | None = None


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DetectionItem(BaseModel):
    frame_index: int
    frame_sec: float
    class_name: str
    confidence: float
    bbox: BBox
    zone: str  # road_area | full_frame
    camera_condition: str | None = None  # day | night | dusty


class FrameSnapshot(BaseModel):
    frame_index: int
    frame_sec: float
    snapshot_url: str


class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    video_url: str
    conditions: str | None
    condition_summary: dict[str, int] | None = None
    frames_analyzed: int | None
    processing_time_sec: float | None = None
    detections: list[DetectionItem]
    snapshots: list[FrameSnapshot]


class WebhookProgressPayload(BaseModel):
    task_id: str
    event: str              # started | progress | done | failed
    status: str             # processing | done | failed
    frames_processed: int
    frames_total: int
    progress_pct: float
    condition_summary: dict[str, int]
    timestamp: str          # ISO 8601
    detections_count: int | None = None
    dominant_condition: str | None = None
    processing_time_sec: float | None = None
    error: str | None = None
