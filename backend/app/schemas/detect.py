from pydantic import BaseModel


class TaskCreatedResponse(BaseModel):
    task_id: str
    frames_to_process: int | None = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # pending | processing | done | failed


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


class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    conditions: str | None
    frames_analyzed: int | None
    processing_time_sec: float | None = None
    detections: list[DetectionItem]
