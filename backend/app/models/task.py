from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DetectionTask(Base):
    __tablename__ = "detection_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # provided by external system
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending | processing | done | failed
    conditions: Mapped[str | None] = mapped_column(String(20), nullable=True)  # day | night | dusty
    webhook_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    frames_to_process: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frames_processed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frames_analyzed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    video_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rendered_video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("detection_tasks.id", ondelete="CASCADE")
    )
    frame_index: Mapped[int] = mapped_column(Integer)
    frame_sec: Mapped[float] = mapped_column(Float)
    class_name: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float)
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_w: Mapped[float] = mapped_column(Float)
    bbox_h: Mapped[float] = mapped_column(Float)
    zone: Mapped[str] = mapped_column(String(20))  # road_area | full_frame
    camera_condition: Mapped[str | None] = mapped_column(String(10), nullable=True)  # day | night | dusty
    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    task: Mapped["DetectionTask"] = relationship(back_populates="detections")
