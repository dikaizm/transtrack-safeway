from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    from app.models import task  # noqa: F401 — ensures models are registered
    Base.metadata.create_all(bind=engine)


def upgrade_schema():
    """Add/remove columns on existing tables without Alembic."""
    with engine.connect() as conn:
        for stmt in [
            # additions
            "ALTER TABLE detection_tasks ADD COLUMN IF NOT EXISTS webhook_url TEXT",
            "ALTER TABLE detection_tasks ADD COLUMN IF NOT EXISTS frames_processed INTEGER",
            "ALTER TABLE detection_tasks ADD COLUMN IF NOT EXISTS rendered_video_url TEXT",
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS camera_condition VARCHAR(10)",
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS snapshot_url TEXT",
            # removals / renames (idempotent)
            "ALTER TABLE detection_tasks DROP COLUMN IF EXISTS render_video",
            "ALTER TABLE detection_tasks DROP COLUMN IF EXISTS video_url",
            "ALTER TABLE detection_tasks DROP COLUMN IF EXISTS rendered_video_path",
            "ALTER TABLE detections DROP COLUMN IF EXISTS snapshot_path",
        ]:
            conn.execute(text(stmt))
        conn.commit()
