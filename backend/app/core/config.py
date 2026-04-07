from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/transtrack"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Storage
    temp_video_dir: str = "/tmp/transtrack/videos"
    rendered_video_dir: str = "/tmp/transtrack/rendered"
    max_video_size_mb: int = 50
    max_video_duration_sec: int = 120

    # Model weights
    model_segmentation_weights: str = "models/segmentation/best.pt"
    model_detection_weights: str = "models/detection/best.pt"

    # Inference
    frame_sample_fps: int = 3
    device: str = "cpu"
    detection_confidence: float = 0.25
    segmentation_confidence: float = 0.5
    vehicle_confidence: float = 0.4

    # Task
    task_result_ttl_sec: int = 86400
    worker_timeout_sec: int = 300

    class Config:
        env_file = ".env"


settings = Settings()
