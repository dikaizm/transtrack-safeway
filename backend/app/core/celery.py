import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "transtrack",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.detect"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=settings.worker_timeout_sec,
    task_soft_time_limit=settings.worker_timeout_sec - 30,
    result_expires=settings.task_result_ttl_sec,
    worker_prefetch_multiplier=1,   # one task at a time per worker (GPU constraint)
    worker_pool="solo",             # avoids fork()/Objective-C crash on macOS
)
