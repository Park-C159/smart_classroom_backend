"""Celery app configuration and periodic task schedule."""
from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "math_qa",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.document_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=False,
    worker_concurrency=1,  # GPU tasks are serial, one at a time
    task_track_started=True,
)

# Periodic tasks
celery_app.conf.beat_schedule = {
    # "nightly-pdf-parse": {
    #     "task": "app.tasks.document_tasks.parse_pending_pdfs",
    #     "schedule": crontab(hour=2, minute=0),  # 2 AM daily
    # },
}
