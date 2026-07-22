"""Cloud Tasks enqueue client for the public API."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google.api_core import exceptions as gcp_exceptions
from google.cloud import tasks_v2
from google.protobuf import duration_pb2

from src.config.constants import settings
from src.shared.schemas.tasks import TranslateTaskPayload

logger = logging.getLogger(__name__)

_TASK_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


class CloudTasksService:
    """Enqueue translation jobs onto a Cloud Tasks HTTP queue."""

    def __init__(self, client: tasks_v2.CloudTasksClient | None = None):
        self._client = client

    @property
    def client(self) -> tasks_v2.CloudTasksClient:
        if self._client is None:
            self._client = tasks_v2.CloudTasksClient()
        return self._client

    def _queue_path(self) -> str:
        project = settings.CLOUD_TASKS_PROJECT or settings.GOOGLE_CLOUD_PROJECT
        return self.client.queue_path(
            project,
            settings.CLOUD_TASKS_LOCATION,
            settings.CLOUD_TASKS_QUEUE,
        )

    @staticmethod
    def task_id_for_job(job_id: str) -> str:
        """Derive a Cloud Tasks task id from job_id (idempotent create)."""
        safe = _TASK_NAME_SAFE.sub("-", job_id)
        return f"translate-{safe}"[:500]

    def enqueue_translate(
        self,
        job_id: str,
        *,
        traceparent: str | None = None,
        tracestate: str | None = None,
    ) -> str:
        """Create an HTTP task targeting the worker. Returns the task name.

        Raises on failure so the caller can compensate (mark job failed).
        """
        if not settings.CLOUD_TASKS_QUEUE:
            raise RuntimeError("CLOUD_TASKS_QUEUE is not configured")
        if not settings.CLOUD_TASKS_WORKER_URL:
            raise RuntimeError("CLOUD_TASKS_WORKER_URL is not configured")
        if not settings.CLOUD_TASKS_OIDC_SERVICE_ACCOUNT:
            raise RuntimeError("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT is not configured")

        payload = TranslateTaskPayload(
            job_id=job_id,
            traceparent=traceparent,
            tracestate=tracestate,
        )
        body = json.dumps(payload.model_dump(exclude_none=True)).encode("utf-8")

        task: dict[str, Any] = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": settings.CLOUD_TASKS_WORKER_URL,
                "headers": {"Content-Type": "application/json"},
                "body": body,
                "oidc_token": {
                    "service_account_email": settings.CLOUD_TASKS_OIDC_SERVICE_ACCOUNT,
                    "audience": settings.CLOUD_TASKS_WORKER_URL,
                },
            }
        }

        deadline = int(settings.CLOUD_TASKS_DISPATCH_DEADLINE_SECONDS)
        if deadline > 0:
            task["dispatch_deadline"] = duration_pb2.Duration(seconds=deadline)

        parent = self._queue_path()
        task_name = f"{parent}/tasks/{self.task_id_for_job(job_id)}"
        task["name"] = task_name

        try:
            created = self.client.create_task(
                request={"parent": parent, "task": task}
            )
            logger.info("Enqueued Cloud Task %s for job %s", created.name, job_id)
            return created.name
        except gcp_exceptions.AlreadyExists:
            logger.info(
                "Cloud Task already exists for job %s (%s); treating as success",
                job_id,
                task_name,
            )
            return task_name
