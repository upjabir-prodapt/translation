"""Cloud Tasks enqueue client for the public API."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from google.api_core import exceptions as gcp_exceptions
from google.cloud import tasks_v2
from google.protobuf import duration_pb2

from src.config.constants import settings
from src.shared.schemas.tasks import TranslateTaskPayload

logger = logging.getLogger(__name__)

_TASK_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]")

PRIORITY_HIGH = "high"
PRIORITY_STANDARD = "standard"


class CloudTasksService:
    """Enqueue translation jobs onto a Cloud Tasks HTTP queue."""

    def __init__(self, client: tasks_v2.CloudTasksClient | None = None):
        self._client = client

    @property
    def client(self) -> tasks_v2.CloudTasksClient:
        if self._client is None:
            self._client = tasks_v2.CloudTasksClient()
        return self._client

    @staticmethod
    def _resolve_queue_name(priority: str | None = None) -> str:
        """Pick the queue for a priority tier.

        Cloud Tasks has no per-task priority field, so tiers are separate
        queues with independent dispatch budgets. Falls back to the standard
        queue (with a WARNING) when `high` is requested but
        `CLOUD_TASKS_QUEUE_HIGH` is unset -- that way the setting can be
        deployed before the queue is provisioned without failing jobs.
        """
        if str(priority or "").lower() == PRIORITY_HIGH:
            high = (settings.CLOUD_TASKS_QUEUE_HIGH or "").strip()
            if high:
                return high
            logger.warning(
                "priority=high requested but CLOUD_TASKS_QUEUE_HIGH is unset; "
                "falling back to the standard queue %s",
                settings.CLOUD_TASKS_QUEUE,
            )
        return settings.CLOUD_TASKS_QUEUE

    def _queue_path(self, priority: str | None = None) -> str:
        project = settings.CLOUD_TASKS_PROJECT or settings.GOOGLE_CLOUD_PROJECT
        return self.client.queue_path(
            project,
            settings.CLOUD_TASKS_LOCATION,
            self._resolve_queue_name(priority),
        )

    @staticmethod
    def task_id_for_job(job_id: str) -> str:
        """Derive a Cloud Tasks task id from job_id (idempotent create)."""
        safe = _TASK_NAME_SAFE.sub("-", job_id)
        return f"translate-{safe}"[:500]

    def _is_local_http_target(self) -> bool:
        """Return True when the configured worker URL is a local HTTP target.

        In that case we bypass Cloud Tasks and POST directly to the worker so
        local development does not require an HTTPS endpoint or OIDC token.
        """
        url = settings.CLOUD_TASKS_WORKER_URL
        return settings.IS_LOCAL and url.startswith("http://")

    def _enqueue_local_http(
        self,
        job_id: str,
        *,
        traceparent: str | None = None,
        tracestate: str | None = None,
        priority: str | None = None,
        doc_format: str | None = None,
    ) -> str:
        """Direct HTTP dispatch for local development.

        Priority is still resolved and logged so routing can be verified
        locally without provisioning any Cloud Tasks queue.
        """
        logger.info(
            "Local dispatch for job %s (would use queue=%s priority=%s doc_format=%s)",
            job_id,
            self._resolve_queue_name(priority),
            priority,
            doc_format,
        )
        payload = TranslateTaskPayload(
            job_id=job_id,
            traceparent=traceparent,
            tracestate=tracestate,
            priority=priority,
            doc_format=doc_format,
        )
        body = json.dumps(payload.model_dump(exclude_none=True)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if traceparent:
            headers["traceparent"] = traceparent
        if tracestate:
            headers["tracestate"] = tracestate

        url = settings.CLOUD_TASKS_WORKER_URL
        response = httpx.post(url, content=body, headers=headers, timeout=3600.0)
        response.raise_for_status()
        logger.info("Dispatched job %s directly to local worker at %s", job_id, url)
        return f"local-http/{job_id}"

    def enqueue_translate(
        self,
        job_id: str,
        *,
        traceparent: str | None = None,
        tracestate: str | None = None,
        priority: str | None = None,
        doc_format: str | None = None,
    ) -> str:
        """Create an HTTP task targeting the worker. Returns the task name.

        `priority` selects which queue the task is created on -- see
        `_resolve_queue_name`. It is decided server-side from the document
        format, never by the client.

        Raises on failure so the caller can compensate (mark job failed).
        """
        if not settings.CLOUD_TASKS_WORKER_URL:
            raise RuntimeError("CLOUD_TASKS_WORKER_URL is not configured")

        if self._is_local_http_target():
            return self._enqueue_local_http(
                job_id,
                traceparent=traceparent,
                tracestate=tracestate,
                priority=priority,
                doc_format=doc_format,
            )

        if not settings.CLOUD_TASKS_QUEUE:
            raise RuntimeError("CLOUD_TASKS_QUEUE is not configured")
        if not settings.CLOUD_TASKS_OIDC_SERVICE_ACCOUNT:
            raise RuntimeError("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT is not configured")

        payload = TranslateTaskPayload(
            job_id=job_id,
            traceparent=traceparent,
            tracestate=tracestate,
            priority=priority,
            doc_format=doc_format,
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

        parent = self._queue_path(priority)
        queue_name = self._resolve_queue_name(priority)
        task_name = f"{parent}/tasks/{self.task_id_for_job(job_id)}"
        task["name"] = task_name

        try:
            created = self.client.create_task(request={"parent": parent, "task": task})
            logger.info(
                "Enqueued Cloud Task %s for job %s queue=%s priority=%s doc_format=%s",
                created.name,
                job_id,
                queue_name,
                priority or PRIORITY_STANDARD,
                doc_format,
            )
            return created.name
        except gcp_exceptions.AlreadyExists:
            logger.info(
                "Cloud Task already exists for job %s (%s); treating as success",
                job_id,
                task_name,
            )
            return task_name

    def delete_translate_task(
        self, job_id: str, priority: str | None = None
    ) -> bool:
        """Best-effort removal of a queued task, used when a job is cancelled.

        Cancelling only flips the BigQuery status; without this the task is
        still dispatched and the worker no-ops. That wastes a dispatch slot --
        and on the high-priority queue those slots are deliberately scarce.

        Never raises: the BigQuery status is the source of truth, and the
        worker's own cancellation check remains the correctness guarantee.
        Returns True when a task was actually deleted.

        When `priority` is unknown, both queues are tried, since the task could
        be on either.
        """
        if self._is_local_http_target():
            return False  # nothing was ever enqueued in local dispatch mode

        if not settings.CLOUD_TASKS_QUEUE:
            return False

        task_id = self.task_id_for_job(job_id)
        project = settings.CLOUD_TASKS_PROJECT or settings.GOOGLE_CLOUD_PROJECT

        if priority:
            candidates = [self._resolve_queue_name(priority)]
        else:
            candidates = [settings.CLOUD_TASKS_QUEUE]
            high = (settings.CLOUD_TASKS_QUEUE_HIGH or "").strip()
            if high:
                candidates.append(high)

        for queue_name in candidates:
            if not queue_name:
                continue
            task_path = self.client.task_path(
                project, settings.CLOUD_TASKS_LOCATION, queue_name, task_id
            )
            try:
                self.client.delete_task(request={"name": task_path})
            except gcp_exceptions.NotFound:
                # Already dispatched, already deleted, or on the other queue.
                logger.debug(
                    "No queued task %s in %s for job %s", task_id, queue_name, job_id
                )
                continue
            except Exception as exc:  # noqa: BLE001 - must never fail a cancel
                logger.warning(
                    "Failed to delete Cloud Task for job %s from queue %s: %s",
                    job_id,
                    queue_name,
                    exc,
                )
                continue
            logger.info(
                "Deleted queued Cloud Task for cancelled job %s from queue %s",
                job_id,
                queue_name,
            )
            return True
        return False
