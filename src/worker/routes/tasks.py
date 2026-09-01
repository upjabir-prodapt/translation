"""Internal Cloud Tasks routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import status

from src.shared.schemas.tasks import TranslateTaskPayload
from src.worker.auth.oidc import require_cloud_tasks_oidc
from src.worker.dependencies import get_translate_task_handler
from src.worker.handlers.translate_task_handler import LEASED_ELSEWHERE_STATUS
from src.worker.handlers.translate_task_handler import SHUTTING_DOWN_STATUS
from src.worker.handlers.translate_task_handler import TranslateTaskHandler
from src.worker.lifecycle import job_in_flight

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/internal/tasks/translate",
    tags=["tasks"],
    dependencies=[Depends(require_cloud_tasks_oidc)],
)
async def translate_task(
    request: Request,
    payload: TranslateTaskPayload,
    handler: TranslateTaskHandler = Depends(get_translate_task_handler),  # noqa: B008
):
    """Cloud Tasks target: run translation pipeline for job_id."""
    logger.info(
        "Received translate task request job_id=%s traceparent=%s headers=%s",
        payload.job_id,
        payload.traceparent,
        dict(request.headers),
    )
    try:
        # Counted for the duration of the pipeline so the shutdown log can
        # report exactly how many jobs a SIGTERM/SIGKILL interrupted.
        with job_in_flight(payload.job_id):
            result = await handler.handle(payload)
    except Exception as e:
        # Non-2xx → Cloud Tasks retries
        logger.exception("Translate task failed for job %s: %s", payload.job_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {e}",
        ) from e

    logger.info("Translate task completed for job %s: %s", payload.job_id, result)

    if result.get("status") == SHUTTING_DOWN_STATUS:
        # The worker is draining and never started this job. 503 → Cloud
        # Tasks re-dispatches it onto a healthy instance instead of the
        # job being lost to the imminent SIGKILL.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker is shutting down; retry this task.",
        )

    if result.get("status") == LEASED_ELSEWHERE_STATUS:
        # Another live instance is already translating this job. 503 makes
        # Cloud Tasks back off instead of a second instance re-running the
        # whole pipeline at another ~6-7 GB. The retry either finds the job
        # terminal (200) or finds an expired lease and takes over.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job is already being processed by another worker; retry this task.",
        )

    if result.get("status") == "not_found":
        # Permanent — 200 so Tasks does not retry forever
        return result

    return result
