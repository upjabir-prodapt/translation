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
from src.worker.handlers.translate_task_handler import TranslateTaskHandler

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
        result = await handler.handle(payload)
    except Exception as e:
        # Non-2xx → Cloud Tasks retries
        logger.exception("Translate task failed for job %s: %s", payload.job_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {e}",
        ) from e

    logger.info("Translate task completed for job %s: %s", payload.job_id, result)

    if result.get("status") == "not_found":
        # Permanent — 200 so Tasks does not retry forever
        return result

    return result
