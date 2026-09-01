"""How handler statuses map onto HTTP, which is what Cloud Tasks reacts to.

Both deferral statuses must produce 503, not 200: a 200 tells Cloud Tasks
the job is done and stops the retry, silently losing the work.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from src.shared.schemas.tasks import TranslateTaskPayload
from src.worker.handlers.translate_task_handler import LEASED_ELSEWHERE_STATUS
from src.worker.handlers.translate_task_handler import SHUTTING_DOWN_STATUS
from src.worker.routes.tasks import translate_task


class _Request:
    headers: dict[str, str] = {}


def _handler(status: str):
    handler = AsyncMock()
    handler.handle = AsyncMock(
        return_value={"job_id": "j1", "status": status, "action": "deferred"}
    )
    return handler


@pytest.mark.parametrize(
    "status",
    [SHUTTING_DOWN_STATUS, LEASED_ELSEWHERE_STATUS],
    ids=["shutdown", "leased"],
)
async def test_deferral_statuses_map_to_503(status):
    with pytest.raises(HTTPException) as exc_info:
        await translate_task(
            _Request(), TranslateTaskPayload(job_id="j1"), _handler(status)
        )
    assert exc_info.value.status_code == 503


async def test_not_found_maps_to_200_so_tasks_stops_retrying():
    handler = AsyncMock()
    handler.handle = AsyncMock(
        return_value={"job_id": "j1", "status": "not_found", "action": "noop"}
    )
    result = await translate_task(
        _Request(), TranslateTaskPayload(job_id="j1"), handler
    )
    assert result["status"] == "not_found"


async def test_successful_run_is_returned_unchanged():
    handler = AsyncMock()
    handler.handle = AsyncMock(
        return_value={"job_id": "j1", "status": "completed", "action": "ran"}
    )
    result = await translate_task(
        _Request(), TranslateTaskPayload(job_id="j1"), handler
    )
    assert result == {"job_id": "j1", "status": "completed", "action": "ran"}
