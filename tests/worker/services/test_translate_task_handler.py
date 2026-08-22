"""Worker translate-task handler tests."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.shared.schemas.tasks import TranslateTaskPayload
from src.worker.handlers.translate_task_handler import TranslateTaskHandler


@pytest.fixture
def handler():
    bq = AsyncMock()
    storage = AsyncMock()
    orch = AsyncMock()
    orch.run = AsyncMock()
    h = TranslateTaskHandler(bigquery=bq, storage=storage, orchestrator=orch)
    h._workspaces = MagicMock()
    return h, bq, orch


async def test_noop_when_completed(handler):
    h, bq, orch = handler
    bq.get_translation_job.return_value = {"job_id": "j1", "status": "completed"}
    result = await h.handle(TranslateTaskPayload(job_id="j1"))
    assert result["action"] == "noop"
    orch.run.assert_not_called()


async def test_noop_when_cancelled(handler):
    h, bq, orch = handler
    bq.get_translation_job.return_value = {"job_id": "j1", "status": "cancelled"}
    result = await h.handle(TranslateTaskPayload(job_id="j1"))
    assert result["action"] == "noop"
    orch.run.assert_not_called()


async def test_runs_when_queued(handler):
    h, bq, orch = handler
    job = {"job_id": "j1", "status": "queued"}
    bq.get_translation_job.side_effect = [
        job,
        job,
        {"job_id": "j1", "status": "completed"},
    ]
    result = await h.handle(TranslateTaskPayload(job_id="j1"))
    assert result["action"] == "ran"
    orch.run.assert_awaited_once()


async def test_processing_retry_cleans_workspace(handler):
    h, bq, orch = handler
    job = {"job_id": "j1", "status": "processing"}
    bq.get_translation_job.side_effect = [
        job,
        job,
        {"job_id": "j1", "status": "completed"},
    ]
    result = await h.handle(TranslateTaskPayload(job_id="j1"))
    assert result["action"] == "ran"
    h._workspaces.cleanup.assert_called_once_with("j1")


async def test_not_found(handler):
    h, bq, orch = handler
    bq.get_translation_job.return_value = None
    result = await h.handle(TranslateTaskPayload(job_id="missing"))
    assert result["status"] == "not_found"
    orch.run.assert_not_called()
