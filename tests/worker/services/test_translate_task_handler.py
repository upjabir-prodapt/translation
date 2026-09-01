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


# ---------------------------------------------------------------------------
# Cross-instance job lease
# ---------------------------------------------------------------------------


async def test_deferred_when_another_instance_holds_the_lease(handler):
    """The crash-retry branch rmtree's the job's scratch directory.

    Doing that to a job another instance is still translating is exactly the
    failure this lease exists to prevent, so the lease is taken *before* that
    branch is reached.
    """
    from unittest.mock import patch

    from src.worker.handlers.translate_task_handler import LEASED_ELSEWHERE_STATUS

    h, bq, orch = handler
    bq.get_translation_job.return_value = {"job_id": "j1", "status": "processing"}
    with patch(
        "src.worker.handlers.translate_task_handler.job_lease.acquire",
        return_value=False,
    ):
        result = await h.handle(TranslateTaskPayload(job_id="j1"))

    assert result["status"] == LEASED_ELSEWHERE_STATUS
    assert result["action"] == "deferred"
    orch.run.assert_not_called()
    h._workspaces.cleanup.assert_not_called()


async def test_lease_is_released_after_a_successful_run(handler):
    from unittest.mock import patch

    h, bq, orch = handler
    job = {"job_id": "j1", "status": "queued"}
    bq.get_translation_job.side_effect = [
        job,
        job,
        {"job_id": "j1", "status": "completed"},
    ]
    with (
        patch(
            "src.worker.handlers.translate_task_handler.job_lease.acquire",
            return_value=True,
        ),
        patch(
            "src.worker.handlers.translate_task_handler.job_lease.release"
        ) as mock_release,
    ):
        await h.handle(TranslateTaskPayload(job_id="j1"))
    mock_release.assert_called_once_with("j1")


async def test_lease_is_released_when_the_pipeline_raises(handler):
    """A job whose pipeline failed must be retryable immediately, not after
    JOB_LEASE_TTL_SECONDS."""
    from unittest.mock import patch

    h, bq, orch = handler
    job = {"job_id": "j1", "status": "queued"}
    bq.get_translation_job.side_effect = [job, job]
    orch.run.side_effect = RuntimeError("pipeline exploded")

    with (
        patch(
            "src.worker.handlers.translate_task_handler.job_lease.acquire",
            return_value=True,
        ),
        patch(
            "src.worker.handlers.translate_task_handler.job_lease.release"
        ) as mock_release,
        pytest.raises(RuntimeError),
    ):
        await h.handle(TranslateTaskPayload(job_id="j1"))
    mock_release.assert_called_once_with("j1")


async def test_lease_is_refreshed_while_the_pipeline_runs(handler, monkeypatch):
    """The heartbeat is a background task, not a progress-callback hook.

    `_PipelineProgressTracker.update` is not called densely enough during
    typesetting and PDF rendering -- precisely the long stages -- so hanging
    the heartbeat off progress would let the lease lapse mid-job and invite
    the duplicate execution the lease exists to prevent.
    """
    import asyncio
    from unittest.mock import patch

    from src.worker.handlers import translate_task_handler as mod

    monkeypatch.setattr(mod.settings, "JOB_LEASE_REFRESH_SECONDS", 0.01)

    h, bq, orch = handler
    job = {"job_id": "j1", "status": "queued"}
    bq.get_translation_job.side_effect = [
        job,
        job,
        {"job_id": "j1", "status": "completed"},
    ]

    refreshed: list[str] = []
    first_refresh = asyncio.Event()

    def _refresh(job_id):
        refreshed.append(job_id)
        first_refresh.set()
        return True

    async def _run_until_first_heartbeat(**_kwargs):
        await asyncio.wait_for(first_refresh.wait(), timeout=5)

    orch.run.side_effect = _run_until_first_heartbeat

    with (
        patch(
            "src.worker.handlers.translate_task_handler.job_lease.acquire",
            return_value=True,
        ),
        patch("src.worker.handlers.translate_task_handler.job_lease.release"),
        patch(
            "src.worker.handlers.translate_task_handler.job_lease.refresh",
            side_effect=_refresh,
        ),
    ):
        await h.handle(TranslateTaskPayload(job_id="j1"))

    assert refreshed, "the lease was never refreshed during the pipeline run"
    assert refreshed[0] == "j1"


async def test_heartbeat_task_is_cancelled_when_the_pipeline_finishes(handler):
    """No orphaned heartbeat must outlive the job it was beating for."""
    import asyncio
    from unittest.mock import patch

    h, bq, orch = handler
    job = {"job_id": "j1", "status": "queued"}
    bq.get_translation_job.side_effect = [
        job,
        job,
        {"job_id": "j1", "status": "completed"},
    ]
    before = {t.get_name() for t in asyncio.all_tasks()}

    with (
        patch(
            "src.worker.handlers.translate_task_handler.job_lease.acquire",
            return_value=True,
        ),
        patch("src.worker.handlers.translate_task_handler.job_lease.release"),
        patch("src.worker.handlers.translate_task_handler.job_lease.refresh"),
    ):
        await h.handle(TranslateTaskPayload(job_id="j1"))

    leftover = {t.get_name() for t in asyncio.all_tasks() if not t.done()} - before
    assert not any(name.startswith("job-lease-") for name in leftover)
