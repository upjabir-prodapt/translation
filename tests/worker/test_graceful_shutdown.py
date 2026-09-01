"""Graceful-shutdown accounting for the worker.

Before this, the worker had no signal handling at all: the pipeline runs
synchronously inside `loop.run_in_executor(...)`, which uvicorn cannot
cancel, so every deploy/scale-down with a job in flight ended in a
SIGKILL that looked identical to an OOM kill in Cloud Logging.
"""

from __future__ import annotations

import signal
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.worker import lifecycle
from src.worker.handlers.translate_task_handler import SHUTTING_DOWN_STATUS
from src.worker.handlers.translate_task_handler import TranslateTaskHandler


@pytest.fixture(autouse=True)
def _clean_lifecycle_state():
    lifecycle.reset_for_tests()
    yield
    lifecycle.reset_for_tests()


class TestInFlightAccounting:
    def test_counter_starts_at_zero(self):
        assert lifecycle.in_flight_job_count() == 0
        assert lifecycle.in_flight_job_ids() == []

    def test_counts_job_for_the_duration_of_the_block(self):
        with lifecycle.job_in_flight("job-1"):
            assert lifecycle.in_flight_job_count() == 1
            assert lifecycle.in_flight_job_ids() == ["job-1"]
        assert lifecycle.in_flight_job_count() == 0

    def test_decrements_even_when_the_job_raises(self):
        with pytest.raises(RuntimeError), lifecycle.job_in_flight("job-1"):
            raise RuntimeError("pipeline blew up")
        assert lifecycle.in_flight_job_count() == 0

    def test_tracks_concurrent_jobs(self):
        with lifecycle.job_in_flight("job-a"), lifecycle.job_in_flight("job-b"):
            assert lifecycle.in_flight_job_ids() == ["job-a", "job-b"]


class TestShutdownFlag:
    def test_flag_is_unset_by_default(self):
        assert lifecycle.is_shutting_down() is False

    def test_request_shutdown_logs_the_in_flight_census(self, caplog):
        with lifecycle.job_in_flight("job-7"), caplog.at_level("WARNING"):
            lifecycle.request_shutdown("SIGTERM")

        assert lifecycle.is_shutting_down() is True
        assert "worker received shutdown" in caplog.text
        assert "in_flight_jobs=1" in caplog.text
        assert "job-7" in caplog.text

    def test_request_shutdown_is_idempotent(self, caplog):
        with caplog.at_level("WARNING"):
            lifecycle.request_shutdown("SIGTERM")
            lifecycle.request_shutdown("lifespan")

        assert caplog.text.count("worker received shutdown") == 1

    def test_signal_handler_sets_the_flag_and_chains(self):
        called = []
        original = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_a: called.append("previous"))
        try:
            lifecycle.install_signal_handlers()
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)
        finally:
            signal.signal(signal.SIGTERM, original)

        assert lifecycle.is_shutting_down() is True
        assert called == ["previous"]


def _make_handler(job_status_value: str = "queued") -> TranslateTaskHandler:
    handler = TranslateTaskHandler.__new__(TranslateTaskHandler)
    handler.bigquery = MagicMock()
    handler.bigquery.get_translation_job = AsyncMock(
        return_value={"job_id": "job-1", "status": job_status_value}
    )
    handler.storage = MagicMock()
    handler.orchestrator = MagicMock()
    handler.orchestrator.run = AsyncMock()
    handler._workspaces = MagicMock()
    return handler


class TestHandlerDefersOnShutdown:
    @pytest.mark.asyncio
    async def test_unstarted_job_is_deferred_for_redelivery(self):
        from src.shared.schemas.tasks import TranslateTaskPayload

        handler = _make_handler()
        lifecycle.request_shutdown("SIGTERM")

        result = await handler.handle(TranslateTaskPayload(job_id="job-1"))

        assert result["status"] == SHUTTING_DOWN_STATUS
        assert result["action"] == "deferred"
        handler.orchestrator.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_job_runs_normally_when_not_shutting_down(self):
        from src.shared.schemas.tasks import TranslateTaskPayload

        handler = _make_handler()

        result = await handler.handle(TranslateTaskPayload(job_id="job-1"))

        assert result["status"] != SHUTTING_DOWN_STATUS
        handler.orchestrator.run.assert_awaited_once()
