"""
Unit tests for worker/services/progress.py — ProgressTracker.

Firestore repository is fully mocked; no real GCP connection is made.
"""

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from worker.services.progress import ProgressTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(
    job_id: str = "job-1",
    firestore=None,
    min_update_interval: float = 0.0,  # 0 disables rate-limiting in most tests
) -> ProgressTracker:
    fs = firestore or AsyncMock()
    return ProgressTracker(firestore=fs, job_id=job_id, min_update_interval=min_update_interval)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


class TestProgressTrackerUpdate:
    async def test_returns_true_on_success(self):
        tracker = _make_tracker()
        result = await tracker.update(0.5)
        assert result is True

    async def test_calls_firestore_update_job(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs)
        await tracker.update(0.5)
        fs.update_job.assert_called_once()

    async def test_progress_clamped_to_zero(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs)
        await tracker.update(-0.5)
        call_data = fs.update_job.call_args[0][1]
        assert call_data["progress"] == 0.0

    async def test_progress_clamped_to_one(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs)
        await tracker.update(1.5)
        call_data = fs.update_job.call_args[0][1]
        assert call_data["progress"] == 1.0

    async def test_stage_included_when_provided(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs)
        await tracker.update(0.3, current_stage="parsing")
        call_data = fs.update_job.call_args[0][1]
        assert call_data["current_stage"] == "parsing"

    async def test_stage_not_included_when_none(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs)
        await tracker.update(0.3, current_stage=None)
        call_data = fs.update_job.call_args[0][1]
        assert "current_stage" not in call_data

    async def test_progress_none_not_included(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs)
        await tracker.update(None)
        call_data = fs.update_job.call_args[0][1]
        assert "progress" not in call_data

    async def test_updated_at_always_included(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs)
        await tracker.update(0.5)
        call_data = fs.update_job.call_args[0][1]
        assert "updated_at" in call_data

    async def test_last_update_set_after_call(self):
        tracker = _make_tracker()
        assert tracker.last_update is None
        await tracker.update(0.5)
        assert tracker.last_update is not None

    async def test_rate_limiting_skips_update(self):
        fs = AsyncMock()
        # High interval means second call is skipped
        tracker = _make_tracker(firestore=fs, min_update_interval=60.0)
        await tracker.update(0.1)  # first call — sets last_update
        result = await tracker.update(0.2)  # second call — rate-limited
        assert result is False
        fs.update_job.assert_called_once()  # only first call went through

    async def test_force_bypasses_rate_limit(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs, min_update_interval=60.0)
        await tracker.update(0.1)
        result = await tracker.update(0.2, force=True)
        assert result is True
        assert fs.update_job.call_count == 2

    async def test_firestore_exception_returns_false(self):
        fs = AsyncMock()
        fs.update_job.side_effect = Exception("Firestore down")
        tracker = _make_tracker(firestore=fs)
        result = await tracker.update(0.5)
        assert result is False


# ---------------------------------------------------------------------------
# set_error()
# ---------------------------------------------------------------------------


class TestProgressTrackerSetError:
    async def test_calls_firestore_with_failed_status(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs)
        await tracker.set_error("something went wrong")
        fs.update_job.assert_called_once()
        call_data = fs.update_job.call_args[0][1]
        assert call_data["status"] == "failed"

    async def test_error_message_included(self):
        fs = AsyncMock()
        tracker = _make_tracker(firestore=fs)
        await tracker.set_error("translation timed out")
        call_data = fs.update_job.call_args[0][1]
        assert call_data["error_message"] == "translation timed out"

    async def test_firestore_exception_swallowed(self):
        fs = AsyncMock()
        fs.update_job.side_effect = Exception("Firestore down")
        tracker = _make_tracker(firestore=fs)
        # Should not raise
        await tracker.set_error("oops")
