from unittest.mock import AsyncMock

import pytest
from src.api.services.progress_tracker import ProgressTracker


@pytest.fixture
def mock_updater():
    return AsyncMock()


@pytest.fixture
def tracker(mock_updater):
    return ProgressTracker(updater=mock_updater, job_id="job1", min_update_interval=0.1)


class TestProgressTracker:
    async def test_update_success(self, tracker, mock_updater):
        res = await tracker.update(0.5, "Stage1")
        assert res is True
        mock_updater.update_job.assert_called_once()
        assert tracker.last_update is not None

    async def test_update_rate_limit(self, tracker, mock_updater):
        await tracker.update(0.1)
        res = await tracker.update(0.2)  # Too soon
        assert res is False
        assert mock_updater.update_job.call_count == 1

    async def test_update_force(self, tracker, mock_updater):
        await tracker.update(0.1)
        res = await tracker.update(0.2, force=True)
        assert res is True
        assert mock_updater.update_job.call_count == 2

    async def test_set_error(self, tracker, mock_updater):
        await tracker.set_error("fail")
        mock_updater.update_job.assert_called_once()
        args = mock_updater.update_job.call_args[0][1]
        assert args["status"] == "failed"

    async def test_update_failure_exception(self, tracker, mock_updater):
        mock_updater.update_job.side_effect = Exception("db down")
        res = await tracker.update(0.5)
        assert res is False
