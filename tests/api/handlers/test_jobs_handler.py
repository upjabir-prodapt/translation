from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.api.handlers.jobs_handler import JobsHandler


@pytest.fixture
def mock_service():
    return AsyncMock()


@pytest.fixture
def handler(mock_service):
    return JobsHandler(job_service=mock_service)


class TestJobsHandler:
    async def test_get_job_status(self, handler, mock_service):
        await handler.get_job_status("job1", "user@example.com")
        mock_service.get_job_status.assert_called_once_with("job1", "user@example.com")

    async def test_list_jobs(self, handler, mock_service):
        await handler.list_jobs(
            status="completed", limit=10, offset=0, user_id="user@example.com"
        )
        mock_service.list_jobs.assert_called_once_with(
            "completed", 10, 0, user_id="user@example.com"
        )

    async def test_cancel_job(self, handler, mock_service):
        req = MagicMock()
        await handler.cancel_job("job1", req, "user@example.com")
        mock_service.cancel_job.assert_called_once_with("job1", req, "user@example.com")

    async def test_download_output(self, handler, mock_service):
        await handler.download_output("job1", "user@example.com")
        mock_service.get_download_url.assert_called_once_with(
            "job1", "mono", "user@example.com"
        )
