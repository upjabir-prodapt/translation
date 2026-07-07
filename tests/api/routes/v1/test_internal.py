"""
Route tests for /internal/storage/cleanup-expired-outputs and the
expired-download (410 Gone) error mapping.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_file_lifecycle_service, get_job_service
from api.exceptions import OutputFileExpiredError
from api.main import app


@pytest.fixture()
def cleanup_client():
    """TestClient with a mocked FileLifecycleService."""
    service = AsyncMock()
    service.cleanup_expired_outputs.return_value = {
        "expired_outputs_scanned": 2,
        "terminal_jobs_scanned": 1,
        "files_deleted": 4,
        "jobs_cleaned": 3,
        "failures": 0,
    }
    app.dependency_overrides[get_file_lifecycle_service] = lambda: service

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, service

    app.dependency_overrides.clear()


class TestCleanupEndpoint:
    def test_cleanup_returns_summary(self, cleanup_client):
        client, service = cleanup_client

        response = client.post("/api/v1/internal/storage/cleanup-expired-outputs")

        assert response.status_code == 200
        body = response.json()
        assert body["files_deleted"] == 4
        assert body["jobs_cleaned"] == 3
        assert body["failures"] == 0
        service.cleanup_expired_outputs.assert_awaited_once()


class TestExpiredDownloadMapping:
    def test_expired_download_returns_410(self):
        job_service = AsyncMock()
        job_service.get_download_url.side_effect = OutputFileExpiredError("job-x")
        app.dependency_overrides[get_job_service] = lambda: job_service

        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/api/v1/jobs/job-x/download")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 410
        assert response.json()["error"]["code"] == "OUTPUT_FILE_EXPIRED"
