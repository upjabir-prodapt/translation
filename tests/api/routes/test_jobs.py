from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from src.api.core.security import AuthenticatedUser
from src.api.core.security import get_current_user_context
from src.api.dependencies import get_jobs_handler
from src.api.main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_current_user_context] = lambda: AuthenticatedUser(
        email="test@example.com", business_unit="bu1", organization="org1"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_job_status():
    return {
        "job_id": "job1",
        "status": "completed",
        "progress": 1.0,
        "user": "test@example.com",
        "department": "bu1",
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }


class TestJobsRoutes:
    def test_get_job_status(self, client, mock_job_status):
        mock_handler = AsyncMock()
        mock_handler.get_job_status.return_value = mock_job_status
        app.dependency_overrides[get_jobs_handler] = lambda: mock_handler

        response = client.get("/api/v1/jobs/job1")
        assert response.status_code == 200
        assert response.json()["status"] == "completed"

    def test_list_jobs(self, client, mock_job_status):
        mock_handler = AsyncMock()
        mock_handler.list_jobs.return_value = {
            "jobs": [mock_job_status],
            "total": 1,
            "limit": 10,
            "offset": 0,
        }
        app.dependency_overrides[get_jobs_handler] = lambda: mock_handler

        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        assert len(response.json()["jobs"]) == 1

    def test_cancel_job(self, client):
        mock_handler = AsyncMock()
        app.dependency_overrides[get_jobs_handler] = lambda: mock_handler

        response = client.delete("/api/v1/jobs/job1")
        assert response.status_code == 200
        mock_handler.cancel_job.assert_called_once()
