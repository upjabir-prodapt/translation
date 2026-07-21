"""Worker app smoke tests."""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_worker_healthz():
    with patch(
        "src.worker.auth.oidc.settings.WORKER_SKIP_OIDC_VERIFICATION",
        True,
    ):
        from src.worker.main import app

        with TestClient(app) as client:
            response = client.get("/healthz")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"


def test_worker_task_requires_oidc_by_default():
    from src.worker.main import app

    with TestClient(app) as client:
        response = client.post(
            "/internal/tasks/translate",
            json={"job_id": "x"},
        )
        assert response.status_code == 401
