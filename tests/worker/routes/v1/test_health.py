"""
Integration tests for worker health endpoints:
  GET /health
  GET /ready

Uses FastAPI TestClient with a minimal app built from the health router.
WorkerLifecycle is mocked so no real GCS warmup occurs.

NOTE: worker.routes.__init__ imports the process router which depends on
worker.handlers.translation_services (a missing module). We stub it out
in sys.modules before importing to avoid the ModuleNotFoundError.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Stub missing handler module so that worker.routes can be imported
_fake_handlers = types.ModuleType("worker.handlers.translation_services")
_fake_handlers.handle_translation_task = AsyncMock(return_value={"success": True})
sys.modules.setdefault("worker.handlers.translation_services", _fake_handlers)

from worker.routes.v1.health import router as health_router


# ---------------------------------------------------------------------------
# Minimal app with only the health router
# ---------------------------------------------------------------------------


def _make_worker_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)
    return app


@pytest.fixture(scope="module")
def worker_client():
    with TestClient(_make_worker_app(), raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestWorkerHealth:
    def test_returns_200(self, worker_client):
        resp = worker_client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_healthy(self, worker_client):
        resp = worker_client.get("/health")
        assert resp.json()["status"] == "healthy"

    def test_response_has_service_name(self, worker_client):
        resp = worker_client.get("/health")
        assert "service" in resp.json()


# ---------------------------------------------------------------------------
# GET /ready — when worker IS initialized
# ---------------------------------------------------------------------------


class TestWorkerReadyInitialized:
    def test_returns_200_when_initialized(self, worker_client):
        with patch("worker.routes.v1.health.WorkerLifecycle") as mock_lc:
            mock_lc.is_initialized.return_value = True
            resp = worker_client.get("/ready")
        assert resp.status_code == 200

    def test_response_has_status_ready(self, worker_client):
        with patch("worker.routes.v1.health.WorkerLifecycle") as mock_lc:
            mock_lc.is_initialized.return_value = True
            resp = worker_client.get("/ready")
        assert resp.json()["status"] == "ready"

    def test_response_has_models_loaded(self, worker_client):
        with patch("worker.routes.v1.health.WorkerLifecycle") as mock_lc:
            mock_lc.is_initialized.return_value = True
            resp = worker_client.get("/ready")
        assert resp.json()["models_loaded"] is True


# ---------------------------------------------------------------------------
# GET /ready — when worker is NOT initialized
# ---------------------------------------------------------------------------


class TestWorkerReadyNotInitialized:
    def test_returns_503_when_not_initialized(self, worker_client):
        with patch("worker.routes.v1.health.WorkerLifecycle") as mock_lc:
            mock_lc.is_initialized.return_value = False
            resp = worker_client.get("/ready")
        assert resp.status_code == 503

    def test_returns_503_on_unexpected_error(self, worker_client):
        with patch("worker.routes.v1.health.WorkerLifecycle") as mock_lc:
            mock_lc.is_initialized.side_effect = RuntimeError("unexpected")
            resp = worker_client.get("/ready")
        assert resp.status_code == 503
