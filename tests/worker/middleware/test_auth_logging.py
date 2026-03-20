"""
Unit tests for worker/middleware/auth_logging.py.

WorkerAuthMiddleware and RequestLoggingMiddleware are tested via a minimal
FastAPI app and Starlette TestClient.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from worker.middleware.auth_logging import RequestLoggingMiddleware, WorkerAuthMiddleware


# ---------------------------------------------------------------------------
# Minimal apps
# ---------------------------------------------------------------------------


def _app_with_auth_middleware() -> FastAPI:
    app = FastAPI()
    app.add_middleware(WorkerAuthMiddleware)

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/process")
    def process():
        return {"status": "processed"}

    return app


def _app_with_logging_middleware() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    @app.get("/error")
    def error():
        raise RuntimeError("intentional error")

    return app


# ---------------------------------------------------------------------------
# WorkerAuthMiddleware
# ---------------------------------------------------------------------------


class TestWorkerAuthMiddleware:
    def test_health_endpoint_passes_through(self):
        client = TestClient(_app_with_auth_middleware(), raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_ready_endpoint_passes_through(self):
        app = FastAPI()
        app.add_middleware(WorkerAuthMiddleware)

        @app.get("/ready")
        def ready():
            return {"status": "ready"}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/ready")
        assert resp.status_code == 200

    def test_non_health_endpoint_proceeds(self):
        client = TestClient(_app_with_auth_middleware(), raise_server_exceptions=False)
        resp = client.get("/process")
        assert resp.status_code == 200

    def test_response_body_intact(self):
        client = TestClient(_app_with_auth_middleware(), raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# RequestLoggingMiddleware
# ---------------------------------------------------------------------------


class TestRequestLoggingMiddleware:
    def test_request_passes_through(self):
        client = TestClient(_app_with_logging_middleware(), raise_server_exceptions=False)
        resp = client.get("/ping")
        assert resp.status_code == 200

    def test_response_body_intact(self):
        client = TestClient(_app_with_logging_middleware(), raise_server_exceptions=False)
        resp = client.get("/ping")
        assert resp.json() == {"ok": True}

    def test_exception_propagates(self):
        client = TestClient(_app_with_logging_middleware(), raise_server_exceptions=False)
        resp = client.get("/error")
        assert resp.status_code == 500
