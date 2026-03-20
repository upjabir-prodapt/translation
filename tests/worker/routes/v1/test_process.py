"""
Integration tests for worker/routes/v1/process.py — POST /process.

worker.handlers.translation_services is stubbed out so no real translation occurs.
"""

import sys
import types
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Stub the missing handler module before importing the route
_fake_handlers = types.ModuleType("worker.handlers.translation_services")
_fake_handlers.handle_translation_task = AsyncMock(return_value={"success": True})
sys.modules.setdefault("worker.handlers.translation_services", _fake_handlers)

from worker.routes.v1.process import router  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal app
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture(scope="module")
def process_client():
    with TestClient(_make_app(), raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Successful task
# ---------------------------------------------------------------------------


class TestProcessSuccess:
    def test_returns_200_on_success(self, process_client):
        payload = {"job_id": "job-1", "config": {"lang_in": "en", "lang_out": "fr"}}
        with patch(
            "worker.routes.v1.process.handle_translation_task",
            new=AsyncMock(return_value={"success": True}),
        ):
            resp = process_client.post("/process", json=payload)
        assert resp.status_code == 200

    def test_returns_empty_body_on_success(self, process_client):
        payload = {"job_id": "job-1", "config": {"lang_in": "en", "lang_out": "fr"}}
        with patch(
            "worker.routes.v1.process.handle_translation_task",
            new=AsyncMock(return_value={"success": True}),
        ):
            resp = process_client.post("/process", json=payload)
        assert resp.text == '""'


# ---------------------------------------------------------------------------
# Task returns failure
# ---------------------------------------------------------------------------


class TestProcessFailure:
    def test_returns_500_when_success_false(self, process_client):
        payload = {"job_id": "job-1", "config": {"lang_in": "en", "lang_out": "fr"}}
        with patch(
            "worker.routes.v1.process.handle_translation_task",
            new=AsyncMock(return_value={"success": False, "error": "translation failed"}),
        ):
            resp = process_client.post("/process", json=payload)
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Validation errors (missing fields)
# ---------------------------------------------------------------------------


class TestProcessValidation:
    def test_missing_job_id_returns_400(self, process_client):
        resp = process_client.post("/process", json={"config": {}})
        assert resp.status_code == 400

    def test_missing_config_returns_400(self, process_client):
        resp = process_client.post("/process", json={"job_id": "j1"})
        assert resp.status_code == 400

    def test_empty_payload_returns_400(self, process_client):
        resp = process_client.post("/process", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Unexpected exception
# ---------------------------------------------------------------------------


class TestProcessUnexpectedException:
    def test_unexpected_exception_returns_500(self, process_client):
        payload = {"job_id": "job-1", "config": {"lang_in": "en", "lang_out": "fr"}}
        with patch(
            "worker.routes.v1.process.handle_translation_task",
            new=AsyncMock(side_effect=RuntimeError("unexpected crash")),
        ):
            resp = process_client.post("/process", json=payload)
        assert resp.status_code == 500
