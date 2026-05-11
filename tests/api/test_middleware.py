"""
Unit tests for api/middleware.py — LoggingMiddleware, SecurityHeadersMiddleware.

Uses FastAPI TestClient with a minimal app so dispatch() is exercised end-to-end.
"""

import importlib.util
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# api/middleware.py is shadowed by the api/middleware/ package, so we must
# load it directly via its file path.
_mw_path = Path(__file__).parent.parent.parent / "src" / "api" / "middleware.py"
_spec = importlib.util.spec_from_file_location("api_middleware_module", str(_mw_path))
_mw_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mw_module)
LoggingMiddleware = _mw_module.LoggingMiddleware
SecurityHeadersMiddleware = _mw_module.SecurityHeadersMiddleware


# ---------------------------------------------------------------------------
# Helpers — minimal apps with middleware
# ---------------------------------------------------------------------------


def _app_with_logging_middleware() -> FastAPI:
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


def _app_with_security_middleware() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------


class TestLoggingMiddleware:
    def test_request_passes_through(self):
        client = TestClient(_app_with_logging_middleware(), raise_server_exceptions=True)
        resp = client.get("/ping")
        assert resp.status_code == 200

    def test_response_body_intact(self):
        client = TestClient(_app_with_logging_middleware(), raise_server_exceptions=True)
        resp = client.get("/ping")
        assert resp.json() == {"ok": True}

    def test_x_process_time_header_present(self):
        client = TestClient(_app_with_logging_middleware(), raise_server_exceptions=True)
        resp = client.get("/ping")
        assert "x-process-time" in resp.headers

    def test_x_process_time_is_numeric(self):
        client = TestClient(_app_with_logging_middleware(), raise_server_exceptions=True)
        resp = client.get("/ping")
        val = resp.headers.get("x-process-time", "")
        assert float(val) >= 0.0


# ---------------------------------------------------------------------------
# SecurityHeadersMiddleware
# ---------------------------------------------------------------------------


class TestSecurityHeadersMiddleware:
    def test_request_passes_through(self):
        client = TestClient(_app_with_security_middleware(), raise_server_exceptions=True)
        resp = client.get("/ping")
        assert resp.status_code == 200

    def test_x_content_type_options(self):
        client = TestClient(_app_with_security_middleware(), raise_server_exceptions=True)
        resp = client.get("/ping")
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_x_frame_options(self):
        client = TestClient(_app_with_security_middleware(), raise_server_exceptions=True)
        resp = client.get("/ping")
        assert resp.headers.get("x-frame-options") == "DENY"

    def test_x_xss_protection(self):
        client = TestClient(_app_with_security_middleware(), raise_server_exceptions=True)
        resp = client.get("/ping")
        assert resp.headers.get("x-xss-protection") == "1; mode=block"

    def test_strict_transport_security(self):
        client = TestClient(_app_with_security_middleware(), raise_server_exceptions=True)
        resp = client.get("/ping")
        hsts = resp.headers.get("strict-transport-security", "")
        assert "max-age=31536000" in hsts
        assert "includeSubDomains" in hsts
