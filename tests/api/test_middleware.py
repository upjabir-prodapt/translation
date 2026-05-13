import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.middleware import LoggingMiddleware
from src.api.middleware import SecurityHeadersMiddleware

app = FastAPI()
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


@app.get("/test")
async def test_route():
    return {"ok": True}


class TestMiddleware:
    def test_logging_and_security_middleware(self):
        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"


def _load_middleware_file():
    """Load src/api/middleware.py directly (bypasses the package which shadows it)."""
    src_root = Path(__file__).parent.parent.parent
    middleware_path = src_root / "src" / "api" / "middleware.py"
    spec = importlib.util.spec_from_file_location("_standalone_middleware", str(middleware_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestStandaloneMiddlewareFile:
    """Exercise the standalone middleware.py file (shadowed by the package at runtime)."""

    def test_classes_defined(self):
        mod = _load_middleware_file()
        assert hasattr(mod, "LoggingMiddleware")
        assert hasattr(mod, "SecurityHeadersMiddleware")

    def test_logging_middleware_dispatch(self):
        mod = _load_middleware_file()
        standalone_app = FastAPI()
        standalone_app.add_middleware(mod.LoggingMiddleware)
        standalone_app.add_middleware(mod.SecurityHeadersMiddleware)

        @standalone_app.get("/ping")
        async def ping():
            return {"pong": True}

        client = TestClient(standalone_app)
        response = client.get("/ping")
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "includeSubDomains" in response.headers["Strict-Transport-Security"]
