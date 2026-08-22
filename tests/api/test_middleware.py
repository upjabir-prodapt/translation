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
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "includeSubDomains" in response.headers["Strict-Transport-Security"]
