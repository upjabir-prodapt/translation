import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import MagicMock, AsyncMock, patch

class TestMainApp:
    def test_health_check(self):
        with TestClient(app) as client:
            response = client.get("/api/v1/healthz")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    def test_root_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert "service" in response.json()
