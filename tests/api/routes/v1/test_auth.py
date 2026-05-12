"""Tests for `POST /api/v1/auth/token`."""

from fastapi.testclient import TestClient
from src.api.main import app


class TestAuthTokenRoute:
    def test_returns_token_for_colt_email(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/auth/token",
                json={
                    "email": "user@colt.net",
                    "business_unit": "engineering",
                    "organization": "colt",
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"  # noqa: S105
        assert "access_token" in body
        assert body["expires_in"] > 0

    def test_rejects_non_colt_email(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/auth/token",
                json={
                    "email": "user@example.com",
                    "business_unit": "engineering",
                    "organization": "colt",
                },
            )
        assert response.status_code == 403
