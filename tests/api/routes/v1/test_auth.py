"""Tests for auth routes (`/api/v1/auth/token`, `/api/v1/auth/whoami`)."""

from fastapi.testclient import TestClient
from src.api.main import app

# TRANSLATION_REQUIRED_GROUP in tests/test.env is "ai-translation-users".
DEV_IAP_HEADER = {
    "X-Dev-IAP-User-Email": "user@colt.net",
    "x-dev-iap-user-groups": "ai-translation-users",
}


class TestAuthTokenRoute:
    def test_returns_token_for_colt_email(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/auth/token",
                headers=DEV_IAP_HEADER,
                json={
                    "business_unit": "engineering",
                    "organization": "colt",
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"  # noqa: S105
        assert "access_token" in body
        assert body["expires_in"] > 0
        assert body["email"] == "user@colt.net"

    def test_rejects_non_colt_email(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/auth/token",
                headers={
                    "X-Dev-IAP-User-Email": "user@example.com",
                    "x-dev-iap-user-groups": "ai-translation-users",
                },
                json={
                    "business_unit": "engineering",
                    "organization": "colt",
                },
            )
        assert response.status_code == 403

    def test_rejects_colt_email_without_required_group(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/auth/token",
                headers={"X-Dev-IAP-User-Email": "user@colt.net"},
                json={
                    "business_unit": "engineering",
                    "organization": "colt",
                },
            )
        assert response.status_code == 403

    def test_whoami_returns_verified_email(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/auth/whoami", headers=DEV_IAP_HEADER)
        assert response.status_code == 200
        assert response.json() == {"email": "user@colt.net", "entitled": True}

    def test_whoami_rejects_missing_required_group(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get(
                "/api/v1/auth/whoami",
                headers={"X-Dev-IAP-User-Email": "user@colt.net"},
            )
        assert response.status_code == 403
