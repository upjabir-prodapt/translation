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


class TestAuthRefreshRoute:
    @staticmethod
    def _issue(client):
        return client.post(
            "/api/v1/auth/token",
            headers=DEV_IAP_HEADER,
            json={"business_unit": "engineering", "organization": "colt"},
        ).json()

    def test_token_response_includes_refresh_token(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            body = self._issue(client)
        assert body["refresh_token"]
        assert body["refresh_expires_in"] > body["expires_in"]

    def test_refresh_with_body_token_returns_new_access_token(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            issued = self._issue(client)
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": issued["refresh_token"]},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["email"] == "user@colt.net"
        assert body["token_type"] == "bearer"  # noqa: S105
        assert body["expires_in"] > 0
        # Absolute session cap: refreshing renews access, never the window.
        assert body["refresh_token"] == issued["refresh_token"]
        assert body["refresh_expires_in"] <= issued["refresh_expires_in"]

    def test_refresh_uses_cookie_when_body_omits_token(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            self._issue(client)  # sets the colt_refresh cookie on the client
            response = client.post("/api/v1/auth/refresh", json={})
        assert response.status_code == 200
        assert response.json()["email"] == "user@colt.net"

    def test_refresh_without_any_token_is_401(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/v1/auth/refresh", json={})
        assert response.status_code == 401

    def test_refresh_rejects_access_token(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            issued = self._issue(client)
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": issued["access_token"]},
            )
        assert response.status_code == 401

    def test_refreshed_access_token_is_accepted_by_protected_route(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            issued = self._issue(client)
            refreshed = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": issued["refresh_token"]},
            ).json()
            response = client.get(
                "/api/v1/jobs",
                headers={"x-app-auth": f"Bearer {refreshed['access_token']}"},
            )
        # The handler itself 500s here (no Firestore under test); what matters
        # is that the request got past the auth dependency at all. Paired with
        # test_refresh_token_is_rejected_as_a_credential, which does 401.
        assert response.status_code != 401

    def test_refresh_token_is_rejected_as_a_credential(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            issued = self._issue(client)
            response = client.get(
                "/api/v1/jobs",
                headers={"x-app-auth": f"Bearer {issued['refresh_token']}"},
            )
        assert response.status_code == 401
