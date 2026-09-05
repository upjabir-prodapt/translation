"""Tests for Apigee-based authentication (src/api/core/apigee_auth.py)."""

from unittest.mock import patch

from fastapi import Depends
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.core import apigee_auth
from src.api.core.apigee_auth import AuthenticatedUser
from src.api.core.apigee_auth import get_current_apigee_user
from src.config.constants import settings

VALID_HEADERS = {
    "x-colt-user-oid": "oid-123",
    "x-colt-user-email": "user@colt.net",
    "x-colt-user-roles": "Translation.User,Translation.Admin",
    "x-colt-user-department": "engineering",
    "x-colt-user-company": "colt",
}

EXPECTED_SA_EMAIL = "apigee-int-runtime@gclt-aicoe-dev-apigee.iam.gserviceaccount.com"
SERVICE_URL = "https://translation-api-xxxxx.run.app"


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/whoami")
    def whoami(
        user: AuthenticatedUser = Depends(get_current_apigee_user),  # noqa: B008
    ):
        return user.model_dump()

    return app


class TestLocalDevPath:
    def test_uses_headers_when_present(self, monkeypatch):
        monkeypatch.setattr(settings, "IS_LOCAL", True)
        with TestClient(_make_app()) as client:
            resp = client.get("/whoami", headers=VALID_HEADERS)

        assert resp.status_code == 200
        body = resp.json()
        assert body["oid"] == "oid-123"
        assert body["email"] == "user@colt.net"
        assert body["roles"] == ["Translation.User", "Translation.Admin"]
        assert body["business_unit"] == "engineering"
        assert body["organization"] == "colt"

    def test_falls_back_to_fake_identity_when_headers_absent(self, monkeypatch):
        monkeypatch.setattr(settings, "IS_LOCAL", True)
        with TestClient(_make_app()) as client:
            resp = client.get("/whoami")

        assert resp.status_code == 200
        body = resp.json()
        assert body["oid"] == "local-dev"
        assert body["email"] == "local-dev@example.com"
        assert body["roles"] == ["Translation.User"]
        assert body["business_unit"] == ""
        assert body["organization"] == ""

    def test_missing_department_and_company_default_to_empty_not_crash(
        self, monkeypatch
    ):
        monkeypatch.setattr(settings, "IS_LOCAL", True)
        headers = {
            k: v
            for k, v in VALID_HEADERS.items()
            if k not in ("x-colt-user-department", "x-colt-user-company")
        }
        with TestClient(_make_app()) as client:
            resp = client.get("/whoami", headers=headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["business_unit"] == ""
        assert body["organization"] == ""
        # Identity headers are still honored even though department/company aren't set.
        assert body["oid"] == "oid-123"
        assert body["email"] == "user@colt.net"


class TestApigeeIdentityVerification:
    def _apply_prod_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "IS_LOCAL", False)
        monkeypatch.setattr(settings, "CLOUD_RUN_SERVICE_URL", SERVICE_URL)
        monkeypatch.setattr(settings, "APIGEE_RUNTIME_SA_EMAIL", EXPECTED_SA_EMAIL)

    def test_valid_token_via_authorization_header_succeeds(self, monkeypatch):
        self._apply_prod_settings(monkeypatch)
        with (
            patch.object(
                apigee_auth.id_token,
                "verify_oauth2_token",
                return_value={"email": EXPECTED_SA_EMAIL},
            ) as mock_verify,
            TestClient(_make_app()) as client,
        ):
            resp = client.get(
                "/whoami",
                headers={**VALID_HEADERS, "Authorization": "Bearer fake-google-token"},
            )

        assert resp.status_code == 200
        assert resp.json()["email"] == "user@colt.net"
        mock_verify.assert_called_once()
        assert mock_verify.call_args.kwargs["audience"] == SERVICE_URL

    def test_valid_token_without_bearer_prefix_succeeds(self, monkeypatch):
        """Bearer prefix is optional -- Apigee may send the raw token."""
        self._apply_prod_settings(monkeypatch)
        with (
            patch.object(
                apigee_auth.id_token,
                "verify_oauth2_token",
                return_value={"email": EXPECTED_SA_EMAIL},
            ),
            TestClient(_make_app()) as client,
        ):
            resp = client.get(
                "/whoami",
                headers={**VALID_HEADERS, "Authorization": "raw-google-token"},
            )

        assert resp.status_code == 200

    def test_valid_token_via_x_serverless_authorization_header_succeeds(
        self, monkeypatch
    ):
        self._apply_prod_settings(monkeypatch)
        with (
            patch.object(
                apigee_auth.id_token,
                "verify_oauth2_token",
                return_value={"email": EXPECTED_SA_EMAIL},
            ),
            TestClient(_make_app()) as client,
        ):
            resp = client.get(
                "/whoami",
                headers={
                    **VALID_HEADERS,
                    "X-Serverless-Authorization": "Bearer fake-google-token",
                },
            )

        assert resp.status_code == 200

    def test_missing_token_returns_401(self, monkeypatch):
        self._apply_prod_settings(monkeypatch)
        with TestClient(_make_app()) as client:
            resp = client.get("/whoami", headers=VALID_HEADERS)

        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, monkeypatch):
        self._apply_prod_settings(monkeypatch)
        with (
            patch.object(
                apigee_auth.id_token,
                "verify_oauth2_token",
                side_effect=ValueError("bad signature"),
            ),
            TestClient(_make_app()) as client,
        ):
            resp = client.get(
                "/whoami",
                headers={**VALID_HEADERS, "Authorization": "Bearer bad-token"},
            )

        assert resp.status_code == 401

    def test_wrong_service_account_email_returns_403(self, monkeypatch):
        self._apply_prod_settings(monkeypatch)
        with (
            patch.object(
                apigee_auth.id_token,
                "verify_oauth2_token",
                return_value={"email": "someone-else@example.com"},
            ),
            TestClient(_make_app()) as client,
        ):
            resp = client.get(
                "/whoami",
                headers={**VALID_HEADERS, "Authorization": "Bearer fake-token"},
            )

        assert resp.status_code == 403

    def test_missing_oid_header_returns_401_even_with_valid_token(self, monkeypatch):
        self._apply_prod_settings(monkeypatch)
        headers = {k: v for k, v in VALID_HEADERS.items() if k != "x-colt-user-oid"}
        with (
            patch.object(
                apigee_auth.id_token,
                "verify_oauth2_token",
                return_value={"email": EXPECTED_SA_EMAIL},
            ),
            TestClient(_make_app()) as client,
        ):
            resp = client.get(
                "/whoami",
                headers={**headers, "Authorization": "Bearer fake-token"},
            )

        assert resp.status_code == 401
