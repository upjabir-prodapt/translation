"""Tests for auth routes (`/api/v1/auth/token`, `/whoami`, `/refresh`, `/logout`)."""

import time
from datetime import timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from src.api.core import security
from src.api.core.security import SESSION_COOKIE_NAME
from src.api.core.security import create_access_token
from src.api.main import app
from src.config.constants import settings

# TRANSLATION_REQUIRED_GROUP in tests/test.env is "ai-translation-users".
DEV_IAP_HEADER = {
    "X-Dev-IAP-User-Email": "user@colt.net",
    "x-dev-iap-user-groups": "ai-translation-users",
}

BASE_CLAIMS = {
    "sub": "user@colt.net",
    "business_unit": "engineering",
    "organization": "colt",
}


def _decode(token: str) -> dict:
    return jwt.decode(
        token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
    )


def _session_token(**overrides) -> str:
    """Mint a session token with sane defaults, overridable per test."""
    claims = {
        **BASE_CLAIMS,
        "auth_time": int(time.time()),
        "scopes": ["translation"],
        **overrides,
    }
    # `None` means "omit this claim entirely" -- distinct from an empty list,
    # which is what a user with no entitlements would legitimately carry.
    claims = {k: v for k, v in claims.items() if v is not None}
    return create_access_token(claims=claims, expires_delta=timedelta(minutes=30))


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


class TestMintedClaims:
    def test_token_carries_auth_time_and_own_scope(self):
        before = int(time.time())
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/auth/token",
                headers=DEV_IAP_HEADER,
                json={"business_unit": "engineering", "organization": "colt"},
            )
        assert response.status_code == 200
        payload = _decode(response.json()["access_token"])
        assert payload["scopes"] == ["translation"]
        assert before <= payload["auth_time"] <= int(time.time())

    def test_token_carries_sales_scope_when_entitled_to_both(self, monkeypatch):
        """The cookie is shared, so a dual-entitled user must get both scopes.

        If Translation stamped only its own scope, hubLogin()'s Promise.all
        would leave whichever response landed last in charge of the cookie and
        the other service would 403 on every call.
        """

        async def _mock_scopes(_identity):
            return {"translation", "sales"}

        monkeypatch.setattr(
            "src.api.routes.v1.auth.resolve_session_scopes", _mock_scopes
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/auth/token",
                headers=DEV_IAP_HEADER,
                json={"business_unit": "engineering", "organization": "colt"},
            )
        assert response.status_code == 200
        payload = _decode(response.json()["access_token"])
        assert payload["scopes"] == ["sales", "translation"]


class TestScopeEnforcement:
    def test_data_route_rejects_token_scoped_to_other_service_only(self):
        """The cross-service bypass this claim exists to close."""
        token = _session_token(scopes=["sales"])
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.get("/api/v1/jobs")
        assert response.status_code == 403

    def test_legacy_token_without_scopes_allowed_while_lenient(self):
        assert settings.REQUIRE_SCOPE_CLAIM is False
        token = _session_token(scopes=None)
        payload = security.decode_and_verify_token(token)
        assert payload["sub"] == "user@colt.net"

    def test_legacy_token_without_scopes_rejected_when_strict(self, monkeypatch):
        monkeypatch.setattr(settings, "REQUIRE_SCOPE_CLAIM", True)
        token = _session_token(scopes=None)
        with pytest.raises(Exception) as exc:
            security.decode_and_verify_token(token)
        assert getattr(exc.value, "status_code", None) == 403

    def test_empty_scopes_rejected_even_while_lenient(self):
        """Leniency covers *absent* scopes only.

        A token that carries `scopes: []` was minted by a scope-aware service
        for a user with no entitlements -- honouring it would reopen the very
        hole the leniency window is meant to close gradually.
        """
        token = _session_token(scopes=[])
        with pytest.raises(Exception) as exc:
            security.decode_and_verify_token(token)
        assert getattr(exc.value, "status_code", None) == 403


class TestRefreshRoute:
    def test_refresh_with_no_body_returns_new_token_and_cookie(self):
        """A body-less POST must not 422: the UI renews on a timer with nothing to send."""
        token = _session_token()
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 200
        body = response.json()
        assert body["email"] == "user@colt.net"
        assert body["expires_in"] == settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        assert SESSION_COOKIE_NAME in response.headers.get("set-cookie", "")

    def test_refresh_preserves_auth_time_and_extends_expiry(self):
        original_auth_time = int(time.time()) - 3600
        # Deliberately short-lived, so "the new token outlives the old one" is
        # observable rather than two identical 30-minute expiries in the same
        # wall-clock second.
        token = create_access_token(
            claims={
                **BASE_CLAIMS,
                "auth_time": original_auth_time,
                "scopes": ["translation"],
            },
            expires_delta=timedelta(minutes=5),
        )
        old = _decode(token)

        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")

        assert response.status_code == 200
        new = _decode(response.json()["access_token"])
        # Preserved, not restamped -- otherwise the absolute cap would slide
        # forward with every renewal and never be reached.
        assert new["auth_time"] == original_auth_time
        assert new["exp"] > old["exp"]
        assert new["sub"] == old["sub"]
        assert new["business_unit"] == old["business_unit"]
        assert new["organization"] == old["organization"]

    def test_refresh_rejects_expired_token(self):
        token = create_access_token(
            claims={
                **BASE_CLAIMS,
                "auth_time": int(time.time()) - 60,
                "scopes": ["translation"],
            },
            expires_delta=timedelta(minutes=-5),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 401

    def test_refresh_rejects_unauthenticated_caller(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 401

    def test_refresh_past_absolute_cap_401s_and_clears_cookie(self):
        beyond_cap = (
            int(time.time()) - (settings.SESSION_ABSOLUTE_MAX_MINUTES * 60) - 60
        )
        token = _session_token(auth_time=beyond_cap)
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 401
        set_cookie = response.headers.get("set-cookie", "")
        assert SESSION_COOKIE_NAME in set_cookie
        assert "Max-Age=0" in set_cookie or "expires=" in set_cookie.lower()

    def test_refresh_falls_back_to_iat_for_legacy_token_past_cap(self):
        """Pre-`auth_time` tokens still get a ceiling, derived from `iat`."""
        token = create_access_token(
            claims={**BASE_CLAIMS, "scopes": ["translation"]},
            expires_delta=timedelta(minutes=30),
        )
        stale_iat = int(time.time()) - (settings.SESSION_ABSOLUTE_MAX_MINUTES * 60) - 60
        payload = _decode(token)
        payload["iat"] = stale_iat
        forged = jwt.encode(
            payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, forged)
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 401

    def test_refresh_403s_and_clears_cookie_when_entitlement_revoked(self, monkeypatch):
        """Revocation takes effect at the next renewal, not the next login."""
        monkeypatch.setattr(settings, "IS_LOCAL", False)

        async def _no_scopes(email: str) -> set[str]:
            assert email == "user@colt.net"
            return set()

        monkeypatch.setattr("src.api.routes.v1.auth.resolve_scopes", _no_scopes)

        token = _session_token()
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")

        assert response.status_code == 403
        set_cookie = response.headers.get("set-cookie", "")
        assert SESSION_COOKIE_NAME in set_cookie
        assert "Max-Age=0" in set_cookie or "expires=" in set_cookie.lower()

    def test_refresh_restamps_freshly_resolved_scopes(self, monkeypatch):
        """Scopes come from Firestore each time, so a newly granted service appears."""
        monkeypatch.setattr(settings, "IS_LOCAL", False)

        async def _both(_email: str) -> set[str]:
            return {"translation", "sales"}

        monkeypatch.setattr("src.api.routes.v1.auth.resolve_scopes", _both)

        token = _session_token(scopes=["translation"])
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")

        assert response.status_code == 200
        assert _decode(response.json()["access_token"])["scopes"] == [
            "sales",
            "translation",
        ]

    def test_refresh_normalises_email_before_lookup(self, monkeypatch):
        monkeypatch.setattr(settings, "IS_LOCAL", False)
        seen: list[str] = []

        async def _record(email: str) -> set[str]:
            seen.append(email)
            return {"translation"}

        monkeypatch.setattr("src.api.routes.v1.auth.resolve_scopes", _record)

        token = _session_token(sub="  User@Colt.NET ")
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")

        assert response.status_code == 200
        assert seen == ["user@colt.net"]


class TestLogoutRoute:
    def test_logout_clears_cookie_without_authentication(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/v1/auth/logout")
        assert response.status_code == 204
        set_cookie = response.headers.get("set-cookie", "")
        assert SESSION_COOKIE_NAME in set_cookie
        assert "Max-Age=0" in set_cookie or "expires=" in set_cookie.lower()


import time
from datetime import timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from src.api.core import security
from src.api.core.security import SESSION_COOKIE_NAME
from src.api.core.security import create_access_token
from src.api.main import app
from src.config.constants import settings

# TRANSLATION_REQUIRED_GROUP in tests/test.env is "ai-translation-users".
DEV_IAP_HEADER = {
    "X-Dev-IAP-User-Email": "user@colt.net",
    "x-dev-iap-user-groups": "ai-translation-users",
}

BASE_CLAIMS = {
    "sub": "user@colt.net",
    "business_unit": "engineering",
    "organization": "colt",
}


def _decode(token: str) -> dict:
    return jwt.decode(
        token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
    )


def _session_token(**overrides) -> str:
    """Mint a session token with sane defaults, overridable per test."""
    claims = {
        **BASE_CLAIMS,
        "auth_time": int(time.time()),
        "scopes": ["translation"],
        **overrides,
    }
    # `None` means "omit this claim entirely" -- distinct from an empty list,
    # which is what a user with no entitlements would legitimately carry.
    claims = {k: v for k, v in claims.items() if v is not None}
    return create_access_token(claims=claims, expires_delta=timedelta(minutes=30))


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


class TestMintedClaims:
    def test_token_carries_auth_time_and_own_scope(self):
        before = int(time.time())
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/auth/token",
                headers=DEV_IAP_HEADER,
                json={"business_unit": "engineering", "organization": "colt"},
            )
        assert response.status_code == 200
        payload = _decode(response.json()["access_token"])
        assert payload["scopes"] == ["translation"]
        assert before <= payload["auth_time"] <= int(time.time())

    def test_token_carries_sales_scope_when_entitled_to_both(self, monkeypatch):
        """The cookie is shared, so a dual-entitled user must get both scopes.

        If Translation stamped only its own scope, hubLogin()'s Promise.all
        would leave whichever response landed last in charge of the cookie and
        the other service would 403 on every call.
        """

        async def _mock_scopes(_identity):
            return {"translation", "sales"}

        monkeypatch.setattr(
            "src.api.routes.v1.auth.resolve_session_scopes", _mock_scopes
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/auth/token",
                headers=DEV_IAP_HEADER,
                json={"business_unit": "engineering", "organization": "colt"},
            )
        assert response.status_code == 200
        payload = _decode(response.json()["access_token"])
        assert payload["scopes"] == ["sales", "translation"]


class TestScopeEnforcement:
    def test_data_route_rejects_token_scoped_to_other_service_only(self):
        """The cross-service bypass this claim exists to close."""
        token = _session_token(scopes=["sales"])
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.get("/api/v1/jobs")
        assert response.status_code == 403

    def test_legacy_token_without_scopes_allowed_while_lenient(self):
        assert settings.REQUIRE_SCOPE_CLAIM is False
        token = _session_token(scopes=None)
        payload = security.decode_and_verify_token(token)
        assert payload["sub"] == "user@colt.net"

    def test_legacy_token_without_scopes_rejected_when_strict(self, monkeypatch):
        monkeypatch.setattr(settings, "REQUIRE_SCOPE_CLAIM", True)
        token = _session_token(scopes=None)
        with pytest.raises(Exception) as exc:
            security.decode_and_verify_token(token)
        assert getattr(exc.value, "status_code", None) == 403

    def test_empty_scopes_rejected_even_while_lenient(self):
        """Leniency covers *absent* scopes only.

        A token that carries `scopes: []` was minted by a scope-aware service
        for a user with no entitlements -- honouring it would reopen the very
        hole the leniency window is meant to close gradually.
        """
        token = _session_token(scopes=[])
        with pytest.raises(Exception) as exc:
            security.decode_and_verify_token(token)
        assert getattr(exc.value, "status_code", None) == 403


class TestRefreshRoute:
    def test_refresh_with_no_body_returns_new_token_and_cookie(self):
        """A body-less POST must not 422: the UI renews on a timer with nothing to send."""
        token = _session_token()
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 200
        body = response.json()
        assert body["email"] == "user@colt.net"
        assert body["expires_in"] == settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        assert SESSION_COOKIE_NAME in response.headers.get("set-cookie", "")

    def test_refresh_preserves_auth_time_and_extends_expiry(self):
        original_auth_time = int(time.time()) - 3600
        # Deliberately short-lived, so "the new token outlives the old one" is
        # observable rather than two identical 30-minute expiries in the same
        # wall-clock second.
        token = create_access_token(
            claims={
                **BASE_CLAIMS,
                "auth_time": original_auth_time,
                "scopes": ["translation"],
            },
            expires_delta=timedelta(minutes=5),
        )
        old = _decode(token)

        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")

        assert response.status_code == 200
        new = _decode(response.json()["access_token"])
        # Preserved, not restamped -- otherwise the absolute cap would slide
        # forward with every renewal and never be reached.
        assert new["auth_time"] == original_auth_time
        assert new["exp"] > old["exp"]
        assert new["sub"] == old["sub"]
        assert new["business_unit"] == old["business_unit"]
        assert new["organization"] == old["organization"]

    def test_refresh_rejects_expired_token(self):
        token = create_access_token(
            claims={
                **BASE_CLAIMS,
                "auth_time": int(time.time()) - 60,
                "scopes": ["translation"],
            },
            expires_delta=timedelta(minutes=-5),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 401

    def test_refresh_rejects_unauthenticated_caller(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 401

    def test_refresh_past_absolute_cap_401s_and_clears_cookie(self):
        beyond_cap = (
            int(time.time()) - (settings.SESSION_ABSOLUTE_MAX_MINUTES * 60) - 60
        )
        token = _session_token(auth_time=beyond_cap)
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 401
        set_cookie = response.headers.get("set-cookie", "")
        assert SESSION_COOKIE_NAME in set_cookie
        assert "Max-Age=0" in set_cookie or "expires=" in set_cookie.lower()

    def test_refresh_falls_back_to_iat_for_legacy_token_past_cap(self):
        """Pre-`auth_time` tokens still get a ceiling, derived from `iat`."""
        token = create_access_token(
            claims={**BASE_CLAIMS, "scopes": ["translation"]},
            expires_delta=timedelta(minutes=30),
        )
        stale_iat = int(time.time()) - (settings.SESSION_ABSOLUTE_MAX_MINUTES * 60) - 60
        payload = _decode(token)
        payload["iat"] = stale_iat
        forged = jwt.encode(
            payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, forged)
            response = client.post("/api/v1/auth/refresh")
        assert response.status_code == 401

    def test_refresh_403s_and_clears_cookie_when_entitlement_revoked(self, monkeypatch):
        """Revocation takes effect at the next renewal, not the next login."""
        monkeypatch.setattr(settings, "IS_LOCAL", False)

        async def _no_scopes(email: str) -> set[str]:
            assert email == "user@colt.net"
            return set()

        monkeypatch.setattr("src.api.routes.v1.auth.resolve_scopes", _no_scopes)

        token = _session_token()
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")

        assert response.status_code == 403
        set_cookie = response.headers.get("set-cookie", "")
        assert SESSION_COOKIE_NAME in set_cookie
        assert "Max-Age=0" in set_cookie or "expires=" in set_cookie.lower()

    def test_refresh_restamps_freshly_resolved_scopes(self, monkeypatch):
        """Scopes come from Firestore each time, so a newly granted service appears."""
        monkeypatch.setattr(settings, "IS_LOCAL", False)

        async def _both(_email: str) -> set[str]:
            return {"translation", "sales"}

        monkeypatch.setattr("src.api.routes.v1.auth.resolve_scopes", _both)

        token = _session_token(scopes=["translation"])
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")

        assert response.status_code == 200
        assert _decode(response.json()["access_token"])["scopes"] == [
            "sales",
            "translation",
        ]

    def test_refresh_normalises_email_before_lookup(self, monkeypatch):
        monkeypatch.setattr(settings, "IS_LOCAL", False)
        seen: list[str] = []

        async def _record(email: str) -> set[str]:
            seen.append(email)
            return {"translation"}

        monkeypatch.setattr("src.api.routes.v1.auth.resolve_scopes", _record)

        token = _session_token(sub="  User@Colt.NET ")
        with TestClient(app, raise_server_exceptions=False) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            response = client.post("/api/v1/auth/refresh")

        assert response.status_code == 200
        assert seen == ["user@colt.net"]


class TestLogoutRoute:
    def test_logout_clears_cookie_without_authentication(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/v1/auth/logout")
        assert response.status_code == 204
        set_cookie = response.headers.get("set-cookie", "")
        assert SESSION_COOKIE_NAME in set_cookie
        assert "Max-Age=0" in set_cookie or "expires=" in set_cookie.lower()
