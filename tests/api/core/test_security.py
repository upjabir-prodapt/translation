"""Tests for the security.py delegation seam.

Authentication logic itself lives in apigee_auth.py (see
tests/api/core/test_apigee_auth.py); these tests only confirm that
get_current_user / get_current_user_context still delegate to
get_current_apigee_user and expose the same AuthenticatedUser shape that
existing route code depends on.
"""

from fastapi import Depends
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.core import security
from src.api.core.security import AuthenticatedUser
from src.api.core.security import get_current_user
from src.api.core.security import get_current_user_context


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/context")
    def context_route(
        user: AuthenticatedUser = Depends(get_current_user_context),  # noqa: B008
    ):
        return user.model_dump()

    @app.get("/user")
    def user_route(
        user: AuthenticatedUser = Depends(get_current_user),  # noqa: B008
    ):
        return user.model_dump()

    return app


_FAKE_USER = AuthenticatedUser(
    oid="oid-1",
    email="user@colt.net",
    roles=["Translation.User"],
    business_unit="engineering",
    organization="colt",
)


class TestGetCurrentUserContext:
    def test_delegates_to_apigee_dependency_and_returns_user(self):
        app = _make_app()
        app.dependency_overrides[security.get_current_apigee_user] = lambda: _FAKE_USER

        with TestClient(app) as client:
            resp = client.get("/context")

        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "user@colt.net"
        assert body["business_unit"] == "engineering"
        assert body["organization"] == "colt"
        assert body["oid"] == "oid-1"
        assert body["roles"] == ["Translation.User"]


class TestGetCurrentUser:
    def test_delegates_to_apigee_dependency_and_returns_user(self):
        app = _make_app()
        app.dependency_overrides[security.get_current_apigee_user] = lambda: _FAKE_USER

        with TestClient(app) as client:
            resp = client.get("/user")

        assert resp.status_code == 200
        assert resp.json()["email"] == "user@colt.net"


class TestAuthenticatedUserShape:
    def test_matches_pre_existing_field_names_and_defaults(self):
        """Route code (translate.py, jobs.py, reviews_handler.py) only reads
        .email / .business_unit / .organization -- these must keep working
        unchanged, and the new .oid / .roles fields must be safely optional."""
        user = AuthenticatedUser(
            email="test@example.com", business_unit="bu1", organization="org1"
        )
        assert user.email == "test@example.com"
        assert user.business_unit == "bu1"
        assert user.organization == "org1"
        assert user.oid == ""
        assert user.roles == []
