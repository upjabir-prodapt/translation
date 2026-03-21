"""
Unit tests for api/core/security.py — verify_token(), get_current_user().
"""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from api.core.security import verify_token, get_current_user


# ---------------------------------------------------------------------------
# verify_token
# ---------------------------------------------------------------------------


class TestVerifyToken:
    async def test_raises_401_when_no_credentials(self):
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(None)
        assert exc_info.value.status_code == 401

    async def test_raises_401_detail_not_authenticated(self):
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(None)
        assert exc_info.value.detail == "Not authenticated"

    async def test_raises_401_with_bearer_header(self):
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(None)
        assert "WWW-Authenticate" in exc_info.value.headers

    async def test_returns_dict_with_valid_credentials(self):
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="sometoken")
        result = await verify_token(creds)
        assert isinstance(result, dict)

    async def test_returns_sub_field(self):
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="anytoken")
        result = await verify_token(creds)
        assert "sub" in result

    async def test_returns_email_field(self):
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="anytoken")
        result = await verify_token(creds)
        assert "email" in result

    async def test_returns_name_field(self):
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="anytoken")
        result = await verify_token(creds)
        assert "name" in result


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    def test_returns_coroutine_when_no_credentials(self):
        import inspect
        result = get_current_user(None)
        assert inspect.isawaitable(result)
        result.close()

    def test_returns_coroutine_with_credentials(self):
        import inspect
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tok")
        result = get_current_user(creds)
        assert inspect.isawaitable(result)
        result.close()
