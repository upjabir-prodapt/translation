"""Unit tests for JWT auth helpers in `api/core/security.py`."""

from datetime import timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.api.core.security import create_access_token
from src.api.core.security import decode_and_verify_token
from src.api.core.security import get_current_user_context
from src.api.core.security import verify_token


def _claims() -> dict[str, str]:
    return {
        "sub": "user@colt.net",
        "business_unit": "engineering",
        "organization": "colt",
    }


class TestVerifyToken:
    async def test_raises_401_when_no_credentials(self):
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(None)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Not authenticated"

    async def test_valid_token_returns_payload(self):
        token = create_access_token(_claims())
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        payload = await verify_token(creds)
        assert payload["sub"] == "user@colt.net"
        assert payload["business_unit"] == "engineering"
        assert payload["organization"] == "colt"

    async def test_invalid_token_raises_401(self):
        creds = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid.token.value"
        )
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(creds)
        assert exc_info.value.status_code == 401


class TestTokenFunctions:
    def test_decode_and_verify_token_accepts_valid_claims(self):
        token = create_access_token(_claims())
        payload = decode_and_verify_token(token)
        assert payload["sub"] == "user@colt.net"

    def test_decode_and_verify_token_rejects_missing_required_claim(self):
        token = create_access_token({"sub": "user@colt.net"})
        with pytest.raises(HTTPException) as exc_info:
            decode_and_verify_token(token)
        assert exc_info.value.status_code == 401

    def test_create_access_token_respects_custom_expiry(self):
        token = create_access_token(_claims(), expires_delta=timedelta(minutes=1))
        payload = decode_and_verify_token(token)
        assert payload["exp"] > payload["iat"]


class TestCurrentUserContext:
    async def test_returns_normalized_context(self):
        token = create_access_token(_claims())
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        user = await get_current_user_context(creds)
        assert user.email == "user@colt.net"
        assert user.business_unit == "engineering"
        assert user.organization == "colt"
