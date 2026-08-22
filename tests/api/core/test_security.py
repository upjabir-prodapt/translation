from datetime import timedelta

import pytest
from fastapi import HTTPException
from src.api.core.security import create_access_token
from src.api.core.security import decode_and_verify_token


class TestSecurity:
    def test_create_and_verify_token_success(self):
        claims = {
            "sub": "user@test.com",
            "business_unit": "bu1",
            "organization": "org1",
        }
        token = create_access_token(claims)

        payload = decode_and_verify_token(token)
        assert payload["sub"] == "user@test.com"
        assert payload["business_unit"] == "bu1"

    def test_decode_invalid_token(self):
        with pytest.raises(HTTPException) as exc:
            decode_and_verify_token("invalid-token")
        assert exc.value.status_code == 401

    def test_decode_missing_claims(self):
        claims = {"sub": "user@test.com"}  # missing bu and org
        token = create_access_token(claims)
        with pytest.raises(HTTPException) as exc:
            decode_and_verify_token(token)
        assert "Missing required token claims" in exc.value.detail

    def test_create_token_custom_expiry(self):
        token = create_access_token({"sub": "u"}, expires_delta=timedelta(seconds=1))
        assert token is not None
