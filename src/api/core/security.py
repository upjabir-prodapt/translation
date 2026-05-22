"""Security utilities for authentication and authorization."""

import logging
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

import jwt
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from fastapi.security import APIKeyHeader
from jwt import InvalidTokenError
from pydantic import BaseModel

from src.config.constants import settings

logger = logging.getLogger(__name__)

# x-app-auth token scheme
app_auth_scheme = APIKeyHeader(name="x-app-auth", auto_error=False)


class AuthenticatedUser(BaseModel):
    """Authenticated user context extracted from JWT claims."""

    email: str
    business_unit: str
    organization: str


def _jwt_secret() -> str:
    """Return configured JWT secret with safe local fallback."""
    if settings.JWT_SECRET_KEY:
        return settings.JWT_SECRET_KEY
    if settings.IS_LOCAL:
        return "local-dev-insecure-jwt-secret-that-is-at-least-32-bytes-long"
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="JWT secret is not configured",
    )


def create_access_token(
    claims: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """Create an HS256 signed JWT access token."""
    now = datetime.now(UTC)
    expires = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        **claims,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=settings.JWT_ALGORITHM)


def decode_and_verify_token(token: str) -> dict[str, Any]:
    """Decode and verify JWT token, including required claims."""
    try:
        payload = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=[settings.JWT_ALGORITHM],
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    required_claims = ("sub", "business_unit", "organization")
    missing = [claim for claim in required_claims if not payload.get(claim)]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing required token claims: {', '.join(missing)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def verify_token(
    api_key: str | None = Depends(app_auth_scheme),  # noqa: B008
) -> dict[str, Any]:
    """Verify bearer token and return JWT payload."""
    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = api_key[7:]
    return decode_and_verify_token(token)


def get_current_user(
    api_key: str | None = Depends(app_auth_scheme),  # noqa: B008
) -> dict[str, Any]:
    """Backward-compatible dependency returning raw token payload."""
    return verify_token(api_key)


def get_current_user_context(
    api_key: str | None = Depends(app_auth_scheme),  # noqa: B008
) -> AuthenticatedUser:
    """FastAPI dependency to extract normalized user context from JWT."""
    payload = verify_token(api_key)
    return AuthenticatedUser(
        email=str(payload["sub"]),
        business_unit=str(payload["business_unit"]),
        organization=str(payload["organization"]),
    )
