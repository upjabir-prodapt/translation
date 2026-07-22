"""Verify Cloud Tasks OIDC tokens on worker HTTP handlers."""

from __future__ import annotations

import logging

from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import id_token

from src.config.constants import settings

logger = logging.getLogger(__name__)


def _expected_audience() -> str:
    """Audience must match the URL Cloud Tasks uses when calling the worker."""
    return settings.WORKER_OIDC_AUDIENCE or settings.CLOUD_TASKS_WORKER_URL or ""


async def require_cloud_tasks_oidc(request: Request) -> dict:
    """FastAPI dependency: require a valid Google OIDC bearer token from Cloud Tasks."""
    if settings.WORKER_SKIP_OIDC_VERIFICATION:
        logger.warning("WORKER_SKIP_OIDC_VERIFICATION enabled — skipping OIDC check")
        return {"skipped": True}

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )
    token = auth[len("Bearer ") :].strip()
    audience = _expected_audience()
    if not audience:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Worker OIDC audience is not configured",
        )

    try:
        claims = id_token.verify_oauth2_token(
            token,
            google_auth_requests.Request(),
            audience=audience,
        )
    except Exception as e:
        logger.warning("OIDC verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OIDC token",
        ) from e

    expected_sa = settings.CLOUD_TASKS_OIDC_SERVICE_ACCOUNT.strip()
    email = (claims.get("email") or "").strip()
    if expected_sa and email and email != expected_sa:
        logger.warning("OIDC email mismatch: got %s expected %s", email, expected_sa)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unexpected service account",
        )

    return claims
