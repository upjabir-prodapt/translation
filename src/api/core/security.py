"""Security utilities for authentication and authorization.

Authentication is delegated entirely to Apigee (see `apigee_auth.py`) --
Apigee verifies the caller's Entra identity and its own Google-signed ID
token before this service is ever reached. This module keeps the FastAPI
dependency seam (`get_current_user_context` / `get_current_user`) and the
`AuthenticatedUser` shape stable so route code does not need to change.
"""

import logging

from fastapi import Depends
from fastapi import Request

from src.api.core.apigee_auth import AuthenticatedUser
from src.api.core.apigee_auth import get_current_apigee_user

logger = logging.getLogger(__name__)

__all__ = [
    "AuthenticatedUser",
    "get_current_user",
    "get_current_user_context",
]


def get_current_user(
    user: AuthenticatedUser = Depends(get_current_apigee_user),  # noqa: B008
) -> AuthenticatedUser:
    """Backward-compatible dependency returning the authenticated user context."""
    return user


def get_current_user_context(
    request: Request,
    user: AuthenticatedUser = Depends(get_current_apigee_user),  # noqa: B008
) -> AuthenticatedUser:
    """FastAPI dependency to extract normalized user context from Apigee-verified headers."""
    request.state.user = user
    return user
