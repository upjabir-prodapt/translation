"""Security utilities for authentication and authorization."""

import logging

from fastapi import HTTPException
from fastapi import status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


async def verify_token(credentials: HTTPAuthorizationCredentials | None) -> dict:
    """Verify JWT token and return payload.

    For now, this is a placeholder. In production, integrate with:
    - Google Identity Platform
    - Firebase Auth
    - Custom OAuth provider

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        Dict with user information

    Raises:
        HTTPException: If token is invalid
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # TODO: Implement actual JWT verification
    # For development, accept any token
    return {"sub": "user123", "email": "user@example.com", "name": "Test User"}


def get_current_user(credentials: HTTPAuthorizationCredentials | None = None):
    """FastAPI dependency to get current user."""
    return verify_token(credentials)
