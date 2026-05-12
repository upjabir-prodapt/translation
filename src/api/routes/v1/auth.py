"""Authentication endpoints."""

from datetime import timedelta

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from src.api.core.security import create_access_token
from src.api.schemas.requests import AuthTokenRequest
from src.api.schemas.responses import AuthTokenResponse
from src.config.constants import settings

router = APIRouter()


@router.post("/auth/token", tags=["auth"])
async def create_auth_token(request: AuthTokenRequest) -> AuthTokenResponse:
    """Issue JWT for users with `@colt.net` email addresses."""
    normalized_email = request.email.strip().lower()
    if not normalized_email.endswith("@colt.net"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only @colt.net email addresses are allowed",
        )

    expires_in = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    access_token = create_access_token(
        claims={
            "sub": normalized_email,
            "business_unit": request.business_unit,
            "organization": request.organization,
        },
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return AuthTokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
    )
