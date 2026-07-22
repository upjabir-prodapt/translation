"""Authentication endpoints."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends

from src.api.core.iap_auth import get_iap_user
from src.api.core.security import create_access_token
from src.api.schemas.requests import AuthTokenRequest
from src.api.schemas.responses import AuthTokenResponse
from src.api.schemas.responses import WhoamiResponse
from src.config.constants import settings

router = APIRouter()


@router.get("/auth/whoami", tags=["auth"])
async def whoami(
    iap_email: Annotated[str, Depends(get_iap_user)],
) -> WhoamiResponse:
    """Return verified user email from IAP JWT (entitlement probe)."""
    return WhoamiResponse(email=iap_email)


@router.post("/auth/token", tags=["auth"])
async def create_auth_token(
    request: AuthTokenRequest,
    iap_email: Annotated[str, Depends(get_iap_user)],
) -> AuthTokenResponse:
    """Issue JWT using verified IAP identity and user-provided cost attribution."""
    expires_in = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    access_token = create_access_token(
        claims={
            "sub": iap_email,
            "business_unit": request.business_unit.strip(),
            "organization": request.organization.strip(),
        },
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    token_type = "bearer"  # noqa: S105
    return AuthTokenResponse(
        access_token=access_token,
        token_type=token_type,
        expires_in=expires_in,
        email=iap_email,
    )
