"""Authentication endpoints."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends

from src.api.core.iap_auth import IapIdentity
from src.api.core.iap_auth import require_group
from src.api.core.security import create_access_token
from src.api.schemas.requests import AuthTokenRequest
from src.api.schemas.responses import AuthTokenResponse
from src.api.schemas.responses import WhoamiResponse
from src.config.constants import settings

router = APIRouter()

_require_translation_group = require_group(settings.TRANSLATION_REQUIRED_GROUP)


@router.get("/auth/whoami", tags=["auth"])
async def whoami(
    identity: Annotated[IapIdentity, Depends(_require_translation_group)],
) -> WhoamiResponse:
    """Return verified user email + entitlement status.

    Raises 403 (via the require_group dependency) when the user is not a member
    of the Entra security group configured in TRANSLATION_REQUIRED_GROUP.
    """
    return WhoamiResponse(email=identity.email, entitled=True)


@router.post("/auth/token", tags=["auth"])
async def create_auth_token(
    request: AuthTokenRequest,
    identity: Annotated[IapIdentity, Depends(_require_translation_group)],
) -> AuthTokenResponse:
    """Issue JWT using verified IAP identity and user-provided cost attribution."""
    expires_in = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    access_token = create_access_token(
        claims={
            "sub": identity.email,
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
        email=identity.email,
    )
