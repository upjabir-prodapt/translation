"""Authentication endpoints."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Response

from src.api.core.iap_auth import IapIdentity
from src.api.core.iap_auth import require_translation_entitlement
from src.api.core.security import SESSION_COOKIE_NAME
from src.api.core.security import create_access_token
from src.api.schemas.requests import AuthTokenRequest
from src.api.schemas.responses import AuthTokenResponse
from src.api.schemas.responses import WhoamiResponse
from src.config.constants import settings

router = APIRouter()

_require_translation_group = require_translation_entitlement()


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
    response: Response,
    identity: Annotated[IapIdentity, Depends(_require_translation_group)],
) -> AuthTokenResponse:
    """Issue JWT using verified IAP identity and user-provided cost attribution.

    Sets the JWT as an httpOnly session cookie (primary, XSS-resistant
    transport) in addition to returning it in the response body. The body
    value is kept for backward compatibility with clients still sending it
    via the `x-app-auth` header during the migration to cookie-based auth;
    once that migration is confirmed complete, returning the raw token in
    the body can be dropped as a cleanup pass.
    """
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

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        max_age=expires_in,
        httponly=True,
        secure=not settings.IS_LOCAL,
        samesite="strict",
        path="/",
    )

    return AuthTokenResponse(
        access_token=access_token,
        token_type=token_type,
        expires_in=expires_in,
        email=identity.email,
    )
