"""Authentication endpoints."""

import logging
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import status

from src.api.core.entitlements import has_translation_access
from src.api.core.iap_auth import IapIdentity
from src.api.core.iap_auth import require_translation_entitlement
from src.api.core.security import REFRESH_COOKIE_NAME
from src.api.core.security import REFRESH_COOKIE_PATH
from src.api.core.security import SESSION_COOKIE_NAME
from src.api.core.security import create_access_token
from src.api.core.security import create_refresh_token
from src.api.core.security import decode_and_verify_refresh_token
from src.api.schemas.requests import AuthTokenRequest
from src.api.schemas.requests import RefreshTokenRequest
from src.api.schemas.responses import AuthTokenResponse
from src.api.schemas.responses import WhoamiResponse
from src.config.constants import settings

logger = logging.getLogger(__name__)

router = APIRouter()

_require_translation_group = require_translation_entitlement()

TOKEN_TYPE = "bearer"  # noqa: S105


def _set_session_cookie(response: Response, access_token: str, max_age: int) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        max_age=max_age,
        httponly=True,
        secure=not settings.IS_LOCAL,
        samesite="strict",
        path="/",
    )


def _set_refresh_cookie(response: Response, refresh_token: str, max_age: int) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        max_age=max_age,
        httponly=True,
        secure=not settings.IS_LOCAL,
        samesite="strict",
        path=REFRESH_COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=REFRESH_COOKIE_NAME,
        httponly=True,
        secure=not settings.IS_LOCAL,
        samesite="strict",
        path=REFRESH_COOKIE_PATH,
    )


def _remaining_seconds(payload: dict) -> int:
    """Seconds left on a verified token, floored at 0."""
    expires_at = int(payload.get("exp", 0))
    now = int(datetime.now(UTC).timestamp())
    return max(0, expires_at - now)


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
    """
    Issue JWT using verified IAP identity and user-provided cost attribution.

    Sets the JWT as an httpOnly session cookie (primary, XSS-resistant
    transport) in addition to returning it in the response body. The body
    value is kept for backward compatibility with clients still sending it
    via the `x-app-auth` header during the migration to cookie-based auth;
    once that migration is confirmed complete, returning the raw token in
    the body can be dropped as a cleanup pass.

    Also issues a longer-lived refresh token (same transports) so the UI can
    renew the access token via POST /auth/refresh instead of re-prompting for
    business unit / organization every time the access token lapses.
    """
    expires_in = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    refresh_expires_in = settings.JWT_REFRESH_TOKEN_EXPIRE_MINUTES * 60
    claims = {
        "sub": identity.email,
        "business_unit": request.business_unit.strip(),
        "organization": request.organization.strip(),
    }
    access_token = create_access_token(
        claims=claims,
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(
        claims=claims,
        expires_delta=timedelta(minutes=settings.JWT_REFRESH_TOKEN_EXPIRE_MINUTES),
    )

    _set_session_cookie(response, access_token, expires_in)
    _set_refresh_cookie(response, refresh_token, refresh_expires_in)

    return AuthTokenResponse(
        access_token=access_token,
        token_type=TOKEN_TYPE,
        expires_in=expires_in,
        email=identity.email,
        refresh_token=refresh_token,
        refresh_expires_in=refresh_expires_in,
    )


@router.post("/auth/refresh", tags=["auth"])
async def refresh_auth_token(
    request: Request,
    body: RefreshTokenRequest,
    response: Response,
) -> AuthTokenResponse:
    """Exchange a refresh token for a fresh access token.

    Deliberately does NOT require an IAP assertion -- that is the whole point
    of the endpoint: the UI renews silently while the browser's IAP session is
    being re-established or the user is mid-upload. Entitlement is re-checked
    against Firestore on every refresh instead, so revoking a user takes effect
    within one access-token lifetime rather than one refresh-token lifetime.

    The refresh token is returned unchanged with its remaining lifetime, so a
    session has an absolute cap: refreshing renews the access token but never
    extends the window. When it runs out the UI must send the user back
    through POST /auth/token.
    """
    token = body.refresh_token or request.cookies.get(REFRESH_COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_and_verify_refresh_token(token.strip())
    email = str(payload["sub"])

    # Skip in local dev: there is no live Firestore, and the dev-header group
    # simulation that stands in for it is not carried on a refresh call.
    if not settings.IS_LOCAL and not await has_translation_access(email):
        logger.warning(
            "Refresh denied for %s -- Firestore entitlement no longer present", email
        )
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this service. Contact your administrator.",
        )

    expires_in = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    refresh_expires_in = _remaining_seconds(payload)
    access_token = create_access_token(
        claims={
            "sub": email,
            "business_unit": str(payload["business_unit"]),
            "organization": str(payload["organization"]),
        },
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    _set_session_cookie(response, access_token, expires_in)

    logger.info("Refreshed access token for %s", email)
    return AuthTokenResponse(
        access_token=access_token,
        token_type=TOKEN_TYPE,
        expires_in=expires_in,
        email=email,
        refresh_token=token,
        refresh_expires_in=refresh_expires_in,
    )
