"""Authentication endpoints."""

import logging
import time
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Annotated
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response
from fastapi import status

from src.api.core.entitlements import resolve_scopes
from src.api.core.iap_auth import IapIdentity
from src.api.core.iap_auth import require_translation_entitlement
from src.api.core.iap_auth import resolve_session_scopes
from src.api.core.security import REFRESH_COOKIE_NAME
from src.api.core.security import REFRESH_COOKIE_PATH
from src.api.core.security import SERVICE_SCOPE
from src.api.core.security import SESSION_COOKIE_NAME
from src.api.core.security import create_access_token
from src.api.core.security import verify_token
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


def _set_session_cookie(response: Response, token: str, max_age: int) -> None:
    """Write the shared `colt_session` cookie.

    Attributes must stay byte-identical to Sales-Agent's, and to every other
    place this cookie is written or deleted here: the two services share one
    cookie, so any divergence in path/samesite/secure produces a second,
    shadowing cookie rather than an overwrite.
    """
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=max_age,
        httponly=True,
        secure=not settings.IS_LOCAL,
        samesite="strict",
        path="/",
    )


def _delete_session_cookie(response: Response) -> None:
    """Clear the shared `colt_session` cookie using the same attributes."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        secure=not settings.IS_LOCAL,
        samesite="strict",
        path="/",
    )


def _clearing_headers() -> dict[str, str]:
    """Set-Cookie header that clears the session, for use on an HTTPException.

    Raising discards whatever was written to the injected `Response`, because
    the exception handler builds a fresh one. When a refresh is refused the
    session is definitively over, so the cookie must be cleared on the way
    out -- otherwise the browser keeps re-presenting a credential that can
    only ever be rejected, and the UI has no signal to stop retrying.
    """
    scratch = Response()
    _delete_session_cookie(scratch)
    return {"set-cookie": scratch.headers["set-cookie"]}


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

    Two claims beyond the identity ones are stamped here:

    * `auth_time` -- when this human actually authenticated with IAP. It is
      copied forward unchanged by every subsequent renewal, so it is what
      bounds the session's absolute lifetime no matter how many times the
      token is re-minted.
    * `scopes` -- which services the session may reach. Both services' scopes
      are resolved and stamped, because this one cookie is shared with
      Sales-Agent.
    """
    expires_in = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    scopes = await resolve_session_scopes(identity)
    access_token = create_access_token(
        claims={
            "sub": identity.email,
            "business_unit": request.business_unit.strip(),
            "organization": request.organization.strip(),
            "auth_time": int(time.time()),
            "scopes": sorted(scopes),
        },
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    token_type = "bearer"  # noqa: S105

    _set_session_cookie(response, access_token, expires_in)

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


@router.post("/auth/refresh", tags=["auth"])
async def refresh_auth_token(
    response: Response,
    payload: Annotated[dict[str, Any], Depends(verify_token)],
) -> AuthTokenResponse:
    """Slide the session forward, re-minting from the still-valid current token.

    Deliberately takes NO request body: the UI renews on a timer and has
    nothing to send, and declaring a Pydantic body parameter would make a
    body-less POST fail validation with 422 instead of succeeding.

    Authentication is the caller's existing `colt_session` cookie (or
    `x-app-auth` header), validated by the usual `verify_token` dependency.
    An already-expired token therefore 401s here and cannot be renewed --
    that is the intended limit of a sliding session with no refresh token:
    renewal only works while the current token still lives.

    Two things are re-checked on every renewal, which is what makes a 30
    minute token acceptable in the first place:

    * the absolute cap, from the preserved `auth_time`; and
    * the Firestore entitlement, so a revoked user loses access within one
      token lifetime rather than at their next login.
    """
    now = int(time.time())
    # Legacy tokens predate `auth_time`; `iat` is the closest honest stand-in
    # and errs toward ending the session sooner, never later.
    auth_time = payload.get("auth_time") or payload.get("iat")
    if auth_time is None:
        logger.warning("Refresh denied: token carries neither auth_time nor iat")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session cannot be renewed. Please sign in again.",
            headers=_clearing_headers(),
        )

    auth_time = int(auth_time)
    if now - auth_time > settings.SESSION_ABSOLUTE_MAX_MINUTES * 60:
        logger.info("Refresh denied: session exceeded absolute maximum lifetime")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please sign in again.",
            headers=_clearing_headers(),
        )

    email = str(payload["sub"]).strip().lower()

    if settings.IS_LOCAL:
        # No live Firestore locally; carry the existing scopes forward so the
        # renewal path is exercisable offline.
        scopes = set(payload.get("scopes") or [SERVICE_SCOPE])
    else:
        scopes = await resolve_scopes(email)
        if SERVICE_SCOPE not in scopes:
            logger.warning(
                "Refresh denied: entitlement no longer grants %s", SERVICE_SCOPE
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this service. Contact your administrator.",
                headers=_clearing_headers(),
            )

    expires_in = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    access_token = create_access_token(
        claims={
            "sub": payload["sub"],
            "business_unit": payload["business_unit"],
            "organization": payload["organization"],
            # Preserved, never restamped -- restamping it here would turn the
            # absolute cap into another sliding window and remove the ceiling.
            "auth_time": auth_time,
            "scopes": sorted(scopes),
        },
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    _set_session_cookie(response, access_token, expires_in)

    return AuthTokenResponse(
        access_token=access_token,
        token_type="bearer",  # noqa: S106
        expires_in=expires_in,
        email=str(payload["sub"]),
    )


@router.post(
    "/auth/logout",
    tags=["auth"],
    status_code=status.HTTP_204_NO_CONTENT,
)
async def logout(response: Response) -> None:
    """Clear the shared session cookie.

    Unauthenticated by design: the only effect is removing a credential, so
    requiring a valid one would just make logout fail exactly when it is most
    needed (an expired or malformed session the user wants to be rid of).
    """
    _delete_session_cookie(response)
