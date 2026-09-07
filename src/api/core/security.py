"""Security utilities for authentication and authorization."""

import logging
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

import jwt
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.security import APIKeyHeader
from jwt import InvalidTokenError
from pydantic import BaseModel

from src.api.core.entitlements import SCOPE_TRANSLATION
from src.config.constants import settings

logger = logging.getLogger(__name__)

# The scope this service requires in every session token it accepts.
SERVICE_SCOPE = SCOPE_TRANSLATION

# Value of the `typ` claim that separates the two token kinds. Without it a
# long-lived refresh token would also pass as a short-lived access token,
# silently extending the access-token lifetime to the refresh window.
ACCESS_TOKEN_TYPE = "access"  # noqa: S105
REFRESH_TOKEN_TYPE = "refresh"  # noqa: S105

REQUIRED_CLAIMS = ("sub", "business_unit", "organization")

# x-app-auth token scheme (shown as "XAppAuth" in Swagger Authorize)
app_auth_scheme = APIKeyHeader(
    name="x-app-auth",
    scheme_name="XAppAuth",
    auto_error=False,
    description=(
        "JWT from POST /api/v1/auth/token. "
        "Enter `Bearer <access_token>` or paste the token alone."
    ),
)

# httpOnly session cookie name — preferred credential transport. The
# `x-app-auth` header above is kept as a fallback during the migration
# window so any older/cached clients continue to work; see verify_token().
SESSION_COOKIE_NAME = "colt_session"  # noqa: S105

# Refresh token cookie. Scoped to the refresh endpoint only: the browser then
# never attaches this long-lived credential to ordinary API traffic, so a
# logging or proxy mishap on a translate call cannot leak a whole session.
REFRESH_COOKIE_NAME = "colt_refresh"  # noqa: S105
REFRESH_COOKIE_PATH = "/api/v1/auth"


class AuthenticatedUser(BaseModel):
    """Authenticated user context extracted from JWT claims."""

    email: str
    business_unit: str
    organization: str


def _jwt_secret() -> str:
    """Return configured JWT secret with safe local fallback.

    LOAD-BEARING: this secret is intentionally IDENTICAL to Sales-Agent's
    `SECRET_KEY`. One IAP login mints one `colt_session` cookie that both
    services accept, and that only works while both sign and verify with the
    same key. Rotating one side independently breaks cross-service SSO
    immediately and silently -- every call to the un-rotated service starts
    401ing. Rotate both together, or not at all. What keeps the shared secret
    safe is the `scopes` claim (see _enforce_service_scope), not key
    separation.
    """
    if settings.JWT_SECRET_KEY:
        return settings.JWT_SECRET_KEY
    if settings.IS_LOCAL:
        return "local-dev-insecure-jwt-secret-that-is-at-least-32-bytes-long"
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="JWT secret is not configured",
    )


def _encode(claims: dict[str, Any], token_type: str, expires_delta: timedelta) -> str:
    """Sign a JWT of `token_type` carrying `claims`."""
    now = datetime.now(UTC)
    expires = now + expires_delta
    payload = {
        **claims,
        "typ": token_type,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=settings.JWT_ALGORITHM)


def normalize_scopes(raw: Any) -> set[str]:
    """Normalize a `scopes` claim into a lowercase set.

    Accepts the list form this service mints as well as a space/comma
    separated string, so a token produced by a differently-serialising client
    is still understood rather than silently treated as unscoped.
    """
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
        return {p.lower() for p in parts}
    if isinstance(raw, (list, tuple, set)):
        return {str(s).strip().lower() for s in raw if str(s).strip()}
    return set()


def _enforce_service_scope(payload: dict[str, Any]) -> None:
    """Require this service's own scope in the token's `scopes` claim.

    Translation and Sales-Agent share one HS256 secret and one `colt_session`
    cookie so that a single IAP login covers both. A valid signature therefore
    proves only "some Colt service minted this", not "the caller is entitled
    to *this* service" -- this check is what actually separates the two.

    While `REQUIRE_SCOPE_CLAIM` is False, a token with no `scopes` claim at
    all is accepted so sessions minted before scopes existed keep working
    through the rollout window. A token that *does* carry `scopes` is always
    held to them, in both modes -- otherwise the bypass would remain wide open
    for exactly the tokens the claim was added to constrain.
    """
    if "scopes" not in payload:
        if settings.REQUIRE_SCOPE_CLAIM:
            logger.warning("Token has no scopes claim and REQUIRE_SCOPE_CLAIM is on")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this service. Contact your administrator.",
            )
        return

    scopes = normalize_scopes(payload.get("scopes"))
    if SERVICE_SCOPE not in scopes:
        logger.warning(
            "Token rejected: scope %r absent from token scopes %s",
            SERVICE_SCOPE,
            sorted(scopes),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this service. Contact your administrator.",
        )


def create_access_token(
    claims: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """Create an HS256 signed JWT access token."""
    return _encode(
        claims,
        ACCESS_TOKEN_TYPE,
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(
    claims: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """Create an HS256 signed JWT refresh token.

    Carries the same identity/cost-attribution claims as the access token so
    /auth/refresh can re-mint one without a second trip through IAP.
    """
    return _encode(
        claims,
        REFRESH_TOKEN_TYPE,
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.JWT_REFRESH_TOKEN_EXPIRE_MINUTES),
    )


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _decode(token: str) -> dict[str, Any]:
    """Verify signature/expiry and the claims every token kind must carry."""
    try:
        payload = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=[settings.JWT_ALGORITHM],
        )
    except InvalidTokenError as exc:
        raise _unauthorized("Invalid or expired token") from exc

    missing = [claim for claim in REQUIRED_CLAIMS if not payload.get(claim)]
    if missing:
        raise _unauthorized(f"Missing required token claims: {', '.join(missing)}")

    _enforce_service_scope(payload)

    return payload


def decode_and_verify_token(token: str) -> dict[str, Any]:
    """Decode and verify an access token, including required claims."""
    payload = _decode(token)
    # Tokens minted before `typ` existed have no such claim and are still
    # valid access tokens; only an explicit refresh token is turned away, so
    # the long-lived credential can never stand in for the short-lived one.
    if payload.get("typ") == REFRESH_TOKEN_TYPE:
        raise _unauthorized("Refresh token cannot be used as an access token")
    return payload


def decode_and_verify_refresh_token(token: str) -> dict[str, Any]:
    """Decode and verify a refresh token, including required claims."""
    payload = _decode(token)
    # Strict the other way round: an access token (or a legacy token with no
    # `typ`) must not buy a fresh session.
    if payload.get("typ") != REFRESH_TOKEN_TYPE:
        raise _unauthorized("Not a refresh token")
    return payload


def _extract_bearer_token(header_value: str | None) -> str:
    """Parse JWT from x-app-auth or the session cookie (Bearer prefix optional)."""
    if not header_value or not header_value.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    value = header_value.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value


def verify_token(
    request: Request,
    api_key: str | None = Depends(app_auth_scheme),  # noqa: B008
) -> dict[str, Any]:
    """Verify bearer token and return JWT payload.

    Accepts the JWT from either the legacy `x-app-auth` header or the
    httpOnly `colt_session` cookie (preferred, set by POST /auth/token).
    The header takes priority so older/cached clients keep working during
    the migration window; once all clients are confirmed cookie-only, the
    header path can be removed as a cleanup pass.
    """
    token_source = api_key or request.cookies.get(SESSION_COOKIE_NAME)
    return decode_and_verify_token(_extract_bearer_token(token_source))


def get_current_user(
    payload: dict[str, Any] = Depends(verify_token),  # noqa: B008
) -> dict[str, Any]:
    """Backward-compatible dependency returning raw token payload."""
    return payload


def get_current_user_context(
    request: Request,
    payload: dict[str, Any] = Depends(verify_token),  # noqa: B008
) -> AuthenticatedUser:
    """FastAPI dependency to extract normalized user context from JWT."""
    user = AuthenticatedUser(
        email=str(payload["sub"]),
        business_unit=str(payload["business_unit"]),
        organization=str(payload["organization"]),
    )
    request.state.user = user
    return user
