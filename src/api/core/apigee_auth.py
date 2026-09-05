"""Apigee-based authentication for the Translation API.

Replaces the deleted IAP-JWT / shared-session-cookie auth stack
(`iap_auth.py`, `entitlements.py`, and the JWT machinery formerly in
`security.py`). Apigee is now the sole authority for authentication and
authorization for this service:

* Browser -> aihub-bff (Entra login / IAP) -> Apigee `int` proxy -> this
  Cloud Run service.
* Apigee verifies the caller's Entra JWT itself and re-derives
  `x-colt-user-oid`, `x-colt-user-email`, and `x-colt-user-roles` from that
  verified token -- these three headers cannot be forged by the caller.
* `x-colt-user-department` and `x-colt-user-company` are passed through from
  the BFF UNCHANGED (Apigee does not verify them) and map to
  `business_unit` / `organization` respectively.
* Apigee also authenticates itself to this Cloud Run service with a
  Google-signed ID token whose audience is this service's own Cloud Run URL
  (`CLOUD_RUN_SERVICE_URL`). Apigee's default `GoogleIDToken` policy behavior
  sends this in the `Authorization` header; an earlier design doc assumed
  `X-Serverless-Authorization`. Which one Apigee actually populates has not
  been empirically confirmed, so both are accepted here and the header that
  actually supplied the token is logged at INFO so this can be pinned down
  later via Apigee Trace.

Modeled after the working `verify_oauth2_token` + expected-SA-email pattern
in Sales-Agent's `src/worker/api/auth.py` (Cloud Tasks OIDC verification),
adapted for this service's audience/expected-SA and header-fallback logic.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import id_token
from pydantic import BaseModel
from pydantic import Field

from src.config.constants import settings

logger = logging.getLogger(__name__)

# Google-signed ID token Apigee uses to authenticate itself to this service.
# Accept either header -- see module docstring.
SERVERLESS_AUTHORIZATION_HEADER = "X-Serverless-Authorization"
AUTHORIZATION_HEADER = "Authorization"

# Canonical Apigee-injected user-context headers (see module docstring).
OID_HEADER = "x-colt-user-oid"
EMAIL_HEADER = "x-colt-user-email"
ROLES_HEADER = "x-colt-user-roles"
DEPARTMENT_HEADER = "x-colt-user-department"
COMPANY_HEADER = "x-colt-user-company"


class AuthenticatedUser(BaseModel):
    """Authenticated user context resolved from Apigee-injected headers."""

    oid: str = ""
    email: str
    roles: list[str] = Field(default_factory=list)
    business_unit: str
    organization: str


_LOCAL_DEV_USER = AuthenticatedUser(
    oid="local-dev",
    email="local-dev@example.com",
    roles=["Translation.User"],
    business_unit="",
    organization="",
)


def _strip_bearer_prefix(value: str) -> str:
    """Strip a leading `Bearer ` prefix if present; return the value unchanged otherwise."""
    value = value.strip()
    if value.lower().startswith("bearer "):
        return value[len("Bearer ") :].strip()
    return value


def _extract_google_id_token(request: Request) -> tuple[str, str]:
    """Read the Google ID token from X-Serverless-Authorization, else Authorization.

    Returns (token, header_name_used). header_name_used is "" when neither
    header carried a token, so the caller can log which header (if any)
    actually supplied it -- this is what lets the real behavior be confirmed
    later via Apigee Trace.
    """
    serverless_value = request.headers.get(SERVERLESS_AUTHORIZATION_HEADER)
    if serverless_value:
        return _strip_bearer_prefix(serverless_value), SERVERLESS_AUTHORIZATION_HEADER

    authorization_value = request.headers.get(AUTHORIZATION_HEADER)
    if authorization_value:
        return _strip_bearer_prefix(authorization_value), AUTHORIZATION_HEADER

    return "", ""


def _verify_apigee_identity(request: Request) -> None:
    """Verify Apigee's own Google-signed ID token. Raises 401/403 on failure."""
    token, header_used = _extract_google_id_token(request)
    if not token:
        logger.warning(
            "No Google ID token found on %s or %s for %s %s",
            SERVERLESS_AUTHORIZATION_HEADER,
            AUTHORIZATION_HEADER,
            request.method,
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Apigee identity token",
        )

    logger.info(
        "Google ID token supplied via %s header for %s %s",
        header_used,
        request.method,
        request.url.path,
    )

    try:
        claims = id_token.verify_oauth2_token(
            token,
            google_auth_requests.Request(),
            audience=settings.CLOUD_RUN_SERVICE_URL,
        )
    except Exception as exc:
        logger.warning("Apigee Google ID token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Apigee identity token",
        ) from exc

    expected_sa = settings.APIGEE_RUNTIME_SA_EMAIL.strip()
    email = str(claims.get("email") or "").strip()
    if not expected_sa or email != expected_sa:
        logger.warning(
            "Apigee identity token email mismatch: got %r, expected %r",
            email,
            expected_sa,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unexpected caller identity",
        )


def _user_from_headers(request: Request) -> AuthenticatedUser:
    """Build the user context from Apigee-injected `x-colt-user-*` headers."""
    oid = request.headers.get(OID_HEADER)
    if not oid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {OID_HEADER} header",
        )

    email = request.headers.get(EMAIL_HEADER) or ""
    roles_raw = request.headers.get(ROLES_HEADER) or ""
    roles = [role.strip() for role in roles_raw.split(",") if role.strip()]
    # Passed through from the BFF unchanged (not verified by Apigee); absent
    # is a valid case, not an error, so default to empty rather than crash.
    business_unit = request.headers.get(DEPARTMENT_HEADER) or ""
    organization = request.headers.get(COMPANY_HEADER) or ""

    return AuthenticatedUser(
        oid=oid,
        email=email,
        roles=roles,
        business_unit=business_unit,
        organization=organization,
    )


def get_current_apigee_user(request: Request) -> AuthenticatedUser:
    """FastAPI dependency: verify Apigee's identity and resolve the caller's user context.

    Local-dev (`IS_LOCAL=true`) skips Google ID-token verification entirely:
    the `x-colt-user-*` headers are used directly if present, otherwise a
    fixed fake identity is returned, so local runs and tests keep working
    without a real Google-signed token.
    """
    if settings.IS_LOCAL:
        if request.headers.get(OID_HEADER):
            logger.info("[local] Building user context from x-colt-user-* headers")
            return _user_from_headers(request)
        logger.info(
            "[local] No %s header present -- using fake local-dev identity",
            OID_HEADER,
        )
        return _LOCAL_DEV_USER

    _verify_apigee_identity(request)
    return _user_from_headers(request)
