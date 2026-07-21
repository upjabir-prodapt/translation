"""GCP IAP JWT verification for Entra-federated user identity."""

import logging

from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from src.config.constants import settings

logger = logging.getLogger(__name__)

IAP_JWT_HEADER = "x-goog-iap-jwt-assertion"
DEV_IAP_USER_HEADER = "x-dev-iap-user-email"
IAP_CERTS_URL = "https://www.gstatic.com/iap/verify/public_key"

_GOOGLE_REQUEST = google_requests.Request()


def _normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if not normalized.endswith("@colt.net"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only @colt.net email addresses are allowed",
        )
    return normalized


def verify_iap_jwt(assertion: str, audience: str) -> str:
    """Verify IAP JWT and return normalized email."""
    try:
        claims = id_token.verify_token(
            assertion,
            _GOOGLE_REQUEST,
            audience=audience,
            certs_url=IAP_CERTS_URL,
        )
    except Exception as exc:
        logger.warning("IAP JWT verification failed for audience %s: %s", audience, exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing IAP identity token",
        ) from exc

    email = claims.get("email") or claims.get("sub")
    if not email or not isinstance(email, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="IAP token missing email claim",
        )
    return _normalize_email(email)


def _verify_iap_assertion(assertion: str) -> str:
    """Accept this service's IAP audience, or the AI Hub ILB audience when called via aihub hostname."""
    audiences = [a for a in [settings.IAP_AUDIENCE, settings.HUB_IAP_AUDIENCE] if a]
    last_exc: HTTPException | None = None
    for audience in dict.fromkeys(audiences):
        try:
            return verify_iap_jwt(assertion, audience)
        except HTTPException as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing IAP audience configuration",
    )


def get_iap_user(request: Request) -> str:
    """FastAPI dependency returning verified user email from IAP JWT."""
    if settings.IS_LOCAL:
        dev_email = request.headers.get(DEV_IAP_USER_HEADER)
        if dev_email:
            return _normalize_email(dev_email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"Local dev: set {DEV_IAP_USER_HEADER} header "
                "with a @colt.net email address"
            ),
        )

    assertion = request.headers.get(IAP_JWT_HEADER)
    if not assertion:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Goog-IAP-JWT-Assertion header",
        )
    return _verify_iap_assertion(assertion)
