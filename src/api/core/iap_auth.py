"""GCP IAP JWT verification for Entra-federated user identity."""

import logging
from typing import Any

from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from src.config.constants import settings

logger = logging.getLogger(__name__)

IAP_JWT_HEADER = "x-goog-iap-jwt-assertion"
# Architecture B fallback: Google Front End strips inbound X-Goog-*/X-Google-*
# request headers at EVERY Cloud Run service's own public ingress edge (anti-
# spoofing) — only IAP itself, sitting directly in front of a given backend,
# is trusted to set those headers for that backend. When the AI Hub's own
# nginx relays the hub's inbound IAP JWT to this service's Cloud Run URL, GFE
# strips X-Goog-IAP-JWT-Assertion before it reaches this app. nginx therefore
# additionally relays the same JWT value under this non-reserved header name,
# which GFE does not strip. Same verification path (HUB_IAP_AUDIENCE) applies.
HUB_FORWARDED_IAP_JWT_HEADER = "x-colt-hub-iap-assertion"
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


def _normalize_groups(raw_groups: Any) -> set[str]:
    """Normalize the IAP JWT `groups` claim into a set of lowercase group identifiers.

    The workforce pool attributeMapping (`google.groups: assertion.groups`) can surface
    this claim as a list of strings, a comma/space separated string, or absent entirely.
    """
    if not raw_groups:
        return set()
    if isinstance(raw_groups, str):
        parts = [p.strip() for p in raw_groups.replace(",", " ").split() if p.strip()]
        return {p.lower() for p in parts}
    if isinstance(raw_groups, (list, tuple, set)):
        return {str(g).strip().lower() for g in raw_groups if str(g).strip()}
    return set()


def verify_iap_jwt_claims(assertion: str, audience: str) -> dict[str, Any]:
    """Verify IAP JWT and return the full claims dict."""
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
    return claims


def verify_iap_jwt(assertion: str, audience: str) -> str:
    """Verify IAP JWT and return normalized email (backward-compatible helper)."""
    claims = verify_iap_jwt_claims(assertion, audience)
    return _normalize_email(str(claims.get("email") or claims.get("sub")))


def _verify_iap_assertion_claims(assertion: str) -> dict[str, Any]:
    """Accept this service's IAP audience, or the AI Hub ILB audience when called via aihub hostname."""
    audiences = [a for a in [settings.IAP_AUDIENCE, settings.HUB_IAP_AUDIENCE] if a]
    last_exc: HTTPException | None = None
    for audience in dict.fromkeys(audiences):
        try:
            return verify_iap_jwt_claims(assertion, audience)
        except HTTPException as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing IAP audience configuration",
    )


class IapIdentity:
    """Verified IAP identity: normalized email + Entra group memberships."""

    def __init__(self, email: str, groups: set[str]) -> None:
        self.email = email
        self.groups = groups

    def has_group(self, required_group: str) -> bool:
        if not required_group:
            # No group restriction configured — do not silently grant access.
            return False
        return required_group.strip().lower() in self.groups


def get_iap_identity(request: Request) -> IapIdentity:
    """FastAPI dependency returning verified email + group claims from the IAP JWT."""
    if settings.IS_LOCAL:
        dev_email = request.headers.get(DEV_IAP_USER_HEADER)
        if dev_email:
            dev_groups_raw = request.headers.get("x-dev-iap-user-groups", "")
            return IapIdentity(
                _normalize_email(dev_email), _normalize_groups(dev_groups_raw)
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"Local dev: set {DEV_IAP_USER_HEADER} header "
                "with a @colt.net email address"
            ),
        )

    assertion = request.headers.get(IAP_JWT_HEADER) or request.headers.get(
        HUB_FORWARDED_IAP_JWT_HEADER
    )
    if not assertion:
        # Debug aid (Architecture B rollout): dump every header NAME we actually
        # received so we can see exactly what nginx/GFE delivered when the
        # assertion is missing. Never logs header VALUES (JWTs, bearer tokens,
        # cookies) to avoid leaking credentials into logs.
        logger.warning(
            "IAP assertion missing on %s %s -- received header names: %s",
            request.method,
            request.url.path,
            sorted(request.headers.keys()),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Goog-IAP-JWT-Assertion header",
        )
    claims = _verify_iap_assertion_claims(assertion)
    email = _normalize_email(str(claims.get("email") or claims.get("sub")))
    groups = _normalize_groups(claims.get("groups"))
    return IapIdentity(email=email, groups=groups)


def require_group(required_group: str):
    """Dependency factory: verify IAP identity AND require membership in a specific Entra group."""

    def _dependency(request: Request) -> IapIdentity:
        identity = get_iap_identity(request)
        if not identity.has_group(required_group):
            logger.warning(
                "User %s denied — missing required group %r (has: %s)",
                identity.email,
                required_group,
                sorted(identity.groups),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this service. Contact your administrator.",
            )
        return identity

    return _dependency


def get_iap_user(request: Request) -> str:
    """FastAPI dependency returning verified user email from IAP JWT (no group check)."""
    return get_iap_identity(request).email
