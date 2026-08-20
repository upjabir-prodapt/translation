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
        logger.warning("Rejected non-colt.net email/sub claim: %r", normalized)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only @colt.net email addresses are allowed",
        )
    logger.info("Normalized IAP email: %s", normalized)
    return normalized


def _normalize_groups(raw_groups: Any) -> set[str]:
    """Normalize the IAP JWT `groups` claim into a set of lowercase group identifiers.

    The workforce pool attributeMapping (`google.groups: assertion.groups`) can surface
    this claim as a list of strings, a comma/space separated string, or absent entirely.
    """
    logger.info("Raw groups claim from IAP token: %r", raw_groups)
    if not raw_groups:
        logger.info("No groups claim present -- normalized to empty set")
        return set()
    if isinstance(raw_groups, str):
        parts = [p.strip() for p in raw_groups.replace(",", " ").split() if p.strip()]
        result = {p.lower() for p in parts}
        logger.info("Normalized groups (from string claim): %s", sorted(result))
        return result
    if isinstance(raw_groups, (list, tuple, set)):
        result = {str(g).strip().lower() for g in raw_groups if str(g).strip()}
        logger.info("Normalized groups (from list/tuple/set claim): %s", sorted(result))
        return result
    logger.warning(
        "Unrecognized groups claim type %s -- normalized to empty set", type(raw_groups)
    )
    return set()


def verify_iap_jwt_claims(assertion: str, audience: str) -> dict[str, Any]:
    """Verify IAP JWT and return the full claims dict."""
    logger.info(
        "Verifying JWT signature (len=%d, prefix=%s...) against audience %s",
        len(assertion),
        assertion[:12],
        audience,
    )
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

    logger.info(
        "JWT signature verified successfully. Claim keys present: %s",
        sorted(claims.keys()),
    )

    email = claims.get("email") or claims.get("sub")
    if not email or not isinstance(email, str):
        logger.warning("Verified JWT has no usable email/sub claim: claims=%s", claims)
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
    """Verify against HUB_IAP_AUDIENCE (the aihub-be IAP resource).

    Architecture B: IAP is disabled on this service's own backend service, so
    settings.IAP_AUDIENCE can never match a real token anymore -- the only
    IAP-protected resource left in the chain is the AI Hub's own aihub-be,
    whose audience is HUB_IAP_AUDIENCE. This service's Cloud Run application
    itself was never independently IAP-protected in the first place (IAP only
    ever applies at the GCLB backend-service layer); Cloud Run IAM
    (roles/run.invoker) is the network-level gate now, entirely separate from
    this JWT-audience check, which exists purely to verify the caller's real
    Entra-federated identity/groups.
    """
    logger.info(
        "Verifying IAP assertion against HUB_IAP_AUDIENCE=%s (IAP_AUDIENCE=%s is no longer checked)",
        settings.HUB_IAP_AUDIENCE,
        settings.IAP_AUDIENCE or "<unset>",
    )
    if not settings.HUB_IAP_AUDIENCE:
        logger.warning(
            "HUB_IAP_AUDIENCE is not configured -- cannot verify any IAP assertion"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing IAP audience configuration",
        )
    return verify_iap_jwt_claims(assertion, settings.HUB_IAP_AUDIENCE)


class IapIdentity:
    """Verified IAP identity: normalized email + Entra group memberships."""

    def __init__(self, email: str, groups: set[str]) -> None:
        self.email = email
        self.groups = groups

    def has_group(self, required_group: str) -> bool:
        if not required_group:
            # No group restriction configured — do not silently grant access.
            logger.warning(
                "has_group() called with empty required_group -- denying by default"
            )
            return False
        result = required_group.strip().lower() in self.groups
        logger.info(
            "has_group check: required=%s, user_groups=%s, result=%s",
            required_group.strip().lower(),
            sorted(self.groups),
            result,
        )
        return result


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
    logger.info("Processing IAP for headers: %s", sorted(request.headers.keys()))
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
    logger.info(
        "Assertion selected from header %s (len=%d, prefix=%s...)",
        IAP_JWT_HEADER
        if request.headers.get(IAP_JWT_HEADER)
        else HUB_FORWARDED_IAP_JWT_HEADER,
        len(assertion),
        assertion[:12],
    )
    claims = _verify_iap_assertion_claims(assertion)
    # Debug aid (Architecture B rollout): log the full verified claims dict so we
    # can see exactly what IAP/Entra populated (email, sub, groups, aud, iss,
    # etc.) when downstream checks (email domain, group membership) reject a
    # request. This is the DECODED JWT PAYLOAD, not the raw token -- it does not
    # include the token's cryptographic signature, but does include identity
    # claims (email, groups) which are not secret and are already visible to
    # this service via the header value itself.
    logger.warning("IAP verified claims: %s", claims)
    email = _normalize_email(str(claims.get("email") or claims.get("sub")))
    groups = _normalize_groups(claims.get("groups"))
    logger.info("Resolved IapIdentity: email=%s, groups=%s", email, sorted(groups))
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
