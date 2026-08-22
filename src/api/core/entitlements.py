"""Firestore-backed access entitlements, keyed by verified IAP email.

Why this exists: Entra ID's OIDC ID token (the only claim source available to
this workforce identity pool provider, per its `assertionClaimsBehavior:
ONLY_ID_TOKEN_CLAIMS` setting) does not reliably include a `groups` claim for
every user/session -- Entra omits `groups` from the ID token entirely once a
user exceeds the per-token group-count limit (a well-documented "group
overage" behavior), and confirmed via production logs that this app's real
IAP-verified JWTs for at least one legitimate user contained zero groups.

Rather than depend on Microsoft Graph API "extra attributes" (which requires
Entra admin work: a client secret, API permissions, and admin consent) to
reliably surface `groups` into the token, entitlement here is instead looked
up directly in Firestore by the already-reliably-verified email address --
`_normalize_email()` never had a groups-claim reliability problem, only the
`groups` claim itself did.

Collection layout (Firestore, native mode):
    user_entitlements/{email}
        {
          "translation_access": bool,
          "sales_agent_access": bool,
        }

`{email}` is the fully-normalized (lowercased) email as returned by
`_normalize_email()` -- e.g. "jabir.mohammed@colt.net" or
"jmohammed@internal.colt.net".
"""

from __future__ import annotations

import logging

from google.cloud import firestore

from src.config.constants import settings

logger = logging.getLogger(__name__)

ENTITLEMENTS_COLLECTION = "user_entitlements"
TRANSLATION_ACCESS_FIELD = "translation_access"

_client: firestore.AsyncClient | None = None


def _get_client() -> firestore.AsyncClient:
    """Lazily construct a module-level singleton AsyncClient.

    Constructed lazily (not at import time) so that importing this module
    never requires live GCP credentials -- useful for local/dev/test runs
    that don't exercise the entitlement lookup path at all.
    """
    global _client
    if _client is None:
        _client = firestore.AsyncClient(project=settings.GOOGLE_CLOUD_PROJECT)
    return _client


async def has_translation_access(email: str) -> bool:
    """Return True if `email` (already normalized/lowercased) is entitled to Translation.

    Fails closed: any Firestore error, missing document, or missing/false
    field value results in False (no access), never an exception bubbling up
    as a false-positive grant.
    """
    try:
        doc_ref = _get_client().collection(ENTITLEMENTS_COLLECTION).document(email)
        snapshot = await doc_ref.get()
    except Exception:
        logger.exception(
            "Firestore entitlement lookup failed for %s -- denying access (fail closed)",
            email,
        )
        return False

    if not snapshot.exists:
        logger.info("No entitlement document found for %s -- denying access", email)
        return False

    data = snapshot.to_dict() or {}
    entitled = bool(data.get(TRANSLATION_ACCESS_FIELD, False))
    logger.info(
        "Entitlement lookup for %s: %s=%s -> entitled=%s",
        email,
        TRANSLATION_ACCESS_FIELD,
        data.get(TRANSLATION_ACCESS_FIELD),
        entitled,
    )
    return entitled
