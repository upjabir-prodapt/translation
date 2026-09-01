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
SALES_AGENT_ACCESS_FIELD = "sales_agent_access"

# Scope identifiers stamped into the shared `colt_session` JWT's `scopes`
# claim. Translation and Sales-Agent sign with the SAME HS256 secret and share
# one cookie (deliberate, so a single login covers both services), which means
# signature validity alone says nothing about *which* service a token was
# minted for. Each service therefore additionally requires its own scope to be
# present -- that check, not the signature, is what keeps a Translation-only
# user out of Sales-Agent and vice versa. Keep these strings identical in both
# repos.
SCOPE_TRANSLATION = "translation"
SCOPE_SALES = "sales"

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


async def _fetch_entitlement_doc(email: str) -> dict[str, object] | None:
    """Read `user_entitlements/{email}` once, or None on any failure.

    Fails closed: Firestore errors and missing documents are both reported as
    None so every caller denies rather than accidentally granting.
    """
    try:
        doc_ref = _get_client().collection(ENTITLEMENTS_COLLECTION).document(email)
        snapshot = await doc_ref.get()
    except Exception:
        logger.exception(
            "Firestore entitlement lookup failed -- denying access (fail closed)"
        )
        return None

    if not snapshot.exists:
        logger.info("No entitlement document found -- denying access")
        return None

    return snapshot.to_dict() or {}


async def resolve_scopes(email: str) -> set[str]:
    """Return the set of session scopes granted to `email`.

    Reads the entitlement document exactly ONCE and derives both scopes from
    it, so minting or renewing a session costs a single Firestore read rather
    than one per service. `email` must already be normalized/lowercased.

    Fails closed: an unreadable or absent document yields an empty set.
    """
    data = await _fetch_entitlement_doc(email)
    if data is None:
        return set()

    scopes: set[str] = set()
    if bool(data.get(TRANSLATION_ACCESS_FIELD, False)):
        scopes.add(SCOPE_TRANSLATION)
    if bool(data.get(SALES_AGENT_ACCESS_FIELD, False)):
        scopes.add(SCOPE_SALES)

    logger.info("Resolved session scopes: %s", sorted(scopes))
    return scopes


async def has_translation_access(email: str) -> bool:
    """Return True if `email` (already normalized/lowercased) is entitled to Translation.

    Fails closed: any Firestore error, missing document, or missing/false
    field value results in False (no access), never an exception bubbling up
    as a false-positive grant.
    """
    data = await _fetch_entitlement_doc(email)
    if data is None:
        return False

    entitled = bool(data.get(TRANSLATION_ACCESS_FIELD, False))
    logger.info(
        "Entitlement lookup: %s=%s -> entitled=%s",
        TRANSLATION_ACCESS_FIELD,
        data.get(TRANSLATION_ACCESS_FIELD),
        entitled,
    )
    return entitled
