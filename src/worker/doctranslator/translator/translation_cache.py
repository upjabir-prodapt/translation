"""Redis-backed cross-instance LLM translation cache.

Replaces the need for a local SQLite cache (as used by upstream)
with a Redis (Google Cloud Memorystore, reached over Private Service
Connect) instance so that cache hits are shared across all Cloud Run
worker instances and across separate jobs/model-attempts, not just
within a single process.

Cache key = sha256(prompt_version|provider|model|lang_in|lang_out|text).
Every cache entry is written with a native Redis TTL (``EXPIRE``) of
``settings.REDIS_CACHE_TTL_SECONDS`` (default: 7 days), so expiry is
enforced by Redis itself -- this replaces the previous Firestore-backed
implementation, which relied on an out-of-band TTL policy configured on
the Firestore collection rather than a native per-key expiry.

See docs/infra/redis-memorystore-psc-setup.md for how the Memorystore
instance and PSC endpoint backing REDIS_HOST/REDIS_PORT are provisioned.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from src.config.constants import settings
from src.worker.doctranslator.utils.atomic_integer import AtomicInteger

logger = logging.getLogger(__name__)

# Bump this when the prompt templates change materially so stale cached
# translations produced by an old prompt are not served under the new one.
PROMPT_VERSION = "v1"

_client: Any = None
_cache_enabled: bool | None = None


def _get_client():
    """Lazily construct a module-level singleton Redis client.

    Constructed lazily so importing this module never requires live
    network access to Memorystore. Fails closed (caching disabled) if
    the client cannot be constructed, consistent with the fail-open
    behavior of get()/set() below.
    """
    global _client, _cache_enabled
    if _cache_enabled is False:
        return None
    if _client is not None:
        return _client
    if not settings.REDIS_HOST:
        logger.warning(
            "[translation_cache] REDIS_HOST not configured; caching disabled",
        )
        _cache_enabled = False
        return None
    try:
        import redis

        # redis-py's ConnectionPool does not accept a bare `ssl=` kwarg --
        # TLS must be requested via `connection_class=SSLConnection` instead
        # (the plain `Connection` class doesn't know the `ssl` argument).
        # Memorystore's PSC endpoint presents a certificate that isn't
        # verifiable via the system trust store from arbitrary VPC clients,
        # so certificate verification is disabled here (network isolation via
        # PSC is the actual security boundary, per
        # docs/infra/redis-memorystore-psc-setup.md) — this matches the
        # documented connectivity smoke test (`ssl_cert_reqs=None`).
        extra_tls_kwargs: dict[str, Any] = {}
        if settings.REDIS_TLS_ENABLED:
            connection_class = redis.SSLConnection
            extra_tls_kwargs["ssl_cert_reqs"] = None
        else:
            connection_class = redis.Connection
        pool = redis.ConnectionPool(
            connection_class=connection_class,
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD or None,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
            socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
            decode_responses=True,
            **extra_tls_kwargs,
        )
        _client = redis.Redis(connection_pool=pool)
        # Fail fast at construction time so a dead Memorystore endpoint
        # disables caching immediately rather than on the first miss.
        _client.ping()
        _cache_enabled = True
        return _client
    except Exception:
        logger.warning(
            "[translation_cache] Redis client init failed; caching disabled",
            exc_info=True,
        )
        _client = None
        _cache_enabled = False
        return None


def build_cache_key(
    *,
    provider: str,
    model: str,
    lang_in: str,
    lang_out: str,
    text: str,
    domain: str | None = None,
) -> str:
    """Return a stable, namespaced cache key for one (engine, params, text)
    combination.

    The key is prefixed with ``settings.REDIS_KEY_PREFIX`` so this service's
    entries cannot collide with keys written by any other service sharing
    the same Redis Cluster. Redis Cluster mode has no DB/AUTH-based
    isolation (only DB 0, ``SELECT`` disabled), so this prefix is the only
    isolation mechanism available -- see
    docs/infra/redis-memorystore-psc-setup.md.
    """
    parts = [
        PROMPT_VERSION,
        str(provider),
        str(model),
        str(lang_in),
        str(lang_out),
    ]
    if domain and str(domain).strip():
        parts.append(str(domain).strip().lower())
    parts.append(text)
    payload = "|".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()
    return f"{settings.REDIS_KEY_PREFIX}{digest}"


class TranslationCache:
    """Get/set cached LLM translation results, backed by Redis.

    Fails open on any Redis error: cache misses/errors never block a
    translation call, they just fall through to invoking the LLM as usual.

    Tracks lightweight in-process hit/miss counters (E1 instrumentation)
    so pipelines can log cache effectiveness without needing a dedicated
    metrics backend (no Counter/statsd/prometheus infra exists yet -- see
    docs/architecture/pdf-vs-docx-translation-architecture.md).
    """

    def __init__(self) -> None:
        self.hit_count = AtomicInteger()
        self.miss_count = AtomicInteger()

    def get(self, cache_key: str) -> str | None:
        client = _get_client()
        if client is None:
            self.miss_count.inc()
            return None
        try:
            value = client.get(cache_key)
            if value is not None:
                self.hit_count.inc()
                return str(value)
            self.miss_count.inc()
            return None
        except Exception:
            logger.debug("[translation_cache] cache get failed", exc_info=True)
            self.miss_count.inc()
            return None

    def get_many(self, cache_keys: list[str]) -> dict[str, str]:
        """Batch-fetch multiple cache keys in a single round trip (MGET).

        Returns a dict of only the keys that were present (cache hits).
        Fails open (returns an empty dict) on any Redis error, same
        semantics as get(). Used to avoid N serial round trips when
        pre-checking cache membership for a whole document's worth of
        units before batching (docs/plan.md Section 4.4).
        """
        if not cache_keys:
            return {}
        client = _get_client()
        if client is None:
            self.miss_count.inc(len(cache_keys))
            return {}
        try:
            values = client.mget(cache_keys)
            hits = {
                key: str(value)
                for key, value in zip(cache_keys, values, strict=False)
                if value is not None
            }
            self.hit_count.inc(len(hits))
            self.miss_count.inc(len(cache_keys) - len(hits))
            return hits
        except Exception:
            logger.debug("[translation_cache] cache mget failed", exc_info=True)
            self.miss_count.inc(len(cache_keys))
            return {}

    def stats_snapshot(self) -> dict[str, int]:
        """Return current cumulative hit/miss counters for logging."""
        hits = self.hit_count.value
        misses = self.miss_count.value
        total = hits + misses
        hit_rate = round(hits / total, 4) if total else 0.0
        return {"cache_hits": hits, "cache_misses": misses, "cache_hit_rate": hit_rate}

    def set(
        self,
        cache_key: str,
        translation: str,
        *,
        provider: str,
        model: str,
        lang_in: str,
        lang_out: str,
    ) -> None:
        client = _get_client()
        if client is None:
            return
        try:
            client.set(cache_key, translation, ex=settings.REDIS_CACHE_TTL_SECONDS)
        except Exception:
            logger.debug("[translation_cache] cache set failed", exc_info=True)


_translation_cache = TranslationCache()


def get_translation_cache() -> TranslationCache:
    return _translation_cache
