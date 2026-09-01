"""Shared lazy Redis client for every Redis consumer in the worker.

Extracted verbatim (behaviour-wise) from `translation_cache._get_client`,
which was the only Redis consumer until the job lease arrived. Two modules
constructing their own pool against the same Memorystore/PSC endpoint would
double the connection count and duplicate the TLS quirks documented in
docs/infra/redis-memorystore-psc-setup.md, so there is exactly one pooled
singleton here and both consumers share it.

Failure semantics are unchanged and deliberate:

* construction fails closed -- a dead endpoint disables Redis-backed
  features immediately rather than on the first operation;
* individual operations fail *open* at the call site (a cache miss, an
  ungated job) -- Redis is an optimisation and a safety net, never a
  dependency of correctness.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from src.config.constants import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client: Any = None
_enabled: bool | None = None


def _build_client() -> Any:
    import redis

    # redis-py's ConnectionPool does not accept a bare `ssl=` kwarg --
    # TLS must be requested via `connection_class=SSLConnection` instead
    # (the plain `Connection` class doesn't know the `ssl` argument).
    # Memorystore's PSC endpoint presents a certificate that isn't
    # verifiable via the system trust store from arbitrary VPC clients,
    # so certificate verification is disabled here (network isolation via
    # PSC is the actual security boundary, per
    # docs/infra/redis-memorystore-psc-setup.md) -- this matches the
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
    client = redis.Redis(connection_pool=pool)
    # Fail fast at construction time so a dead Memorystore endpoint
    # disables Redis-backed features immediately rather than on first use.
    client.ping()
    return client


def get_redis_client() -> Any:
    """Return the process-wide Redis client, or None when unavailable.

    Constructed lazily so importing this module never requires live network
    access. Once construction has failed the result is cached: the caller is
    on a hot path and must not retry a dead endpoint per operation.
    """
    global _client, _enabled
    if _enabled is False:
        return None
    if _client is not None:
        return _client
    with _lock:
        if _enabled is False:
            return None
        if _client is not None:
            return _client
        if not settings.REDIS_HOST:
            logger.warning("[redis] REDIS_HOST not configured; Redis features disabled")
            _enabled = False
            return None
        try:
            _client = _build_client()
            _enabled = True
            return _client
        except Exception:
            logger.warning(
                "[redis] client init failed; Redis features disabled", exc_info=True
            )
            _client = None
            _enabled = False
            return None


def is_redis_enabled() -> bool:
    """True when a client has been successfully constructed."""
    return get_redis_client() is not None


def reset_for_tests() -> None:
    """Drop the cached client/enabled state. Test-only."""
    global _client, _enabled
    with _lock:
        _client = None
        _enabled = None
