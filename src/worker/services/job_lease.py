"""Cross-instance job lease, so one job runs on one worker at a time.

`TranslateTaskHandler` treats a job already in `PROCESSING` as a crash
retry: it wipes the local scratch directory and re-runs the whole pipeline.
That is right when the previous owner really did crash, and catastrophic
when it did not. With `--concurrency=1` a Cloud Tasks retry cannot be
delivered to the instance that is busy with the job, so Cloud Run starts a
second instance -- and a large PDF costs ~6-7 GB on each of them.

Nothing prevented that: no lease, no owner, no heartbeat. This module adds
one, keyed on job ID, in the Redis that already backs the translation cache.

Design points that are not incidental:

* **Compare-and-swap release and refresh.** Both run as Lua via `EVAL` so
  the read and the write are atomic. A bare `DEL`/`EXPIRE` would let an
  instance whose lease had already expired -- and been taken over -- delete
  or extend somebody else's.
* **Fail open.** `REDIS_HOST` is set in production but absent locally, and
  Redis being down must not stop translations. A worker that cannot reach
  Redis behaves exactly as it did before this module existed. Phases 1-3
  remove the timeout-driven retries that create duplicate execution in the
  first place; this is defence in depth, not the primary fix.
"""

from __future__ import annotations

import logging

from src.config.constants import settings
from src.worker.lifecycle import owner_id
from src.worker.services.redis_client import get_redis_client

logger = logging.getLogger(__name__)

_KEY_PREFIX = "job_lease:"

# Extend the TTL only if we still own the key. Returns 1 on success.
_REFRESH_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
end
return 0
"""

# Delete the key only if we still own it. Returns 1 on success.
_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


def _key(job_id: str) -> str:
    return f"{settings.REDIS_KEY_PREFIX}{_KEY_PREFIX}{job_id}"


def acquire(job_id: str, *, ttl_seconds: int | None = None) -> bool:
    """Try to take the lease for `job_id`. True when this process holds it.

    Returns True when Redis is unavailable: an ungated job is strictly
    better than a job that never runs.
    """
    client = get_redis_client()
    if client is None:
        logger.debug(f"[job_lease] Redis unavailable; running job {job_id} ungated")
        return True
    ttl = int(ttl_seconds or settings.JOB_LEASE_TTL_SECONDS)
    try:
        acquired = client.set(_key(job_id), owner_id(), nx=True, ex=ttl)
    except Exception:
        logger.warning(
            f"[job_lease] acquire failed for {job_id}; running ungated", exc_info=True
        )
        return True
    if acquired:
        logger.info(f"[job_lease] Acquired lease for {job_id} (ttl={ttl}s)")
        return True
    logger.warning(
        f"[job_lease] Job {job_id} is already leased by another worker instance"
    )
    return False


def holder(job_id: str) -> str | None:
    """Return the current lease owner, or None when unleased/unavailable."""
    client = get_redis_client()
    if client is None:
        return None
    try:
        value = client.get(_key(job_id))
    except Exception:
        logger.debug(f"[job_lease] holder lookup failed for {job_id}", exc_info=True)
        return None
    return str(value) if value is not None else None


def refresh(job_id: str, *, ttl_seconds: int | None = None) -> bool:
    """Extend our lease. False when it is no longer ours (or Redis is down)."""
    client = get_redis_client()
    if client is None:
        return False
    ttl = int(ttl_seconds or settings.JOB_LEASE_TTL_SECONDS)
    try:
        return bool(client.eval(_REFRESH_SCRIPT, 1, _key(job_id), owner_id(), ttl))
    except Exception:
        logger.debug(f"[job_lease] refresh failed for {job_id}", exc_info=True)
        return False


def release(job_id: str) -> bool:
    """Release our lease. A non-owner's release is a no-op, never a delete."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        released = bool(client.eval(_RELEASE_SCRIPT, 1, _key(job_id), owner_id()))
    except Exception:
        logger.debug(f"[job_lease] release failed for {job_id}", exc_info=True)
        return False
    if released:
        logger.info(f"[job_lease] Released lease for {job_id}")
    return released
