"""Shared LLM batch-size planning for the DOCX and PDF pipelines.

Both pipelines batch paragraphs/units into a single LLM prompt, and both
previously read `settings.LLM_*_BATCH_MAX_TOKENS` directly. Measured against
the 2026-08-24 baseline, those flat caps (40000 / 80000) were so large that a
414-unit document produced only 1-3 batches for a 12-worker thread pool, so
the pools sat ~60% idle while each individual call ran 100-140s.

Empirical model fitted over 114 logged production calls:

    latency(s) ~= 16.6 + 12.7 * (payload_tokens / 1000)

The ~16.6s fixed per-call overhead means tiny batches waste time on overhead,
while huge batches serialise work that could have run in parallel. Minimising
*wall-clock* (not per-call throughput) puts the optimum at roughly
`total_payload_tokens / pool_max_workers` -- i.e. exactly enough batches to
fill the pool in one wave -- clamped into the empirically flat 600-2500 token
region so small documents don't fragment and huge ones don't serialise.

`compute_batch_plan()` is the single place that decision is made; both
pipelines call it so their batching behaviour cannot drift apart again.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from src.config.constants import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-wide ceiling on concurrently in-flight LLM calls.
#
# The DOCX path nests thread pools: translate_all() runs up to
# TRANSLATION_POOL_MAX_WORKERS batches, and *each* of those may open its own
# fallback pool of the same size. At 12 workers that is 12 + 12x12 = 156
# potential simultaneous Vertex calls from a single job, on a Cloud Run
# instance sized cpu=4 / containerConcurrency=1.
#
# Threads themselves are cheap (~12 MB RSS for 144 idle threads), so this is
# not primarily a memory guard -- it bounds the *working set* each in-flight
# call carries (TLS connection, request/response JSON, tiktoken buffers) and
# stops a fallback storm from silently exceeding the QPS budget the rate
# limiter is trying to enforce.
#
# A semaphore is used rather than shrinking the pools because the pool sizes
# are what give a single wave its parallelism; only the *total* needs capping.
# ---------------------------------------------------------------------------
_llm_slots_lock = threading.Lock()
_llm_slots: threading.BoundedSemaphore | None = None


def _get_llm_slots() -> threading.BoundedSemaphore | None:
    """Lazily build the shared in-flight-call semaphore (None = unbounded)."""
    global _llm_slots
    limit = int(getattr(settings, "LLM_MAX_INFLIGHT_CALLS", 0) or 0)
    if limit <= 0:
        return None
    with _llm_slots_lock:
        if _llm_slots is None:
            _llm_slots = threading.BoundedSemaphore(limit)
        return _llm_slots


@contextmanager
def llm_call_slot():
    """Acquire one slot from the process-wide in-flight LLM call budget.

    No-ops when `LLM_MAX_INFLIGHT_CALLS <= 0`, so the guard can be disabled
    without touching call sites.
    """
    slots = _get_llm_slots()
    if slots is None:
        yield
        return
    slots.acquire()
    try:
        yield
    finally:
        slots.release()


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Resolved batch-size caps for one translation/extraction stage."""

    max_tokens: int
    max_items: int
    total_payload_tokens: int
    pool_max_workers: int
    adaptive: bool


def compute_batch_plan(
    total_payload_tokens: int,
    pool_max_workers: int,
    *,
    base_max_tokens: int,
    base_max_items: int,
    token_multiplier: float = 1.0,
) -> BatchPlan:
    """Resolve the token/item caps to use for one batching pass.

    `token_multiplier` is the pre-existing CJK scaling factor (see
    `get_token_multiplier`): tiktoken's gpt-4o encoding under-counts CJK
    tokens relative to what Gemini/Claude actually consume, so item caps are
    scaled down for CJK target languages. It is applied in both adaptive and
    fixed modes so this function is a drop-in for the previous behaviour.

    When `LLM_ADAPTIVE_BATCHING_ENABLED` is false, the configured caps are
    returned unchanged (with the CJK multiplier still applied to the item
    cap), preserving the exact legacy semantics for rollback.
    """
    safe_workers = max(1, int(pool_max_workers))
    scaled_items = max(1, int(base_max_items * token_multiplier))

    if not settings.LLM_ADAPTIVE_BATCHING_ENABLED or total_payload_tokens <= 0:
        return BatchPlan(
            max_tokens=int(base_max_tokens),
            max_items=scaled_items,
            total_payload_tokens=int(total_payload_tokens),
            pool_max_workers=safe_workers,
            adaptive=False,
        )

    floor = max(1, int(settings.LLM_ADAPTIVE_BATCH_MIN_TOKENS))
    ceiling = max(floor, int(settings.LLM_ADAPTIVE_BATCH_MAX_TOKENS))

    # One wave: enough batches to saturate the pool exactly once. Round the
    # target UP -- truncating (e.g. 10400/12 = 866.67 -> 866) would spill a
    # 13th batch onto a 12-worker pool and cost an entire extra wave.
    target = -(-total_payload_tokens // safe_workers)
    adaptive_tokens = int(min(max(target, floor), ceiling))
    # Never exceed the configured hard cap -- adaptive sizing may only make
    # batches smaller than the operator-configured ceiling, never larger.
    adaptive_tokens = min(adaptive_tokens, int(base_max_tokens))
    adaptive_tokens = max(1, int(adaptive_tokens * token_multiplier))

    return BatchPlan(
        max_tokens=adaptive_tokens,
        max_items=scaled_items,
        total_payload_tokens=int(total_payload_tokens),
        pool_max_workers=safe_workers,
        adaptive=True,
    )


def log_batch_plan(stage: str, plan: BatchPlan, batch_count: int) -> None:
    """Emit the Phase-0 batching/concurrency instrumentation line.

    Makes "is the worker pool actually being filled?" answerable directly
    from a log file instead of by reconstructing it from call timestamps.
    """
    waves = -(-batch_count // plan.pool_max_workers) if batch_count else 0
    utilisation = (
        round(min(batch_count, plan.pool_max_workers) / plan.pool_max_workers, 3)
        if plan.pool_max_workers
        else 0.0
    )
    median_payload = (
        int(plan.total_payload_tokens / batch_count) if batch_count else 0
    )
    logger.info(
        f"batch_plan stage={stage} batch_count={batch_count} "
        f"max_tokens={plan.max_tokens} max_items={plan.max_items} "
        f"total_payload_tokens={plan.total_payload_tokens} "
        f"mean_batch_payload_tokens={median_payload} "
        f"pool_max_workers={plan.pool_max_workers} waves={waves} "
        f"pool_utilisation={utilisation} adaptive={plan.adaptive}"
    )
