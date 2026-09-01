"""One place that composes the three LLM call primitives.

Three orthogonal concerns wrap every outbound LLM request:

* `llm_call_slot()` (`doctranslator/batching.py`) -- the process-wide
  `LLM_MAX_INFLIGHT_CALLS` budget;
* `llm_retry` (`src/config/retry.py`) -- tenacity with the shared
  retryable-exception predicate and exponential backoff;
* `instrumented_llm_call` (`translator/instrumentation.py`) -- the OTel
  CLIENT span with latency and token-usage attributes.

They already existed but were wired up by hand at each call site, and the
quality judge only ever wired up two of the three: it retried and traced,
but never acquired a slot. That was harmless while the judge made exactly
one call per attempt; the chunked judge fans out, and an uncapped second
fan-out running alongside the translator's would blow straight through the
budget the semaphore exists to enforce.

Ordering matters and is fixed here: the slot is acquired *inside* the retry,
so a call waiting out a 429 backoff releases its slot to a call that can
make progress instead of holding the budget hostage for the whole backoff.

Scope: new call sites (currently the judge). The translator providers are
deliberately not migrated -- see the plan's "Out of scope".
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.config.retry import llm_retry
from src.worker.doctranslator.batching import llm_call_slot
from src.worker.doctranslator.translator.instrumentation import instrumented_llm_call
from src.worker.doctranslator.translator.usage import TokenUsage

_default_logger = logging.getLogger(__name__)


def invoke_llm[T](
    call: Callable[[], T],
    *,
    span_name: str,
    attributes: dict[str, Any] | None = None,
    usage_fn: Callable[[T], TokenUsage] | None = None,
    log_prefix: str = "llm",
    logger: logging.Logger | None = None,
    retry_on: tuple[type[BaseException], ...] = (),
) -> T:
    """Run one blocking LLM SDK call under slot + retry + instrumentation.

    `call` must be idempotent: tenacity may invoke it several times. It is
    re-invoked on transient failures only (see `is_retryable_llm_exception`),
    plus any type listed in `retry_on`; everything else propagates on the
    first raise.

    Putting response *parsing* inside `call` is deliberate and supported --
    that is what lets a malformed structured response be retried instead of
    being swallowed into a fabricated fallback score.
    """
    retry_logger = logger or _default_logger

    @llm_retry(logger=retry_logger, also_retry_on=retry_on)
    def _attempt() -> T:
        # Acquired inside the retry, not outside: a call sleeping through
        # exponential backoff must not occupy one of the
        # LLM_MAX_INFLIGHT_CALLS slots while doing nothing.
        with llm_call_slot():
            return instrumented_llm_call(
                span_name=span_name,
                attributes=dict(attributes or {}),
                call=call,
                usage_fn=usage_fn,
                log_prefix=log_prefix,
            )

    return _attempt()
