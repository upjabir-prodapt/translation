"""Per-provider translation rate limiters.

Previously this module exposed a single process-wide mutable RateLimiter
singleton, and `create_translator()` called `set_translate_rate_limiter()`
on every translator construction (i.e. every model attempt, every job).
Since Cloud Run can run multiple concurrent in-process pipelines (dev mode)
or multiple translators within one process/job (multi-target-language
fan-out), that meant concurrent translations fought over -- and could
silently reset -- one shared QPS budget.

This module now keeps one RateLimiter per provider name, created once with
its configured QPS and never silently overwritten afterwards. Concurrent
jobs/languages sharing the same provider still correctly share that
provider's QPS budget (which is the intent -- it reflects a real upstream
API rate limit), but no longer stomp on each other's configuration.
"""

from __future__ import annotations

import threading
import time

from src.config.constants import settings


class RateLimiter:
    """Thread-safe leaky-bucket rate limiter."""

    def __init__(self, max_qps: int):
        if max_qps <= 0:
            raise ValueError("max_qps must be a positive number")
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.lock = threading.Lock()
        self.next_request_time = time.monotonic()

    def wait(self, _rate_limit_params: dict | None = None) -> None:
        with self.lock:
            now = time.monotonic()
            wait_duration = self.next_request_time - now
            if wait_duration > 0:
                time.sleep(wait_duration)
            now = time.monotonic()
            self.next_request_time = (
                max(self.next_request_time, now) + self.min_interval
            )

    def set_max_qps(self, max_qps: int) -> None:
        if max_qps <= 0:
            raise ValueError("max_qps must be a positive number")
        with self.lock:
            self.max_qps = max_qps
            self.min_interval = 1.0 / max_qps


_registry_lock = threading.Lock()
_rate_limiters: dict[str, RateLimiter] = {}

# Default/legacy limiter key used when callers don't specify a provider.
_DEFAULT_KEY = "__default__"


def get_translate_rate_limiter(provider: str | None = None) -> RateLimiter:
    """Return the shared RateLimiter for a provider, creating it on first use.

    Each provider gets its own limiter so concurrent translations across
    different providers never share (or fight over) the same QPS budget.
    """
    key = provider or _DEFAULT_KEY
    with _registry_lock:
        limiter = _rate_limiters.get(key)
        if limiter is None:
            limiter = RateLimiter(max(int(settings.TRANSLATION_MAX_QPS), 1))
            _rate_limiters[key] = limiter
        return limiter


def set_translate_rate_limiter(max_qps: int, provider: str | None = None) -> None:
    """Ensure a rate limiter exists for `provider` with at least `max_qps`.

    Unlike the old global-singleton behavior, this never lowers/resets an
    already-initialized limiter out from under other in-flight callers --
    it only creates the limiter on first use (using the configured qps) and
    otherwise leaves an existing limiter's rate untouched. Pass an explicit
    `provider` to scope the limiter; omitting it preserves the legacy
    single-limiter behavior for callers that don't care about per-provider
    isolation.
    """
    key = provider or _DEFAULT_KEY
    with _registry_lock:
        limiter = _rate_limiters.get(key)
        if limiter is None:
            _rate_limiters[key] = RateLimiter(max(int(max_qps), 1))
        # If a limiter already exists for this key, intentionally leave its
        # rate as-is: concurrent jobs/attempts must not silently reset each
        # other's shared QPS budget mid-flight.
