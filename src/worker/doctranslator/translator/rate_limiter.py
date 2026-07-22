"""Global translation rate limiter."""

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


_translate_rate_limiter = RateLimiter(max(int(settings.TRANSLATION_MAX_QPS), 1))


def get_translate_rate_limiter() -> RateLimiter:
    return _translate_rate_limiter


def set_translate_rate_limiter(max_qps: int) -> None:
    _translate_rate_limiter.set_max_qps(max_qps)
