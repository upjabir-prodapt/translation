"""Shared retry policies for external service calls."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from src.config.constants import settings

_RETRYABLE_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_RETRYABLE_ERROR_SUBSTRINGS = (
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "service unavailable",
    "timeout",
    "deadline exceeded",
    "connection reset",
)


def is_retryable_llm_exception(exc: BaseException) -> bool:
    """Return True when an exception is likely transient for LLM calls."""
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            OSError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.HTTPError,
        ),
    ):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in _RETRYABLE_HTTP_STATUS_CODES:
        return True

    error_message = str(exc).lower()
    return any(keyword in error_message for keyword in _RETRYABLE_ERROR_SUBSTRINGS)


def llm_retry(*, logger: logging.Logger) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Build a Tenacity decorator for shared LLM retry policy."""
    return retry(
        stop=stop_after_attempt(settings.LLM_RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=settings.LLM_RETRY_MULTIPLIER,
            min=settings.LLM_RETRY_MIN_SECONDS,
            max=settings.LLM_RETRY_MAX_SECONDS,
        ),
        retry=retry_if_exception(is_retryable_llm_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
