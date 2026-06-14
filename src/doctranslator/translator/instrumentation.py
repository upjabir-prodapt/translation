"""OpenTelemetry instrumentation helpers for LLM SDK calls."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any

from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode

from src.config.tracing import tracer_llm
from src.doctranslator.translator.usage import TokenUsage

logger = logging.getLogger(__name__)

_MAX_CHARS_PROMPT_PREVIEW = 300

# OTel span attribute key constants (avoids S1192 duplicate-literal warnings)
ATTR_LLM_MODEL = "llm.model"
ATTR_LLM_NAME = "llm.name"
ATTR_LLM_PROVIDER = "llm.provider"
ATTR_LLM_TEMPERATURE = "llm.temperature"
ATTR_LLM_INPUT_TOKENS = "llm.input_tokens"
ATTR_LLM_OUTPUT_TOKENS = "llm.output_tokens"
ATTR_LLM_TOTAL_TOKENS = "llm.total_tokens"
ATTR_LLM_PROMPT_CHARS = "llm.prompt_chars"
ATTR_LLM_PROMPT_HASH = "llm.prompt_hash"
ATTR_LLM_PROMPT_PREVIEW = "llm.prompt_preview"
ATTR_LLM_INPUT_CHARS = "llm.input_chars"
ATTR_LLM_OUTPUT_CHARS = "llm.output_chars"
ATTR_LLM_LATENCY_S = "llm.latency_s"
ATTR_LLM_CACHED_TOKENS = "llm.cached_tokens"

# Backward-compatible aliases used by existing code
_ATTR_LLM_MODEL = ATTR_LLM_MODEL
_ATTR_LLM_NAME = ATTR_LLM_NAME
_ATTR_LLM_PROVIDER = ATTR_LLM_PROVIDER
_ATTR_LLM_TEMPERATURE = ATTR_LLM_TEMPERATURE
_ATTR_LLM_INPUT_TOKENS = ATTR_LLM_INPUT_TOKENS
_ATTR_LLM_OUTPUT_TOKENS = ATTR_LLM_OUTPUT_TOKENS
_ATTR_LLM_TOTAL_TOKENS = ATTR_LLM_TOTAL_TOKENS
_ATTR_LLM_PROMPT_CHARS = ATTR_LLM_PROMPT_CHARS
_ATTR_LLM_PROMPT_HASH = ATTR_LLM_PROMPT_HASH
_ATTR_LLM_PROMPT_PREVIEW = ATTR_LLM_PROMPT_PREVIEW
_ATTR_LLM_INPUT_CHARS = ATTR_LLM_INPUT_CHARS
_ATTR_LLM_OUTPUT_CHARS = ATTR_LLM_OUTPUT_CHARS
_ATTR_LLM_LATENCY_S = ATTR_LLM_LATENCY_S


def prompt_fingerprint(contents: str) -> tuple[int, str, str]:
    """Return prompt_chars, prompt_hash, and prompt_preview for tracing."""
    prompt_chars = len(contents)
    prompt_hash = hashlib.sha256(
        contents.encode("utf-8", errors="replace")
    ).hexdigest()[:12]
    prompt_preview = contents[:_MAX_CHARS_PROMPT_PREVIEW].replace("\n", "\\n")
    return prompt_chars, prompt_hash, prompt_preview


def usage_to_span_attributes(usage: TokenUsage) -> dict[str, int]:
    attrs = {
        ATTR_LLM_INPUT_TOKENS: usage.input_tokens,
        ATTR_LLM_OUTPUT_TOKENS: usage.output_tokens,
        ATTR_LLM_TOTAL_TOKENS: usage.total_tokens,
    }
    if usage.cache_hit_input_tokens:
        attrs[ATTR_LLM_CACHED_TOKENS] = usage.cache_hit_input_tokens
    return attrs


def instrumented_llm_call(
    *,
    span_name: str,
    attributes: dict[str, Any],
    call: Callable[[], Any],
    usage_fn: Callable[[Any], TokenUsage] | None = None,
    log_prefix: str = "llm",
) -> Any:
    """Execute an LLM SDK call inside a traced span with latency and usage."""
    t0 = time.monotonic()
    with tracer_llm.start_as_current_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes=attributes,
    ) as span:
        try:
            result = call()
            if usage_fn is not None:
                usage = usage_fn(result)
                for key, value in usage_to_span_attributes(usage).items():
                    span.set_attribute(key, value)
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            elapsed = time.monotonic() - t0
            span.set_attribute(ATTR_LLM_LATENCY_S, round(elapsed, 3))
            logger.debug(
                "%s finished: span=%s latency_s=%.3f",
                log_prefix,
                span_name,
                elapsed,
            )
