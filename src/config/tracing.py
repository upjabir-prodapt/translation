"""Named tracers and @span decorator for all instrumented layers."""

import inspect
from contextvars import ContextVar
from functools import wraps

from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode

# ── Root span propagation ────────────────────────────────────────────────────
# Stores the outermost pipeline span so any code deep in the call stack can
# annotate it directly without needing to pass it down through every layer.
_root_span_var: ContextVar = ContextVar("pipeline_root_span", default=None)


def set_root_span(span) -> None:
    """Register *span* as the root span for the current async/thread context."""
    _root_span_var.set(span)


def set_root_span_attribute(key: str, value) -> None:
    """Set a single attribute on the root span if one is registered."""
    span = _root_span_var.get()
    if span is not None and span.is_recording():
        span.set_attribute(key, value)


def set_root_span_attributes(attrs: dict) -> None:
    """Bulk-set attributes on the root span if one is registered."""
    span = _root_span_var.get()
    if span is not None and span.is_recording():
        for key, value in attrs.items():
            span.set_attribute(key, value)


def get_tracer(name: str):
    return trace.get_tracer(name)


tracer_http = get_tracer("translation_api.http")
tracer_llm = get_tracer("translation_api.llm")
tracer_pipeline = get_tracer("translation_api.pipeline")
tracer_repository = get_tracer("translation_api.repository")


def span(tracer, name: str, kind: SpanKind = SpanKind.INTERNAL, attributes: dict = None):
    """Decorator that wraps a sync or async function in an OTel span."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as s:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    s.set_status(Status(StatusCode.ERROR, str(exc)))
                    s.record_exception(exc)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as s:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    s.set_status(Status(StatusCode.ERROR, str(exc)))
                    s.record_exception(exc)
                    raise

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator
