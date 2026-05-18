"""Named tracers and @span decorator for all instrumented layers."""

import inspect
from functools import wraps

from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode


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
