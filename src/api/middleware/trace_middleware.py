"""Enriches auto-generated FastAPI HTTP spans with business context attributes."""

from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware


class TraceEnrichmentMiddleware(BaseHTTPMiddleware):
    """Injects user/org/job_id attributes into the active OTel HTTP span."""

    async def dispatch(self, request, call_next):
        span = trace.get_current_span()
        user = request.state.__dict__.get("user")
        if user:
            span.set_attribute("user.email", user.email)
            span.set_attribute("user.organization", user.organization)
            span.set_attribute("user.business_unit", user.business_unit)
        job_id = request.path_params.get("job_id")
        if job_id:
            span.set_attribute("translation.job_id", job_id)
        return await call_next(request)
