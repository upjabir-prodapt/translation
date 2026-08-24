"""Cross-service Cloud Tasks payload schemas."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class TranslateTaskPayload(BaseModel):
    """HTTP body posted by Cloud Tasks to the worker."""

    job_id: str = Field(..., min_length=1)
    traceparent: str | None = None
    tracestate: str | None = None
    # Routing/observability metadata only. Deliberately NON-PII: the worker
    # re-reads the full job -- including cost_attribution (user_id,
    # business_unit, organization) -- from BigQuery in
    # TranslateTaskHandler.handle(), so user identity must never be duplicated
    # into the task body. Task bodies are persisted by Cloud Tasks and surface
    # in logs, so putting PII here would widen exposure for no benefit.
    #
    # Both optional so tasks created by a previous revision still deserialise
    # during a rolling deploy.
    priority: str | None = None
    doc_format: str | None = None
