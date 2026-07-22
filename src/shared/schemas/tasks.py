"""Cross-service Cloud Tasks payload schemas."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class TranslateTaskPayload(BaseModel):
    """HTTP body posted by Cloud Tasks to the worker."""

    job_id: str = Field(..., min_length=1)
    traceparent: str | None = None
    tracestate: str | None = None
