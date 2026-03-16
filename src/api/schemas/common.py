"""Common schemas shared across the API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class BaseJobSchema(BaseModel):
    """Base schema for job-related data."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class FileInfo(BaseModel):
    """Information about a file."""

    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type")
    checksum: str | None = Field(None, description="MD5 checksum")


class ProgressInfo(BaseModel):
    """Progress information for a job."""

    percentage: float = Field(..., ge=0.0, le=1.0, description="Progress (0.0 - 1.0)")
    current_stage: str | None = Field(None, description="Current stage name")
    eta_seconds: int | None = Field(None, description="Estimated time remaining")

    class Config:
        json_schema_extra = {
            "example": {
                "percentage": 0.65,
                "current_stage": "IL Translator",
                "eta_seconds": 120,
            }
        }


class OutputFiles(BaseModel):
    """Output file information."""

    mono_pdf: str | None = Field(None, description="GCS URI for monolingual PDF")
    dual_pdf: str | None = Field(None, description="GCS URI for dual-language PDF")
    no_watermark_mono: str | None = Field(
        None, description="GCS URI for no-watermark monolingual PDF"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Additional metadata about outputs"
    )
