"""API response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class TranslateResponse(BaseModel):
    """Response model for translation submission."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Initial job status (always 'queued')")
    created_at: datetime = Field(..., description="Job creation timestamp")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    progress: float = Field(..., description="Progress percentage (0.0 - 1.0)")
    current_stage: str | None = Field(None, description="Current processing stage")
    user: str = Field(..., description="User who submitted the job")
    department: str = Field(..., description="Department that requested the job")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: datetime | None = Field(None, description="Job completion timestamp")
    error_message: str | None = Field(None, description="Error message if job failed")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class JobListResponse(BaseModel):
    """Response model for job listing."""

    jobs: list[JobStatusResponse] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    limit: int = Field(..., description="Jobs per page")
    offset: int = Field(..., description="Jobs skipped")


class DownloadResponse(BaseModel):
    """Response model for file download."""

    download_url: str = Field(..., description="Signed URL for downloading the file")
    expires_in: int = Field(..., description="URL expiration time in seconds")
    filename: str = Field(..., description="Suggested filename")
    file_size: int | None = Field(None, description="File size in bytes")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: dict[str, Any] = Field(..., description="Error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "message": "Validation failed",
                    "code": "VALIDATION_ERROR",
                    "details": {"field": "lang_in"},
                }
            }
        }
