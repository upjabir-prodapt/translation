"""API response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class TranslateResponse(BaseModel):
    """Response model for translation submission (POST /translate)."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Initial job status (always 'queued')")
    status_url: str = Field(..., description="URL to poll for job status")


# ---------------------------------------------------------------------------
# GET /translate/{job_id} — detailed job result
# ---------------------------------------------------------------------------


class TranslatedDocumentResult(BaseModel):
    """Translated document output details."""

    content: str | None = Field(
        None, description="Base64-encoded translated document (null — use download_url)"
    )
    format: str | None = Field(None, description="Output document format")
    filename: str | None = Field(None, description="Suggested output filename")
    download_url: str | None = Field(
        None, description="Signed URL to download the translated document"
    )


class TranslationMetadata(BaseModel):
    """Metadata about the translation process."""

    source_language: str | None = Field(
        None, description="Detected or provided source language"
    )
    target_language: str | None = Field(None, description="Target language")
    domain: str | None = Field(None, description="Translation domain")
    model_used: str | None = Field(None, description="Model ID used for translation")
    model_version: str | None = Field(None, description="Model version")
    quality_score: float | None = Field(None, description="Quality score (0.0–1.0)")
    ab_test_variant: str | None = Field(
        None, description="A/B test variant (A = first model, B = second, etc.)"
    )
    chunks_processed: int | None = Field(None, description="Number of chunks processed")
    retry_attempts: int | None = Field(
        None, description="Number of retry attempts made"
    )


class TranslationLabels(BaseModel):
    """Cost and performance labels for the translation job."""

    translation_intent: str | None = Field(None, description="Routing intent label")
    processing_time_seconds: int | None = Field(
        None, description="Total processing time in seconds"
    )
    token_count: int | None = Field(None, description="Total tokens consumed")
    cost_usd: float | None = Field(None, description="Estimated cost in USD")


class TranslationResult(BaseModel):
    """Full translation result payload."""

    translated_document: TranslatedDocumentResult | None = None
    metadata: TranslationMetadata | None = None
    labels: TranslationLabels | None = None


class JobDetailResponse(BaseModel):
    """Response model for GET /translate/{job_id}."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    submitted_at: datetime | None = Field(None, description="Job submission timestamp")
    completed_at: datetime | None = Field(None, description="Job completion timestamp")
    result: TranslationResult | None = Field(
        None, description="Translation result (populated on completion)"
    )
    error_message: str | None = Field(
        None, description="Failure or cancellation reason when job did not succeed"
    )


# ---------------------------------------------------------------------------
# Existing schemas (unchanged)
# ---------------------------------------------------------------------------


class JobStatusResponse(BaseModel):
    """Response model for job status (legacy /jobs/{job_id} endpoint)."""

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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "message": "Validation failed",
                    "code": "VALIDATION_ERROR",
                    "details": {"field": "lang_in"},
                }
            }
        }
    )

    error: dict[str, Any] = Field(..., description="Error details")
