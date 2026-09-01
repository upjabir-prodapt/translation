"""API response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

JOB_ID_DESCRIPTION = "Unique job identifier"


class TranslateResponse(BaseModel):
    """Response model for translation submission (POST /translate)."""

    job_id: str = Field(..., description=JOB_ID_DESCRIPTION)
    status: str = Field(..., description="Initial job status (always 'queued')")
    status_url: str = Field(..., description="URL to poll for job status")
    is_duplicate: bool = Field(
        False,
        description=(
            "True when job_id refers to an existing job reused for this "
            "submission (implementation_plan.md D.5 / EC-15 idempotency "
            "window) rather than a newly created one."
        ),
    )


class MultiTranslateJobResponse(BaseModel):
    """One ordinary translation job created by a multi-target submission."""

    job_id: str = Field(..., description=JOB_ID_DESCRIPTION)
    target_language: str = Field(..., description="Normalized target language")
    status: str = Field(..., description="Initial job status")
    status_url: str = Field(..., description="URL to poll for this job")
    is_duplicate: bool = Field(
        False,
        description=(
            "True when job_id refers to an existing job reused for this "
            "target language (implementation_plan.md D.5 / EC-15 "
            "idempotency window) rather than a newly created one."
        ),
    )


class MultiTranslateResponse(BaseModel):
    """Response for a multi-target translation submission."""

    batch_id: str = Field(..., description="Grouping ID for this submission")
    jobs: list[MultiTranslateJobResponse] = Field(
        ..., min_length=1, description="Ordinary jobs in requested target order"
    )


class AuthTokenResponse(BaseModel):
    """Response model for access token issuance."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Access token lifetime in seconds")
    email: str = Field(..., description="Verified user email from IAP identity")
    refresh_token: str = Field(
        ..., description="JWT refresh token for POST /auth/refresh"
    )
    refresh_expires_in: int = Field(
        ...,
        description=(
            "Seconds left on the refresh token. Counts down across refreshes: "
            "it is an absolute session cap, not a sliding window."
        ),
    )


class WhoamiResponse(BaseModel):
    """Response model for IAP identity probe."""

    email: str = Field(..., description="Verified user email from IAP identity")
    entitled: bool = Field(
        True,
        description="Whether the user is entitled to this service (Entra group membership)",
    )


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

    job_id: str = Field(..., description=JOB_ID_DESCRIPTION)
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

    job_id: str = Field(..., description=JOB_ID_DESCRIPTION)
    status: str = Field(..., description="Current job status")
    progress: float = Field(..., description="Progress percentage (0.0 - 1.0)")
    current_stage: str | None = Field(None, description="Current processing stage")
    user: str = Field(..., description="User who submitted the job")
    department: str = Field(..., description="Department that requested the job")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: datetime | None = Field(None, description="Job completion timestamp")
    download_url: str | None = Field(
        None, description="Signed URL to download translated output when available"
    )
    error_message: str | None = Field(None, description="Error message if job failed")
    filename: str | None = Field(
        None, description="Original (source) document filename"
    )
    source_language: str | None = Field(
        None, description="Source language, if known/detected"
    )
    target_language: str | None = Field(None, description="Target language")


class JobListResponse(BaseModel):
    """Response model for job listing."""

    jobs: list[JobStatusResponse] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    limit: int = Field(..., description="Jobs per page")
    offset: int = Field(..., description="Jobs skipped")


class MultiJobStatusItemResponse(BaseModel):
    """Status and output metadata for one requested translation job."""

    job_id: str = Field(..., description=JOB_ID_DESCRIPTION)
    target_language: str | None = Field(None, description="Normalized target language")
    status: str = Field(..., description="Current job status")
    download_url: str | None = Field(None, description="Signed output URL")
    download_filename: str | None = Field(None, description="Output filename")
    error_message: str | None = Field(
        None, description="Failure or cancellation reason"
    )


class MultiJobStatusResponse(BaseModel):
    """Ordered statuses for a multi-job lookup."""

    jobs: list[MultiJobStatusItemResponse]


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


class ReviewSubmitResponse(BaseModel):
    """Response model for review submission (POST /reviews/{job_id})."""

    status: str = Field(..., description="'successfully sent' or 'failed'")
    review_id: str | None = Field(
        None, description="Unique review identifier (present on success)"
    )


class ReviewResponse(BaseModel):
    """Response model for a single translation review."""

    review_id: str = Field(..., description="Unique review identifier")
    job_id: str = Field(..., description=JOB_ID_DESCRIPTION)
    rating: int = Field(..., description="Rating from 1 to 5")
    comment: str | None = Field(None, description="Optional review comment")
    reviewer_email: str = Field(..., description="Email of the reviewer")
    created_at: datetime = Field(..., description="Review creation timestamp")
    updated_at: datetime = Field(..., description="Review last updated timestamp")


class ReviewListResponse(BaseModel):
    """Response model for listing reviews of a job."""

    job_id: str = Field(..., description=JOB_ID_DESCRIPTION)
    reviews: list[ReviewResponse] = Field(..., description="List of reviews")
    total: int = Field(..., description="Total number of reviews for this job")


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
