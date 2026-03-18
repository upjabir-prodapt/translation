"""API request schemas."""

from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class TranslateRequest(BaseModel):
    """Request model for PDF translation."""

    VALID_DOMAINS: ClassVar[set[str]] = {
        "commercial",
        "legal",
        "finance",
        "hr",
        "oprations",
    }

    domain: str = Field(
        ...,
        min_length=2,
        max_length=20,
        description="Domain specialization for translation",
    )
    lang_in: str = Field(
        "auto",
        min_length=2,
        max_length=5,
        description="Source language code or 'auto' for automatic detection",
    )
    lang_out: str = Field(
        ...,
        min_length=2,
        max_length=5,
        description="Target language code (e.g., 'en', 'zh')",
    )
    user: str = Field(..., min_length=1, description="User submitting the translation")
    department: str = Field(
        ..., min_length=1, description="Department requesting the translation"
    )

    @field_validator("lang_in", "lang_out")
    @classmethod
    def validate_language_codes(cls, v: str) -> str:
        """Validate language codes."""
        normalized = v.strip().lower()
        if normalized == "auto":
            return normalized
        if not normalized.isalpha() or len(normalized) < 2 or len(normalized) > 5:
            raise ValueError("Invalid language code")
        return normalized

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Validate and normalize domain."""
        normalized = v.strip().lower()
        if normalized not in cls.VALID_DOMAINS:
            raise ValueError(
                "Invalid domain. Allowed: commercial, legal, finance, hr, oprations"
            )
        return normalized


class JobCancelRequest(BaseModel):
    """Request model for job cancellation."""

    reason: str | None = Field(None, description="Optional reason for cancellation")


class JobListRequest(BaseModel):
    """Request model for listing jobs."""

    status: str | None = Field(
        None,
        pattern="^(queued|processing|completed|failed|cancelled)$",
        description="Filter by job status",
    )
    limit: int = Field(10, ge=1, le=100, description="Maximum number of jobs to return")
    offset: int = Field(0, ge=0, description="Number of jobs to skip")
