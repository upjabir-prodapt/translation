"""API request schemas."""

import base64
import binascii
from typing import ClassVar
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from src.config.domain_prompts import GENERIC_DOMAIN
from src.config.domain_prompts import SUPPORTED_DOMAINS
from src.config.domain_prompts import normalize_domain_key
from src.config.translation_routing import normalize_language

MAX_TARGET_LANGUAGES_PER_REQUEST = 5
MAX_BATCH_STATUS_JOB_IDS = 20


class DocumentInput(BaseModel):
    """Document to be translated."""

    content: str = Field(..., description="Base64-encoded document content")
    format: Literal["pdf", "docx"] = Field("pdf", description="Document format")
    filename: str = Field(..., min_length=1, description="Original filename")

    @field_validator("content")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that content is valid base64."""
        try:
            base64.b64decode(v, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError("content must be valid base64-encoded data") from e
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename has supported extension."""
        lower = v.strip().lower()
        if not (lower.endswith(".pdf") or lower.endswith(".docx")):
            raise ValueError("filename must end with .pdf or .docx")
        return v.strip()


class TranslationConfigInput(BaseModel):
    """Translation configuration."""

    # Domains are defined by their prompt profiles, so the API accepts exactly
    # the domains the workflow has a dedicated prompt for.
    VALID_DOMAINS: ClassVar[frozenset[str]] = frozenset(SUPPORTED_DOMAINS)
    SUPPORTED_LANGUAGE_LABELS: ClassVar[str] = (
        "English (en), Spanish (es), Italian (it), French (fr), Japanese (ja), German (de)"
    )

    source_language: str | None = Field(
        None,
        description=(
            "Optional source language (full name or code). "
            "Supported: English, Spanish, Italian, French, Japanese, German. "
            "Auto-detected if omitted. Cannot equal target_language."
        ),
    )
    target_language: str = Field(
        ...,
        min_length=2,
        description=(
            "Target language (full name or code). "
            "Supported: English, Spanish, Italian, French, Japanese, German."
        ),
    )
    domain: str = Field(
        ...,
        min_length=2,
        max_length=20,
        description="Translation domain: commercial, legal, finance, hr, operations",
    )

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Validate and normalize domain, accepting known aliases."""
        normalized = normalize_domain_key(v)
        if normalized == GENERIC_DOMAIN:
            raise ValueError(
                f"Invalid domain. Allowed: {', '.join(sorted(cls.VALID_DOMAINS))}"
            )
        return normalized

    @field_validator("target_language")
    @classmethod
    def validate_language_codes(cls, v: str) -> str:
        """Validate target language input: full name or code only."""
        cleaned = v.strip()
        try:
            normalize_language(cleaned)
        except ValueError as e:
            raise ValueError(
                f"{e}. Supported languages: {cls.SUPPORTED_LANGUAGE_LABELS}"
            ) from e
        return cleaned

    @field_validator("source_language")
    @classmethod
    def validate_source_language(cls, v: str | None) -> str | None:
        """Validate optional source language input: full name or code only."""
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        try:
            normalize_language(cleaned)
        except ValueError as e:
            raise ValueError(
                f"{e}. Supported languages: {cls.SUPPORTED_LANGUAGE_LABELS}"
            ) from e
        return cleaned

    @model_validator(mode="after")
    def validate_source_not_equal_target(self) -> "TranslationConfigInput":
        """Ensure source language does not equal target language."""
        if self.source_language is not None:
            source_normalized = normalize_language(self.source_language)
            target_normalized = normalize_language(self.target_language)
            if source_normalized == target_normalized:
                raise ValueError("Source language cannot equal target language")
        return self


class TranslationTargetsInput(BaseModel):
    """API-boundary selection of one or more target languages."""

    target_languages: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_TARGET_LANGUAGES_PER_REQUEST,
        description="One or more target languages for translation",
    )

    @field_validator("target_languages")
    @classmethod
    def validate_targets(cls, values: list[str]) -> list[str]:
        """Normalize and validate target languages."""
        normalized = [normalize_language(value.strip()) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Target languages must be unique after normalization")
        return normalized

    @property
    def normalized_targets(self) -> list[str]:
        """Return normalized targets in request order."""
        return self.target_languages


class ProcessingOptions(BaseModel):
    """Processing options for translation job."""

    enable_dlp: bool = Field(
        True, description="Enable Data Loss Prevention masking before translation"
    )
    enable_chunking: bool = Field(
        True, description="Enable chunked processing for large documents"
    )
    priority: Literal["standard", "high"] = Field(
        "standard", description="Processing priority"
    )


class CostAttributionInput(BaseModel):
    """Billing ownership metadata for cost attribution."""

    user_id: str = Field(..., min_length=1, description="Submitting user ID")
    business_unit: str = Field(..., min_length=1, description="Business unit name")
    organization: str = Field(..., min_length=1, description="Organization name")


class AuthTokenRequest(BaseModel):
    """Request payload for token issuance (email derived from IAP JWT)."""

    email: str | None = Field(
        default=None,
        min_length=3,
        description="Deprecated — email is taken from verified IAP identity",
    )
    business_unit: str = Field(..., min_length=1, description="Business unit name")
    organization: str = Field(..., min_length=1, description="Organization name")


class TranslateRequest(BaseModel):
    """Request model for document translation."""

    document: DocumentInput
    translation_config: TranslationConfigInput
    cost_attribution: CostAttributionInput
    processing_options: ProcessingOptions = Field(default_factory=ProcessingOptions)


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


class MultiJobStatusRequest(BaseModel):
    """Request status for multiple ordinary translation jobs."""

    job_ids: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_STATUS_JOB_IDS)

    @field_validator("job_ids")
    @classmethod
    def validate_unique_job_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("Job IDs must not be empty")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Job IDs must be unique")
        return cleaned


class CreateReviewRequest(BaseModel):
    """Request model for submitting a translation review."""

    rating: int = Field(
        ..., ge=1, le=5, description="Rating from 1 (worst) to 5 (best)"
    )
    comment: str | None = Field(
        None, max_length=2000, description="Optional review comment"
    )
