"""API request schemas."""

import base64
from typing import ClassVar
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from src.config.constants import settings
from src.config.translation_routing import normalize_language

MAX_TARGET_LANGUAGES_PER_REQUEST = 5
MAX_BATCH_STATUS_JOB_IDS = 20


class DocumentInput(BaseModel):
    """Document to be translated."""

    content: str = Field(..., description="Base64-encoded document content")
    format: Literal["pdf", "docx", "txt"] = Field("pdf", description="Document format")
    filename: str = Field(..., min_length=1, description="Original filename")

    @field_validator("content")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that content is valid base64."""
        try:
            base64.b64decode(v, validate=True)
        except ValueError as e:
            raise ValueError("content must be valid base64-encoded data") from e
        return v

    # Cap on the stored filename length (before the extension), leaving
    # headroom under typical GCS/filesystem path-component limits even
    # after a UUID job-id prefix is added elsewhere in the path.
    _MAX_FILENAME_STEM_LENGTH: ClassVar[int] = 100

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename has a supported extension, then sanitize it
        for safe storage as a GCS blob path component.

        Extension check is driven by `settings.ALLOWED_EXTENSIONS`
        (implementation_plan.md A.4.1) rather than a hardcoded tuple, so
        the two never drift apart again -- `ALLOWED_EXTENSIONS` used to
        list unsupported `.doc` and omit supported `.txt` while being
        read nowhere.

        Sanitization (A.4.4) strips path separators and control
        characters and caps the length, so a 150-char filename with
        accents/emoji is accepted but never used verbatim as a blob path
        component (path traversal / overlong-path safety), while still
        preserving the real extension and enough of the stem to be
        recognizable.
        """
        stripped = v.strip()
        lower = stripped.lower()
        allowed = {str(ext).strip().lower() for ext in settings.ALLOWED_EXTENSIONS}
        matched_ext = next((ext for ext in allowed if lower.endswith(ext)), None)
        if matched_ext is None:
            allowed_display = ", ".join(sorted(allowed))
            raise ValueError(f"filename must end with one of: {allowed_display}")

        # Strip directory components (path traversal) and control chars,
        # keeping the rest (including accents/emoji) intact.
        base_name = stripped.replace("\\", "/").rsplit("/", 1)[-1]
        base_name = "".join(ch for ch in base_name if ch.isprintable())
        stem = base_name[: -len(matched_ext)] if base_name else base_name
        stem = stem[: cls._MAX_FILENAME_STEM_LENGTH] or "document"
        return f"{stem}{matched_ext}"


class TranslationConfigInput(BaseModel):
    """Translation configuration."""

    VALID_DOMAINS: ClassVar[set[str]] = {
        "commercial",
        "legal",
        "finance",
        "hr",
        "operations",
    }
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
        """Validate and normalize domain."""
        normalized = v.strip().lower()

        # Handle legacy alias
        if normalized == "oprations":
            normalized = "operations"

        if normalized not in cls.VALID_DOMAINS:
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


class RefreshTokenRequest(BaseModel):
    """Request payload for POST /auth/refresh.

    `refresh_token` is optional: browser clients leave it out and let the
    httpOnly `colt_refresh` cookie carry the credential, while non-browser
    callers (mobile, scripts, Swagger) pass it explicitly in the body.
    """

    refresh_token: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Refresh token from POST /auth/token. Omit it to use the "
            "httpOnly colt_refresh cookie instead."
        ),
    )


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
