"""API request schemas."""

import base64
import binascii
from typing import ClassVar
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from src.config.domain_prompts import GENERIC_DOMAIN
from src.config.domain_prompts import SUPPORTED_DOMAINS
from src.config.domain_prompts import normalize_domain_key


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

    source_language: str = Field(
        "auto",
        description="Source language (full name like 'English', code like 'en', or 'auto')",
    )
    target_language: str = Field(
        ..., min_length=2, description="Target language (full name or code)"
    )
    domain: str = Field(
        ...,
        min_length=2,
        max_length=20,
        description=(
            "Translation domain (selects the domain-specific prompt profile): "
            + ", ".join(sorted(SUPPORTED_DOMAINS))
        ),
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

    @field_validator("source_language", "target_language")
    @classmethod
    def validate_language_codes(cls, v: str) -> str:
        """Normalize language input — accepts full names or short codes."""
        return v.strip()


class ProcessingOptions(BaseModel):
    """Processing options for translation job."""

    enable_dlp: bool = Field(
        True, description="Enable Data Loss Prevention scanning (always applied)"
    )
    enable_chunking: bool = Field(
        True, description="Enable chunked processing for large documents"
    )
    priority: Literal["standard", "high"] = Field(
        "standard", description="Processing priority"
    )


class TranslateRequest(BaseModel):
    """Request model for document translation."""

    document: DocumentInput
    translation_config: TranslationConfigInput
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
