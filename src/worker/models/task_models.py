"""Pydantic models for worker tasks."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class TaskStatus(StrEnum):
    """Task status enum."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TranslationTaskConfig(BaseModel):
    """Translation task configuration."""

    lang_in: str = Field(..., description="Source language code")
    lang_out: str = Field(..., description="Target language code")
    domain: str | None = Field(None, description="Domain/industry context")
    options: dict[str, Any] = Field(
        default_factory=dict, description="Additional translation options"
    )

    class Config:
        frozen = True


class TranslationTask(BaseModel):
    """Translation task data from Cloud Tasks."""

    job_id: str = Field(..., description="Unique job identifier")
    config: TranslationTaskConfig = Field(..., description="Translation configuration")

    class Config:
        frozen = True


class TranslationResult(BaseModel):
    """Translation processing result."""

    success: bool = Field(..., description="Whether translation succeeded")
    error: str | None = Field(None, description="Error message if failed")
    page_count: int | None = Field(None, description="Number of pages processed")
    output_files: dict[str, str] = Field(
        default_factory=dict, description="Output file paths"
    )

    class Config:
        frozen = True
