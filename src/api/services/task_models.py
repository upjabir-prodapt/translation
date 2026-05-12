"""Pydantic models for translation processing options."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from src.config.constants import settings


class DocTranslatorTranslationConfig(BaseModel):
    """Subset of DocTranslator TranslationConfig parameters used by API pipeline."""

    input_file: str | Path = Field(..., description="Input PDF file path")
    lang_in: str = Field(..., description="Source language code")
    lang_out: str = Field(..., description="Target language code")
    model_list: list[str] = Field(default_factory=list, min_length=1)
    output_dir: str | Path | None = None
    working_dir: str | Path | None = None
    qps: int = Field(default_factory=lambda: settings.TRANSLATION_MAX_QPS)
    no_dual: bool = True
    no_mono: bool = False
    add_cover_page: bool = True
    glossaries: list[dict[str, Any]] | None = None

    @field_validator("lang_in", "lang_out")
    @classmethod
    def validate_normalized_language_codes(cls, value: str) -> str:
        code = value.strip().lower()
        if not code.isalpha() or len(code) not in {2, 3, 4, 5}:
            raise ValueError("Language code must be normalized alphabetic code")
        return code

    @field_validator("model_list")
    @classmethod
    def validate_model_list(cls, value: list[str]) -> list[str]:
        normalized = [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
        if not normalized:
            raise ValueError("model_list must contain at least one model")
        return normalized

    def to_doctranslator_kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    model_config = ConfigDict(frozen=True)
