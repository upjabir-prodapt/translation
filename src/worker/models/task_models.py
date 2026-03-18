"""Pydantic models for worker tasks."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class WatermarkOutputMode(StrEnum):
    """Watermark output mode for BabelDOC."""

    WATERMARKED = "watermarked"
    NO_WATERMARK = "no_watermark"
    BOTH = "both"


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


class BabelDOCTranslationConfig(BaseModel):
    """BabelDOC TranslationConfig parameters.

    Maps to babeldoc.format.pdf.translation_config.TranslationConfig
    """

    # Required
    input_file: str | Path = Field(..., description="Input PDF file path")
    lang_in: str = Field(..., description="Source language code")
    lang_out: str = Field(..., description="Target language code")
    model_list: list[str] = Field(
        default_factory=list,
        min_length=1,
        description="Ordered model candidates for translator selection",
    )

    # Output
    output_dir: str | Path | None = Field(None, description="Output directory")
    working_dir: str | Path | None = Field(None, description="Working directory")
    no_dual: bool = Field(False, description="Skip dual PDF generation")
    no_mono: bool = Field(False, description="Skip mono PDF generation")
    use_alternating_pages_dual: bool = Field(
        False, description="Use alternating pages for dual PDF"
    )

    # Page control
    pages: str | None = Field(
        None, description="Page range to process (e.g., '1-5,7,9-12')"
    )

    # Font
    font: str | Path | None = Field(None, description="Custom font path")
    primary_font_family: str | None = Field(None, description="Primary font family")
    formular_font_pattern: str | None = Field(
        None, description="Formula font pattern regex"
    )
    formular_char_pattern: str | None = Field(
        None, description="Formula character pattern regex"
    )

    # Translation control
    qps: int = Field(4, description="Queries per second limit")
    split_short_lines: bool = Field(False, description="Enable short line splitting")
    short_line_split_factor: float = Field(0.8, description="Short line split factor")
    skip_clean: bool = Field(False, description="Skip cleaning step")
    dual_translate_first: bool = Field(False, description="Translate dual first")
    disable_rich_text_translate: bool = Field(
        False, description="Disable rich text translation"
    )
    enhance_compatibility: bool = Field(False, description="Enhance PDF compatibility")
    min_text_length: int = Field(5, description="Minimum text length to translate")
    custom_system_prompt: str | None = Field(None, description="Custom system prompt")
    disable_same_text_fallback: bool = Field(
        False, description="Disable same text fallback"
    )
    skip_translation: bool = Field(False, description="Skip translation (parse only)")
    add_cover_page: bool = Field(
        True, description="Prepend a translation summary cover page"
    )

    # OCR and detection
    skip_scanned_detection: bool = Field(
        False, description="Skip scanned page detection"
    )
    ocr_workaround: bool = Field(False, description="Enable OCR workaround")
    auto_enable_ocr_workaround: bool = Field(
        False, description="Auto-enable OCR workaround"
    )

    # Visual elements
    show_char_box: bool = Field(False, description="Show character boxes (debug)")
    enable_graphic_element_process: bool = Field(
        True, description="Process graphic elements"
    )
    skip_form_render: bool = Field(False, description="Skip form rendering")
    skip_curve_render: bool = Field(False, description="Skip curve rendering")
    only_parse_generate_pdf: bool = Field(
        False, description="Only parse and generate PDF"
    )

    # Formula handling
    add_formula_placehold_hint: bool = Field(
        False, description="Add formula placeholder hint"
    )
    remove_non_formula_lines: bool = Field(
        False, description="Remove non-formula lines"
    )
    non_formula_line_iou_threshold: float = Field(
        0.9, description="Non-formula line IOU threshold"
    )
    skip_formula_offset_calculation: bool = Field(
        False, description="Skip formula offset calculation"
    )

    # Table and figure
    figure_table_protection_threshold: float = Field(
        0.9, description="Figure/table protection threshold"
    )
    merge_alternating_line_numbers: bool = Field(
        True, description="Merge alternating line numbers"
    )

    # Watermark (always no watermark for worker pipeline)
    watermark_output_mode: WatermarkOutputMode = Field(
        WatermarkOutputMode.NO_WATERMARK, description="Watermark output mode"
    )

    # Glossary
    glossaries: list[dict[str, Any]] | None = Field(
        None, description="Glossary entries"
    )
    auto_extract_glossary: bool = Field(True, description="Auto-extract glossary")
    save_auto_extracted_glossary: bool = Field(
        True, description="Save auto-extracted glossary"
    )

    # Performance
    pool_max_workers: int | None = Field(None, description="Max worker threads")
    term_pool_max_workers: int | None = Field(
        None, description="Term extraction workers"
    )

    # Filtering
    only_include_translated_page: bool = Field(
        False, description="Only include translated pages"
    )

    # Debug
    debug: bool = Field(False, description="Enable debug mode")
    report_interval: float = Field(0.1, description="Progress report interval")
    metadata_extra_data: str | None = Field(None, description="Extra metadata")

    # Asset paths (loaded via loaders.assets, passed to BabelDOC)
    doc_layout_model_path: str | Path | None = Field(
        None, description="Path to DocLayout ONNX model (auto-loaded if not provided)"
    )
    table_model_path: str | Path | None = Field(
        None,
        description="Path to table detection model (future use, currently disabled)",
    )

    @field_validator("lang_in", "lang_out")
    @classmethod
    def validate_normalized_language_codes(cls, value: str) -> str:
        """Ensure language codes are normalized alpha codes."""
        code = value.strip().lower()
        if not code.isalpha() or len(code) not in {2, 3, 4, 5}:
            raise ValueError("Language code must be normalized alphabetic code")
        return code

    @field_validator("model_list")
    @classmethod
    def validate_model_list(cls, value: list[str]) -> list[str]:
        """Normalize and validate model list."""
        normalized = [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
        if not normalized:
            raise ValueError("model_list must contain at least one model")
        return normalized

    def to_babeldoc_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for BabelDOC TranslationConfig."""
        return self.model_dump(exclude_none=True)

    class Config:
        frozen = True


class TranslationTask(BaseModel):
    """Translation task data from Cloud Tasks."""

    job_id: str = Field(..., description="Unique job identifier")
    config: TranslationTaskConfig = Field(..., description="Translation configuration")

    class Config:
        frozen = True
