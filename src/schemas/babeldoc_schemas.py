"""Pydantic schemas for BabelDOC translation service."""

import json
import uuid
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

# Load default configuration from config.json
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.json"


def _load_default_config() -> dict:
    """Load default configuration from config.json."""
    try:
        with _DEFAULT_CONFIG_PATH.open() as f:
            config = json.load(f)
            return config.get("babeldoc", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_DEFAULTS = _load_default_config()


class TranslationModelSchema(BaseModel):
    """Schema for translation model configuration."""

    model_name: str = Field(
        default=_DEFAULTS.get("openai-model", "gpt-4o-mini"),
        description="Name of the translation model to use",
    )
    api_key: str | None = Field(
        default=_DEFAULTS.get("openai-api-key"),
        description="API key for the translation service",
    )
    qps: int = Field(
        default=_DEFAULTS.get("qps", 4),
        ge=1,
        description="Queries per second rate limit",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for failed requests",
    )


class TranslationConfigSchema(BaseModel):
    """Schema for translation job configuration."""

    job_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the translation job",
    )
    input_file: Path = Field(
        ...,
        description="Path to the input PDF file",
    )
    output_dir: Path = Field(
        default=Path(_DEFAULTS.get("output", "output")),
        description="Directory for output files",
    )
    lang_in: str = Field(
        default=_DEFAULTS.get("lang-in", "en"),
        description="Source language code",
    )
    lang_out: str = Field(
        default=_DEFAULTS.get("lang-out", "zh"),
        description="Target language code",
    )
    debug: bool = Field(
        default=_DEFAULTS.get("debug", False),
        description="Enable debug mode",
    )
    pages: str | None = Field(
        default=None,
        description="Page ranges to translate (e.g., '1-5,7,10-')",
    )
    no_dual: bool = Field(
        default=_DEFAULTS.get("no-dual", False),
        description="Disable dual-language PDF generation",
    )
    no_mono: bool = Field(
        default=_DEFAULTS.get("no-mono", False),
        description="Disable monolingual PDF generation",
    )
    watermark_output_mode: str = Field(
        default=_DEFAULTS.get("watermark-output-mode", "watermarked"),
        description="Watermark output mode: watermarked, no_watermark, or both",
    )
    auto_extract_glossary: bool = Field(
        default=_DEFAULTS.get("auto-extract-glossary", True),
        description="Automatically extract glossary terms",
    )
    min_text_length: int = Field(
        default=_DEFAULTS.get("min-text-length", 5),
        ge=1,
        description="Minimum text length to translate",
    )
    split_short_lines: bool = Field(
        default=_DEFAULTS.get("split-short-lines", False),
        description="Enable short line splitting",
    )
    short_line_split_factor: float = Field(
        default=_DEFAULTS.get("short-line-split-factor", 0.8),
        ge=0.0,
        le=1.0,
        description="Factor for short line splitting",
    )
    skip_clean: bool = Field(
        default=_DEFAULTS.get("skip-clean", False),
        description="Skip PDF cleaning step",
    )
    dual_translate_first: bool = Field(
        default=_DEFAULTS.get("dual-translate-first", False),
        description="Translate dual-language first",
    )
    disable_rich_text_translate: bool = Field(
        default=_DEFAULTS.get("disable-rich-text-translate", False),
        description="Disable rich text translation",
    )
    enhance_compatibility: bool = Field(
        default=_DEFAULTS.get("enhance-compatibility", False),
        description="Enhance PDF compatibility",
    )
    use_alternating_pages_dual: bool = Field(
        default=_DEFAULTS.get("use-alternating-pages-dual", False),
        description="Use alternating pages for dual PDF",
    )
    translate_table_text: bool = Field(
        default=_DEFAULTS.get("translate-table-text", False),
        description="Translate text in tables",
    )
    show_char_box: bool = Field(
        default=_DEFAULTS.get("show-char-box", False),
        description="Show character boxes in output",
    )
    skip_scanned_detection: bool = Field(
        default=_DEFAULTS.get("skip-scanned-detection", False),
        description="Skip scanned document detection",
    )
    ocr_workaround: bool = Field(
        default=_DEFAULTS.get("ocr-workaround", False),
        description="Enable OCR workaround",
    )
    add_formula_placehold_hint: bool = Field(
        default=_DEFAULTS.get("add-formula-placehold-hint", False),
        description="Add formula placeholder hints",
    )
    disable_same_text_fallback: bool = Field(
        default=_DEFAULTS.get("disable-same-text-fallback", False),
        description="Disable same text fallback",
    )
    auto_enable_ocr_workaround: bool = Field(
        default=_DEFAULTS.get("auto-enable-ocr-workaround", False),
        description="Auto-enable OCR workaround when needed",
    )
    only_include_translated_page: bool = Field(
        default=_DEFAULTS.get("only-include-translated-page", False),
        description="Only include translated pages in output",
    )
    save_auto_extracted_glossary: bool = Field(
        default=_DEFAULTS.get("save-auto-extracted-glossary", True),
        description="Save auto-extracted glossary to file",
    )
    disable_graphic_element_process: bool = Field(
        default=_DEFAULTS.get("disable-graphic-element-process", False),
        description="Disable graphic element processing",
    )
    merge_alternating_line_numbers: bool = Field(
        default=_DEFAULTS.get("merge-alternating-line-numbers", True),
        description="Merge alternating line numbers",
    )
    skip_translation: bool = Field(
        default=_DEFAULTS.get("skip-translation", False),
        description="Skip translation (for testing)",
    )
    skip_form_render: bool = Field(
        default=_DEFAULTS.get("skip-form-render", False),
        description="Skip form rendering",
    )
    skip_curve_render: bool = Field(
        default=_DEFAULTS.get("skip-curve-render", False),
        description="Skip curve rendering",
    )
    only_parse_generate_pdf: bool = Field(
        default=_DEFAULTS.get("only-parse-generate-pdf", False),
        description="Only parse and generate PDF without translation",
    )
    remove_non_formula_lines: bool = Field(
        default=_DEFAULTS.get("remove-non-formula-lines", False),
        description="Remove non-formula lines",
    )
    non_formula_line_iou_threshold: float = Field(
        default=_DEFAULTS.get("non-formula-line-iou-threshold", 0.9),
        ge=0.0,
        le=1.0,
        description="IoU threshold for non-formula line detection",
    )
    figure_table_protection_threshold: float = Field(
        default=_DEFAULTS.get("figure-table-protection-threshold", 0.9),
        ge=0.0,
        le=1.0,
        description="Threshold for figure/table protection",
    )
    skip_formula_offset_calculation: bool = Field(
        default=_DEFAULTS.get("skip-formula-offset-calculation", False),
        description="Skip formula offset calculation",
    )
    enable_json_mode_if_requested: bool = Field(
        default=_DEFAULTS.get("enable-json-mode-if-requested", False),
        description="Enable JSON mode if requested",
    )
    send_dashscope_header: bool = Field(
        default=_DEFAULTS.get("send-dashscope-header", False),
        description="Send DashScope header",
    )
    no_send_temperature: bool = Field(
        default=_DEFAULTS.get("no-send-temperature", False),
        description="Do not send temperature parameter",
    )
    report_interval: float = Field(
        default=_DEFAULTS.get("report-interval", 0.1),
        ge=0.0,
        description="Progress report interval in seconds",
    )
    custom_system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for translation",
    )
    metadata_extra_data: str | None = Field(
        default=None,
        description="Extra metadata to add to output PDF",
    )

    @field_validator("input_file", "output_dir", mode="before")
    @classmethod
    def validate_paths(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("watermark_output_mode")
    @classmethod
    def validate_watermark_mode(cls, v):
        """Validate watermark output mode."""
        valid_modes = ["watermarked", "no_watermark", "both"]
        if v not in valid_modes:
            raise ValueError(
                f"Invalid watermark_output_mode. Must be one of: {valid_modes}"
            )
        return v


class TranslationJobStatus(BaseModel):
    """Schema for translation job status."""

    job_id: uuid.UUID
    status: str = Field(
        ...,
        description="Job status: pending, running, completed, failed",
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Progress as a percentage (0.0 to 1.0)",
    )
    message: str | None = Field(
        default=None,
        description="Status message or error description",
    )
    result: dict | None = Field(
        default=None,
        description="Translation result data if completed",
    )
