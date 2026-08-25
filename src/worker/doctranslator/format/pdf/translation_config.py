from __future__ import annotations

import enum
import logging
import shutil
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.constants import settings
from src.config.translation_routing import get_language_display_name
from src.worker.doctranslator.format.pdf.split_manager import BaseSplitStrategy
from src.worker.doctranslator.format.pdf.split_manager import PageCountStrategy
from src.worker.doctranslator.glossary import Glossary
from src.worker.doctranslator.glossary import GlossaryEntry
from src.worker.doctranslator.progress_monitor import ProgressMonitor
from src.worker.doctranslator.translator.translator import BaseTranslator

logger = logging.getLogger(__name__)


def _is_cjk_language_code(lang: str | None) -> bool:
    if not lang:
        return False
    normalized = str(lang).strip().lower()
    return normalized.startswith(("zh", "ja", "ko"))


def get_token_multiplier(lang_in: str, lang_out: str) -> float:
    if _is_cjk_language_code(lang_in) or _is_cjk_language_code(lang_out):
        return float(settings.LLM_TOKEN_MULTIPLIER_CJK)
    return float(settings.LLM_TOKEN_MULTIPLIER_DEFAULT)


class WatermarkOutputMode(enum.Enum):
    Watermarked = "watermarked"
    NoWatermark = "no_watermark"
    Both = "both"


class SharedContextCrossSplitPart:
    def __init__(self):
        self.first_paragraph = None
        self.recent_title_paragraph = None
        self._lock = threading.Lock()
        self.user_glossaries: list[Glossary] = []
        self.auto_extracted_glossary: Glossary | None = None
        self.raw_extracted_terms: list[tuple[str, str]] = []
        self.auto_enabled_ocr_workaround = False
        # Statistics for valid characters/text across the whole file
        self.valid_char_count_total: int = 0
        self.total_valid_text_token_count: int = 0
        self._cached_il_docs: dict[int, Any] = {}

    def get_cached_il_doc(self, part_index: int = 0) -> Any | None:
        """Retrieve cached parsed IL document for a part, if previously stored."""
        with self._lock:
            return self._cached_il_docs.get(part_index)

    def set_cached_il_doc(self, part_index: int, doc: Any) -> None:
        """Store a parsed IL document snapshot for cross-attempt reuse."""
        with self._lock:
            self._cached_il_docs[part_index] = doc

    def has_cached_il_docs(self) -> bool:
        """Check whether any parsed IL documents have been cached."""
        with self._lock:
            return bool(self._cached_il_docs)

    def initialize_glossaries(self, initial_glossaries: list[Glossary] | None):
        with self._lock:
            self.user_glossaries = (
                list(initial_glossaries) if initial_glossaries else []
            )
            self.auto_extracted_glossary = None
            self.raw_extracted_terms = []
            self.unique_name = self._generate_unique_auto_glossary_name()
            self.norm_terms = set()
            for g in self.user_glossaries:
                for entity in g.normalized_lookup:
                    self.norm_terms.add(entity)
            # reset statistics buffer when initializing
            self.valid_char_count_total = 0
            self.total_valid_text_token_count = 0

    def add_raw_extracted_term_pair(self, src: str, tgt: str):
        with self._lock:
            self.raw_extracted_terms.append((src, tgt))

    def _generate_unique_auto_glossary_name(self) -> str:
        base_name = "auto_extracted_glossary"
        current_name = base_name
        suffix = 0
        existing_names = {g.name for g in self.user_glossaries}

        while current_name in existing_names:
            suffix += 1
            current_name = f"{base_name}#{suffix}"
        return current_name

    def contains_term(self, term: str) -> bool:
        with self._lock:
            try:
                return term in self.norm_terms
            except Exception:  # noqa: BLE001 - best-effort lookup; any error is non-fatal
                return False

    def finalize_auto_extracted_glossary(self):
        with self._lock:
            self.auto_extracted_glossary = None

            if not self.raw_extracted_terms:
                self.raw_extracted_terms = []
                return

            term_translations: dict[str, list[str]] = {}
            for src, tgt in self.raw_extracted_terms:
                term_translations.setdefault(src, []).append(tgt)

            final_entries: list[GlossaryEntry] = []
            for src, tgts in term_translations.items():
                if not tgts:
                    continue
                most_common_tgt = Counter(tgts).most_common(1)[0][0]
                final_entries.append(GlossaryEntry(src, most_common_tgt))

            if final_entries:
                self.auto_extracted_glossary = Glossary(
                    name=self.unique_name, entries=final_entries
                )

    def get_glossaries(self) -> list[Glossary]:
        with self._lock:
            all_glossaries = list(self.user_glossaries)
            if self.auto_extracted_glossary:
                all_glossaries.append(self.auto_extracted_glossary)
            return all_glossaries

    def get_glossaries_for_translation(
        self, auto_extract_enabled: bool
    ) -> list[Glossary]:
        with self._lock:
            if auto_extract_enabled and self.auto_extracted_glossary:
                return [self.auto_extracted_glossary]
            else:
                all_glossaries = list(self.user_glossaries)
                if self.auto_extracted_glossary:
                    all_glossaries.append(self.auto_extracted_glossary)
                return all_glossaries

    def add_valid_counts(self, char_count: int, token_count: int):
        """Accumulate valid character and token counts in a threadsafe way."""
        if char_count <= 0 and token_count <= 0:
            return
        with self._lock:
            if char_count > 0:
                self.valid_char_count_total += char_count
            if token_count > 0:
                self.total_valid_text_token_count += token_count


@dataclass
class DlpConfig:
    enable_dlp: bool = True
    dlp_job_id: str | None = None
    dlp_source_language: str | None = None
    dlp_post_translation: bool = False


@dataclass(slots=True)
class TranslationCoverPageMetadata:
    original_language: str
    target_language: str
    model_used: str
    domain: str
    translation_date: str
    confidence_score: float | None
    translated_sections: str
    judge_model: str | None = None

    #: Shown on the cover page of every AI-translated output so a human
    #: reviewer is always warned before relying on the translation.
    DISCLAIMER = (
        "This is an AI generated translation that may have mistakes and "
        "needs to be reviewed by a human native language speaker."
    )

    def iter_rows(self) -> list[tuple[str, str]]:
        rows = [
            (
                "Original language",
                get_language_display_name(self.original_language),
            ),
            (
                "Target language",
                get_language_display_name(self.target_language),
            ),
            ("Model used", self.model_used or "N/A"),
            ("Domain", self.domain or "N/A"),
            ("Translation date", self.translation_date or "N/A"),
        ]
        if self.confidence_score is not None:
            confidence = (
                f"{self.confidence_score:.2f} ({self.confidence_score * 100:.1f}%)"
            )
            if self.judge_model:
                confidence = f"{confidence} via {self.judge_model}"
            rows.append(("Confidence score", confidence))

        rows.append(("Sections translated", self.translated_sections or "N/A"))
        return rows


class TranslationConfig:
    @staticmethod
    def create_max_pages_per_part_split_strategy(max_pages_per_part: int):
        return PageCountStrategy(max_pages_per_part)

    def _init_working_dir(
        self, working_dir: str | Path | None, debug: bool, input_file: str | Path
    ) -> Path:
        """Create and return the working directory, setting _is_temp_dir."""
        if working_dir is None:
            if debug:
                base_temp_dir = settings.temp_root_path
                working_dir = base_temp_dir / "working" / Path(input_file).stem
                self._is_temp_dir = False
            else:
                working_dir = tempfile.mkdtemp()
                self._is_temp_dir = True
        else:
            self._is_temp_dir = False
        return working_dir

    def _init_pool_settings(
        self, pool_max_workers: int | None, term_pool_max_workers: int | None
    ) -> None:
        """Configure pool sizes from explicit args or settings defaults."""
        self.pool_max_workers = (
            pool_max_workers
            if pool_max_workers is not None
            else int(settings.TRANSLATION_POOL_MAX_WORKERS)
        )
        self.term_pool_max_workers = (
            term_pool_max_workers
            if term_pool_max_workers is not None
            else int(settings.TERM_EXTRACTION_POOL_MAX_WORKERS)
        )

    def _init_ocr_flags(
        self,
        ocr_workaround: bool,
        enhance_compatibility: bool,
        auto_enable_ocr_workaround: bool,
        use_side_by_side_dual: bool,
        use_alternating_pages_dual: bool,
    ) -> None:
        """Set OCR / compatibility flag overrides in correct priority order."""
        self.skip_clean = self.skip_clean or enhance_compatibility
        self.dual_translate_first = self.dual_translate_first or enhance_compatibility
        self.disable_rich_text_translate = (
            self.disable_rich_text_translate or enhance_compatibility
        )

        if ocr_workaround:
            self.skip_scanned_detection = True
            self.disable_rich_text_translate = True

        # backward-compat: if side-by-side explicitly disabled, use alternating
        if use_side_by_side_dual is False and use_alternating_pages_dual is False:
            self.use_alternating_pages_dual = True

        if auto_enable_ocr_workaround:
            self.ocr_workaround = False
            self.skip_scanned_detection = False

        if self.ocr_workaround:
            self.remove_non_formula_lines = False

    def _init_dirs(
        self,
        working_dir: str | Path | None,
        output_dir: str | Path | None,
        debug: bool,
        input_file: str | Path,
    ) -> None:
        """Resolve, create, and store working and output directories."""
        working_dir = self._init_working_dir(working_dir, debug, input_file)
        self.working_dir = working_dir
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        if output_dir is None:
            output_dir = Path.cwd()
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _init_feature_flags(
        self,
        auto_extract_glossary: bool,
        skip_translation: bool,
        only_parse_generate_pdf: bool,
        primary_font_family: str | None,
        only_include_translated_page: bool | None,
        save_auto_extracted_glossary: bool,
        enable_graphic_element_process: bool,
        skip_form_render: bool,
        skip_curve_render: bool,
        non_formula_line_iou_threshold: float,
        figure_table_protection_threshold: float,
        skip_formula_offset_calculation: bool,
    ) -> None:
        """Set miscellaneous feature flags."""
        self.auto_extract_glossary = auto_extract_glossary
        self.skip_translation = skip_translation
        self.only_parse_generate_pdf = only_parse_generate_pdf
        if self.skip_translation or self.only_parse_generate_pdf:
            self.auto_extract_glossary = False
        if primary_font_family not in [None, "serif", "sans-serif", "script"]:
            raise ValueError(
                f"primary_font_family must be one of None, 'serif', 'sans-serif', 'script'; got {primary_font_family!r}"
            )
        self.primary_font_family = primary_font_family
        self.only_include_translated_page = (
            bool(only_include_translated_page)
            if only_include_translated_page
            else False
        )
        self.save_auto_extracted_glossary = save_auto_extracted_glossary
        self.enable_graphic_element_process = enable_graphic_element_process
        self.skip_form_render = skip_form_render
        self.skip_curve_render = skip_curve_render
        self.non_formula_line_iou_threshold = non_formula_line_iou_threshold
        self.figure_table_protection_threshold = figure_table_protection_threshold
        self.skip_formula_offset_calculation = skip_formula_offset_calculation

    def _init_dlp_fields(self, dlp_config: DlpConfig | None) -> None:
        """Initialise DLP-related instance fields."""
        cfg = dlp_config or DlpConfig()
        self.enable_dlp = bool(cfg.enable_dlp)
        self.dlp_job_id = cfg.dlp_job_id
        self.dlp_source_language = cfg.dlp_source_language
        self.dlp_post_translation = bool(cfg.dlp_post_translation)
        self.dlp_provider: str | None = None
        self.dlp_chunk_mode: str | None = None
        self.dlp_token_rows: list[dict] = []
        self.dlp_applied_pre_translation = False
        self.dlp_token_counter = 0
        self.dlp_chunk_count = 0
        self.dlp_unmask_before_pdf = True

    def _init_llm_batch_limits(
        self, lang_in: str, lang_out: str, disable_same_text_fallback: bool
    ) -> None:
        """Calculate and store LLM batch size limits."""
        self.disable_same_text_fallback = disable_same_text_fallback
        token_multiplier = max(get_token_multiplier(lang_in, lang_out), 0.1)
        self.llm_translation_batch_max_tokens = int(
            settings.LLM_TRANSLATION_BATCH_MAX_TOKENS
        )
        self.llm_translation_batch_max_paragraphs = max(
            int(settings.LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS * token_multiplier), 1
        )
        self.llm_term_extraction_batch_max_tokens = int(
            settings.LLM_TERM_EXTRACTION_BATCH_MAX_TOKENS
        )
        self.llm_term_extraction_batch_max_paragraphs = max(
            int(settings.LLM_TERM_EXTRACTION_BATCH_MAX_PARAGRAPHS * token_multiplier), 1
        )

    def __init__(  # NOSONAR - public configuration object intentionally exposes many backward-compatible keyword parameters
        self,
        translator: BaseTranslator,
        input_file: str | Path,
        lang_in: str,
        lang_out: str,
        doc_layout_model,  # DocLayoutModel
        # for backward compatibility
        font: str | Path | None = None,
        pages: str | None = None,
        output_dir: str | Path | None = None,
        debug: bool = False,
        working_dir: str | Path | None = None,
        no_dual: bool = False,
        no_mono: bool = False,
        formular_font_pattern: str | None = None,
        formular_char_pattern: str | None = None,
        qps: int = settings.TRANSLATION_MAX_QPS,
        split_short_lines: bool = False,
        short_line_split_factor: float = 0.8,
        use_rich_pbar: bool = True,
        progress_monitor: ProgressMonitor | None = None,
        skip_clean: bool = False,
        dual_translate_first: bool = False,
        disable_rich_text_translate: bool = False,
        enhance_compatibility: bool = False,
        report_interval: float = 0.1,
        min_text_length: int = settings.LLM_TRANSLATION_MIN_TEXT_LENGTH,
        use_side_by_side_dual: bool = True,  # Deprecated: 是否使用拼版式双语 PDF（并排显示原文和译文）向下兼容选项，已停用。
        use_alternating_pages_dual: bool = False,
        watermark_output_mode: WatermarkOutputMode = WatermarkOutputMode.Watermarked,
        # Add split-related parameters
        split_strategy: BaseSplitStrategy | None = None,
        table_model=None,
        show_char_box: bool = False,
        skip_scanned_detection: bool = False,
        ocr_workaround: bool = False,
        custom_system_prompt: str | None = None,
        domain: str | None = None,
        add_formula_placehold_hint: bool = False,
        glossaries: list[Glossary] | None = None,
        pool_max_workers: int | None = None,
        auto_extract_glossary: bool = True,
        auto_enable_ocr_workaround: bool = False,
        primary_font_family: str | None = None,
        only_include_translated_page: bool | None = False,
        save_auto_extracted_glossary: bool = True,
        enable_graphic_element_process: bool = True,
        merge_alternating_line_numbers: bool = True,
        skip_translation: bool = False,
        skip_form_render: bool = False,
        skip_curve_render: bool = False,
        only_parse_generate_pdf: bool = False,
        remove_non_formula_lines: bool = False,
        non_formula_line_iou_threshold: float = 0.9,
        figure_table_protection_threshold: float = 0.9,
        skip_formula_offset_calculation: bool = False,
        term_extraction_translator: BaseTranslator | None = None,
        metadata_extra_data: str | None = None,
        term_pool_max_workers: int | None = None,
        disable_same_text_fallback: bool = settings.LLM_DISABLE_SAME_TEXT_FALLBACK,
        add_cover_page: bool = True,
        cover_page_metadata: TranslationCoverPageMetadata | None = None,
        dlp_config: DlpConfig
        | None = None,  # NOSONAR - public configuration object; param count cannot be reduced below 13 without breaking callers
        shared_context_cross_split_part: SharedContextCrossSplitPart | None = None,
    ):
        self.translator = translator
        self.term_extraction_translator = term_extraction_translator or translator
        initial_user_glossaries = list(glossaries) if glossaries else []

        self.input_file = input_file
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.font = None  # just ignore font

        self.pages = pages
        self.page_ranges = self.parse_pages(pages) if pages else None
        self.debug = debug
        self.watermark_output_mode = watermark_output_mode

        self.no_dual = no_dual
        self.no_mono = no_mono
        self.formular_font_pattern = formular_font_pattern
        self.formular_char_pattern = formular_char_pattern
        self.qps = qps
        self._init_pool_settings(pool_max_workers, term_pool_max_workers)
        self.split_short_lines = split_short_lines
        self.short_line_split_factor = short_line_split_factor
        self.use_rich_pbar = use_rich_pbar
        self.progress_monitor = progress_monitor
        self.doc_layout_model = doc_layout_model

        self.skip_clean = skip_clean
        self.skip_scanned_detection = skip_scanned_detection
        self.dual_translate_first = dual_translate_first
        self.disable_rich_text_translate = disable_rich_text_translate
        self.report_interval = report_interval
        self.min_text_length = min_text_length
        self.use_alternating_pages_dual = use_alternating_pages_dual
        self.ocr_workaround = ocr_workaround
        self.merge_alternating_line_numbers = merge_alternating_line_numbers
        self.remove_non_formula_lines = remove_non_formula_lines
        self._init_ocr_flags(
            ocr_workaround,
            enhance_compatibility,
            auto_enable_ocr_workaround,
            use_side_by_side_dual,
            use_alternating_pages_dual,
        )

        if progress_monitor and progress_monitor.cancel_event is None:
            progress_monitor.cancel_event = threading.Event()

        self._init_dirs(working_dir, output_dir, debug, input_file)

        if not doc_layout_model:
            raise ValueError(
                "doc_layout_model is required. "
                "Load model via loaders.assets and pass to TranslationConfig."
            )
        self.doc_layout_model = doc_layout_model

        if shared_context_cross_split_part is not None:
            self.shared_context_cross_split_part = shared_context_cross_split_part
        else:
            self.shared_context_cross_split_part = SharedContextCrossSplitPart()
            self.shared_context_cross_split_part.initialize_glossaries(
                initial_user_glossaries
            )
        self.split_part_index = 0

        self.split_strategy = split_strategy
        self._part_working_dirs: dict[int, Path] = {}
        self._part_output_dirs: dict[int, Path] = {}

        self.table_model = table_model
        self.show_char_box = show_char_box
        self.custom_system_prompt = custom_system_prompt
        self.domain = domain
        self.add_formula_placehold_hint = add_formula_placehold_hint
        self.auto_enable_ocr_workaround = auto_enable_ocr_workaround
        self._init_feature_flags(
            auto_extract_glossary,
            skip_translation,
            only_parse_generate_pdf,
            primary_font_family,
            only_include_translated_page,
            save_auto_extracted_glossary,
            enable_graphic_element_process,
            skip_form_render,
            skip_curve_render,
            non_formula_line_iou_threshold,
            figure_table_protection_threshold,
            skip_formula_offset_calculation,
        )

        # force disable table translate until the new model is ready
        self.table_model = None
        self.metadata_extra_data = metadata_extra_data
        self.add_cover_page = add_cover_page
        self.cover_page_metadata = cover_page_metadata
        self._init_dlp_fields(dlp_config)

        self.term_extraction_token_usage: dict[str, int] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hit_prompt_tokens": 0,
        }
        self._init_llm_batch_limits(lang_in, lang_out, disable_same_text_fallback)

    def _normalize_page_range(
        self, start: int, end: int, total_pages: int
    ) -> tuple[int, int]:
        """Clamp a (start, end) page range to valid page numbers within total_pages.

        Returns (normalized_start, normalized_end); caller should skip if end < start.
        """
        normalized_start = max(int(start), 1)
        normalized_end = total_pages if int(end) == -1 else min(int(end), total_pages)
        return normalized_start, normalized_end

    def get_translated_page_numbers(self, total_pages: int) -> list[int]:
        if total_pages <= 0:
            return []
        if not self.page_ranges:
            return list(range(1, total_pages + 1))

        translated_pages: set[int] = set()
        for start, end in self.page_ranges:
            normalized_start, normalized_end = self._normalize_page_range(
                start, end, total_pages
            )
            if normalized_end < normalized_start:
                continue
            translated_pages.update(range(normalized_start, normalized_end + 1))
        return sorted(translated_pages)

    def get_translated_sections_summary(self, total_pages: int) -> str:
        if total_pages <= 0:
            return "Unknown"
        translated_pages = self.get_translated_page_numbers(total_pages)
        return f"{len(translated_pages)}/{total_pages} pages"

    def parse_pages(self, pages_str: str | None) -> list[tuple[int, int]] | None:
        """解析页码字符串，返回页码范围列表

        Args:
            pages_str: 形如 "1-,2,-3,4" 的页码字符串

        Returns:
            包含 (start, end) 元组的列表，其中 -1 表示无限制
        """
        if not pages_str:
            return None

        ranges: list[tuple[int, int]] = []
        for part in pages_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                start_as_int = int(start) if start else 1
                end_as_int = int(end) if end else -1
                ranges.append((start_as_int, end_as_int))
            else:
                page = int(part)
                ranges.append((page, page))
        return ranges

    def should_translate_page(self, page_number: int) -> bool:
        """判断指定页码是否需要翻译
        Args:
            page_number: 页码
        Returns:
            是否需要翻译该页
        """
        if isinstance(self.page_ranges, list) and len(self.page_ranges) == 0:
            return False
        if not self.page_ranges:
            return True

        for start, end in self.page_ranges:
            if start <= page_number and (end == -1 or page_number <= end):
                return True
        return False

    def get_output_file_path(self, filename: str) -> Path:
        return Path(self.output_dir) / filename

    def get_working_file_path(self, filename: str) -> Path:
        return Path(self.working_dir) / filename

    def get_part_working_dir(self, part_index: int) -> Path:
        """Get working directory for a specific part"""
        if part_index not in self._part_working_dirs:
            if self.working_dir:
                part_dir = Path(self.working_dir) / f"part_{part_index}"
            else:
                part_dir = Path(tempfile.mkdtemp()) / f"part_{part_index}"
            part_dir.mkdir(parents=True, exist_ok=True)
            self._part_working_dirs[part_index] = part_dir
        return self._part_working_dirs[part_index]

    def get_part_output_dir(self, part_index: int) -> Path:
        """Get output directory for a specific part"""
        if part_index not in self._part_output_dirs:
            part_dir = Path(self.working_dir) / f"part_{part_index}_output"
            part_dir.mkdir(parents=True, exist_ok=True)
            self._part_output_dirs[part_index] = part_dir
        return self._part_output_dirs[part_index]

    def cleanup_part_output_dir(self, part_index: int):
        """Clean up output directory for a specific part"""
        if part_index in self._part_output_dirs:
            part_dir = self._part_output_dirs[part_index]
            if part_dir.exists():
                shutil.rmtree(part_dir)
            del self._part_output_dirs[part_index]

    def cleanup_part_working_dir(self, part_index: int):
        """Clean up working directory for a specific part"""
        if part_index in self._part_working_dirs:
            part_dir = self._part_working_dirs[part_index]
            if part_dir.exists():
                shutil.rmtree(part_dir, ignore_errors=True)
            del self._part_working_dirs[part_index]

    def cleanup_temp_files(self):
        """Clean up all temporary files including part working directories"""
        try:
            for part_index in list(self._part_working_dirs.keys()):
                self.cleanup_part_working_dir(part_index)
            if self._is_temp_dir:
                logger.info(f"cleanup temp files: {self.working_dir}")
                shutil.rmtree(self.working_dir, ignore_errors=True)
        except Exception:
            logger.exception("Error cleaning up temporary files")

    def raise_if_cancelled(self):
        if self.progress_monitor is not None:
            self.progress_monitor.raise_if_cancelled()

    def cancel_translation(self):
        if self.progress_monitor is not None:
            self.progress_monitor.cancel()

    def get_term_extraction_translator(self) -> BaseTranslator:
        """Return the translator to use for automatic term extraction."""
        return self.term_extraction_translator

    def record_term_extraction_usage(
        self,
        total_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_prompt_tokens: int,
    ) -> None:
        """Accumulate token usage for automatic term extraction."""
        if total_tokens > 0:
            self.term_extraction_token_usage["total_tokens"] += total_tokens
        if prompt_tokens > 0:
            self.term_extraction_token_usage["prompt_tokens"] += prompt_tokens
        if completion_tokens > 0:
            self.term_extraction_token_usage["completion_tokens"] += completion_tokens
        if cache_hit_prompt_tokens > 0:
            self.term_extraction_token_usage["cache_hit_prompt_tokens"] += (
                cache_hit_prompt_tokens
            )


class TranslateResult:
    original_pdf_path: str
    total_seconds: float
    mono_pdf_path: Path | None
    dual_pdf_path: Path | None
    no_watermark_mono_pdf_path: Path | None
    no_watermark_dual_pdf_path: Path | None
    peak_memory_usage: int | None
    auto_extracted_glossary_path: Path | None
    total_valid_character_count: int | None
    total_valid_text_token_count: int | None

    def __init__(
        self,
        mono_pdf_path: Path | None,
        dual_pdf_path: Path | None,
        auto_extracted_glossary_path: Path | None = None,
    ):
        self.mono_pdf_path = mono_pdf_path
        self.dual_pdf_path = dual_pdf_path

        # For compatibility considerations, if only a non-watermarked PDF is generated,
        # the values of mono_pdf_path and no_watermark_mono_pdf_path are the same.
        self.no_watermark_mono_pdf_path = mono_pdf_path
        self.no_watermark_dual_pdf_path = dual_pdf_path

        self.auto_extracted_glossary_path = auto_extracted_glossary_path
        self.total_valid_character_count = None
        self.total_valid_text_token_count = None

    def _append_path_info(self, result: list):
        """Append path information to result list."""
        if hasattr(self, "original_pdf_path") and self.original_pdf_path:
            result.append(f"\tOriginal PDF: {self.original_pdf_path}")

        if self.mono_pdf_path:
            result.append(f"\tMonolingual PDF: {self.mono_pdf_path}")

        if self.dual_pdf_path:
            result.append(f"\tDual-language PDF: {self.dual_pdf_path}")

        if (
            hasattr(self, "no_watermark_mono_pdf_path")
            and self.no_watermark_mono_pdf_path
            and self.no_watermark_mono_pdf_path != self.mono_pdf_path
        ):
            result.append(
                f"\tNo-watermark Monolingual PDF: {self.no_watermark_mono_pdf_path}"
            )

        if (
            hasattr(self, "no_watermark_dual_pdf_path")
            and self.no_watermark_dual_pdf_path
            and self.no_watermark_dual_pdf_path != self.dual_pdf_path
        ):
            result.append(
                f"\tNo-watermark Dual-language PDF: {self.no_watermark_dual_pdf_path}"
            )

        if (
            hasattr(self, "auto_extracted_glossary_path")
            and self.auto_extracted_glossary_path
        ):
            result.append(
                f"\tAuto-extracted glossary: {self.auto_extracted_glossary_path}"
            )

    def _append_time_and_memory_info(self, result: list):
        """Append timing and memory information to result list."""
        if hasattr(self, "total_seconds") and self.total_seconds:
            result.append(f"\tTotal time: {self.total_seconds:.2f} seconds")

        if hasattr(self, "peak_memory_usage") and self.peak_memory_usage:
            result.append(f"\tPeak memory usage: {self.peak_memory_usage} MB")

    def _append_count_info(self, result: list):
        """Append character and token count information to result list."""
        if hasattr(self, "total_valid_character_count") and isinstance(
            self.total_valid_character_count, int
        ):
            result.append(
                f"\tTotal valid character count: {self.total_valid_character_count}"
            )

        if hasattr(self, "total_valid_text_token_count") and isinstance(
            self.total_valid_text_token_count, int
        ):
            result.append(
                f"\tTotal valid text token count (gpt-4o): {self.total_valid_text_token_count}"
            )

    def __str__(self):
        """Return a human-readable string representation of the translation result."""
        result = []
        self._append_path_info(result)
        self._append_time_and_memory_info(result)
        self._append_count_info(result)

        if result:
            result.insert(0, "Translation results:")

        return "\n".join(result) if result else "No translation results available"
