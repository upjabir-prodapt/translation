"""Translation processing service used by the worker's translation pipeline."""

import logging
import re
import threading
import uuid
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymupdf
from langdetect import DetectorFactory
from opentelemetry.trace import SpanKind

from src.config.constants import settings
from src.config.tracing import tracer_pipeline
from src.config.translation_routing import ModelRoute
from src.repository.translation_storage_repository import (
    get_translation_storage_repository,
)
from src.worker.doctranslator import async_translate
from src.worker.doctranslator.docvision.doclayout import OnnxModel
from src.worker.doctranslator.format.pdf.split_manager import (
    StructureAwareSplitStrategy,
)
from src.worker.doctranslator.format.pdf.translation_config import DlpConfig
from src.worker.doctranslator.format.pdf.translation_config import (
    SharedContextCrossSplitPart,
)
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig
from src.worker.doctranslator.format.pdf.translation_config import (
    TranslationCoverPageMetadata,
)
from src.worker.doctranslator.format.pdf.translation_config import WatermarkOutputMode
from src.worker.doctranslator.glossary import Glossary
from src.worker.doctranslator.translator.factory import create_translator
from src.worker.loaders.assets import get_doclayout_onnx_model_path
from src.worker.services.language_detection_core import aggregate_languages
from src.worker.services.language_detection_core import detect_language_for_text
from src.worker.services.language_detection_core import is_detectable_text
from src.worker.services.language_detection_core import normalize_detected_language
from src.worker.services.model_attempt_orchestrator import ModelAttemptOrchestrator
from src.worker.services.task_models import DocTranslatorTranslationConfig

logger = logging.getLogger(__name__)

DetectorFactory.seed = 0

# Process-wide singleton for the DocLayout ONNX model. Previously each
# JobProcessor instance (created fresh per job in PipelineOrchestrator)
# loaded its own onnxruntime.InferenceSession, meaning N concurrent jobs in
# one process re-read the model file from disk and duplicated the loaded
# session in memory N times. OnnxModel already guards inference calls with
# an internal lock, so sharing one instance across concurrent jobs/threads
# is safe.
_doc_layout_model_lock = threading.Lock()
_doc_layout_model_singleton: OnnxModel | None = None


def _get_shared_doc_layout_model() -> OnnxModel:
    """Return the process-wide DocLayout ONNX model, loading it once."""
    global _doc_layout_model_singleton
    if _doc_layout_model_singleton is None:
        with _doc_layout_model_lock:
            if _doc_layout_model_singleton is None:
                model_path = get_doclayout_onnx_model_path()
                _doc_layout_model_singleton = OnnxModel(str(model_path))
                logger.info(
                    f"Loaded DocLayout model from {model_path} (process-wide singleton)"
                )
    return _doc_layout_model_singleton


def _job_runtime_root(job_id: str) -> Path:
    """Per-job directory under TEMP_DIR/jobs (matches TempWorkspaceService layout)."""
    base = settings.temp_root_path
    return base / settings.TEMP_JOBS_ROOT / str(job_id)


class JobProcessor:
    """Processes translation jobs using DocTranslator with progress tracking."""

    PROGRESS_START = 0.2
    PROGRESS_TRANSLATION_START = 0.3
    PROGRESS_TRANSLATION_RANGE = 0.6
    PROGRESS_FINALIZE = 0.8
    PROGRESS_COMPLETE = 0.9

    # Renamed from MAX_LANGUAGES_PER_PAGE (implementation_plan.md Phase
    # C.3): the guard message used to say "more than 2 languages" while
    # this constant was actually 10 -- a real message/constant
    # contradiction. The name now matches what it actually is, and the
    # message below renders the live value instead of a hardcoded "2".
    MAX_DISTINCT_LANGUAGES_PER_PAGE = 10
    MAX_DETECTION_CHARS = settings.LANGUAGE_DETECTION_MAX_CHARS
    MIN_DETECTION_TEXT_LENGTH = 20
    MIN_DETECTION_ALPHA_CHARS = 5
    MIN_DETECTION_CONFIDENCE = 0.80
    DETECTED_LANGUAGE_ALIASES = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "iw": "he",
    }

    def __init__(self, progress_tracker: Any):
        self.progress_tracker = progress_tracker
        self._current_attempt_chunks: int = 0
        self._model_orchestrator = ModelAttemptOrchestrator(self)

    def _get_doc_layout_model(self) -> OnnxModel:
        return _get_shared_doc_layout_model()

    async def _load_glossary_from_gcs(
        self, job_id: str, glossary_filename: str, lang_out: str
    ) -> list[Glossary]:
        storage = get_translation_storage_repository()
        glossary_dir = _job_runtime_root(job_id) / "glossary"
        glossary_dir.mkdir(parents=True, exist_ok=True)
        local_path = glossary_dir / glossary_filename

        try:
            await storage.download_glossary(job_id, local_path, glossary_filename)
            glossary = Glossary.from_csv(local_path, lang_out)
            return [glossary]
        except Exception:
            logger.exception(
                f"Failed to load glossary '{glossary_filename}' from GCS for job {job_id}",
                glossary_filename,
                job_id,
            )
            return []

    async def translate(self, config: dict[str, Any]) -> dict[str, Any]:
        glossary_filename = config.get("glossary_filename")
        if glossary_filename and not config.get("glossaries"):
            config = dict(config)
            config["glossaries"] = await self._load_glossary_from_gcs(
                job_id=str(config.get("job_id", "")),
                glossary_filename=glossary_filename,
                lang_out=str(config.get("lang_out", "")),
            )

        return await self._model_orchestrator.run_model_chain(config)

    async def _run_single_attempt(
        self, translation_config: TranslationConfig, config: dict[str, Any]
    ) -> dict[str, Any]:
        self._current_attempt_chunks = 0
        async for event in async_translate(translation_config):
            result = await self._handle_translation_event(event, config)
            if result is not None:
                # Carry the real split forward so per-chunk cost attribution
                # matches the parts that actually ran.
                result["split_page_ranges"] = list(
                    getattr(
                        translation_config.shared_context_cross_split_part,
                        "split_page_ranges",
                        [],
                    )
                    or []
                )
                return result
        raise RuntimeError("Translation completed without finish event")

    def _get_total_pdf_pages(self, input_file: str | Path) -> int:
        try:
            with pymupdf.open(str(input_file)) as doc:
                return int(doc.page_count)
        except Exception:
            logger.warning(f"Unable to determine page count for {input_file}")
            return 0

    def _apply_cover_pages(
        self,
        translation_config: TranslationConfig,
        attempt_result: dict[str, Any],
        metadata: TranslationCoverPageMetadata,
    ) -> None:
        if not getattr(translation_config, "add_cover_page", True):
            return

        translation_config.cover_page_metadata = metadata
        output_paths = {
            Path(str(path))
            for path in (
                attempt_result.get("mono_pdf_path"),
                attempt_result.get("dual_pdf_path"),
                attempt_result.get("no_watermark_mono_pdf_path"),
                attempt_result.get("no_watermark_dual_pdf_path"),
            )
            if path
        }
        with tracer_pipeline.start_as_current_span(
            "pipeline.pdf_typeset",
            kind=SpanKind.INTERNAL,
            attributes={"pdf.output_count": len(output_paths)},
        ):
            for output_path in output_paths:
                self._prepend_cover_page(output_path, metadata)

    def _prepend_cover_page(
        self, pdf_path: Path, metadata: TranslationCoverPageMetadata
    ) -> None:
        if not pdf_path.exists():
            return
        original_doc: pymupdf.Document | None = None
        new_doc: pymupdf.Document | None = None
        temp_output_path = pdf_path.with_name(f"{pdf_path.stem}.cover{pdf_path.suffix}")
        try:
            original_doc = pymupdf.open(str(pdf_path))
            if original_doc.page_count == 0:
                return
            first_page_rect = original_doc[0].rect
            new_doc = pymupdf.open()
            cover_page = new_doc.new_page(
                width=first_page_rect.width,
                height=first_page_rect.height,
            )
            self._draw_cover_page(cover_page, metadata)
            new_doc.insert_pdf(original_doc)
            new_doc.save(str(temp_output_path), garbage=4, deflate=True)
        except Exception:
            logger.exception(f"Failed to prepend cover page to {pdf_path}")
            return
        finally:
            if original_doc is not None:
                original_doc.close()
            if new_doc is not None:
                new_doc.close()

        temp_output_path.replace(pdf_path)

    def _draw_cover_page(
        self,
        page: pymupdf.Page,
        metadata: TranslationCoverPageMetadata,
    ) -> None:
        page_rect = page.rect
        margin = 54
        usable_width = page_rect.width - (margin * 2)
        title_rect = pymupdf.Rect(margin, 56, page_rect.width - margin, 98)
        subtitle_rect = pymupdf.Rect(margin, 104, page_rect.width - margin, 130)
        divider_y = 145

        page.insert_textbox(
            title_rect, "AI Translated Document", fontsize=24, fontname="helv"
        )
        page.insert_textbox(
            subtitle_rect,
            "This cover page summarizes the generated translation output.",
            fontsize=11,
            fontname="helv",
            color=(0.35, 0.35, 0.35),
        )
        page.draw_line(
            pymupdf.Point(margin, divider_y),
            pymupdf.Point(page_rect.width - margin, divider_y),
            color=(0.75, 0.75, 0.75),
            width=1,
        )

        y_position = 170
        row_height = 28
        label_width = min(170, usable_width * 0.35)
        for label, value in metadata.iter_rows():
            label_rect = pymupdf.Rect(
                margin, y_position, margin + label_width, y_position + row_height
            )
            value_rect = pymupdf.Rect(
                margin + label_width + 10,
                y_position,
                page_rect.width - margin,
                y_position + row_height,
            )
            page.insert_textbox(
                label_rect,
                f"{label}:",
                fontsize=12,
                fontname="helv",
                color=(0.2, 0.2, 0.2),
            )
            page.insert_textbox(value_rect, value, fontsize=12, fontname="helv")
            y_position += row_height

        disclaimer_rect = pymupdf.Rect(
            margin, y_position + 12, page_rect.width - margin, y_position + 60
        )
        page.insert_textbox(
            disclaimer_rect,
            metadata.DISCLAIMER,
            fontsize=10,
            fontname="helv",
            color=(0.55, 0.15, 0.15),
        )

    def _counter_value(self, value: Any) -> int:
        if hasattr(value, "value"):
            return int(value.value)
        return int(value or 0)

    def detect_source_language(self, input_file: str | Path) -> str:
        """Detect the dominant source language of a PDF.

        Kept backward-compatible (returns just the winning language code)
        for existing callers; see `detect_source_language_with_distribution()`
        for the full char-weighted distribution
        (implementation_plan.md Phase C.5.1).
        """
        detected_language, _ = self.detect_source_language_with_distribution(input_file)
        return detected_language

    def detect_source_language_with_distribution(
        self, input_file: str | Path
    ) -> tuple[str, "Counter[str]"]:
        """Detect source language and return the full per-language distribution.

        Returns `(dominant_language, full_language_counter)` -- the full
        char-weighted `Counter` (not just the winner) is returned so
        callers can persist the complete language distribution for a
        mixed-language document (implementation_plan.md Phase C.5.1)
        instead of discarding everything but `most_common(1)`.
        """
        document_languages: Counter[str] = Counter()
        processed_chars = 0
        with pymupdf.open(str(input_file)) as doc:
            for page_number, page in enumerate(doc, start=1):
                page_languages, page_chars = self._detect_page_languages(page)
                if len(page_languages) > self.MAX_DISTINCT_LANGUAGES_PER_PAGE:
                    detected = ", ".join(sorted(page_languages))
                    raise ValueError(
                        f"Detected more than {self.MAX_DISTINCT_LANGUAGES_PER_PAGE} "
                        f"languages on page {page_number}: {detected}"
                    )
                document_languages.update(page_languages)
                processed_chars += page_chars
                if processed_chars >= self.MAX_DETECTION_CHARS:
                    logger.info(
                        "Language detection reached configured character budget (%s) at page %s",
                        self.MAX_DETECTION_CHARS,
                        page_number,
                    )
                    break
        if not document_languages:
            # This branch fires when no page yielded any detectable text
            # -- the same underlying condition PDFValidator's API-side
            # text-layer probe rejects. Use the identical user-facing
            # wording (implementation_plan.md Phase B.3.3) instead of the
            # internal "Unable to detect source language from PDF text",
            # since a scanned/image-only PDF is the far more common cause
            # than a genuinely undetectable (but present) language.
            raise ValueError(
                "This PDF has no extractable text layer (scanned or "
                "image-only). OCR is not supported — please supply a "
                "text-based PDF."
            )
        detected_language = aggregate_languages(document_languages)
        logger.info(f"Detected source language {detected_language} from {input_file}")
        return detected_language, document_languages

    def _detect_page_languages(self, page: Any) -> tuple[Counter[str], int]:
        page_languages: Counter[str] = Counter()
        page_chars = 0
        for chunk in self._iter_page_text_chunks(page):
            page_chars += len(chunk)
            detected_language = self._detect_language_for_text(chunk)
            if detected_language is None:
                continue
            page_languages[detected_language] += len(chunk)
        return page_languages, page_chars

    def _iter_page_text_chunks(self, page: Any) -> Iterator[str]:
        # Use PyMuPDF page blocks for robust text extraction without pdfminer internals.
        for block in page.get_text("blocks") or []:
            if len(block) < 5:
                continue
            text = self._normalize_detection_text(str(block[4]))
            if self._is_detectable_text(text):
                yield text

    def _normalize_detection_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _is_detectable_text(self, text: str) -> bool:
        # Delegates to the shared core (implementation_plan.md Phase C.1)
        # so PDF and DOCX/TXT detection use identical thresholds; kept as
        # a thin method (not removed) since it is part of this class's
        # existing public-ish surface (tests, LanguageDetectionService).
        return is_detectable_text(text)

    def _detect_language_for_text(self, text: str) -> str | None:
        return detect_language_for_text(text)

    def _normalize_detected_language(self, language: str) -> str:
        return normalize_detected_language(language)

    def _build_translation_config(
        self,
        config: dict[str, Any],
        output_dir: Path,
        shared_context: SharedContextCrossSplitPart | None = None,
    ) -> TranslationConfig:
        doc_layout_model = self._get_doc_layout_model()
        job_id = config.get("job_id", str(uuid.uuid4()))
        attempt_index = int(config.get("attempt_index", 1))
        working_dir = (
            _job_runtime_root(str(job_id)) / "working" / f"iter_{attempt_index}"
        )
        working_dir.mkdir(parents=True, exist_ok=True)

        raw_model_list = config.get("model_list", [])
        model_ids = [
            item.model_id if isinstance(item, ModelRoute) else str(item)
            for item in raw_model_list
        ]
        base_config = DocTranslatorTranslationConfig.model_validate(
            {
                "input_file": Path(config["input_file"]),
                "output_dir": output_dir,
                "lang_in": config["lang_in"],
                "lang_out": config["lang_out"],
                "model_list": model_ids,
                "working_dir": working_dir,
                "add_cover_page": bool(config.get("add_cover_page", True)),
                "no_dual": bool(config.get("no_dual", True)),
                "no_mono": bool(config.get("no_mono", False)),
            }
        )
        selected_model = str(config.get("selected_model", "")).strip()
        domain = config.get("domain")
        # "selected_model_region" is threaded in by TranslationAttemptRunner
        # from the matching ModelRoute (docs/plan.md Section 3.3) so
        # gemini-3.5-flash's europe-west3 pinning survives the trip from
        # model_selection.json through to the Vertex AI client construction.
        selected_region = config.get("selected_model_region")
        translator = create_translator(
            selected_model or base_config.model_list[0],
            lang_in=base_config.lang_in,
            lang_out=base_config.lang_out,
            qps=base_config.qps,
            region=selected_region,
            domain=domain,
        )
        glossaries = config.get("glossaries")

        watermark_mode = WatermarkOutputMode.NoWatermark
        if bool(config.get("pdf_watermark", False)):
            watermark_mode = WatermarkOutputMode.Watermarked

        return TranslationConfig(
            translator=translator,
            term_extraction_translator=translator,
            input_file=Path(str(base_config.input_file)),
            output_dir=output_dir,
            lang_in=base_config.lang_in,
            lang_out=base_config.lang_out,
            domain=domain,
            doc_layout_model=doc_layout_model,
            table_model=None,
            working_dir=working_dir,
            glossaries=glossaries,
            split_strategy=StructureAwareSplitStrategy(
                min_pages_to_split=10,
                overlap_pages=2,
            ),
            qps=base_config.qps,
            pool_max_workers=settings.TRANSLATION_POOL_MAX_WORKERS,
            term_pool_max_workers=settings.TERM_EXTRACTION_POOL_MAX_WORKERS,
            min_text_length=settings.LLM_TRANSLATION_MIN_TEXT_LENGTH,
            disable_same_text_fallback=settings.LLM_DISABLE_SAME_TEXT_FALLBACK,
            no_dual=base_config.no_dual,
            no_mono=base_config.no_mono,
            add_cover_page=base_config.add_cover_page,
            watermark_output_mode=watermark_mode,
            dlp_config=DlpConfig(
                enable_dlp=bool(
                    config.get(
                        "enable_dlp", getattr(settings, "GOOGLE_DLP_ENABLED", True)
                    )
                ),
                dlp_job_id=str(config.get("job_id", "")).strip() or None,
                dlp_source_language=str(config.get("lang_in", "")).strip() or None,
                dlp_post_translation=bool(config.get("dlp_post_translation", False)),
            ),
            shared_context_cross_split_part=shared_context,
        )

    async def _handle_translation_event(
        self, event: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any] | None:
        del config
        event_type = event.get("type")
        if event_type == "progress_start":
            await self.progress_tracker.update(
                self.PROGRESS_TRANSLATION_START, "Starting translation"
            )
        elif event_type == "progress_update":
            await self._handle_progress_update(event)
        elif event_type == "progress_end":
            await self.progress_tracker.update(
                self.PROGRESS_FINALIZE, "Finalizing output"
            )
        elif event_type == "finish":
            return await self._handle_finish_event(event)
        elif event_type == "error":
            # `ProgressMonitor.translate_error()` (progress_monitor.py)
            # passes the *original exception object* through this event,
            # not a string. Re-raise it as-is instead of always wrapping
            # in a generic RuntimeError, so document-shape errors like
            # ScannedPDFError survive with their real type -- required
            # for TranslationAttemptRunner's non-retryable-exception check
            # (implementation_plan.md Phase B.3.1) to actually see them.
            error = event.get("error")
            if isinstance(error, BaseException):
                raise error
            raise RuntimeError(f"Translation failed: {error or 'Unknown error'}")
        return None

    async def _handle_progress_update(self, event: dict[str, Any]) -> None:
        overall_progress = event.get("overall_progress", 0) / 100.0
        mapped_progress = self.PROGRESS_START + (
            overall_progress * self.PROGRESS_TRANSLATION_RANGE
        )
        stage = event.get("stage", "Processing")
        self._current_attempt_chunks = max(
            self._current_attempt_chunks, int(event.get("stage_current", 0))
        )
        await self.progress_tracker.update(
            mapped_progress, f"{stage} ({event.get('overall_progress', 0):.0f}%)"
        )

    async def _handle_finish_event(self, event: dict[str, Any]) -> dict[str, Any]:
        await self.progress_tracker.update(
            self.PROGRESS_COMPLETE, "Translation complete"
        )
        result = event.get("translate_result")
        output_files = {}
        for file_type in [
            "mono_pdf",
            "dual_pdf",
            "no_watermark_mono_pdf",
            "no_watermark_dual_pdf",
        ]:
            attr_name = f"{file_type}_path"
            if isinstance(result, dict):
                file_path = result.get(attr_name) or result.get(file_type)
            else:
                file_path = (
                    getattr(result, attr_name, None) if result is not None else None
                )
            if file_path and Path(file_path).exists():
                output_files[f"{file_type}_path"] = Path(file_path)
        page_count = result.get("page_count", 0) if isinstance(result, dict) else 0
        return {
            **output_files,
            "page_count": page_count,
            "chunks_processed": self._current_attempt_chunks,
        }
