"""Job processing service for the worker."""

import json
import re
import uuid
from collections import Counter
from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
from langdetect import DetectorFactory
from langdetect import LangDetectException
from langdetect import detect_langs

from babeldoc import async_translate
from babeldoc.docvision.doclayout import OnnxModel
from babeldoc.format.pdf.split_manager import StructureAwareSplitStrategy
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import TranslationCoverPageMetadata
from babeldoc.pdfminer.high_level import extract_pages
from babeldoc.pdfminer.layout import LTTextContainer
from babeldoc.translator.factory import create_translator
from babeldoc.translator.factory import create_translator_from_model_list
from config.constants import settings
from config.logging import logger
from loaders.assets import get_doclayout_onnx_model_path
from worker.models.task_models import BabelDOCTranslationConfig
from worker.services.progress import ProgressTracker
from worker.services.quality_judge import GoogleADKJudgeAgent
from worker.services.quality_judge import QualityJudgeResult
from worker.services.quality_judge import extract_attempt_text

DetectorFactory.seed = 0


class JobProcessor:
    """Processes translation jobs using BabelDOC with progress tracking."""

    # Progress mapping constants
    PROGRESS_START = 0.2
    PROGRESS_TRANSLATION_START = 0.3
    PROGRESS_TRANSLATION_RANGE = 0.6  # 0.2 to 0.8
    PROGRESS_FINALIZE = 0.8
    PROGRESS_COMPLETE = 0.9
    AUTO_LANGUAGE = "auto"
    MAX_LANGUAGES_PER_PAGE = 2
    MIN_DETECTION_TEXT_LENGTH = 20
    MIN_DETECTION_ALPHA_CHARS = 5
    MIN_DETECTION_CONFIDENCE = 0.80
    DETECTED_LANGUAGE_ALIASES = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "iw": "he",
    }

    def __init__(self, progress_tracker: ProgressTracker):
        """Initialize processor with progress tracker."""
        self.progress_tracker = progress_tracker
        self._doc_layout_model: OnnxModel | None = None

    def _get_doc_layout_model(self) -> OnnxModel:
        """Lazy load doc layout model via loaders.assets."""
        if self._doc_layout_model is None:
            model_path = get_doclayout_onnx_model_path()
            self._doc_layout_model = OnnxModel(str(model_path))
            logger.info(f"Loaded DocLayout model from {model_path}")
        return self._doc_layout_model

    async def translate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Process a translation job with iterative model fallback."""
        output_base_dir = Path(config["output_dir"])
        output_base_dir.mkdir(parents=True, exist_ok=True)
        model_list = config.get("model_list", [])
        if not model_list:
            raise ValueError("model_list is required for translation")

        max_attempts = min(
            int(config.get("max_model_attempts", settings.MAX_MODEL_ATTEMPTS)),
            len(model_list),
        )
        judge = GoogleADKJudgeAgent(config.get("judge_model"))
        best_attempt_result: dict[str, Any] | None = None
        best_attempt_score = -1.0
        best_attempt_config: dict[str, Any] | None = None
        best_translation_config: TranslationConfig | None = None
        best_quality_result: QualityJudgeResult | None = None
        attempt_reports: list[dict[str, Any]] = []

        for model_index in range(max_attempts):
            attempt_index = model_index + 1
            selected_model = model_list[model_index]
            attempt_output_dir = output_base_dir / f"iter_{attempt_index}"
            attempt_output_dir.mkdir(parents=True, exist_ok=True)
            attempt_config = {
                **config,
                "selected_model": selected_model,
                "attempt_index": attempt_index,
            }
            translation_config = self._build_translation_config(
                attempt_config, attempt_output_dir
            )

            await self.progress_tracker.update(
                self.PROGRESS_START,
                f"Attempt {attempt_index}/{max_attempts}: model={selected_model}",
            )
            logger.info(
                "Attempt %s starting with model=%s for %s -> %s",
                attempt_index,
                selected_model,
                config["lang_in"],
                config["lang_out"],
            )

            try:
                attempt_result = await self._run_single_attempt(
                    translation_config, attempt_config
                )
            except Exception:
                logger.exception(
                    "Attempt %s failed with model %s", attempt_index, selected_model
                )
                if attempt_index == max_attempts:
                    raise
                continue

            source_text, translated_text = extract_attempt_text(
                Path(str(translation_config.working_dir))
            )
            quality_result = self._evaluate_attempt_quality(
                judge=judge,
                source_text=source_text,
                translated_text=translated_text,
            )
            token_usage = self._collect_token_usage(translation_config, selected_model)
            attempt_report = {
                "attempt_index": attempt_index,
                "model_id": selected_model,
                "quality": quality_result.to_dict(),
                "token_usage": token_usage,
                "working_dir": str(translation_config.working_dir),
                "output_dir": str(attempt_output_dir),
            }
            attempt_reports.append(attempt_report)
            self._write_quality_report(
                Path(str(translation_config.working_dir)), attempt_report
            )

            final_score = quality_result.final_score
            if final_score > best_attempt_score:
                best_attempt_score = final_score
                best_attempt_result = {
                    **attempt_result,
                    "attempt_index": attempt_index,
                    "model_id": selected_model,
                    "quality_report": quality_result.to_dict(),
                    "token_usage": token_usage,
                }
                best_attempt_config = dict(attempt_config)
                best_translation_config = translation_config
                best_quality_result = quality_result

            if quality_result.pass_fail:
                break

        if best_attempt_result is None:
            raise RuntimeError("All translation attempts failed")
        if best_translation_config and best_attempt_config and best_quality_result:
            cover_page_metadata = self._build_cover_page_metadata(
                best_translation_config,
                best_attempt_config,
                best_quality_result,
            )
            self._apply_cover_pages(
                best_translation_config,
                best_attempt_result,
                cover_page_metadata,
            )
        best_attempt_result["attempts"] = attempt_reports
        return best_attempt_result

    async def _run_single_attempt(
        self, translation_config: TranslationConfig, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one translation attempt and return finish payload."""
        async for event in async_translate(translation_config):
            result = await self._handle_translation_event(event, config)
            if result is not None:
                return result
        raise RuntimeError("Translation completed without finish event")

    def _evaluate_attempt_quality(
        self,
        *,
        judge: GoogleADKJudgeAgent,
        source_text: str,
        translated_text: str,
    ) -> QualityJudgeResult:
        """Evaluate one attempt using judge agent."""
        if not source_text.strip() or not translated_text.strip():
            return QualityJudgeResult(
                alignment_score=0.0,
                omission_score=0.0,
                hallucination_score=0.0,
                final_score=0.0,
                pass_fail=False,
                reasons=["Missing source/translated text for quality evaluation."],
                model=judge.model,
            )
        return judge.evaluate(source_text=source_text, translated_text=translated_text)

    def _collect_token_usage(
        self, translation_config: TranslationConfig, selected_model: str
    ) -> dict[str, Any]:
        """Collect token usage and cost estimates for one attempt."""
        translator = translation_config.translator
        prompt_tokens = self._counter_value(
            getattr(translator, "prompt_token_count", 0)
        )
        completion_tokens = self._counter_value(
            getattr(translator, "completion_token_count", 0)
        )
        total_tokens = self._counter_value(getattr(translator, "token_count", 0))
        cache_hit_tokens = self._counter_value(
            getattr(translator, "cache_hit_prompt_token_count", 0)
        )
        term_usage = dict(translation_config.term_extraction_token_usage)

        estimated_cost = self._estimate_cost(
            model_id=selected_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return {
            "model_id": selected_model,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit_prompt_tokens": cache_hit_tokens,
            "term_extraction_usage": term_usage,
            "estimated_cost_usd": estimated_cost,
        }

    def _estimate_cost(
        self, *, model_id: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate cost using configured per-1k token rates."""
        normalized = model_id.lower()
        if normalized.startswith("gemini"):
            input_rate = settings.GEMINI_INPUT_COST_PER_1K
            output_rate = settings.GEMINI_OUTPUT_COST_PER_1K
        else:
            input_rate = settings.OPENAI_INPUT_COST_PER_1K
            output_rate = settings.OPENAI_OUTPUT_COST_PER_1K

        return round(
            (prompt_tokens / 1000.0) * float(input_rate)
            + (completion_tokens / 1000.0) * float(output_rate),
            6,
        )

    def _write_quality_report(
        self, working_dir: Path, attempt_report: dict[str, Any]
    ) -> None:
        """Write per-attempt quality report for auditability."""
        try:
            quality_path = working_dir / "quality_report.json"
            quality_path.write_text(
                json.dumps(attempt_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to write quality report in %s", working_dir)

    def _build_cover_page_metadata(
        self,
        translation_config: TranslationConfig,
        config: dict[str, Any],
        quality_result: QualityJudgeResult,
    ) -> TranslationCoverPageMetadata:
        total_pages = self._get_total_pdf_pages(translation_config.input_file)
        translated_sections = translation_config.get_translated_sections_summary(
            total_pages
        )
        return TranslationCoverPageMetadata(
            original_language=translation_config.lang_in,
            target_language=translation_config.lang_out,
            model_used=str(config.get("selected_model") or "Unknown"),
            domain=str(config.get("domain") or "N/A"),
            translation_date=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            confidence_score=quality_result.final_score,
            translated_sections=translated_sections,
            judge_model=quality_result.model,
        )

    def _get_total_pdf_pages(self, input_file: str | Path) -> int:
        try:
            with pymupdf.open(str(input_file)) as doc:
                return int(doc.page_count)
        except Exception:
            logger.warning("Unable to determine page count for %s", input_file)
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
            logger.exception("Failed to prepend cover page to %s", pdf_path)
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
            title_rect,
            "AI Translated Document",
            fontsize=24,
            fontname="helv",
            fontfile=None,
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
                margin,
                y_position,
                margin + label_width,
                y_position + row_height,
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
            page.insert_textbox(
                value_rect,
                value,
                fontsize=12,
                fontname="helv",
            )
            y_position += row_height

    def _counter_value(self, value: Any) -> int:
        """Extract integer counter value from AtomicInteger or plain int."""
        if hasattr(value, "value"):
            return int(value.value)
        return int(value or 0)

    def detect_source_language(self, input_file: str | Path) -> str:
        """Detect the dominant source language from PDF page text."""
        document_languages: Counter[str] = Counter()

        for page_number, page in enumerate(extract_pages(str(input_file)), start=1):
            page_languages = self._detect_page_languages(page)
            if len(page_languages) > self.MAX_LANGUAGES_PER_PAGE:
                detected = ", ".join(sorted(page_languages))
                raise ValueError(
                    f"Detected more than 2 languages on page {page_number}: {detected}"
                )
            document_languages.update(page_languages)

        if not document_languages:
            raise ValueError("Unable to detect source language from PDF text")

        detected_language, _ = document_languages.most_common(1)[0]
        logger.info(
            "Detected source language %s from %s", detected_language, input_file
        )
        return detected_language

    def _detect_page_languages(self, page: Any) -> Counter[str]:
        """Detect languages for one page based on text containers."""
        page_languages: Counter[str] = Counter()
        for chunk in self._iter_page_text_chunks(page):
            detected_language = self._detect_language_for_text(chunk)
            if detected_language is None:
                continue
            page_languages[detected_language] += len(chunk)
        return page_languages

    def _iter_page_text_chunks(self, item: Any) -> Iterable[str]:
        """Yield cleaned text chunks suitable for language detection."""
        if isinstance(item, LTTextContainer):
            text = self._normalize_detection_text(item.get_text())
            if self._is_detectable_text(text):
                yield text
            return

        if not hasattr(item, "__iter__"):
            return

        for child in item:
            yield from self._iter_page_text_chunks(child)

    def _normalize_detection_text(self, text: str) -> str:
        """Normalize PDF text for language detection."""
        return re.sub(r"\s+", " ", text).strip()

    def _is_detectable_text(self, text: str) -> bool:
        """Skip short or non-linguistic chunks that confuse langdetect."""
        alpha_count = sum(1 for ch in text if ch.isalpha())
        return (
            len(text) >= self.MIN_DETECTION_TEXT_LENGTH
            and alpha_count >= self.MIN_DETECTION_ALPHA_CHARS
        )

    def _detect_language_for_text(self, text: str) -> str | None:
        """Detect one language for a text chunk."""
        try:
            candidates = detect_langs(text)
        except LangDetectException:
            return None

        if not candidates:
            return None

        best_match = candidates[0]
        if best_match.prob < self.MIN_DETECTION_CONFIDENCE:
            return None

        return self._normalize_detected_language(best_match.lang)

    def _normalize_detected_language(self, language: str) -> str:
        """Normalize langdetect codes into worker-friendly source codes."""
        normalized = str(language).strip().lower()
        return self.DETECTED_LANGUAGE_ALIASES.get(normalized, normalized)

    def _build_translation_config(
        self, config: dict[str, Any], output_dir: Path
    ) -> TranslationConfig:
        """Build TranslationConfig from job config."""
        doc_layout_model = self._get_doc_layout_model()

        # Create working directory: CACHE_FOLDER/job_id/working/iter_N
        job_id = config.get("job_id", str(uuid.uuid4()))
        attempt_index = int(config.get("attempt_index", 1))
        cache_folder = settings.CACHE_FOLDER or Path.cwd()
        working_dir = cache_folder / job_id / "working" / f"iter_{attempt_index}"
        working_dir.mkdir(parents=True, exist_ok=True)

        base_config = BabelDOCTranslationConfig.model_validate(
            {
                "input_file": Path(config["input_file"]),
                "output_dir": output_dir,
                "lang_in": config["lang_in"],
                "lang_out": config["lang_out"],
                "model_list": config.get("model_list", []),
                "working_dir": working_dir,
            }
        )
        merged_input = {**config.get("options", {}), **config}
        updates: dict[str, Any] = {}
        for field_name, field in BabelDOCTranslationConfig.model_fields.items():
            if field_name in {
                "input_file",
                "output_dir",
                "lang_in",
                "lang_out",
                "working_dir",
                "model_list",
            }:
                continue
            if field_name not in merged_input:
                continue
            value = merged_input[field_name]
            if value is None:
                continue
            default = field.default
            if default is not None and value == default:
                continue
            updates[field_name] = value

        resolved_config = base_config.model_copy(update=updates)
        selected_model = str(config.get("selected_model", "")).strip()
        if selected_model:
            translator = create_translator(
                selected_model,
                lang_in=resolved_config.lang_in,
                lang_out=resolved_config.lang_out,
                qps=resolved_config.qps,
            )
        else:
            translator = create_translator_from_model_list(
                resolved_config.model_list,
                lang_in=resolved_config.lang_in,
                lang_out=resolved_config.lang_out,
                qps=resolved_config.qps,
            )
        extra_params = resolved_config.to_babeldoc_kwargs()
        for key in (
            "input_file",
            "output_dir",
            "lang_in",
            "lang_out",
            "working_dir",
            "model_list",
            "doc_layout_model_path",
            "table_model_path",
        ):
            extra_params.pop(key, None)

        # glossaries in the config dict are already Glossary objects (loaded by the
        # handler); pop the raw field so it doesn't conflict with the typed parameter.
        glossaries = config.get("glossaries") or extra_params.pop("glossaries", None)

        return TranslationConfig(
            translator=translator,
            term_extraction_translator=translator,
            input_file=Path(str(resolved_config.input_file)),
            output_dir=output_dir,
            lang_in=resolved_config.lang_in,
            lang_out=resolved_config.lang_out,
            doc_layout_model=doc_layout_model,
            table_model=None,
            working_dir=working_dir,
            glossaries=glossaries,
            split_strategy=StructureAwareSplitStrategy(
                min_pages_to_split=10,
                overlap_pages=2,
            ),
            **extra_params,
        )

    async def _handle_translation_event(
        self, event: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle a single translation event. Returns result dict if finished, None otherwise."""
        event_type = event.get("type")

        if event_type == "progress_start":
            await self.progress_tracker.update(
                self.PROGRESS_TRANSLATION_START, "Starting translation"
            )

        elif event_type == "progress_update":
            await self._handle_progress_update(event, config)

        elif event_type == "progress_end":
            await self.progress_tracker.update(
                self.PROGRESS_FINALIZE, "Finalizing output"
            )

        elif event_type == "finish":
            return await self._handle_finish_event(event)

        elif event_type == "error":
            error_msg = event.get("error", "Unknown error")
            logger.error(f"Translation error: {error_msg}")
            raise RuntimeError(f"Translation failed: {error_msg}")

        return None

    async def _handle_progress_update(
        self, event: dict[str, Any], config: dict[str, Any]
    ) -> None:
        """Handle progress update event."""
        # Map 0-100 progress to our range
        overall_progress = event.get("overall_progress", 0) / 100.0
        mapped_progress = self.PROGRESS_START + (
            overall_progress * self.PROGRESS_TRANSLATION_RANGE
        )

        stage = event.get("stage", "Processing")
        await self.progress_tracker.update(
            mapped_progress, f"{stage} ({event.get('overall_progress', 0):.0f}%)"
        )

    async def _handle_finish_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Handle finish event and return results."""
        await self.progress_tracker.update(
            self.PROGRESS_COMPLETE, "Translation complete"
        )

        result = event.get("translate_result")

        # Extract output file paths
        output_files = {}
        for file_type in ["mono_pdf", "dual_pdf", "no_watermark_mono_pdf"]:
            attr_name = f"{file_type}_path"
            if isinstance(result, dict):
                file_path = result.get(attr_name) or result.get(file_type)
            else:
                file_path = (
                    getattr(result, attr_name, None) if result is not None else None
                )
            if file_path and Path(file_path).exists():
                output_files[f"{file_type}_path"] = Path(file_path)

        if isinstance(result, dict):
            page_count = result.get("page_count", 0)
        else:
            page_count = getattr(result, "page_count", 0) or 0
        logger.info(
            f"Translation finished: {page_count} pages, {len(output_files)} output files"
        )

        return {
            **output_files,
            "page_count": page_count,
        }
