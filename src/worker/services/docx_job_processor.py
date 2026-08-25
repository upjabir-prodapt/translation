"""DOCX-native translation job processor: model-attempt loop + quality judge.

Lighter-weight counterpart to processor_service.JobProcessor /
model_attempt_orchestrator.ModelAttemptOrchestrator for native DOCX
translation. Much cheaper per-attempt than the PDF flow: no PDF re-parsing,
no ONNX inference, no font subsetting -- each attempt just re-runs
translation batches against the same extracted units.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document

from src.config.constants import settings
from src.config.translation_routing import ModelRoute
from src.worker.doctranslator.format.docx.cover_page import add_cover_page
from src.worker.doctranslator.format.docx.docx_translator import DocxTranslationResult
from src.worker.doctranslator.format.docx.docx_translator import translate_docx
from src.worker.doctranslator.format.pdf.translation_config import (
    TranslationCoverPageMetadata,
)
from src.worker.doctranslator.translator.factory import create_translator
from src.worker.services.dlp_service import DlpResult
from src.worker.services.glossary_service import GlossaryService
from src.worker.services.llm_cost_service import get_vertex_llm_cost_service
from src.worker.services.quality_judge_service import GoogleADKJudgeAgent

logger = logging.getLogger(__name__)


class DocxJobProcessor:
    """Runs the model-attempt loop for native DOCX translation."""

    def __init__(self, glossary_service: GlossaryService | None = None):
        self.glossary_service = glossary_service or GlossaryService()
        self._cost_service = get_vertex_llm_cost_service()

    def _counter_value(self, value: Any) -> int:
        if hasattr(value, "value"):
            return int(value.value)
        return int(value or 0)

    def _collect_token_usage(
        self, translator, selected_model: str, selected_region: str | None = None
    ) -> dict[str, Any]:
        prompt_tokens = self._counter_value(getattr(translator, "prompt_token_count", 0))
        completion_tokens = self._counter_value(
            getattr(translator, "completion_token_count", 0)
        )
        total_tokens = self._counter_value(getattr(translator, "token_count", 0))
        cache_hit_tokens = self._counter_value(
            getattr(translator, "cache_hit_prompt_token_count", 0)
        )
        breakdown = self._cost_service.calculate_attempt_cost(
            model_id=selected_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=cache_hit_tokens,
            region=selected_region,
        )
        return {
            "model_id": selected_model,
            "provider": breakdown.provider,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit_prompt_tokens": cache_hit_tokens,
            "estimated_cost_usd": breakdown.total_cost_usd,
            "cost_breakdown": breakdown.to_dict(),
        }

    async def translate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run the model-attempt loop for one DOCX translation job.

        `config` keys: job_id, input_file, output_dir, lang_in, lang_out,
        domain, model_list, max_model_attempts, enable_dlp,
        auto_extract_glossary.
        """
        job_id = str(config["job_id"])
        input_path = Path(config["input_file"])
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        lang_in = config["lang_in"]
        lang_out = config["lang_out"]
        domain = config.get("domain", "")
        model_list: list[ModelRoute] | list[str] = config.get("model_list", [])
        if not model_list:
            raise ValueError("model_list is required for DOCX translation")
        max_attempts = min(
            int(config.get("max_model_attempts", settings.MAX_MODEL_ATTEMPTS)),
            len(model_list),
        )
        enable_dlp = bool(
            config.get("enable_dlp", getattr(settings, "GOOGLE_DLP_ENABLED", True))
        )
        auto_extract_glossary = bool(config.get("auto_extract_glossary", True))
        enable_judge = bool(
            config.get("enable_judge", getattr(settings, "QUALITY_JUDGE_ENABLED", True))
        )

        judge: GoogleADKJudgeAgent | None = None
        if enable_judge:
            judge = GoogleADKJudgeAgent(
                config.get("judge_model"),
                region=config.get("judge_model_region"),
            )

        best_result: DocxTranslationResult | None = None
        best_score = -1.0
        best_model_id: str | None = None
        best_quality: dict[str, Any] | None = None
        best_token_usage: dict[str, Any] | None = None
        best_attempt_index = 0

        cached_extracted_terms: list[tuple[str, str]] | None = None
        cached_dlp_result: DlpResult | None = None

        for attempt_index in range(1, max_attempts + 1):
            selected_route = model_list[attempt_index - 1]
            if isinstance(selected_route, ModelRoute):
                selected_model = selected_route.model_id
                selected_region = selected_route.region
            else:
                selected_model = str(selected_route)
                selected_region = None
            attempt_output_dir = output_dir / f"iter_{attempt_index}"
            attempt_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = attempt_output_dir / input_path.name

            try:
                translator = create_translator(
                    selected_model,
                    lang_in=lang_in,
                    lang_out=lang_out,
                    qps=settings.TRANSLATION_MAX_QPS,
                    region=selected_region,
                )
                # translate_docx() is fully synchronous and CPU/network bound
                # (python-docx parsing plus many blocking LLM calls). Calling
                # it directly from this coroutine blocked the worker's event
                # loop for the whole job: in the 2026-08-24 baseline a second
                # DOCX job sat unacknowledged for 203s because uvicorn could
                # not service the incoming request. The PDF path already
                # offloads via run_in_executor; this brings DOCX to parity.
                result = await asyncio.to_thread(
                    translate_docx,
                    input_path=input_path,
                    output_path=output_path,
                    translator=translator,
                    lang_out=lang_out,
                    job_id=job_id,
                    source_language=lang_in,
                    enable_dlp=enable_dlp,
                    auto_extract_glossary=auto_extract_glossary,
                    extracted_terms=cached_extracted_terms,
                    dlp_result=cached_dlp_result,
                )
                if cached_extracted_terms is None and result.extracted_terms:
                    cached_extracted_terms = result.extracted_terms
                if cached_dlp_result is None and result.dlp_result is not None:
                    cached_dlp_result = result.dlp_result
            except Exception:
                logger.exception(
                    f"DOCX attempt {attempt_index} failed with model {selected_model}"
                )
                if attempt_index == max_attempts:
                    raise
                continue

            token_usage = self._collect_token_usage(
                translator, selected_model, selected_region
            )

            if not enable_judge or judge is None:
                best_result = result
                best_model_id = selected_model
                best_quality = None
                best_token_usage = token_usage
                best_attempt_index = attempt_index
                break

            if not result.translated_text.strip() or not result.source_text.strip():
                quality_result = None
                final_score = 0.0
            else:
                quality_result = await judge.evaluate_async(
                    source_text=result.source_text,
                    translated_text=result.translated_text,
                )
                final_score = quality_result.final_score

            if final_score > best_score:
                best_score = final_score
                best_result = result
                best_model_id = selected_model
                best_quality = quality_result.to_dict() if quality_result else None
                best_token_usage = token_usage
                best_attempt_index = attempt_index

            if quality_result is not None and quality_result.pass_fail:
                break

            # Mirror ModelAttemptOrchestrator: honour the previously-unused
            # QUALITY_EARLY_ACCEPT_THRESHOLD so a near-miss does not cost a
            # full extra translation pass.
            early_accept = float(settings.QUALITY_EARLY_ACCEPT_THRESHOLD)
            if quality_result is not None and 0 < early_accept <= final_score:
                logger.info(
                    f"Early-accepting DOCX attempt {attempt_index} "
                    f"(score={final_score:.3f} >= "
                    f"QUALITY_EARLY_ACCEPT_THRESHOLD={early_accept}); "
                    "skipping remaining model attempts"
                )
                break

        if best_result is None:
            raise RuntimeError("All DOCX translation attempts failed")

        if config.get("add_cover_page", True):
            self._apply_cover_page(
                best_result=best_result,
                lang_in=lang_in,
                lang_out=lang_out,
                domain=domain,
                model_used=best_model_id,
                best_quality=best_quality,
            )

        # Glossary write-back only happens after the whole job has picked a
        # winning attempt -- callers should invoke this only once the job's
        # BigQuery status is being marked "completed".
        return {
            "output_path": str(best_result.output_path),
            "attempt_index": best_attempt_index,
            "model_id": best_model_id,
            "quality_report": best_quality,
            "token_usage": best_token_usage,
            "dlp_provider": (
                best_result.dlp_provider.value if best_result.dlp_provider else None
            ),
            "dlp_token_rows": best_result.dlp_token_rows,
            "extracted_terms": best_result.extracted_terms,
            "domain": domain,
            "source_language": lang_in,
            "target_language": lang_out,
        }

    def _apply_cover_page(
        self,
        *,
        best_result: DocxTranslationResult,
        lang_in: str,
        lang_out: str,
        domain: str,
        model_used: str | None,
        best_quality: dict[str, Any] | None,
    ) -> None:
        """Prepend an AI-translation cover page to the winning DOCX output.

        Mirrors the PDF pipeline's cover page (see
        `ModelAttemptOrchestrator._build_cover_page_metadata` /
        `JobProcessor._prepend_cover_page`) -- same metadata fields and
        disclaimer wording, just rendered as native DOCX paragraphs instead
        of a redrawn PDF page.
        """
        best_quality = best_quality or {}
        metadata = TranslationCoverPageMetadata(
            original_language=lang_in,
            target_language=lang_out,
            model_used=str(model_used or "Unknown"),
            domain=str(domain or "N/A"),
            translation_date=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            confidence_score=best_quality.get("final_score"),
            translated_sections="All paragraphs",
            judge_model=best_quality.get("model"),
        )
        try:
            document = Document(str(best_result.output_path))
            add_cover_page(document, metadata)
            document.save(str(best_result.output_path))
        except Exception:
            logger.exception(
                f"Failed to prepend cover page to {best_result.output_path}"
            )

    def persist_extracted_terms(self, attempt_result: dict[str, Any]) -> bool:
        """Write auto-extracted terms into the shared domain glossary.

        Must only be called after the job has been marked completed
        successfully -- terms from failed/low-quality jobs must never reach
        the shared glossary (see docs/architecture/pdf-vs-docx-translation-architecture.md).
        """
        terms = attempt_result.get("extracted_terms") or []
        if not terms:
            return False
        domain = attempt_result.get("domain")
        if not domain:
            return False
        return self.glossary_service.merge_new_terms_into_domain_glossary(
            domain=domain,
            source_language=attempt_result.get("source_language", ""),
            target_language_name=attempt_result.get("target_language", ""),
            new_terms=terms,
        )
