"""DOCX-native translation job processor: model-attempt loop + quality judge.

Lighter-weight counterpart to processor_service.JobProcessor /
model_attempt_orchestrator.ModelAttemptOrchestrator for native DOCX
translation. Much cheaper per-attempt than the PDF flow: no PDF re-parsing,
no ONNX inference, no font subsetting -- each attempt just re-runs
translation batches against the same extracted units.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document

from src.config.constants import settings
from src.config.translation_routing import ModelRoute
from src.repository.translation_storage_repository import (
    get_translation_storage_repository,
)
from src.worker.doctranslator.format.docx.cover_page import add_cover_page
from src.worker.doctranslator.format.docx.docx_translator import DocxTranslationResult
from src.worker.doctranslator.format.docx.docx_translator import translate_docx
from src.worker.doctranslator.format.pdf.translation_config import (
    TranslationCoverPageMetadata,
)
from src.worker.doctranslator.translator.factory import create_translator
from src.worker.services.attempt_decision import AttemptDecision
from src.worker.services.attempt_decision import decide_after_attempt
from src.worker.services.dlp_service import DlpResult
from src.worker.services.glossary_service import GlossaryService
from src.worker.services.llm_cost_service import get_vertex_llm_cost_service
from src.worker.services.quality_judge_service import GoogleADKJudgeAgent
from src.worker.services.quality_judge_service import QualityJudgeResult
from src.worker.services.token_verification_service import (
    log_token_verification_warning,
)
from src.worker.services.token_verification_service import verify_protected_tokens

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
        # Minority languages detection found. Carried on the translator
        # rather than threaded through translate_docx(): the DOCX prompt
        # builder already reads `domain` off the engine when not given one
        # explicitly, so this follows the same route.
        secondary_languages = [
            (str(code), float(share))
            for code, share in (config.get("secondary_languages") or [])
        ]
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
        attempt_reports: list[dict[str, Any]] = []

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
                    domain=domain,
                    secondary_languages=secondary_languages,
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
                    domain=domain,
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

            # implementation_plan.md D.6.1 (EC-01/02/14): best-effort,
            # non-blocking check that protected tokens (URLs, emails,
            # currency, CIDR blocks, ticket IDs, CLI flags, bare digit
            # sequences) present in the source still appear verbatim in
            # the translation. Logged as a quality warning and recorded
            # in the attempt report; never fails the attempt or the job.
            token_verification = verify_protected_tokens(
                result.source_text, result.translated_text
            )
            log_token_verification_warning(
                token_verification, job_id=job_id, attempt_index=attempt_index
            )

            if not enable_judge or judge is None:
                attempt_reports.append(
                    {
                        "attempt_number": attempt_index,
                        "model_id": selected_model,
                        "alignment_score": None,
                        "omission_score": None,
                        "hallucination_score": None,
                        "final_score": None,
                        "pass_fail": None,
                        "is_fallback": False,
                        "total_tokens": token_usage.get("total_tokens", 0),
                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "completion_tokens": token_usage.get("completion_tokens", 0),
                        "cache_hit_prompt_tokens": token_usage.get(
                            "cache_hit_prompt_tokens", 0
                        ),
                        "cost_usd": token_usage.get("estimated_cost_usd", 0.0),
                        "docx_path": str(output_path),
                        "token_verification": token_verification.to_dict(),
                    }
                )
                best_result = result
                best_model_id = selected_model
                best_quality = None
                best_token_usage = token_usage
                best_attempt_index = attempt_index
                break

            # `result.segments` are the aligned (source, translation) unit
            # pairs, handed over in memory -- DOCX is single-pass in-process,
            # so unlike the PDF path there is no per-part tracking JSON to
            # reconstruct them from and a round trip would be pure overhead.
            if not result.segments:
                # An inconclusive verdict, never a numeric 0.0: the previous
                # `quality_result = None` here collided with the "judge
                # disabled" meaning of None, and because both exit conditions
                # below were guarded on `quality_result is not None`, neither
                # could fire -- so an unjudgeable DOCX ran the entire model
                # chain and then reported no quality at all on the cover page.
                quality_result = QualityJudgeResult(
                    alignment_score=0.0,
                    omission_score=0.0,
                    hallucination_score=0.0,
                    final_score=0.0,
                    pass_fail=False,
                    reasons=["No translated segments were available to judge."],
                    model=judge.model,
                    is_fallback=True,
                    inconclusive=True,
                    coverage_ratio=0.0,
                )
            else:
                quality_result = await judge.evaluate_segments_async(
                    result.segments, job_id=job_id
                )
            final_score = quality_result.final_score

            attempt_reports.append(
                {
                    "attempt_number": attempt_index,
                    "model_id": selected_model,
                    "alignment_score": quality_result.alignment_score,
                    "omission_score": quality_result.omission_score,
                    "hallucination_score": quality_result.hallucination_score,
                    "final_score": quality_result.final_score,
                    "pass_fail": quality_result.pass_fail,
                    "is_fallback": quality_result.is_fallback,
                    "inconclusive": quality_result.inconclusive,
                    "coverage_ratio": quality_result.coverage_ratio,
                    "total_tokens": token_usage.get("total_tokens", 0),
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "cache_hit_prompt_tokens": token_usage.get(
                        "cache_hit_prompt_tokens", 0
                    ),
                    "cost_usd": token_usage.get("estimated_cost_usd", 0.0),
                    "docx_path": str(output_path),
                    "token_verification": token_verification.to_dict(),
                }
            )

            try:
                q_path = attempt_output_dir / "quality_report.json"
                q_path.write_text(
                    json.dumps(quality_result.to_dict(), indent=2), encoding="utf-8"
                )
                dlp_applied = bool(
                    enable_dlp
                    and (
                        cached_dlp_result is not None or result.dlp_provider is not None
                    )
                )
                sample_rate = int(getattr(settings, "TRACKING_SAMPLE_PERCENTAGE", 10))
                is_sampled = sample_rate >= 100 or (
                    sample_rate > 0
                    and int(hashlib.sha256(str(job_id).encode()).hexdigest()[:8], 16)
                    % 100
                    < sample_rate
                )
                if dlp_applied and is_sampled:
                    storage = get_translation_storage_repository()
                    await storage.upload_attempt_artifacts(
                        job_id=job_id,
                        attempt_index=attempt_index,
                        quality_report_path=q_path,
                    )
            except Exception:
                logger.debug(
                    f"Failed to upload DOCX quality report for attempt {attempt_index}",
                    exc_info=True,
                )

            # `best_quality` always records a real verdict now, including an
            # inconclusive one. It previously became None whenever the judge
            # failed, which silently emptied the cover page's confidence
            # score and the returned quality_report.
            if final_score > best_score or best_result is None:
                best_score = final_score
                best_result = result
                best_model_id = selected_model
                best_quality = quality_result.to_dict()
                best_token_usage = token_usage
                best_attempt_index = attempt_index

            # Shared with ModelAttemptOrchestrator. This loop used to carry a
            # hand-written copy of the rule prefixed "Mirror
            # ModelAttemptOrchestrator", and the copy had already drifted:
            # both of its exits were guarded on `quality_result is not None`,
            # so a judge failure fell through and ran the whole model chain.
            decision = decide_after_attempt(
                quality_result, attempt_index=attempt_index, pipeline="docx"
            )
            if decision is AttemptDecision.INCONCLUSIVE_ACCEPT:
                logger.warning(
                    f"DOCX job {job_id}: accepting attempt {attempt_index} on an "
                    "inconclusive judge verdict"
                )
            if decision.should_stop:
                break

        if best_result is None:
            raise RuntimeError("All DOCX translation attempts failed")

        for rep in attempt_reports:
            rep["is_selected"] = rep["attempt_number"] == best_attempt_index

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
            "attempts": attempt_reports,
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
