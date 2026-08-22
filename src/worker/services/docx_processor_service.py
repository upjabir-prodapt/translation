"""Model-chain translation of DOCX jobs, mirroring the PDF JobProcessor contract."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from opentelemetry.trace import SpanKind

from src.config.constants import settings
from src.config.domain_prompts import normalize_domain_key
from src.config.tracing import tracer_pipeline
from src.worker.doctranslator.translator.factory import create_translator
from src.worker.docxtranslator.package import DocxPackage
from src.worker.docxtranslator.segments import collect_segments
from src.worker.docxtranslator.segments import extract_text
from src.worker.services.docx_translation_service import DocxTranslationResult
from src.worker.services.docx_translation_service import DocxTranslationService
from src.worker.services.llm_cost_service import get_vertex_llm_cost_service
from src.worker.services.quality_judge_service import GoogleADKJudgeAgent
from src.worker.services.quality_judge_service import QualityJudgeResult

logger = logging.getLogger(__name__)

DLP_CHUNK_MODE = "docx_run_group"


def extract_docx_text(input_file: str | Path, max_chars: int | None = None) -> str:
    """Read the visible text of a .docx for language detection or previews."""
    package = DocxPackage.open(Path(str(input_file)))
    return extract_text(collect_segments(package.text_parts()), max_chars=max_chars)


class DocxJobProcessor:
    """Run a DOCX job through the model chain and pick the best attempt.

    Exposes the same ``translate(config) -> attempt result`` contract as
    :class:`~src.worker.services.processor_service.JobProcessor`, so the
    pipeline orchestrator treats both formats the same way. Outputs stay .docx:
    no cover page is prepended and no PDF is produced, because the deliverable
    is the original Word document with translated text.
    """

    PROGRESS_START = 0.2
    PROGRESS_TRANSLATION_START = 0.3
    PROGRESS_FINALIZE = 0.8
    PROGRESS_COMPLETE = 0.9

    def __init__(
        self,
        progress_tracker: Any,
        translation_service: DocxTranslationService | None = None,
    ) -> None:
        self.progress_tracker = progress_tracker
        self.translation_service = translation_service or DocxTranslationService()
        self._cost_service = get_vertex_llm_cost_service()

    async def translate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Translate a DOCX across the model chain, returning the best attempt."""
        model_list = list(config.get("model_list") or [])
        if not model_list:
            raise ValueError("model_list is required for translation")

        output_base_dir = Path(config["output_dir"])
        output_base_dir.mkdir(parents=True, exist_ok=True)
        max_attempts = min(
            int(config.get("max_model_attempts", settings.MAX_MODEL_ATTEMPTS)),
            len(model_list),
        )
        judge = GoogleADKJudgeAgent(
            config.get("judge_model"), domain=config.get("domain")
        )

        best: dict[str, Any] | None = None
        best_score = -1.0
        for model_index in range(max_attempts):
            attempt = await self._run_attempt(
                config=config,
                model_index=model_index,
                model_list=model_list,
                max_attempts=max_attempts,
                output_base_dir=output_base_dir,
                judge=judge,
            )
            if attempt is None:
                continue
            score = float(attempt["quality_report"].get("final_score") or 0.0)
            if score > best_score:
                best_score = score
                best = attempt
            if attempt["quality_passed"]:
                break

        if best is None:
            raise RuntimeError("All DOCX translation attempts failed")
        best.pop("quality_passed", None)
        return best

    async def _run_attempt(
        self,
        *,
        config: dict[str, Any],
        model_index: int,
        model_list: list[str],
        max_attempts: int,
        output_base_dir: Path,
        judge: GoogleADKJudgeAgent,
    ) -> dict[str, Any] | None:
        attempt_index = model_index + 1
        selected_model = model_list[model_index]
        input_path = Path(str(config["input_file"]))
        attempt_dir = output_base_dir / f"iter_{attempt_index}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        output_path = attempt_dir / input_path.name

        await self.progress_tracker.update(
            self.PROGRESS_START,
            f"Attempt {attempt_index}/{max_attempts}: model={selected_model}",
        )

        domain = normalize_domain_key(config.get("domain"))
        translate_engine = create_translator(
            selected_model,
            lang_in=str(config["lang_in"]),
            lang_out=str(config["lang_out"]),
            qps=int(config.get("qps", settings.TRANSLATION_MAX_QPS)),
            domain=domain,
        )

        try:
            with tracer_pipeline.start_as_current_span(
                "pipeline.docx_translation",
                kind=SpanKind.INTERNAL,
                attributes={
                    "translation.attempt": attempt_index,
                    "translation.model": selected_model,
                    "translation.format": "docx",
                },
            ):
                await self.progress_tracker.update(
                    self.PROGRESS_TRANSLATION_START, "Starting translation"
                )
                result = await self.translation_service.translate_document(
                    input_path=input_path,
                    output_path=output_path,
                    translate_engine=translate_engine,
                    lang_in=str(config["lang_in"]),
                    lang_out=str(config["lang_out"]),
                    domain=domain,
                    job_id=str(config.get("job_id", "")),
                    glossaries=config.get("glossaries"),
                    enable_dlp=bool(config.get("enable_dlp", False)),
                )
        except Exception:
            logger.exception(
                f"DOCX attempt {attempt_index} failed with model {selected_model}"
            )
            if attempt_index == max_attempts:
                raise
            return None

        await self.progress_tracker.update(self.PROGRESS_FINALIZE, "Finalizing output")
        with tracer_pipeline.start_as_current_span(
            "pipeline.quality_judge",
            kind=SpanKind.INTERNAL,
            attributes={"translation.attempt": attempt_index},
        ):
            quality = await self._judge(judge, result)
        await self.progress_tracker.update(
            self.PROGRESS_COMPLETE, "Translation complete"
        )

        return {
            "output_path": str(result.output_path),
            "attempt_index": attempt_index,
            "model_id": selected_model,
            "quality_report": quality.to_dict(),
            "quality_passed": quality.pass_fail,
            "token_usage": self._collect_token_usage(translate_engine, selected_model),
            "chunks_processed": result.batch_count,
            "batch_token_counts": result.batch_token_counts,
            "page_count": 0,
            "paragraph_count": result.segment_count,
            "dlp_token_rows": result.dlp_token_rows,
            "dlp_provider": result.dlp_provider,
            "dlp_chunk_mode": DLP_CHUNK_MODE if result.dlp_token_rows else None,
        }

    async def _judge(
        self, judge: GoogleADKJudgeAgent, result: DocxTranslationResult
    ) -> QualityJudgeResult:
        if not result.source_text.strip() or not result.translated_text.strip():
            return QualityJudgeResult(
                alignment_score=0.0,
                omission_score=0.0,
                hallucination_score=0.0,
                final_score=0.0,
                pass_fail=False,
                reasons=["Missing source/translated text for quality evaluation."],
                model=judge.model,
            )
        return await judge.evaluate_async(
            source_text=result.source_text,
            translated_text=result.translated_text,
        )

    def _collect_token_usage(
        self, translate_engine: Any, selected_model: str
    ) -> dict[str, Any]:
        prompt_tokens = _counter_value(
            getattr(translate_engine, "prompt_token_count", 0)
        )
        completion_tokens = _counter_value(
            getattr(translate_engine, "completion_token_count", 0)
        )
        total_tokens = _counter_value(getattr(translate_engine, "token_count", 0))
        cache_hit_tokens = _counter_value(
            getattr(translate_engine, "cache_hit_prompt_token_count", 0)
        )
        breakdown = self._cost_service.calculate_attempt_cost(
            model_id=selected_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=cache_hit_tokens,
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


def _counter_value(value: Any) -> int:
    if hasattr(value, "value"):
        return int(value.value)
    return int(value or 0)
