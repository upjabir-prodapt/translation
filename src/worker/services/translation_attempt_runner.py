"""Run a single translation model attempt."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from opentelemetry.trace import SpanKind

from src.worker.services.llm_cost_service import get_vertex_llm_cost_service
from src.worker.services.quality_judge_service import GoogleADKJudgeAgent
from src.worker.services.quality_judge_service import QualityJudgeResult
from src.worker.services.quality_judge_service import extract_attempt_text
from src.config.tracing import tracer_pipeline
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig

if TYPE_CHECKING:
    from src.worker.services.processor_service import JobProcessor

logger = logging.getLogger(__name__)


class TranslationAttemptRunner:
    """Execute one model attempt: translate, judge quality, collect usage."""

    def __init__(self, processor: JobProcessor):
        self._processor = processor
        self._cost_service = get_vertex_llm_cost_service()

    async def run_attempt(
        self,
        *,
        model_index: int,
        model_list: list[str],
        config: dict[str, Any],
        output_base_dir: Path,
        max_attempts: int,
        judge: GoogleADKJudgeAgent,
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any],
        TranslationConfig | None,
        QualityJudgeResult | None,
        dict[str, Any] | None,
    ]:
        attempt_index = model_index + 1
        selected_model = model_list[model_index]
        attempt_output_dir = output_base_dir / f"iter_{attempt_index}"
        attempt_output_dir.mkdir(parents=True, exist_ok=True)
        attempt_config = {
            **config,
            "selected_model": selected_model,
            "attempt_index": attempt_index,
        }
        logger.info(
            f"Attempt {attempt_index}/{max_attempts}: attempt_config={attempt_config}"
        )
        translation_config = self._processor._build_translation_config(
            attempt_config, attempt_output_dir
        )

        await self._processor.progress_tracker.update(
            self._processor.PROGRESS_START,
            f"Attempt {attempt_index}/{max_attempts}: model={selected_model}",
        )
        try:
            with tracer_pipeline.start_as_current_span(
                "pipeline.translation",
                kind=SpanKind.INTERNAL,
                attributes={
                    "translation.attempt": attempt_index,
                    "translation.model": selected_model,
                },
            ):
                attempt_result = await self._processor._run_single_attempt(
                    translation_config, attempt_config
                )
        except Exception as exc:
            logger.exception(
                f"Attempt {attempt_index} failed with model {selected_model} with exception: {exc}"
            )
            if attempt_index == max_attempts:
                raise
            return None, attempt_config, None, None, None

        source_text, translated_text = extract_attempt_text(
            Path(str(translation_config.working_dir))
        )
        with tracer_pipeline.start_as_current_span(
            "pipeline.quality_judge",
            kind=SpanKind.INTERNAL,
            attributes={"translation.attempt": attempt_index},
        ):
            quality_result = await self._evaluate_attempt_quality(
                judge=judge,
                source_text=source_text,
                translated_text=translated_text,
            )
        token_usage = self.collect_token_usage(translation_config, selected_model)
        attempt_report = {
            "attempt_index": attempt_index,
            "model_id": selected_model,
            "quality": quality_result.to_dict(),
            "token_usage": token_usage,
            "working_dir": str(translation_config.working_dir),
            "output_dir": str(attempt_output_dir),
        }
        self._write_quality_report(
            Path(str(translation_config.working_dir)), attempt_report
        )
        return (
            attempt_result,
            attempt_config,
            translation_config,
            quality_result,
            attempt_report,
        )

    async def _evaluate_attempt_quality(
        self,
        *,
        judge: GoogleADKJudgeAgent,
        source_text: str,
        translated_text: str,
    ) -> QualityJudgeResult:
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
        return await judge.evaluate_async(
            source_text=source_text, translated_text=translated_text
        )

    def collect_token_usage(
        self, translation_config: TranslationConfig, selected_model: str
    ) -> dict[str, Any]:
        translator = translation_config.translator
        prompt_tokens = self._processor._counter_value(
            getattr(translator, "prompt_token_count", 0)
        )
        completion_tokens = self._processor._counter_value(
            getattr(translator, "completion_token_count", 0)
        )
        total_tokens = self._processor._counter_value(
            getattr(translator, "token_count", 0)
        )
        cache_hit_tokens = self._processor._counter_value(
            getattr(translator, "cache_hit_prompt_token_count", 0)
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

    def _write_quality_report(
        self, working_dir: Path, attempt_report: dict[str, Any]
    ) -> None:
        try:
            quality_path = working_dir / "quality_report.json"
            quality_path.write_text(
                json.dumps(attempt_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.exception(f"Failed to write quality report in {working_dir}")
