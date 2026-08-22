"""Orchestrate model-chain translation attempts with quality selection."""

from __future__ import annotations

import logging
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from src.config.constants import settings
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig
from src.worker.doctranslator.format.pdf.translation_config import (
    TranslationCoverPageMetadata,
)
from src.worker.services.quality_judge_service import GoogleADKJudgeAgent
from src.worker.services.quality_judge_service import QualityJudgeResult
from src.worker.services.translation_attempt_runner import TranslationAttemptRunner

if TYPE_CHECKING:
    from src.worker.services.processor_service import JobProcessor

logger = logging.getLogger(__name__)


class ModelAttemptOrchestrator:
    """Loop model chain attempts and pick the best passing or highest-scoring result."""

    def __init__(self, processor: JobProcessor):
        self._processor = processor
        self._attempt_runner = TranslationAttemptRunner(processor)

    async def run_model_chain(self, config: dict[str, Any]) -> dict[str, Any]:
        output_base_dir = Path(config["output_dir"])
        output_base_dir.mkdir(parents=True, exist_ok=True)
        model_list = config.get("model_list", [])
        if not model_list:
            raise ValueError("model_list is required for translation")

        max_attempts = min(
            int(config.get("max_model_attempts", settings.MAX_MODEL_ATTEMPTS)),
            len(model_list),
        )
        judge = GoogleADKJudgeAgent(
            config.get("judge_model"), domain=config.get("domain")
        )
        best_attempt_result: dict[str, Any] | None = None
        best_attempt_score = -1.0
        best_attempt_config: dict[str, Any] | None = None
        best_translation_config: TranslationConfig | None = None
        best_quality_result: QualityJudgeResult | None = None
        attempt_reports: list[dict[str, Any]] = []

        for model_index in range(max_attempts):
            (
                attempt_result,
                attempt_config,
                translation_config,
                quality_result,
                attempt_report,
            ) = await self._attempt_runner.run_attempt(
                model_index=model_index,
                model_list=model_list,
                config=config,
                output_base_dir=output_base_dir,
                max_attempts=max_attempts,
                judge=judge,
            )

            if not attempt_result or not quality_result or not attempt_report:
                continue

            attempt_reports.append(attempt_report)
            final_score = quality_result.final_score

            if final_score > best_attempt_score:
                best_attempt_score = final_score
                best_attempt_result = {
                    **attempt_result,
                    "attempt_index": attempt_config["attempt_index"],
                    "model_id": attempt_config["selected_model"],
                    "quality_report": quality_result.to_dict(),
                    "token_usage": attempt_report["token_usage"],
                }
                best_attempt_config = attempt_config
                best_translation_config = translation_config
                best_quality_result = quality_result

            if quality_result.pass_fail:
                break

        if best_attempt_result is None:
            raise RuntimeError("All translation attempts failed")
        if best_translation_config:
            best_attempt_result["dlp_provider"] = best_translation_config.dlp_provider
            best_attempt_result["dlp_chunk_mode"] = (
                best_translation_config.dlp_chunk_mode
            )
            best_attempt_result["dlp_token_rows"] = list(
                best_translation_config.dlp_token_rows
            )
        if best_translation_config and best_attempt_config and best_quality_result:
            cover_page_metadata = self._build_cover_page_metadata(
                best_translation_config,
                best_attempt_config,
                best_quality_result,
            )
            self._processor._apply_cover_pages(
                best_translation_config,
                best_attempt_result,
                cover_page_metadata,
            )

        return best_attempt_result

    def _build_cover_page_metadata(
        self,
        translation_config: TranslationConfig,
        config: dict[str, Any],
        quality_result: QualityJudgeResult,
    ) -> TranslationCoverPageMetadata:
        total_pages = self._processor._get_total_pdf_pages(
            translation_config.input_file
        )
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
