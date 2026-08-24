"""Orchestrate model-chain translation attempts with quality selection."""

from __future__ import annotations

import logging
import shutil
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from src.config.constants import settings
from src.worker.doctranslator.format.pdf.translation_config import (
    SharedContextCrossSplitPart,
)
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


def _cleanup_attempt_working_dir(translation_config: TranslationConfig | None) -> None:
    """Best-effort removal of a superseded/losing attempt's working directory.

    Each model attempt parses the full document into its own working_dir
    (IL tree, tracking JSON, etc). Previously all attempt working_dirs stayed
    on disk until the whole job finished (TempWorkspaceService.cleanup),
    meaning peak disk usage scaled with MAX_MODEL_ATTEMPTS. Removing a
    superseded attempt's working_dir as soon as we know it lost keeps at
    most two attempts' worth of intermediates on disk at once.
    """
    if translation_config is None:
        return
    working_dir = getattr(translation_config, "working_dir", None)
    if not working_dir:
        return
    try:
        shutil.rmtree(str(working_dir), ignore_errors=True)
    except Exception:
        logger.debug(
            "Failed to clean up superseded attempt working_dir %s", working_dir
        )



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
            config.get("judge_model"),
            region=config.get("judge_model_region"),
        )
        best_attempt_result: dict[str, Any] | None = None
        best_attempt_score = -1.0
        best_attempt_config: dict[str, Any] | None = None
        best_translation_config: TranslationConfig | None = None
        best_quality_result: QualityJudgeResult | None = None
        attempt_reports: list[dict[str, Any]] = []
        shared_context: SharedContextCrossSplitPart | None = None

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
                shared_context=shared_context,
            )

            if translation_config is not None and shared_context is None:
                shared_context = getattr(
                    translation_config, "shared_context_cross_split_part", None
                )

            if not attempt_result or not quality_result or not attempt_report:
                continue

            attempt_reports.append(attempt_report)
            final_score = quality_result.final_score

            if final_score > best_attempt_score:
                # The previous best attempt (if any) has just been
                # superseded -- its working_dir (full parsed IL tree,
                # tracking JSON, etc) is no longer needed, so free the disk
                # space now instead of waiting for the whole job to finish.
                _cleanup_attempt_working_dir(best_translation_config)
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
            else:
                # This attempt scored lower than the current best -- clean
                # up its working_dir immediately rather than at job end.
                _cleanup_attempt_working_dir(translation_config)

            if quality_result.pass_fail:
                break

            # QUALITY_EARLY_ACCEPT_THRESHOLD was declared in Settings but
            # never read anywhere, so a near-miss score still burned a full
            # extra attempt (re-parsing and re-translating the whole
            # document). Accept a score that is already comfortably good
            # rather than paying 2-3x latency chasing a marginal gain.
            early_accept = float(settings.QUALITY_EARLY_ACCEPT_THRESHOLD)
            if 0 < early_accept <= final_score:
                logger.info(
                    f"Early-accepting attempt {attempt_config['attempt_index']} "
                    f"(score={final_score:.3f} >= "
                    f"QUALITY_EARLY_ACCEPT_THRESHOLD={early_accept}); "
                    "skipping remaining model attempts"
                )
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
