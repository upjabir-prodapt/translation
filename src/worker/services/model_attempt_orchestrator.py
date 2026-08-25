"""Orchestrate model-chain translation attempts with quality selection."""

from __future__ import annotations

import hashlib
import logging
import shutil
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from src.config.constants import settings
from src.repository.translation_storage_repository import (
    get_translation_storage_repository,
)
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


def is_job_sampled_for_tracking(job_id: str) -> bool:
    """Deterministically sample tracking upload based on job_id hash."""
    if not job_id:
        return False
    sample_rate = int(getattr(settings, "TRACKING_SAMPLE_PERCENTAGE", 10))
    if sample_rate <= 0:
        return False
    if sample_rate >= 100:
        return True
    return (
        int(hashlib.sha256(str(job_id).encode("utf-8")).hexdigest()[:8], 16) % 100
        < sample_rate
    )


async def _upload_sampled_attempt_artifacts(
    *,
    job_id: str,
    attempt_index: int,
    translation_config: TranslationConfig | None,
) -> None:
    """Upload attempt tracking and quality report JSON to GCS if sampled and DLP-masked."""
    if translation_config is None or not job_id:
        return
    dlp_applied = bool(
        getattr(translation_config, "dlp_applied_pre_translation", False)
        or getattr(translation_config, "dlp_provider", None)
    )
    if not dlp_applied:
        logger.debug(
            "Skipping tracking upload for job %s: DLP masking was not applied", job_id
        )
        return
    if not is_job_sampled_for_tracking(job_id):
        logger.debug("Skipping tracking upload for job %s: not sampled", job_id)
        return

    working_dir = getattr(translation_config, "working_dir", None)
    if not working_dir:
        return
    work_path = Path(str(working_dir))
    tracking_path = work_path / "translate_tracking.json"
    quality_path = work_path / "quality_report.json"

    storage = get_translation_storage_repository()
    await storage.upload_attempt_artifacts(
        job_id=job_id,
        attempt_index=attempt_index,
        tracking_path=tracking_path if tracking_path.exists() else None,
        quality_report_path=quality_path if quality_path.exists() else None,
    )


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
        enable_judge = bool(
            config.get("enable_judge", getattr(settings, "QUALITY_JUDGE_ENABLED", True))
        )
        judge: GoogleADKJudgeAgent | None = None
        if enable_judge:
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

            if not attempt_result or not attempt_report:
                continue

            if not enable_judge or quality_result is None:
                attempt_reports.append(attempt_report)
                best_attempt_result = {
                    **attempt_result,
                    "attempt_index": attempt_config["attempt_index"],
                    "model_id": attempt_config["selected_model"],
                    "quality_report": None,
                    "token_usage": attempt_report["token_usage"],
                }
                best_attempt_config = attempt_config
                best_translation_config = translation_config
                best_quality_result = None
                break

            attempt_reports.append(attempt_report)
            final_score = quality_result.final_score

            job_id = str(config.get("job_id", ""))
            await _upload_sampled_attempt_artifacts(
                job_id=job_id,
                attempt_index=int(attempt_config.get("attempt_index", model_index + 1)),
                translation_config=translation_config,
            )

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

        winning_attempt_idx = (
            best_attempt_config.get("attempt_index") if best_attempt_config else 1
        )
        formatted_attempts = []
        for rep in attempt_reports:
            q = rep.get("quality") or {}
            t = rep.get("token_usage") or {}
            att_idx = rep.get("attempt_index")
            formatted_attempts.append(
                {
                    "attempt_number": att_idx,
                    "model_id": rep.get("model_id"),
                    "alignment_score": q.get("alignment_score"),
                    "omission_score": q.get("omission_score"),
                    "hallucination_score": q.get("hallucination_score"),
                    "final_score": q.get("final_score"),
                    "pass_fail": q.get("pass_fail"),
                    "is_fallback": q.get("is_fallback", False),
                    "total_tokens": t.get("total_tokens", 0),
                    "prompt_tokens": t.get("prompt_tokens", 0),
                    "completion_tokens": t.get("completion_tokens", 0),
                    "cache_hit_prompt_tokens": t.get("cache_hit_prompt_tokens", 0),
                    "cost_usd": t.get("estimated_cost_usd", 0.0),
                    "is_selected": (att_idx == winning_attempt_idx),
                    "docx_path": rep.get("docx_path"),
                }
            )
        best_attempt_result["attempts"] = formatted_attempts
        best_attempt_result["attempt_reports"] = attempt_reports

        if best_translation_config:
            best_attempt_result["dlp_provider"] = best_translation_config.dlp_provider
            best_attempt_result["dlp_chunk_mode"] = (
                best_translation_config.dlp_chunk_mode
            )
            best_attempt_result["dlp_token_rows"] = list(
                best_translation_config.dlp_token_rows
            )
        if best_translation_config and best_attempt_config:
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
        quality_result: QualityJudgeResult | None,
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
            confidence_score=quality_result.final_score if quality_result else None,
            translated_sections=translated_sections,
            judge_model=quality_result.model if quality_result else None,
        )
