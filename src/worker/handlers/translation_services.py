"""Translation task handler for Cloud Tasks."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from config.constants import settings
from config.logging import logger
from repository import get_bigquery_repository
from repository import get_firestore_repository
from worker.repository.worker_storage_repository import get_worker_storage_repository
from worker.services.processor import JobProcessor
from worker.services.progress import ProgressTracker

# Module-level singleton repositories (initialized once, reused across requests)
_firestore = get_firestore_repository()
_storage = get_worker_storage_repository()
_bigquery = get_bigquery_repository()


async def handle_translation_task(task_data: dict[str, Any]) -> dict[str, Any]:
    """Handle a translation task from Cloud Tasks."""
    job_id = task_data.get("job_id")
    config = task_data.get("config", {})

    if not job_id:
        logger.error("Task received without job_id")
        return {"success": False, "error": "Missing job_id"}

    logger.info(f"Processing translation task for job {job_id}")

    try:
        result = await _process_translation_job(job_id, config)
        return result

    except Exception as e:
        logger.exception(f"Critical error processing job {job_id}")
        return {"success": False, "error": str(e)}


async def _process_translation_job(
    job_id: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Process a translation job asynchronously."""
    # Initialize services with singleton repositories
    progress_tracker = ProgressTracker(_firestore, job_id)
    processor = JobProcessor(progress_tracker)

    start_time = datetime.utcnow()
    job_data: dict[str, Any] | None = None

    try:
        # Update job status to processing
        await _firestore.update_job(
            job_id, {"status": "processing", "progress": 0.0, "updated_at": start_time}
        )
        logger.debug(f"Job {job_id} marked as processing")

        # Get job details
        job_data = await _firestore.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found in Firestore")

        # Download input PDF
        await progress_tracker.update(0.1, "Downloading input PDF")
        input_filename = Path(str(job_data.get("original_filename", "input.pdf"))).name
        temp_base = Path(str(settings.TEMP_DIR)) / job_id
        input_path = temp_base / "input" / input_filename
        input_path.parent.mkdir(parents=True, exist_ok=True)
        await _storage.download_input_pdf(job_id, input_path, filename=input_filename)
        logger.debug(f"Downloaded input PDF for job {job_id}")

        # Prepare translation configuration
        output_dir = temp_base / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        translation_config = {
            "input_file": str(input_path),
            "output_dir": output_dir,
            "job_id": job_id,
            "lang_in": config["lang_in"],
            "lang_out": config["lang_out"],
            "model_list": config.get("model_list", []),
            **config.get("options", {}),
        }

        # Run translation
        logger.debug(f"Starting translation for job {job_id}")
        result = await processor.translate(translation_config)
        logger.debug(f"Translation completed for job {job_id}")

        # Upload output files (mono only; no-watermark mode)
        await progress_tracker.update(0.9, "Uploading results")
        mono_path = result.get("no_watermark_mono_pdf_path") or result.get(
            "mono_pdf_path"
        )
        output_files = {"mono": mono_path} if mono_path else {}

        output_uris = (
            await _storage.upload_output_files(job_id, output_files)
            if output_files
            else {}
        )
        logger.debug(f"Uploaded {len(output_files)} output files for job {job_id}")

        # Calculate processing time and update job as completed
        completed_at = datetime.utcnow()
        processing_time = (completed_at - start_time).total_seconds()

        completion_data = {
            "status": "completed",
            "progress": 1.0,
            "output_gs_uris": output_uris,
            "completed_at": completed_at,
            "updated_at": completed_at,
            "processing_seconds": int(processing_time),
            "pages_processed": result.get("page_count", 0),
            "quality_report": result.get("quality_report"),
            "attempts": result.get("attempts", []),
            "token_usage": result.get("token_usage", {}),
            "selected_model": result.get("model_id"),
        }

        await _firestore.update_job(job_id, completion_data)
        logger.info(f"Job {job_id} completed in {processing_time:.2f}s")

        # Write analytics to BigQuery
        await _write_job_analytics(
            job_id,
            "completed",
            config,
            job_data,
            processing_time,
            result,
            completed_at,
            output_uris,
        )

        # Cleanup temporary files
        await _cleanup_temp_files(job_id)

        return {"success": True}

    except Exception as e:
        logger.exception(f"Job {job_id} failed during processing")

        # Update job as failed
        await _firestore.update_job(
            job_id,
            {
                "status": "failed",
                "error_message": str(e),
                "updated_at": datetime.utcnow(),
            },
        )

        # Write failure analytics (best effort)
        try:
            await _write_job_analytics(
                job_id, "failed", config, job_data, error_message=str(e)
            )
        except Exception as analytics_error:
            logger.error(
                f"Failed to write failure analytics for job {job_id}: {analytics_error}"
            )

        # Cleanup temporary files
        await _cleanup_temp_files(job_id)

        raise


async def _write_job_analytics(
    job_id: str,
    status: str,
    config: dict[str, Any],
    job_data: dict[str, Any] | None,
    processing_time: float | None = None,
    result: dict[str, Any] | None = None,
    completed_at: datetime | None = None,
    output_uris: dict[str, str] | None = None,
    error_message: str | None = None,
) -> None:
    """Write job analytics to BigQuery."""
    attempts = list(result.get("attempts", [])) if result else []
    total_token_usage = 0
    attempt_costs: dict[int, float] = {}
    attempt_tokens: dict[int, int] = {}
    for attempt in attempts:
        idx = int(attempt.get("attempt_index", 0))
        usage = attempt.get("token_usage", {})
        tokens = int(usage.get("total_tokens", 0) or 0)
        total_token_usage += tokens
        attempt_tokens[idx] = tokens
        attempt_costs[idx] = float(usage.get("estimated_cost_usd", 0.0) or 0.0)

    analytics_data = {
        "job_id": job_id,
        "status": status,
        "domain": config.get("domain"),
        "lang_in": config.get("lang_in"),
        "lang_out": config.get("lang_out"),
        "user": config.get("user"),
        "department": config.get("department"),
        "token_usage": total_token_usage,
        "total_cost_usd": round(sum(attempt_costs.values()), 6),
        "iteration_details": attempts,
    }

    if job_data:
        analytics_data["file_size_bytes"] = job_data.get("file_size_bytes")
        analytics_data["created_at"] = job_data.get("created_at")

    if status == "completed" and result:
        analytics_data["processing_seconds"] = (
            int(processing_time) if processing_time else 0
        )
        analytics_data["pages_processed"] = result.get("page_count", 0)
        analytics_data["completed_at"] = completed_at
        analytics_data["output_gs_uris"] = output_uris or {}
        analytics_data["quality_report"] = result.get("quality_report", {})
        analytics_data["selected_model"] = result.get("model_id")
        analytics_data["attempt_count"] = len(attempts)

    if status == "failed":
        analytics_data["error_message"] = error_message

    await _bigquery.write_job_completion(analytics_data)


async def _cleanup_temp_files(job_id: str) -> None:
    """Clean up temporary files for a job."""
    temp_dir = Path(str(settings.TEMP_DIR)) / job_id

    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp files for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files for job {job_id}: {e}")
