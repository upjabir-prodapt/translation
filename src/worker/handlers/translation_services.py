"""Translation task handler for Cloud Tasks."""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

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
        await _handle_job_failure(job_id, str(e))
        return {"success": False, "error": str(e)}


async def _handle_job_failure(job_id: str, error_message: str) -> None:
    """Handle job failure by updating status in Firestore."""
    try:
        await _firestore.update_job(
            job_id,
            {
                "status": "failed",
                "error_message": error_message,
                "updated_at": datetime.utcnow(),
            },
        )
        logger.info(f"Marked job {job_id} as failed")
    except Exception as e:
        logger.error(f"Failed to update job {job_id} failure status: {e}")


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
        input_path = Path(tempfile.gettempdir()) / job_id / input_filename
        input_path.parent.mkdir(parents=True, exist_ok=True)
        await _storage.download_input_pdf(job_id, input_path, filename=input_filename)
        logger.debug(f"Downloaded input PDF for job {job_id}")

        # Prepare translation configuration
        output_dir = Path(tempfile.gettempdir()) / job_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        translation_config = {
            "input_file": str(input_path),
            "output_dir": output_dir,
            "lang_in": config["lang_in"],
            "lang_out": config["lang_out"],
            **config.get("options", {}),
        }

        # Run translation
        logger.debug(f"Starting translation for job {job_id}")
        result = await processor.translate(translation_config)
        logger.debug(f"Translation completed for job {job_id}")

        # Upload output files
        await progress_tracker.update(0.9, "Uploading results")
        output_files = {"mono": result.get("mono_pdf_path")}
        output_files = {k: v for k, v in output_files.items() if v is not None}

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

        # Write detailed translation reports if available
        if "translations" in result:
            await _bigquery.write_translation_report(job_id, result["translations"])

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
    analytics_data = {
        "job_id": job_id,
        "status": status,
        "domain": config.get("domain"),
        "lang_in": config.get("lang_in"),
        "lang_out": config.get("lang_out"),
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

    if status == "failed":
        analytics_data["error_message"] = error_message

    await _bigquery.write_job_completion(analytics_data)


async def _cleanup_temp_files(job_id: str) -> None:
    """Clean up temporary files for a job."""
    temp_dir = Path(tempfile.gettempdir()) / job_id

    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp files for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files for job {job_id}: {e}")
