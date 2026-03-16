"""New async translation service for BabelDOC.

This module provides a clean, service-oriented API for translating PDFs
with Pydantic-based configuration and UUID-based job tracking.
"""

import uuid
from pathlib import Path

from loguru import logger

from babeldoc.const import CACHE_FOLDER
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.format.pdf import high_level
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode
from babeldoc.translator.translator import OpenAITranslator
from babeldoc.translator.translator import set_translate_rate_limiter
from schemas.babeldoc_schemas import TranslationConfigSchema
from schemas.babeldoc_schemas import TranslationJobStatus
from schemas.babeldoc_schemas import TranslationModelSchema

# In-memory job storage (could be replaced with BigQuery for production)
_job_store: dict[uuid.UUID, TranslationJobStatus] = {}


def _get_watermark_mode(mode: str) -> WatermarkOutputMode:
    """Convert string mode to WatermarkOutputMode enum."""
    mode_map = {
        "watermarked": WatermarkOutputMode.Watermarked,
        "no_watermark": WatermarkOutputMode.NoWatermark,
        "both": WatermarkOutputMode.Both,
    }
    return mode_map.get(mode, WatermarkOutputMode.Watermarked)


def create_translation_job(
    pdf_path: Path,
    model_config: TranslationModelSchema | None = None,
    options: dict | None = None,
) -> uuid.UUID:
    """Create a new translation job.

    Args:
        pdf_path: Path to the input PDF file
        model_config: Translation model configuration (uses defaults if not provided)
        options: Optional dictionary to override config.json defaults

    Returns:
        UUID of the created job
    """
    # Create config schema with defaults from config.json
    config_data = {"input_file": pdf_path}
    if options:
        config_data.update(options)

    config = TranslationConfigSchema(**config_data)

    # Create UUID-based directory inside cache_folder for output
    job_output_dir = CACHE_FOLDER / str(config.job_id)
    job_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created job output directory: {job_output_dir}")

    # Set output_dir to the UUID folder (override if not explicitly provided)
    if "output_dir" not in config_data:
        config_data["output_dir"] = job_output_dir
        # Recreate config with updated output_dir
        config = TranslationConfigSchema(**config_data)

    # Initialize job status
    job_status = TranslationJobStatus(
        job_id=config.job_id,
        status="pending",
        message="Job created, waiting to start",
    )
    _job_store[config.job_id] = job_status

    logger.info(f"Created translation job {config.job_id} for {pdf_path}")
    return config.job_id


async def run_translation_job(
    job_id: uuid.UUID,
    model_config: TranslationModelSchema | None = None,
) -> TranslationJobStatus:
    """Run a translation job asynchronously.

    Args:
        job_id: UUID of the job to run
        model_config: Translation model configuration

    Returns:
        Final job status after completion
    """
    # Find job in store
    if job_id not in _job_store:
        raise ValueError(f"Job {job_id} not found")

    job_status = _job_store[job_id]
    job_status.status = "running"
    job_status.message = "Translation started"

    try:
        # Get the config from job store or reconstruct it
        # For now, we need to pass the actual config schema to this function
        # The job store should store the full config schema
        raise NotImplementedError(
            "run_translation_job requires storing full config in job store. "
            "Use translate_pdf() for now or pass config schema directly."
        )
    except NotImplementedError as e:
        job_status.status = "failed"
        job_status.message = str(e)
        logger.error(f"Translation job {job_id} failed: {e}")
    except Exception as e:
        logger.exception(f"Translation job {job_id} failed")
        job_status.status = "failed"
        job_status.message = str(e)

    return job_status


async def _execute_translation(
    job_id: uuid.UUID,
    config_schema: TranslationConfigSchema,
    model_config: TranslationModelSchema,
) -> dict:
    """Execute the actual translation.

    Args:
        job_id: UUID for job tracking
        config_schema: Translation configuration schema
        model_config: Model configuration schema

    Returns:
        Dictionary with translation results
    """
    # Set up rate limiter
    set_translate_rate_limiter(model_config.qps)

    # Create translator
    translator = OpenAITranslator(
        lang_in=config_schema.lang_in,
        lang_out=config_schema.lang_out,
        ignore_cache=True,  # Disable cache as per user's recent changes
        model=model_config.model_name,
        api_key=model_config.api_key,
    )

    # Initialize document layout model
    doc_layout_model = DocLayoutModel.load_available()

    # Create legacy TranslationConfig
    translation_config = TranslationConfig(
        translator=translator,
        input_file=config_schema.input_file,
        lang_in=config_schema.lang_in,
        lang_out=config_schema.lang_out,
        doc_layout_model=doc_layout_model,
        output_dir=config_schema.output_dir,
        debug=config_schema.debug,
        pages=config_schema.pages,
        no_dual=config_schema.no_dual,
        no_mono=config_schema.no_mono,
        watermark_output_mode=_get_watermark_mode(config_schema.watermark_output_mode),
        auto_extract_glossary=config_schema.auto_extract_glossary,
        min_text_length=config_schema.min_text_length,
        split_short_lines=config_schema.split_short_lines,
        short_line_split_factor=config_schema.short_line_split_factor,
        skip_clean=config_schema.skip_clean,
        dual_translate_first=config_schema.dual_translate_first,
        disable_rich_text_translate=config_schema.disable_rich_text_translate,
        enhance_compatibility=config_schema.enhance_compatibility,
        use_alternating_pages_dual=config_schema.use_alternating_pages_dual,
        show_char_box=config_schema.show_char_box,
        skip_scanned_detection=config_schema.skip_scanned_detection,
        ocr_workaround=config_schema.ocr_workaround,
        add_formula_placehold_hint=config_schema.add_formula_placehold_hint,
        disable_same_text_fallback=config_schema.disable_same_text_fallback,
        auto_enable_ocr_workaround=config_schema.auto_enable_ocr_workaround,
        only_include_translated_page=config_schema.only_include_translated_page,
        save_auto_extracted_glossary=config_schema.save_auto_extracted_glossary,
        enable_graphic_element_process=not config_schema.disable_graphic_element_process,
        merge_alternating_line_numbers=config_schema.merge_alternating_line_numbers,
        skip_translation=config_schema.skip_translation,
        skip_form_render=config_schema.skip_form_render,
        skip_curve_render=config_schema.skip_curve_render,
        only_parse_generate_pdf=config_schema.only_parse_generate_pdf,
        remove_non_formula_lines=config_schema.remove_non_formula_lines,
        non_formula_line_iou_threshold=config_schema.non_formula_line_iou_threshold,
        figure_table_protection_threshold=config_schema.figure_table_protection_threshold,
        skip_formula_offset_calculation=config_schema.skip_formula_offset_calculation,
        report_interval=config_schema.report_interval,
        custom_system_prompt=config_schema.custom_system_prompt,
        metadata_extra_data=config_schema.metadata_extra_data,
    )

    # Run async translation
    result = None
    current_stage = None
    async for event in high_level.async_translate(translation_config):
        event_type = event.get("type")
        stage = event.get("stage", current_stage)
        current_stage = stage

        # Enhanced logging based on event type
        if event_type == "progress_start":
            logger.info(
                f"[Stage Start] {stage}: {event.get('stage_current', 0)}/{event.get('stage_total', 0)} items "
                f"(Part {event.get('part_index', 1)}/{event.get('total_parts', 1)})"
            )

        elif event_type == "progress_update":
            # Use debug for frequent updates to avoid log spam
            logger.debug(
                f"[Progress] {stage}: {event.get('stage_progress', 0):.1f}% "
                f"({event.get('stage_current', 0)}/{event.get('stage_total', 0)}), "
                f"Overall: {event.get('overall_progress', 0):.1f}%"
            )
            # Update job progress with overall_progress if available
            if job_id in _job_store:
                _job_store[job_id].progress = event.get("overall_progress", 0.0) / 100.0
            continue  # Skip the generic progress update below

        elif event_type == "progress_end":
            logger.info(
                f"[Stage Complete] {stage}: {event.get('stage_current', 0)}/{event.get('stage_total', 0)} "
                f"Overall: {event.get('overall_progress', 0):.1f}%"
            )

        elif event_type == "error":
            error_msg = event.get("error", "Unknown error")
            logger.error(f"[Translation Failed] {error_msg}")
            raise RuntimeError(f"Translation error: {error_msg}")

        elif event_type == "finish":
            result = event.get("translate_result")
            if result:
                logger.success(
                    f"[Translation Complete] Mono: {result.mono_pdf_path}, "
                    f"Dual: {result.dual_pdf_path or 'N/A'}"
                )
            else:
                logger.warning("[Translation Complete] No result returned")
            break

        else:
            logger.debug(f"[Event] {event_type}: {event}")

        # Generic progress update for non-progress_update events
        if job_id in _job_store and "overall_progress" in event:
            _job_store[job_id].progress = event.get("overall_progress", 0.0) / 100.0

    # Clean up temporary files
    translation_config.cleanup_temp_files()
    logger.info("Cleaned up temporary files")

    if result is None:
        raise RuntimeError("Translation completed but no result was returned")

    return {
        "mono_pdf_path": str(result.mono_pdf_path) if result.mono_pdf_path else None,
        "dual_pdf_path": str(result.dual_pdf_path) if result.dual_pdf_path else None,
        "no_watermark_mono_pdf_path": str(result.no_watermark_mono_pdf_path)
        if result.no_watermark_mono_pdf_path
        else None,
        "no_watermark_dual_pdf_path": str(result.no_watermark_dual_pdf_path)
        if result.no_watermark_dual_pdf_path
        else None,
        "auto_extracted_glossary_path": str(result.auto_extracted_glossary_path)
        if result.auto_extracted_glossary_path
        else None,
        "total_seconds": getattr(result, "total_seconds", None),
        "peak_memory_usage": getattr(result, "peak_memory_usage", None),
    }


def get_job_status(job_id: uuid.UUID) -> TranslationJobStatus | None:
    """Get the status of a translation job.

    Args:
        job_id: UUID of the job

    Returns:
        Job status if found, None otherwise
    """
    return _job_store.get(job_id)


def list_jobs() -> list[TranslationJobStatus]:
    """List all translation jobs.

    Returns:
        List of all job statuses
    """
    return list(_job_store.values())


def cancel_job(job_id: uuid.UUID) -> bool:
    """Cancel a running translation job.

    Args:
        job_id: UUID of the job to cancel

    Returns:
        True if job was cancelled, False otherwise
    """
    if job_id not in _job_store:
        return False

    job_status = _job_store[job_id]
    if job_status.status == "running":
        job_status.status = "failed"
        job_status.message = "Job cancelled by user"
        return True

    return False


def cleanup_job(job_id: uuid.UUID, remove_output: bool = False) -> bool:
    """Remove a completed/failed job from the store and optionally delete output folder.

    Args:
        job_id: UUID of the job to remove
        remove_output: If True, also delete the job's output directory from cache_folder

    Returns:
        True if job was removed, False otherwise
    """
    if job_id not in _job_store:
        return False

    # Optionally remove the output directory
    if remove_output:
        job_output_dir = CACHE_FOLDER / str(job_id)
        if job_output_dir.exists():
            import shutil

            shutil.rmtree(job_output_dir, ignore_errors=True)
            logger.info(f"Removed job output directory: {job_output_dir}")

    del _job_store[job_id]
    return True


def cleanup_job_directory(job_id: uuid.UUID) -> bool:
    """Remove the output directory for a specific job.

    Args:
        job_id: UUID of the job whose directory should be removed

    Returns:
        True if directory was removed, False otherwise
    """
    job_output_dir = CACHE_FOLDER / str(job_id)
    if job_output_dir.exists():
        import shutil

        shutil.rmtree(job_output_dir, ignore_errors=True)
        logger.info(f"Removed job output directory: {job_output_dir}")
        return True
    return False


# Convenience function for simple use cases
async def translate_pdf(
    pdf_path: Path,
    output_dir: Path | None = None,
    lang_in: str = "en",
    lang_out: str = "zh",
    model_config: TranslationModelSchema | None = None,
    **kwargs,
) -> dict:
    """Translate a PDF file asynchronously with minimal configuration.

    This is a convenience function for simple use cases. For more control,
    use create_translation_job() and run_translation_job().

    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory for output files (uses config.json default if not provided)
        lang_in: Source language code
        lang_out: Target language code
        model_config: Translation model configuration
        **kwargs: Additional options to override config.json defaults

    Returns:
        Dictionary with translation results
    """
    # Create config data
    config_data = {
        "input_file": pdf_path,
        "lang_in": lang_in,
        "lang_out": lang_out,
    }
    config_data.update(kwargs)

    # Create config schema first to get the UUID
    config_schema = TranslationConfigSchema(**config_data)

    # Create UUID-based directory inside cache_folder for output
    job_output_dir = CACHE_FOLDER / str(config_schema.job_id)
    job_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created job output directory: {job_output_dir}")

    # Set output_dir to the UUID folder (override if not explicitly provided)
    if output_dir:
        config_data["output_dir"] = output_dir
    else:
        config_data["output_dir"] = job_output_dir

    # Recreate config schema with updated output_dir
    config_schema = TranslationConfigSchema(**config_data)

    # Execute translation directly
    return await _execute_translation(
        job_id=config_schema.job_id,
        config_schema=config_schema,
        model_config=model_config or TranslationModelSchema(),
    )
