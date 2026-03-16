"""Job processing service for the worker."""

import asyncio
from pathlib import Path
from typing import Any

from babeldoc import async_translate
from babeldoc.format.pdf.translation_config import TranslationConfig
from config.logging import logger
from worker.services.progress import ProgressTracker


class JobProcessor:
    """Processes translation jobs using BabelDOC with progress tracking."""

    # Progress mapping constants
    PROGRESS_START = 0.2
    PROGRESS_TRANSLATION_START = 0.3
    PROGRESS_TRANSLATION_RANGE = 0.6  # 0.2 to 0.8
    PROGRESS_FINALIZE = 0.8
    PROGRESS_COMPLETE = 0.9

    def __init__(self, progress_tracker: ProgressTracker):
        """Initialize processor with progress tracker."""
        self.progress_tracker = progress_tracker
        self._translations: list[dict[str, Any]] = []

    async def translate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Process a translation job with progress tracking."""
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build translation config
        translation_config = self._build_translation_config(config, output_dir)

        # Reset state
        self._translations = []

        # Start processing
        await self.progress_tracker.update(
            self.PROGRESS_START, "Initializing translation"
        )
        logger.info(f"Starting translation: {config['lang_in']} → {config['lang_out']}")

        # Process translation events
        try:
            async for event in async_translate(translation_config):
                result = await self._handle_translation_event(event, config)
                if result is not None:
                    return result

            # No finish event received
            raise RuntimeError("Translation completed without finish event")

        except Exception as e:
            logger.exception("Translation processing failed")
            raise

    def _build_translation_config(
        self, config: dict[str, Any], output_dir: Path
    ) -> TranslationConfig:
        """Build TranslationConfig from job config."""
        # Extract known parameters
        known_params = {"input_file", "output_dir", "lang_in", "lang_out"}
        extra_params = {k: v for k, v in config.items() if k not in known_params}

        return TranslationConfig(
            input_file=Path(config["input_file"]),
            output_dir=output_dir,
            lang_in=config["lang_in"],
            lang_out=config["lang_out"],
            **extra_params,
        )

    async def _handle_translation_event(
        self, event: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle a single translation event. Returns result dict if finished, None otherwise."""
        event_type = event.get("type")

        if event_type == "progress_start":
            await self.progress_tracker.update(
                self.PROGRESS_TRANSLATION_START, "Starting translation"
            )

        elif event_type == "progress_update":
            await self._handle_progress_update(event)

        elif event_type == "progress_end":
            await self.progress_tracker.update(
                self.PROGRESS_FINALIZE, "Finalizing output"
            )

        elif event_type == "finish":
            return await self._handle_finish_event(event)

        elif event_type == "error":
            error_msg = event.get("error", "Unknown error")
            logger.error(f"Translation error: {error_msg}")
            raise RuntimeError(f"Translation failed: {error_msg}")

        return None

    async def _handle_progress_update(self, event: dict[str, Any]) -> None:
        """Handle progress update event."""
        # Map 0-100 progress to our range
        overall_progress = event.get("overall_progress", 0) / 100.0
        mapped_progress = self.PROGRESS_START + (
            overall_progress * self.PROGRESS_TRANSLATION_RANGE
        )

        stage = event.get("stage", "Processing")
        await self.progress_tracker.update(
            mapped_progress, f"{stage} ({event.get('overall_progress', 0):.0f}%)"
        )

        # Collect translation data for analytics
        if "translation" in event:
            self._translations.append(
                {
                    "original_text": event["translation"].get("original", ""),
                    "translated_text": event["translation"].get("translated", ""),
                    "engine": "openai",
                }
            )

    async def _handle_finish_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Handle finish event and return results."""
        await self.progress_tracker.update(
            self.PROGRESS_COMPLETE, "Translation complete"
        )

        result = event.get("translate_result", {})

        # Extract output file paths
        output_files = {}
        for file_type in ["mono_pdf", "dual_pdf", "no_watermark_mono_pdf"]:
            file_path = result.get(file_type)
            if file_path and Path(file_path).exists():
                output_files[f"{file_type}_path"] = Path(file_path)

        page_count = result.get("page_count", 0)
        logger.info(
            f"Translation finished: {page_count} pages, {len(output_files)} output files"
        )

        return {
            **output_files,
            "page_count": page_count,
            "translations": self._translations,
        }

    async def translate_with_timeout(
        self, config: dict[str, Any], timeout_seconds: int = 3600
    ) -> dict[str, Any]:
        """Process translation with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.translate(config), timeout=timeout_seconds
            )
        except TimeoutError:
            logger.error(f"Translation timed out after {timeout_seconds}s")
            raise RuntimeError(
                f"Translation timed out after {timeout_seconds} seconds"
            ) from None
