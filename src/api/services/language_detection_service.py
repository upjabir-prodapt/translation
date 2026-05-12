"""Language detection service."""

from pathlib import Path
from typing import Any

from src.api.services.processor_service import JobProcessor
from src.api.services.progress_tracker import ProgressTracker


class _NoopUpdater:
    async def update_job(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        import asyncio

        await asyncio.sleep(0)


class LanguageDetectionService:
    """Detect source language using langdetect-backed processor."""

    def __init__(self):
        tracker = ProgressTracker(updater=_NoopUpdater(), job_id="detector")
        self._processor = JobProcessor(progress_tracker=tracker)

    def detect(self, input_pdf: Path) -> str:
        return self._processor.detect_source_language(input_pdf)
