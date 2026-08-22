"""Language detection service."""

from pathlib import Path
from typing import Any

from src.worker.services.processor_service import JobProcessor
from src.worker.services.progress_tracker import ProgressTracker


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

    def detect_docx(self, input_docx: Path) -> str:
        """Detect the source language of a .docx from its OOXML paragraph text."""
        from src.worker.services.docx_processor_service import extract_docx_text

        text = extract_docx_text(
            input_docx, max_chars=self._processor.MAX_DETECTION_CHARS
        )
        return self._processor.detect_source_language_from_text(text)
