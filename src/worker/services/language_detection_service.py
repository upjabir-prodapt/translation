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

    def detect(
        self, input_path: Path, *, is_docx: bool = False, is_txt: bool = False
    ) -> str:
        """Detect the source language of a PDF, DOCX, or plain-text file.

        `is_docx=True` extracts text via python-docx instead of the default
        pymupdf-based PDF text extraction. `is_txt=True` reads the file's raw
        text content directly.
        """
        if is_txt:
            return self._detect_txt(input_path)
        if is_docx:
            return self._detect_docx(input_path)
        return self._processor.detect_source_language(input_path)

    def _detect_text(self, text: str, *, source_label: str) -> str:
        if not text.strip():
            raise ValueError(f"Unable to detect source language from {source_label}")

        from langdetect import detect_langs

        candidates = detect_langs(text[: self._processor.MAX_DETECTION_CHARS])
        if not candidates:
            raise ValueError(f"Unable to detect source language from {source_label}")
        best_match = candidates[0]
        return self._processor._normalize_detected_language(best_match.lang)  # noqa: SLF001

    def _detect_docx(self, input_path: Path) -> str:
        from docx import Document as open_docx

        from src.worker.doctranslator.format.docx.units import extract_units

        document = open_docx(str(input_path))
        units = extract_units(document)
        text = "\n".join(unit.text for unit in units)
        return self._detect_text(text, source_label="DOCX text")

    def _detect_txt(self, input_path: Path) -> str:
        text = input_path.read_text(encoding="utf-8", errors="replace")
        return self._detect_text(text, source_label="TXT text")
