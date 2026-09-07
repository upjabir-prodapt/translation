"""Language detection service."""

from collections import Counter
from pathlib import Path
from typing import Any

from src.worker.services.language_detection_core import count_unit_languages
from src.worker.services.language_detection_core import detect_language_of_joined_text
from src.worker.services.language_detection_core import resolve_dominant_language
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
        winner, _ = self.detect_with_distribution(
            input_path, is_docx=is_docx, is_txt=is_txt
        )
        return winner

    def detect_with_distribution(
        self, input_path: Path, *, is_docx: bool = False, is_txt: bool = False
    ) -> tuple[str, "Counter[str]"]:
        """Detect source language and return the full per-language distribution.

        implementation_plan.md Phase C.5.1: the full char-weighted
        `Counter` (not just the winner) is returned so callers can
        persist the complete language distribution for a mixed-language
        document instead of discarding everything but the winner.
        """
        if is_txt:
            return self._detect_txt_with_distribution(input_path)
        if is_docx:
            return self._detect_docx_with_distribution(input_path)
        return self._processor.detect_source_language_with_distribution(input_path)

    def _detect_units(
        self, texts: list[str], *, source_label: str
    ) -> tuple[str, "Counter[str]"]:
        """Detect the dominant language across a list of text units.

        implementation_plan.md Phase C.2.1: brings DOCX/TXT detection to
        parity with the PDF pipeline's per-block, confidence-floored,
        char-weighted aggregation. Previously this ran a single
        `detect_langs()` call over the whole concatenated document with
        **no confidence floor at all** -- the root cause of the EC-09
        defect (short/ambiguous text like "Information" or "OK" got a
        confident-looking but unreliable guess accepted outright).

        Returns `(dominant_language, full_language_counter)` -- see
        Phase C.5.1.

        The old "more than N distinct languages" guard is gone; a
        document is now rejected only when no single language dominates
        its text, decided by the shared `resolve_dominant_language()` so
        DOCX/TXT and PDF apply one rule (see language_detection_core.py).
        """
        normalized_units = [" ".join(text.split()) for text in texts]
        language_counter, _ = count_unit_languages(
            normalized_units, max_chars=self._processor.MAX_DETECTION_CHARS
        )
        if not language_counter:
            # Every unit was too short or too name-heavy to trust on its
            # own -- common for form-like DOCX and bullet-only decks.
            # Retry across the concatenated text before giving up.
            language_counter = detect_language_of_joined_text(normalized_units)

        # Raises MixedLanguageError for a genuinely multilingual
        # document, or ValueError asking the caller to choose the source
        # language explicitly when nothing detectable was found (C.2.3 /
        # EC-09).
        winner = resolve_dominant_language(language_counter, source_label=source_label)
        return winner, language_counter

    def _detect_docx(self, input_path: Path) -> str:
        winner, _ = self._detect_docx_with_distribution(input_path)
        return winner

    def _detect_docx_with_distribution(
        self, input_path: Path
    ) -> tuple[str, "Counter[str]"]:
        from docx import Document as DocxDocument

        from src.worker.doctranslator.format.docx.units import extract_units

        document = DocxDocument(str(input_path))
        units, _note_parts = extract_units(document)
        texts = [unit.text for unit in units]
        return self._detect_units(texts, source_label="DOCX text")

    def _detect_txt(self, input_path: Path) -> str:
        winner, _ = self._detect_txt_with_distribution(input_path)
        return winner

    def _detect_txt_with_distribution(
        self, input_path: Path
    ) -> tuple[str, "Counter[str]"]:
        text = input_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines() or [text]
        return self._detect_units(lines, source_label="TXT text")
