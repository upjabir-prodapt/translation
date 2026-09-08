"""Language detection service."""

from collections import Counter
from pathlib import Path
from typing import Any

from src.worker.services.language_detection_core import aggregate_languages
from src.worker.services.language_detection_core import (
    build_unclassifiable_text_message,
)
from src.worker.services.language_detection_core import detect_language_for_text
from src.worker.services.language_detection_core import is_detectable_text
from src.worker.services.processor_service import JobProcessor
from src.worker.services.progress_tracker import ProgressTracker

# MAX_DISTINCT_LANGUAGES_PER_DOCUMENT (a "more than 10 distinct languages"
# rejection, the DOCX/TXT counterpart of the PDF pipeline's per-page guard)
# was removed when mixed-language documents became translatable. See the
# note in `processor_service.JobProcessor`: a distinct-language count
# measures detection noise, not multilingualism. The replacement is the
# share-based supported-language coverage gate in
# `PipelineOrchestrator._assert_language_supported`.


class _NoopUpdater:
    async def update_job(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        import asyncio

        await asyncio.sleep(0)


class LanguageDetectionService:
    """Detect source language using the lingua-backed processor."""

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
        """
        language_counter: Counter[str] = Counter()
        # Counts every unit that was worth classifying, whether or not a
        # language came back. This is both the detection budget (matching
        # the PDF path, which has always charged the budget for candidate
        # rather than classified characters -- an undetermined block costs
        # the same work as a determined one) and the signal that separates
        # "nothing to classify" from "nothing classifiable" below.
        candidate_chars = 0
        max_chars = self._processor.MAX_DETECTION_CHARS
        for text in texts:
            normalized = " ".join(text.split())
            if not is_detectable_text(normalized):
                continue
            candidate_chars += len(normalized)
            detected = detect_language_for_text(normalized)
            if detected is not None:
                language_counter[detected] += len(normalized)
            if candidate_chars >= max_chars:
                break

        winner = aggregate_languages(language_counter)
        if winner is None:
            # Mirrors the PDF pipeline's split (see
            # `JobProcessor.detect_source_language_with_distribution`):
            # `candidate_chars` counts text that passed
            # `is_detectable_text`, so "no text worth classifying" and
            # "text present but never classifiable" are reported as the
            # two different problems they are. Previously both produced
            # the same message, which under lingua's relative-distance
            # floor would misdescribe a text-rich but ambiguous document.
            if candidate_chars == 0:
                raise ValueError(
                    f"Unable to detect a source language: no sufficiently "
                    f"long text was found in {source_label}. Please supply a "
                    "document containing readable text."
                )
            raise ValueError(build_unclassifiable_text_message(subject=source_label))
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
