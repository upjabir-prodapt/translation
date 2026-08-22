"""End-to-end translation of a single DOCX document."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from src.worker.doctranslator.glossary import Glossary
from src.worker.doctranslator.translator.base import BaseTranslator
from src.worker.docxtranslator.package import DocxPackage
from src.worker.docxtranslator.segment_translator import DocxSegmentTranslator
from src.worker.docxtranslator.segments import Segment
from src.worker.docxtranslator.segments import collect_segments
from src.worker.docxtranslator.segments import extract_text
from src.worker.services.dlp_service import DlpProvider
from src.worker.services.dlp_service import DlpService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocxTranslationResult:
    """Outcome of translating one .docx file into another .docx file."""

    output_path: Path
    source_text: str
    translated_text: str
    segment_count: int
    translated_segment_count: int
    batch_token_counts: list[int] = field(default_factory=list)
    dlp_token_rows: list[dict] = field(default_factory=list)
    dlp_provider: DlpProvider | None = None
    failed_batches: int = 0
    untranslated_segments: int = 0
    tag_fallback_segments: int = 0

    @property
    def batch_count(self) -> int:
        return len(self.batch_token_counts)


class DocxTranslationService:
    """Translate a Word document in place, preserving its OOXML structure.

    The document is never converted to another format. Only ``w:t`` text nodes
    change, so tables remain editable Word tables, and styles, numbering,
    headers, footers, footnotes, and text boxes are carried through as-is.
    """

    def __init__(self, dlp_service: DlpService | None = None) -> None:
        self.dlp_service = dlp_service or DlpService()

    async def translate_document(
        self,
        *,
        input_path: Path,
        output_path: Path,
        translate_engine: BaseTranslator,
        lang_in: str,
        lang_out: str,
        domain: str | None,
        job_id: str,
        glossaries: list[Glossary] | None = None,
        enable_dlp: bool = False,
        max_concurrency: int | None = None,
    ) -> DocxTranslationResult:
        """Translate ``input_path`` and write the result to ``output_path``."""
        package = DocxPackage.open(input_path)
        segments = collect_segments(package.text_parts())
        source_text = extract_text(segments)
        logger.info(
            f"job_id={job_id} DOCX parsed: parts={len(package.text_part_names())} "
            f"paragraphs={len(segments)} chars={len(source_text)}"
        )

        token_rows: list[dict] = []
        dlp_provider: DlpProvider | None = None
        if enable_dlp:
            token_rows, dlp_provider = self._mask_segments(
                segments=segments, job_id=job_id, source_language=lang_in
            )

        translator = DocxSegmentTranslator(
            translate_engine=translate_engine,
            lang_out=lang_out,
            domain=domain,
            glossaries=glossaries,
            max_concurrency=max_concurrency,
        )
        outcome = await translator.translate(segments)

        token_map = self._build_token_map(token_rows)
        applied = self._apply_translations(segments, outcome.translations, token_map)
        if token_rows:
            self._restore_untranslated(segments, outcome.translations, token_map)

        package.save(output_path)
        translated_text = extract_text(segments)

        result = DocxTranslationResult(
            output_path=output_path,
            source_text=source_text,
            translated_text=translated_text,
            segment_count=len(segments),
            translated_segment_count=applied,
            batch_token_counts=[batch.token_count for batch in outcome.batches],
            dlp_token_rows=token_rows,
            dlp_provider=dlp_provider,
            failed_batches=outcome.failed_batches,
            untranslated_segments=outcome.untranslated_segments,
            tag_fallback_segments=outcome.tag_fallback_segments,
        )
        logger.info(
            f"job_id={job_id} DOCX translated: applied={applied}/{len(segments)} "
            f"batches={result.batch_count} failed_batches={result.failed_batches} "
            f"untranslated={result.untranslated_segments} "
            f"tag_fallbacks={result.tag_fallback_segments}"
        )
        return result

    def _mask_segments(
        self,
        *,
        segments: list[Segment],
        job_id: str,
        source_language: str,
    ) -> tuple[list[dict], DlpProvider | None]:
        """Mask PII in every run group before anything reaches the model.

        Masking is per run group, matching the unit that gets translated, so a
        token never straddles a formatting boundary.
        """
        groups = [group for segment in segments for group in segment.groups]
        chunks = [group.text for group in groups]
        if not any(chunk.strip() for chunk in chunks):
            return [], self.dlp_service.select_provider()

        dlp_result = self.dlp_service.mask_chunks(
            job_id=job_id,
            chunks=chunks,
            source_language=source_language,
        )
        for group, masked in zip(groups, dlp_result.masked_chunks, strict=True):
            if masked != group.text:
                group.write(masked)
        logger.info(
            f"job_id={job_id} DOCX DLP masking applied: provider={dlp_result.dlp_provider} "
            f"groups={len(groups)} tokens={len(dlp_result.token_rows)}"
        )
        return dlp_result.token_rows, dlp_result.dlp_provider

    @staticmethod
    def _build_token_map(token_rows: list[dict]) -> dict[str, str]:
        return {
            str(row.get("token")): str(row.get("original_value"))
            for row in token_rows
            if row.get("token") and row.get("original_value")
        }

    @staticmethod
    def _unmask(text: str, token_map: dict[str, str]) -> str:
        for token, original in token_map.items():
            if token in text:
                text = text.replace(token, original)
        return text

    def _apply_translations(
        self,
        segments: list[Segment],
        translations: dict[int, list[str]],
        token_map: dict[str, str],
    ) -> int:
        """Write translations back, restoring masked values as we go."""
        applied = 0
        for segment in segments:
            group_texts = translations.get(segment.index)
            if group_texts is None:
                continue
            if token_map:
                group_texts = [self._unmask(text, token_map) for text in group_texts]
            try:
                segment.write(group_texts)
            except ValueError:
                logger.warning(
                    f"Group count mismatch for DOCX paragraph {segment.index}; "
                    "writing translation into the first run"
                )
                segment.write_joined("".join(group_texts))
            applied += 1
        return applied

    def _restore_untranslated(
        self,
        segments: list[Segment],
        translations: dict[int, list[str]],
        token_map: dict[str, str],
    ) -> None:
        """Unmask paragraphs the model never saw, so no token leaks to output."""
        if not token_map:
            return
        for segment in segments:
            if segment.index in translations:
                continue
            for group in segment.groups:
                restored = self._unmask(group.text, token_map)
                if restored != group.text:
                    group.write(restored)
