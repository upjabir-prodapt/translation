"""IL-aware DLP helpers for DocTranslator translation flow."""

from __future__ import annotations

from dataclasses import dataclass

from src.doctranslator.format.pdf.document_il import il_version_1
from src.api.services.dlp_service import DlpService


@dataclass(slots=True)
class ILDlpApplyResult:
    """Result of applying DLP masking to IL paragraph text."""

    applied: bool
    chunk_count: int
    token_rows: list[dict]
    dlp_provider: str | None


def _iter_paragraphs(docs: il_version_1.Document):
    """Yield IL paragraphs in deterministic page/order sequence.

    Chunk indexing for persisted DLP tokens follows this iterator order:
    page 0 paragraph 0..N, page 1 paragraph 0..N, etc.
    """
    for page in docs.page:
        for paragraph in page.pdf_paragraph:
            yield paragraph


def apply_dlp_to_document(
    *,
    docs: il_version_1.Document,
    dlp_service: DlpService,
    job_id: str,
    source_language: str,
    token_counter_start: int = 0,
) -> ILDlpApplyResult:
    """Mask IL paragraph text in-place using existing DLP service."""
    paragraph_refs: list[il_version_1.PdfParagraph] = []
    chunks: list[str] = []
    for paragraph in _iter_paragraphs(docs):
        text = paragraph.unicode
        if not isinstance(text, str):
            continue
        if not text.strip():
            continue
        paragraph_refs.append(paragraph)
        chunks.append(text)

    if not chunks:
        return ILDlpApplyResult(
            applied=False,
            chunk_count=0,
            token_rows=[],
            dlp_provider=dlp_service.select_provider(source_language),
        )

    dlp_result = dlp_service.mask_chunks(
        job_id=job_id,
        chunks=chunks,
        source_language=source_language,
        token_counter_start=token_counter_start,
    )
    for paragraph, masked in zip(paragraph_refs, dlp_result.masked_chunks, strict=False):
        paragraph.unicode = masked

    return ILDlpApplyResult(
        applied=True,
        chunk_count=len(chunks),
        token_rows=dlp_result.token_rows,
        dlp_provider=dlp_result.dlp_provider,
    )


def unmask_document_with_tokens(
    *,
    docs: il_version_1.Document,
    token_rows: list[dict],
) -> int:
    """Restore original values by replacing DLP tokens in paragraph text."""
    if not token_rows:
        return 0
    token_map = {
        str(row.get("token")): str(row.get("original_value"))
        for row in token_rows
        if row.get("token") and row.get("original_value")
    }
    if not token_map:
        return 0

    replacements = 0
    for paragraph in _iter_paragraphs(docs):
        text = paragraph.unicode
        if not isinstance(text, str) or not text:
            continue
        restored = text
        for token, original in token_map.items():
            if token in restored:
                replacements += restored.count(token)
                restored = restored.replace(token, original)
        paragraph.unicode = restored
    return replacements

