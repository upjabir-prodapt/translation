"""IL-aware DLP helpers for DocTranslator translation flow."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.api.services.dlp_service import DlpProvider
from src.api.services.dlp_service import DlpService
from src.doctranslator.format.pdf.document_il import il_version_1

logger = logging.getLogger(__name__)

_DLP_TOKEN_RE = re.compile(r"__DLP_TOKEN_\d{4,}__")


@dataclass(slots=True)
class ILDlpApplyResult:
    """Result of applying DLP masking to IL paragraph text."""

    applied: bool
    chunk_count: int
    token_rows: list[dict]
    dlp_provider: DlpProvider | None


def _iter_paragraphs(docs: il_version_1.Document):
    """Yield IL paragraphs in deterministic page/order sequence.

    Chunk indexing for persisted DLP tokens follows this iterator order:
    page 0 paragraph 0..N, page 1 paragraph 0..N, etc.
    """
    for page in docs.page:
        yield from page.pdf_paragraph


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
            dlp_provider=dlp_service.select_provider(),
        )

    dlp_result = dlp_service.mask_chunks(
        job_id=job_id,
        chunks=chunks,
        source_language=source_language,
        token_counter_start=token_counter_start,
    )
    for paragraph, masked in zip(
        paragraph_refs, dlp_result.masked_chunks, strict=False
    ):
        paragraph.unicode = masked

    return ILDlpApplyResult(
        applied=True,
        chunk_count=len(chunks),
        token_rows=dlp_result.token_rows,
        dlp_provider=dlp_result.dlp_provider,
    )


def _replace_in_str(text: str, token_map: dict[str, str]) -> tuple[str, int]:
    """Replace all tokens in a string; return (result, count_replaced)."""
    count = 0
    for token, original in token_map.items():
        if token in text:
            count += text.count(token)
            text = text.replace(token, original)
    return text, count


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
        if isinstance(paragraph.unicode, str) and paragraph.unicode:
            restored, count = _replace_in_str(paragraph.unicode, token_map)
            paragraph.unicode = restored
            replacements += count

        for composition in paragraph.pdf_paragraph_composition:
            ssuc = composition.pdf_same_style_unicode_characters
            if ssuc is not None and isinstance(ssuc.unicode, str) and ssuc.unicode:
                restored, count = _replace_in_str(ssuc.unicode, token_map)
                ssuc.unicode = restored
                replacements += count

    return replacements


def strip_leaked_tokens(docs: il_version_1.Document) -> int:
    """Remove any DLP tokens that survived unmasking and log each as a critical error.

    Returns the number of leaked tokens stripped. A non-zero return means the
    unmasking step failed to restore that many placeholders — callers should treat
    this as a pipeline fault even though the output is now token-free.
    """
    leaked = 0

    def _strip(text: str) -> tuple[str, int]:
        found = _DLP_TOKEN_RE.findall(text)
        if not found:
            return text, 0
        for token in found:
            logger.critical(
                "DLP token leaked into translated output and was stripped: token=%r — "
                "the original sensitive value could not be restored. "
                "Check that dlp_token_rows is populated and the LLM did not alter the token.",
                token,
            )
        return _DLP_TOKEN_RE.sub("", text), len(found)

    for paragraph in _iter_paragraphs(docs):
        if isinstance(paragraph.unicode, str) and paragraph.unicode:
            cleaned, count = _strip(paragraph.unicode)
            paragraph.unicode = cleaned
            leaked += count

        for composition in paragraph.pdf_paragraph_composition:
            ssuc = composition.pdf_same_style_unicode_characters
            if ssuc is not None and isinstance(ssuc.unicode, str) and ssuc.unicode:
                cleaned, count = _strip(ssuc.unicode)
                ssuc.unicode = cleaned
                leaked += count

    return leaked
