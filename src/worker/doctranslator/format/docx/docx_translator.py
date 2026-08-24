"""Top-level native DOCX translation orchestration.

Wires together: unit extraction, DLP masking/unmasking, domain glossary
application, automatic term extraction, batch translation, and write-back
into the DOCX -- all using the shared services (DlpService, GlossaryService,
BaseTranslator subclasses) that the PDF pipeline also uses.

No PDF conversion, no LibreOffice subprocess, no ONNX layout/OCR models --
see docs/architecture/pdf-vs-docx-translation-architecture.md for the full
comparison and rationale.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from docx import Document

from src.worker.doctranslator.format.docx.paragraph_translator import (
    DocxParagraphTranslator,
)
from src.worker.doctranslator.format.docx.term_extractor import DocxTermExtractor
from src.worker.doctranslator.format.docx.units import extract_units
from src.worker.doctranslator.format.docx.units import write_translated_text
from src.worker.doctranslator.translator.translator import BaseTranslator
from src.worker.services.dlp_service import DlpProvider
from src.worker.services.dlp_service import DlpResult
from src.worker.services.dlp_service import DlpService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocxTranslationResult:
    """Result of translating one DOCX document."""

    output_path: Path
    source_text: str
    translated_text: str
    dlp_provider: DlpProvider | None
    dlp_token_rows: list[dict]
    extracted_terms: list[tuple[str, str]]
    dlp_result: DlpResult | None = None


def translate_docx(
    *,
    input_path: Path,
    output_path: Path,
    translator: BaseTranslator,
    lang_out: str,
    job_id: str,
    source_language: str,
    enable_dlp: bool = False,
    auto_extract_glossary: bool = True,
    extracted_terms: list[tuple[str, str]] | None = None,
    dlp_result: DlpResult | None = None,
) -> DocxTranslationResult:
    """Translate a DOCX document in place (writing to `output_path`).

    Returns a DocxTranslationResult with the source/translated text
    (for quality judging) and any auto-extracted glossary terms (for the
    caller to persist into the shared domain glossary only after the whole
    job/attempt succeeds).
    """
    document = Document(str(input_path))
    units = extract_units(document)
    logger.info(f"[docx] job={job_id} extracted {len(units)} translatable units")

    dlp_provider: DlpProvider | None = None
    dlp_token_rows: list[dict] = []
    if enable_dlp and units:
        if dlp_result is not None:
            for unit, masked_text in zip(units, dlp_result.masked_chunks, strict=False):
                unit.text = masked_text
            dlp_provider = dlp_result.dlp_provider
            dlp_token_rows = dlp_result.token_rows
            logger.info(
                f"[docx] job={job_id} reused DLP masking via {dlp_provider}: "
                f"{len(dlp_token_rows)} token(s) masked"
            )
        else:
            dlp_service = DlpService()
            dlp_res = dlp_service.mask_chunks(
                job_id=job_id,
                chunks=[unit.text for unit in units],
                source_language=source_language,
            )
            for unit, masked_text in zip(units, dlp_res.masked_chunks, strict=False):
                unit.text = masked_text
            dlp_provider = dlp_res.dlp_provider
            dlp_token_rows = dlp_res.token_rows
            dlp_result = dlp_res
            logger.info(
                f"[docx] job={job_id} DLP masking applied via {dlp_provider}: "
                f"{len(dlp_token_rows)} token(s) masked"
            )

    # Automatic term extraction and paragraph translation are run
    # concurrently on first attempt; on retry attempts, previously extracted
    # candidate terms can be passed in directly to avoid duplicate LLM calls.
    paragraph_translator = DocxParagraphTranslator(translator, lang_out)

    if extracted_terms is not None:
        logger.info(
            f"[docx] job={job_id} reusing {len(extracted_terms)} previously extracted term(s)"
        )
        translations = paragraph_translator.translate_all(units)
    elif auto_extract_glossary and units:
        term_extractor = DocxTermExtractor(translator, lang_out)
        with ThreadPoolExecutor(max_workers=2) as executor:
            term_future = executor.submit(term_extractor.extract, units)
            translate_future = executor.submit(paragraph_translator.translate_all, units)
            translations = translate_future.result()
            extracted_terms = term_future.result()
        logger.info(
            f"[docx] job={job_id} auto-extracted {len(extracted_terms)} candidate term(s)"
        )
    else:
        extracted_terms = []
        translations = paragraph_translator.translate_all(units)

    source_parts: list[str] = []
    translated_parts: list[str] = []
    for unit in units:
        translated_text = translations.get(unit.unit_id, unit.text)
        source_parts.append(unit.text)
        translated_parts.append(translated_text)

    if enable_dlp and dlp_token_rows:
        token_map = {
            str(row.get("token")): str(row.get("original_value"))
            for row in dlp_token_rows
            if row.get("token") and row.get("original_value")
        }
        for i, _unit in enumerate(units):
            translated_text = translated_parts[i]
            for token, original in token_map.items():
                if token in translated_text:
                    translated_text = translated_text.replace(token, original)
            translated_parts[i] = translated_text

    for unit, translated_text in zip(units, translated_parts, strict=False):
        write_translated_text(unit, translated_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output_path))
    logger.info(f"[docx] job={job_id} saved translated document to {output_path}")

    return DocxTranslationResult(
        output_path=output_path,
        source_text="\n".join(source_parts),
        translated_text="\n".join(translated_parts),
        dlp_provider=dlp_provider,
        dlp_token_rows=dlp_token_rows,
        extracted_terms=extracted_terms,
        dlp_result=dlp_result,
    )
