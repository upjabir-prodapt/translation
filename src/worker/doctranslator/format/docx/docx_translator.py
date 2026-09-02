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

from src.config.constants import settings
from src.worker.doctranslator.format.docx.paragraph_translator import (
    DocxParagraphTranslator,
)
from src.worker.doctranslator.format.docx.term_extractor import DocxTermExtractor
from src.worker.doctranslator.format.docx.units import extract_units
from src.worker.doctranslator.format.docx.units import flush_note_parts
from src.worker.doctranslator.format.docx.units import write_translated_text
from src.worker.doctranslator.translator.translator import BaseTranslator
from src.worker.services.dlp_service import DlpProvider
from src.worker.services.dlp_service import DlpResult
from src.worker.services.dlp_service import DlpService
from src.worker.services.dlp_tokens import build_token_map
from src.worker.services.dlp_tokens import restore_tokens
from src.worker.services.dlp_tokens import strip_leaked_tokens_from_text

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
    domain: str | None = None,
    enable_dlp: bool = True,
    auto_extract_glossary: bool = True,
    extracted_terms: list[tuple[str, str]] | None = None,
    glossary_terms: list[tuple[str, str]] | None = None,
    dlp_result: DlpResult | None = None,
) -> DocxTranslationResult:
    """Translate a DOCX document in place (writing to `output_path`).

    `glossary_terms` are the business's approved source->target pairs for
    this domain and target language (a pair whose target equals its source
    is a do-not-translate entry). They are pinned into every batch prompt
    ahead of any auto-extracted term, so an approved rendering always wins
    over one the extractor guessed. Until UAT EC-01/EC-08 the domain
    glossary was write-only on this path -- terms were merged into it after
    a job and never read back for the next one.

    Returns a DocxTranslationResult with the source/translated text
    (for quality judging) and any auto-extracted glossary terms (for the
    caller to persist into the shared domain glossary only after the whole
    job/attempt succeeds).
    """
    document = Document(str(input_path))
    units, note_parts = extract_units(document)
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

    # Term extraction feeds translation, so when it runs it must finish
    # first: its output is pinned into every batch prompt as binding
    # terminology, which is what stops the same defined term being rendered
    # two different ways in two different batches of one long document (UAT
    # EC-08, D-03). On a retry attempt the caller passes the previously
    # extracted terms straight in, so the extraction call is made at most
    # once per job.
    #
    # `TERM_CONSISTENCY_PREPASS_ENABLED=false` restores the older behaviour
    # (extraction and translation concurrent, terms used only for the
    # persisted glossary) for anyone who would rather have the latency back.
    prepass = bool(getattr(settings, "TERM_CONSISTENCY_PREPASS_ENABLED", True))
    paragraph_translator: DocxParagraphTranslator | None = None
    translations: dict[int, str] = {}

    if extracted_terms is not None:
        logger.info(
            f"[docx] job={job_id} reusing {len(extracted_terms)} previously extracted term(s)"
        )
    elif auto_extract_glossary and units:
        term_extractor = DocxTermExtractor(translator, lang_out, domain=domain)
        if prepass:
            extracted_terms = term_extractor.extract(units)
            logger.info(
                f"[docx] job={job_id} auto-extracted {len(extracted_terms)} "
                "candidate term(s) before translation; pinned as binding "
                "terminology for every batch"
            )
        else:
            paragraph_translator = DocxParagraphTranslator(
                translator, lang_out, domain=domain
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                term_future = executor.submit(term_extractor.extract, units)
                translate_future = executor.submit(
                    paragraph_translator.translate_all, units
                )
                translations = translate_future.result()
                extracted_terms = term_future.result()
            logger.info(
                f"[docx] job={job_id} auto-extracted {len(extracted_terms)} candidate term(s)"
            )
    else:
        extracted_terms = []

    if paragraph_translator is None:
        # Approved glossary terms first: `build_binding_terminology_block`
        # keeps the first rendering it sees for a term, so anything the
        # business has agreed beats anything the extractor inferred. An
        # approved entry whose target equals its source is a
        # do-not-translate term; the same shape coming back from automatic
        # extraction is only a missing translation and must not protect an
        # ordinary word (UAT S-03).
        approved = list(glossary_terms or [])
        do_not_translate = [src for src, tgt in approved if src == tgt]
        pinned_terms = approved + list(extracted_terms or [])
        paragraph_translator = DocxParagraphTranslator(
            translator,
            lang_out,
            domain=domain,
            glossary_terms=pinned_terms,
            do_not_translate=do_not_translate,
        )
        translations = paragraph_translator.translate_all(units)

    source_parts: list[str] = []
    translated_parts: list[str] = []
    for unit in units:
        translated_text = translations.get(unit.unit_id, unit.text)
        source_parts.append(unit.text)
        translated_parts.append(translated_text)

    if enable_dlp and dlp_token_rows:
        token_map = build_token_map(dlp_token_rows)
        leaked_tokens: list[str] = []
        for i, _unit in enumerate(units):
            translated_text, _ = restore_tokens(translated_parts[i], token_map)
            # Parity with the PDF pipeline: a token the model mangled beyond
            # even lenient recovery must never be shipped inside the document
            # (it used to appear welded to the neighbouring word). Strip it and
            # report it as a pipeline fault -- the sensitive value is lost.
            translated_text, unit_leaked = strip_leaked_tokens_from_text(
                translated_text
            )
            leaked_tokens.extend(unit_leaked)
            translated_parts[i] = translated_text
        if leaked_tokens:
            logger.critical(
                f"[docx] job={job_id} CRITICAL: {len(leaked_tokens)} DLP token(s) "
                f"could not be restored and were stripped from the output "
                f"({', '.join(sorted(set(leaked_tokens))[:10])}). "
                "Sensitive data may be unrecoverable — review the translation pipeline."
            )

    for unit, translated_text in zip(units, translated_parts, strict=False):
        write_translated_text(unit, translated_text)
    # D.2.1: footnotes.xml/endnotes.xml are not modeled by python-docx --
    # `write_translated_text()` above only mutated a detached lxml tree for
    # any footnote/endnote units, so it must be re-serialized back into the
    # part's blob before `document.save()` or those edits are silently lost.
    flush_note_parts(note_parts)

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
