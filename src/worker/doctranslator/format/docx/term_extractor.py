"""Automatic term/glossary extraction for DOCX units.

Ported from the PDF pipeline's AutomaticTermExtractor
(src/worker/doctranslator/format/pdf/document_il/midend/automatic_term_extractor.py),
adapted to operate on plain TranslatableUnit text instead of PdfParagraph.
Batches units by the same token/paragraph-count caps used everywhere else
and asks the LLM for {"src":..., "tgt":...} term pairs relative to the
target language. Extracted pairs are returned to the caller (not persisted
here) -- persistence into the shared domain glossary in GCS only happens
after the whole translation job completes successfully (see
GlossaryService.merge_new_terms_into_domain_glossary).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import tiktoken

from src.config.constants import settings
from src.worker.doctranslator.batching import compute_batch_plan
from src.worker.doctranslator.batching import log_batch_plan
from src.worker.doctranslator.format.docx.units import TranslatableUnit
from src.worker.doctranslator.format.pdf.translation_config import get_token_multiplier
from src.worker.doctranslator.translator.schemas import TermExtractionResponse
from src.worker.doctranslator.translator.translator import BaseTranslator

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """
You are an expert multilingual terminologist. Extract key terms from the text and translate them into {target_language}.

### Extraction Rules
1. Include only: named entities (people, orgs, locations, theorem/algorithm names, dates) and domain-specific nouns/noun phrases essential to meaning.
2. No full sentences. Ignore function words.
3. Use minimal noun phrases (<=5 words unless a named entity). No generic academic nouns (e.g., model, case, property) unless part of a standard term.
4. Extract each term once. Keep order of first appearance.

### Translation Rules
1. Translate each term into {target_language}.
2. If in the reference glossary, use its translation exactly.
3. Keep proper names in original language unless a well-known translation exists.
4. Ensure consistent translations.

{reference_glossary_section}

### Output Format
- Return ONLY a valid JSON array.
- Each element: {{"src": "...", "tgt": "..."}}.
- No comments, no backticks, no extra text.
- If no terms: [].

Input Text:
```
{text_to_process}
```

Return JSON ONLY. NO OTHER TEXT.
Result:
"""


def _clean_json_output(llm_output: str) -> str:
    text = llm_output.strip()
    for prefix in ("<json>", "```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    for suffix in ("</json>", "```"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip()


def _parse_terms(llm_output: str) -> list[tuple[str, str]]:
    cleaned = _clean_json_output(llm_output or "")
    if not cleaned:
        return []
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Term extraction JSON parse failed; skipping batch")
        return []

    if isinstance(data, dict):
        for key in ("terms", "items", "results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        return []

    pairs: list[tuple[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src", "")).strip()
        tgt = str(item.get("tgt", "")).strip()
        if src and tgt and len(src) < 100:
            pairs.append((src, tgt))
    return pairs


class DocxTermExtractor:
    """Extract candidate glossary terms from DOCX translatable units."""

    def __init__(self, translate_engine: BaseTranslator, target_language: str):
        self.translate_engine = translate_engine
        self.target_language = target_language
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = None
        # B1 (docs/plan.md batch-sizing tuning): mirror the PDF pipeline's
        # CJK-aware paragraph-count scaling (see
        # DocxParagraphTranslator.__init__ / translation_config.py's
        # get_token_multiplier) so term-extraction batches shrink the same
        # way for CJK language pairs instead of using a flat cap.
        lang_in = str(getattr(translate_engine, "lang_in", "") or "")
        self._token_multiplier = max(get_token_multiplier(lang_in, target_language), 0.1)
        self._last_batch_plan = None

    def _calc_token_count(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, len(text) // 4)
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return max(1, len(text) // 4)

    def _batch_units(self, units: list[TranslatableUnit]) -> list[list[TranslatableUnit]]:
        # Adaptive sizing: the flat 80000-token cap collapsed a 414-unit
        # document into a single batch with max_workers=1, serialising ~107s
        # of work that the 12-worker pool could have absorbed in one wave.
        total_payload = sum(self._calc_token_count(u.text) for u in units)
        plan = compute_batch_plan(
            total_payload,
            settings.TERM_EXTRACTION_POOL_MAX_WORKERS,
            base_max_tokens=settings.LLM_TERM_EXTRACTION_BATCH_MAX_TOKENS,
            base_max_items=settings.LLM_TERM_EXTRACTION_BATCH_MAX_PARAGRAPHS,
            token_multiplier=self._token_multiplier,
        )
        self._last_batch_plan = plan
        max_tokens = plan.max_tokens
        max_paragraphs = plan.max_items
        batches: list[list[TranslatableUnit]] = []
        current: list[TranslatableUnit] = []
        current_tokens = 0
        for unit in units:
            token_count = self._calc_token_count(unit.text)
            current.append(unit)
            current_tokens += token_count
            if current_tokens > max_tokens or len(current) > max_paragraphs:
                batches.append(current)
                current = []
                current_tokens = 0
        if current:
            batches.append(current)
        return batches

    def _extract_batch(self, batch: list[TranslatableUnit]) -> list[tuple[str, str]]:
        text_to_process = "\n\n".join(unit.text for unit in batch)
        if not text_to_process.strip():
            return []
        prompt = _PROMPT_TEMPLATE.format(
            target_language=self.target_language,
            reference_glossary_section="",
            text_to_process=text_to_process,
        )
        try:
            output = self.translate_engine.llm_translate(
                prompt,
                response_schema=TermExtractionResponse,
            )
        except Exception:
            logger.warning("Term extraction LLM call failed for a batch", exc_info=True)
            return []
        return _parse_terms(output)

    def extract(self, units: list[TranslatableUnit]) -> list[tuple[str, str]]:
        """Extract candidate (source_term, target_term) pairs from all units.

        Batches run concurrently, bounded by TERM_EXTRACTION_POOL_MAX_WORKERS
        -- same concurrency model as the PDF pipeline's AutomaticTermExtractor
        (docs/architecture/pdf-vs-docx-translation-architecture.md). Previously
        this ran one batch at a time in a plain `for` loop even though the
        pool-size setting already existed and was simply unused here.
        """
        all_pairs: list[tuple[str, str]] = []
        batches = self._batch_units(units)
        if not batches:
            logger.info("DOCX term extraction: 0 batches (no units)")
            return []
        if self._last_batch_plan is not None:
            log_batch_plan("DocxTermExtraction", self._last_batch_plan, len(batches))
        max_workers = min(len(batches), max(1, int(settings.TERM_EXTRACTION_POOL_MAX_WORKERS)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._extract_batch, batch) for batch in batches]
            for future in as_completed(futures):
                all_pairs.extend(future.result())
        # Deduplicate by (src.lower(), tgt) while preserving first occurrence.
        seen: set[tuple[str, str]] = set()
        deduped: list[tuple[str, str]] = []
        for src, tgt in all_pairs:
            key = (src.strip().lower(), tgt.strip())
            if key in seen:
                continue
            seen.add(key)
            deduped.append((src, tgt))
        logger.info(
            f"DOCX term extraction completed. Units: {len(units)}, "
            f"Batches: {len(batches)}, max_workers={max_workers}, "
            f"Raw pairs: {len(all_pairs)}, Deduped pairs: {len(deduped)}"
        )
        return deduped
