"""Automatic term/glossary extraction for DOCX units.

Ported from the PDF pipeline's AutomaticTermExtractor
(src/worker/doctranslator/format/pdf/document_il/midend/automatic_term_extractor.py),
adapted to operate on plain TranslatableUnit text instead of PdfParagraph.
Batches units by the same token/paragraph-count caps used everywhere else
and asks the LLM for {"src":..., "tgt":..., "src_lang":...} term triples
relative to the target language. Extracted terms are returned to the
caller (not persisted here) -- persistence into the shared domain glossary
in GCS only happens after the whole translation job completes successfully
(see GlossaryService.merge_new_terms_into_domain_glossary).

Extraction is allowed from any language in `language_mapper.json`, not
just the job's declared source language: a mixed-language document is now
translated as-is (never rejected), and its non-declared-language passages
are legitimate domain content too. Restricting extraction to a single
declared language would silently drop them. `ExtractedGlossaryTerm.
source_language` -- reported per-term by the model itself, not assumed --
is what keeps that safe: it decides which per-language bucket a term is
written into, so a future job only has buckets matching the languages it
actually contains applied to it (see GlossaryService.load_domain_glossary's
`source_languages` filter). Without that per-term attribution, broadening
extraction this way would just as easily let one document's foreign-word
literal-string glossary entries collide with unrelated content in a
future, unrelated document.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import tiktoken

from src.config.constants import settings
from src.config.domain_prompts import get_domain_prompt_profile
from src.worker.doctranslator.batching import compute_batch_plan
from src.worker.doctranslator.batching import log_batch_plan
from src.worker.doctranslator.format.docx.units import TranslatableUnit
from src.worker.doctranslator.format.pdf.translation_config import get_token_multiplier
from src.worker.doctranslator.glossary import ExtractedGlossaryTerm
from src.worker.doctranslator.glossary import TermLanguageDropTracker
from src.worker.doctranslator.translator.schemas import TermExtractionResponse
from src.worker.doctranslator.translator.translator import BaseTranslator
from src.worker.services.language_detection_core import get_supported_languages

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """
You are an expert multilingual terminologist. Extract key terms from the text and translate them into {target_language}.

### Extraction Rules
1. Include only: named entities (people, orgs, locations, theorem/algorithm names, dates) and domain-specific nouns/noun phrases essential to meaning.
2. No full sentences. Ignore function words.
3. Use minimal noun phrases (<=5 words unless a named entity). No generic academic nouns (e.g., model, case, property) unless part of a standard term.
4. Extract each term once. Keep order of first appearance.
5. The input may contain passages in more than one language. Extract terms from passages written in any of these languages: {supported_languages}. Ignore passages written in any other language -- extract nothing from them.
6. For each extracted term, set "src_lang" to the ISO 639-1 code (from the list in rule 5) of the language its source passage was actually written in.

### Translation Rules
1. Translate each term into {target_language}.
2. If in the reference glossary, use its translation exactly.
3. Keep proper names in original language unless a well-known translation exists.
4. Ensure consistent translations.

{reference_glossary_section}

### Output Format
- Return ONLY a valid JSON array.
- Each element: {{"src": "...", "tgt": "...", "src_lang": "..."}}.
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


def _parse_terms(
    llm_output: str, *, drop_tracker: TermLanguageDropTracker
) -> list[ExtractedGlossaryTerm]:
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

    terms: list[ExtractedGlossaryTerm] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src", "")).strip()
        tgt = str(item.get("tgt", "")).strip()
        if not (src and tgt and len(src) < 100):
            continue
        # The model is asked to self-report the source language per term
        # (rule 6 above) rather than trusting the batch's declared
        # source_language -- a mixed document can legitimately contribute
        # terms in more than one language from a single batch. Anything
        # that doesn't normalize to one of the supported codes is dropped
        # rather than guessed -- see `TermLanguageDropTracker` (shared with
        # the PDF extractor).
        source_language = drop_tracker.resolve(
            str(item.get("src_lang", "")), source_term=src
        )
        if source_language is None:
            continue
        terms.append(
            ExtractedGlossaryTerm(
                source=src, target=tgt, source_language=source_language
            )
        )
    return terms


class DocxTermExtractor:
    """Extract candidate glossary terms from DOCX translatable units."""

    def __init__(
        self,
        translate_engine: BaseTranslator,
        target_language: str,
        domain: str | None = None,
        detected_languages: Iterable[str] | None = None,
    ):
        self.translate_engine = translate_engine
        self.target_language = target_language
        self.domain = domain or getattr(translate_engine, "domain", None)
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = None
        # B1 (docs/plan.md batch-sizing tuning): mirror the PDF pipeline's
        # CJK-aware paragraph-count scaling (see
        # DocxParagraphTranslator.__init__ / translation_config.py's
        # get_token_multiplier) so term-extraction batches shrink the same
        # way for CJK language pairs instead of using a flat cap.
        # `lang_in` is used only for batch sizing below now. It used to
        # also constrain the extraction prompt to a single declared
        # language, but that meant a mixed document's non-declared-language
        # passages contributed nothing. Extraction now covers every
        # language in language_mapper.json and tags each extracted term
        # with the language it actually came from (see the module
        # docstring and ExtractedGlossaryTerm) instead of restricting scope
        # to stay safe.
        lang_in = str(getattr(translate_engine, "lang_in", "") or "")
        self._token_multiplier = max(
            get_token_multiplier(lang_in, target_language, detected_languages), 0.1
        )
        self._last_batch_plan = None
        # Static for the process (get_supported_languages() is itself
        # lru_cache'd) -- computed once per extractor instance instead of
        # being rebuilt on every batch inside _extract_batch.
        self._supported_languages_str = ", ".join(sorted(get_supported_languages()))
        # Also reset at the start of `extract()`: one tracker must cover
        # exactly one extraction pass so its drop-rate reflects that
        # document, not a stale count carried over from a previous call on
        # a reused instance. Set here too (rather than left `None`) so
        # `_extract_batch()` remains independently callable/testable
        # without going through `extract()` first.
        self._drop_tracker = TermLanguageDropTracker()

    def _calc_token_count(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, len(text) // 4)
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return max(1, len(text) // 4)

    def _batch_units(
        self, units: list[TranslatableUnit]
    ) -> list[list[TranslatableUnit]]:
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

    def _extract_batch(
        self, batch: list[TranslatableUnit]
    ) -> list[ExtractedGlossaryTerm]:
        text_to_process = "\n\n".join(unit.text for unit in batch)
        if not text_to_process.strip():
            return []
        profile = get_domain_prompt_profile(self.domain)
        reference_glossary_section = (
            f"### Domain Context: {profile.display_name}\n"
            f"Focus on extracting {profile.display_name.lower()} terminology, concepts, and domain-specific nouns.\n"
            if profile
            else ""
        )
        prompt = _PROMPT_TEMPLATE.format(
            target_language=self.target_language,
            supported_languages=self._supported_languages_str,
            reference_glossary_section=reference_glossary_section,
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
        return _parse_terms(output, drop_tracker=self._drop_tracker)

    def extract(self, units: list[TranslatableUnit]) -> list[ExtractedGlossaryTerm]:
        """Extract candidate glossary terms, each tagged with its own source language.

        Batches run concurrently, bounded by TERM_EXTRACTION_POOL_MAX_WORKERS
        -- same concurrency model as the PDF pipeline's AutomaticTermExtractor
        (docs/architecture/pdf-vs-docx-translation-architecture.md). Previously
        this ran one batch at a time in a plain `for` loop even though the
        pool-size setting already existed and was simply unused here.
        """
        self._drop_tracker = TermLanguageDropTracker()
        all_pairs: list[ExtractedGlossaryTerm] = []
        batches = self._batch_units(units)
        if not batches:
            logger.info("DOCX term extraction: 0 batches (no units)")
            return []
        if self._last_batch_plan is not None:
            log_batch_plan("DocxTermExtraction", self._last_batch_plan, len(batches))
        max_workers = min(
            len(batches), max(1, int(settings.TERM_EXTRACTION_POOL_MAX_WORKERS))
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._extract_batch, batch) for batch in batches]
            for future in as_completed(futures):
                all_pairs.extend(future.result())
        self._drop_tracker.warn_if_high_drop_rate(extractor_name="DOCX term extraction")
        # Deduplicate by (source_language, src.lower(), tgt) while preserving
        # first occurrence. source_language is part of the key because the
        # same literal string can legitimately appear as a genuine term in
        # more than one language (e.g. "chat" is French for "cat" but also
        # an ordinary English word) with different intended translations --
        # collapsing them together would silently keep only whichever one
        # was extracted first.
        seen: set[tuple[str, str, str]] = set()
        deduped: list[ExtractedGlossaryTerm] = []
        for term in all_pairs:
            key = (
                term.source_language,
                term.source.strip().lower(),
                term.target.strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(term)
        logger.info(
            f"DOCX term extraction completed. Units: {len(units)}, "
            f"Batches: {len(batches)}, max_workers={max_workers}, "
            f"Raw pairs: {len(all_pairs)}, Deduped pairs: {len(deduped)}"
        )
        return deduped
