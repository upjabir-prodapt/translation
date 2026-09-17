"""PDF automatic term extraction: multi-language extraction + src_lang gating.

Mirrors the DOCX extractor's equivalent behaviour
(tests/worker/doctranslator/format/docx/test_term_extractor.py ::
TestMultiLanguageExtraction) -- extraction is not restricted to the job's
declared `lang_in`, and each extracted term's self-reported `src_lang` is
validated against the supported-language set (via the shared
`TermLanguageDropTracker`) before it is accepted into this job's per-job
auto-extracted glossary.

Exercises `_store_valid_term` directly -- the method actually reached by the
live `extract_terms_from_paragraphs()` path -- rather than the now-removed
`_process_llm_response`, which validated `src_lang` too but was never
called from production code (a prior gap: the validation these tests
previously exercised was not actually live).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.worker.doctranslator.format.pdf.document_il.midend.automatic_term_extractor import (
    LLM_PROMPT_TEMPLATE,
)
from src.worker.doctranslator.format.pdf.document_il.midend.automatic_term_extractor import (
    AutomaticTermExtractor,
)
from src.worker.doctranslator.format.pdf.translation_config import (
    SharedContextCrossSplitPart,
)


def _make_extractor(lang_in: str = "en", lang_out: str = "de"):
    translate_engine = MagicMock()
    translate_engine.llm_translate = MagicMock(return_value="[]")
    translation_config = MagicMock()
    translation_config.lang_in = lang_in
    translation_config.lang_out = lang_out
    translation_config.shared_context_cross_split_part = SharedContextCrossSplitPart()
    return AutomaticTermExtractor(translate_engine, translation_config)


class TestPromptTemplate:
    def test_template_no_longer_names_a_single_source_language(self):
        assert "{source_language}" not in LLM_PROMPT_TEMPLATE
        assert "{supported_languages}" in LLM_PROMPT_TEMPLATE
        assert "src_lang" in LLM_PROMPT_TEMPLATE


class TestStoreValidTerm:
    def test_accepts_a_term_with_a_supported_src_lang(self):
        extractor = _make_extractor()
        stored = extractor._store_valid_term(
            {"src": "contract", "tgt": "Vertrag", "src_lang": "en"},
            request_id="req-1",
        )
        assert stored is True
        assert extractor.shared_context.raw_extracted_terms == [("contract", "Vertrag")]

    def test_accepts_multiple_languages_from_one_mixed_batch(self):
        """A mixed EN/FR passage in the same batch contributes terms from
        both languages -- extraction is no longer limited to lang_in."""
        extractor = _make_extractor(lang_in="en")
        assert extractor._store_valid_term(
            {"src": "contract", "tgt": "Vertrag", "src_lang": "en"},
            request_id="req-2",
        )
        assert extractor._store_valid_term(
            {"src": "contrat", "tgt": "Vertrag", "src_lang": "fr"},
            request_id="req-2",
        )
        assert extractor.shared_context.raw_extracted_terms == [
            ("contract", "Vertrag"),
            ("contrat", "Vertrag"),
        ]

    def test_drops_a_term_with_an_unsupported_src_lang(self):
        extractor = _make_extractor()
        stored = extractor._store_valid_term(
            {
                "src": "договор",
                "tgt": "contract",
                "src_lang": "ru",
            },
            request_id="req-3",
        )
        assert stored is False
        assert extractor.shared_context.raw_extracted_terms == []

    def test_drops_a_term_with_a_missing_src_lang(self):
        extractor = _make_extractor()
        stored = extractor._store_valid_term(
            {"src": "contract", "tgt": "Vertrag"},
            request_id="req-4",
        )
        assert stored is False
        assert extractor.shared_context.raw_extracted_terms == []

    def test_drop_is_tracked_for_the_batch_drop_rate_warning(self):
        """`_store_valid_term` must feed the same shared tracker
        `procress()` reads at the end of a document's extraction pass, or a
        systemic src_lang regression would have no operator-visible
        signal (see TermLanguageDropTracker)."""
        extractor = _make_extractor()
        extractor._store_valid_term(
            {"src": "term", "tgt": "target", "src_lang": "xx"},
            request_id="req-5",
        )
        assert extractor._drop_tracker.candidate_count == 1
        assert extractor._drop_tracker.dropped_count == 1
