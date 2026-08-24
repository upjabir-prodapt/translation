"""PDF paragraph batching must be right-sized for the worker pool.

DOCX computes its plan in `_batch_units()` because it has the whole unit list
up front. PDF batches per-page inside `process_page()`, so the document-level
token total is only knowable in `translate()` -- `_apply_adaptive_batch_plan()`
computes it once there and overrides the config caps that `process_page()`
later reads.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.doctranslator.format.pdf.document_il.midend.il_translator_llm_only import (
    ILTranslatorLLMOnly,
)


def _paragraph(text, debug_id=1):
    para = MagicMock()
    para.unicode = text
    para.debug_id = debug_id
    return para


def _docs(paragraphs_per_page, text="word " * 20):
    docs = MagicMock()
    pages = []
    for _ in range(paragraphs_per_page):
        page = MagicMock()
        page.pdf_paragraph = [_paragraph(text) for _ in range(10)]
        pages.append(page)
    docs.page = pages
    return docs


def _translator(pool_workers=12, base_tokens=40_000, base_paragraphs=200):
    inst = ILTranslatorLLMOnly.__new__(ILTranslatorLLMOnly)
    config = MagicMock()
    config.pool_max_workers = pool_workers
    config.llm_translation_batch_max_tokens = base_tokens
    config.llm_translation_batch_max_paragraphs = base_paragraphs
    inst.translation_config = config
    # 1 token per whitespace-separated word keeps the arithmetic checkable.
    inst.calc_token_count = lambda text: len(text.split())
    return inst


def _adaptive():
    return patch.multiple(
        "src.worker.doctranslator.batching.settings",
        LLM_ADAPTIVE_BATCHING_ENABLED=True,
        LLM_ADAPTIVE_BATCH_MIN_TOKENS=600,
        LLM_ADAPTIVE_BATCH_MAX_TOKENS=2500,
    )


class TestPdfAdaptiveBatchPlan:
    def test_shrinks_the_configured_cap_to_one_wave(self):
        translator = _translator()
        docs = _docs(10)  # 10 pages x 10 paragraphs x 20 tokens = 2000 tokens
        with _adaptive():
            translator._apply_adaptive_batch_plan(docs)
        cap = translator.translation_config.llm_translation_batch_max_tokens
        assert cap < 40_000, "adaptive sizing must reduce the flat cap"
        assert cap == 600  # 2000/12 = 167 -> clamped up to the 600 floor

    def test_large_document_clamps_to_ceiling(self):
        translator = _translator()
        docs = _docs(300)  # 300 x 10 x 20 = 60000 tokens
        with _adaptive():
            translator._apply_adaptive_batch_plan(docs)
        assert translator.translation_config.llm_translation_batch_max_tokens == 2500

    def test_never_exceeds_the_operator_configured_cap(self):
        translator = _translator(base_tokens=800)
        docs = _docs(300)
        with _adaptive():
            translator._apply_adaptive_batch_plan(docs)
        assert translator.translation_config.llm_translation_batch_max_tokens == 800

    def test_untranslatable_paragraphs_are_excluded_from_the_total(self):
        """Paragraphs with no debug_id/unicode never reach the LLM."""
        translator = _translator()
        docs = _docs(5)
        for page in docs.page:
            for para in page.pdf_paragraph:
                para.debug_id = None
        with _adaptive():
            translator._apply_adaptive_batch_plan(docs)
        # Zero payload -> adaptive disabled, configured cap left untouched.
        assert translator.translation_config.llm_translation_batch_max_tokens == 40_000

    def test_disabled_adaptive_leaves_config_untouched(self):
        translator = _translator()
        docs = _docs(10)
        with patch.multiple(
            "src.worker.doctranslator.batching.settings",
            LLM_ADAPTIVE_BATCHING_ENABLED=False,
        ):
            translator._apply_adaptive_batch_plan(docs)
        assert translator.translation_config.llm_translation_batch_max_tokens == 40_000
        assert translator.translation_config.llm_translation_batch_max_paragraphs == 200

    def test_emits_a_batch_plan_log_line(self, caplog):
        translator = _translator()
        docs = _docs(10)
        with _adaptive(), caplog.at_level("INFO"):
            translator._apply_adaptive_batch_plan(docs)
        assert "batch_plan stage=PdfTranslateParagraphs" in caplog.text
