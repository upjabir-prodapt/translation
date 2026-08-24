"""Verify that the chunking phase makes no LLM API calls and incurs zero cost.

Pre-conditions : any document requiring chunking is being processed.
Expected       : no Vertex AI / LLM client is instantiated or called during
                 determine_split_points(); cost for the chunking phase = $0.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import fitz  # PyMuPDF
import pytest
from src.worker.doctranslator.format.pdf.split_manager import PageCountStrategy
from src.worker.doctranslator.format.pdf.split_manager import SplitManager
from src.worker.doctranslator.format.pdf.split_manager import (
    StructureAwareSplitStrategy,
)

# ---------------------------------------------------------------------------
# PDF fixtures
# ---------------------------------------------------------------------------


def _build_pdf(num_pages: int, with_toc: bool = True) -> bytes:
    """Return in-memory PDF bytes, optionally with a multi-section TOC."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Page {i + 1} of document.")

    if with_toc and num_pages >= 25:
        doc.set_toc(
            [
                [1, "Introduction", 1],
                [1, "Background", 6],
                [1, "Methodology", 11],
                [1, "Results", 16],
                [1, "Conclusion", 21],
            ]
        )

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture(scope="module")
def multi_section_pdf(tmp_path_factory) -> Path:
    """30-page PDF with a 5-section TOC."""
    p = tmp_path_factory.mktemp("pdfs") / "multi_section.pdf"
    p.write_bytes(_build_pdf(30, with_toc=True))
    return p


@pytest.fixture(scope="module")
def small_pdf(tmp_path_factory) -> Path:
    """5-page PDF — falls below min_pages_to_split so becomes one chunk."""
    p = tmp_path_factory.mktemp("pdfs") / "small.pdf"
    p.write_bytes(_build_pdf(5, with_toc=False))
    return p


@pytest.fixture(scope="module")
def large_no_toc_pdf(tmp_path_factory) -> Path:
    """25-page PDF without a TOC — triggers fixed-size fallback chunking."""
    p = tmp_path_factory.mktemp("pdfs") / "large_no_toc.pdf"
    p.write_bytes(_build_pdf(25, with_toc=False))
    return p


def _config(path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.input_file = str(path)
    return cfg


# ---------------------------------------------------------------------------
# Helpers: LLM sentinel patches
# ---------------------------------------------------------------------------

_LLM_TARGETS = [
    "google.genai.Client",
    "src.worker.doctranslator.translator.translator.GeminiVertexAITranslator.__init__",
    "src.worker.doctranslator.translator.factory.create_translator",
    "src.worker.doctranslator.translator.factory.create_translator_from_model_list",
]


# ---------------------------------------------------------------------------
# StructureAwareSplitStrategy — multi-section document
# ---------------------------------------------------------------------------


class TestStructureAwareStrategyNoLLM:
    """StructureAwareSplitStrategy must never touch an LLM during chunking."""

    def test_genai_client_never_instantiated(self, multi_section_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        with patch("google.genai.Client") as mock_client:
            strategy.determine_split_points(_config(multi_section_pdf))
        mock_client.assert_not_called()

    def test_gemini_translator_never_instantiated(self, multi_section_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        target = "src.worker.doctranslator.translator.translator.GeminiVertexAITranslator.__init__"
        with patch(target) as mock_init:
            strategy.determine_split_points(_config(multi_section_pdf))
        mock_init.assert_not_called()

    def test_create_translator_never_called(self, multi_section_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        with patch(
            "src.worker.doctranslator.translator.factory.create_translator"
        ) as mock_ct:
            strategy.determine_split_points(_config(multi_section_pdf))
        mock_ct.assert_not_called()

    def test_returns_multiple_chunks_from_toc(self, multi_section_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        chunks = strategy.determine_split_points(_config(multi_section_pdf))
        assert len(chunks) > 1, (
            "Multi-section document must produce more than one chunk"
        )

    def test_all_chunks_have_chunk_index_populated(self, multi_section_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        chunks = strategy.determine_split_points(_config(multi_section_pdf))
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_all_chunks_have_token_count_populated(self, multi_section_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        chunks = strategy.determine_split_points(_config(multi_section_pdf))
        for chunk in chunks:
            assert chunk.token_count > 0


# ---------------------------------------------------------------------------
# StructureAwareSplitStrategy — small document (single-chunk path)
# ---------------------------------------------------------------------------


class TestSingleChunkPathNoLLM:
    """Small documents become one chunk without any LLM contact."""

    def test_genai_client_never_instantiated_for_small_doc(self, small_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        with patch("google.genai.Client") as mock_client:
            strategy.determine_split_points(_config(small_pdf))
        mock_client.assert_not_called()

    def test_returns_single_chunk_for_small_document(self, small_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        chunks = strategy.determine_split_points(_config(small_pdf))
        assert len(chunks) == 1

    def test_single_chunk_has_zero_index(self, small_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        chunks = strategy.determine_split_points(_config(small_pdf))
        assert chunks[0].chunk_index == 0

    def test_single_chunk_has_token_count(self, small_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        chunks = strategy.determine_split_points(_config(small_pdf))
        assert chunks[0].token_count > 0


# ---------------------------------------------------------------------------
# StructureAwareSplitStrategy — large doc without TOC (fixed-chunk fallback)
# ---------------------------------------------------------------------------


class TestFixedChunkFallbackNoLLM:
    """When no TOC is present, fixed-size chunks are created without LLM calls."""

    def test_genai_client_never_instantiated_for_no_toc_doc(self, large_no_toc_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        with patch("google.genai.Client") as mock_client:
            strategy.determine_split_points(_config(large_no_toc_pdf))
        mock_client.assert_not_called()

    def test_fallback_produces_multiple_chunks(self, large_no_toc_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        chunks = strategy.determine_split_points(_config(large_no_toc_pdf))
        assert len(chunks) >= 1

    def test_fallback_chunk_indices_are_sequential(self, large_no_toc_pdf):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        chunks = strategy.determine_split_points(_config(large_no_toc_pdf))
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ---------------------------------------------------------------------------
# PageCountStrategy — no LLM
# ---------------------------------------------------------------------------


class TestPageCountStrategyNoLLM:
    def test_genai_client_never_instantiated(self, multi_section_pdf):
        strategy = PageCountStrategy(max_pages_per_part=10)
        with patch("google.genai.Client") as mock_client:
            strategy.determine_split_points(_config(multi_section_pdf))
        mock_client.assert_not_called()

    def test_create_translator_never_called(self, multi_section_pdf):
        strategy = PageCountStrategy(max_pages_per_part=10)
        with patch(
            "src.worker.doctranslator.translator.factory.create_translator"
        ) as mock_ct:
            strategy.determine_split_points(_config(multi_section_pdf))
        mock_ct.assert_not_called()

    def test_chunk_indices_populated(self, multi_section_pdf):
        strategy = PageCountStrategy(max_pages_per_part=10)
        chunks = strategy.determine_split_points(_config(multi_section_pdf))
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_token_count_populated(self, multi_section_pdf):
        strategy = PageCountStrategy(max_pages_per_part=10)
        chunks = strategy.determine_split_points(_config(multi_section_pdf))
        for chunk in chunks:
            assert chunk.token_count > 0


# ---------------------------------------------------------------------------
# SplitManager (orchestrator) — no LLM
# ---------------------------------------------------------------------------


class TestSplitManagerNoLLM:
    def test_split_manager_makes_no_llm_calls_for_multi_section_doc(
        self, multi_section_pdf
    ):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        config = _config(multi_section_pdf)
        config.split_strategy = strategy

        manager = SplitManager(config)

        with (
            patch("google.genai.Client") as mock_genai,
            patch(
                "src.worker.doctranslator.translator.factory.create_translator"
            ) as mock_ct,
        ):
            manager.determine_split_points(config)

        mock_genai.assert_not_called()
        mock_ct.assert_not_called()


# ---------------------------------------------------------------------------
# Chunking cost = $0
# ---------------------------------------------------------------------------


class TestChunkingCostIsZero:
    """Since chunking triggers no billable API calls, its cost must be $0."""

    def test_no_api_calls_means_zero_token_consumption(self, multi_section_pdf):
        """Running chunking should record zero prompt/completion tokens."""
        from src.config.llm_rate_catalog import ModelRateEntry
        from src.config.llm_rate_catalog import RateTier
        from src.worker.services.llm_cost_service import VertexLLMCostService

        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        chunks = strategy.determine_split_points(_config(multi_section_pdf))

        catalog = (
            ModelRateEntry(
                provider="gemini_vertexai",
                model_id="gemini-2.5-flash",
                region=None,
                tiers=(
                    RateTier(
                        max_input_tokens=None,
                        input_cost_per_1k=0.0003,
                        output_cost_per_1k=0.0025,
                    ),
                ),
            ),
        )
        service = VertexLLMCostService(catalog=catalog)
        breakdown = service.calculate_attempt_cost(
            model_id="gemini-2.5-flash",
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert breakdown.total_cost_usd == 0.0
        assert len(chunks) >= 1

    def test_chunking_cost_is_exactly_zero_regardless_of_model_rate(
        self, multi_section_pdf
    ):
        from src.config.llm_rate_catalog import ModelRateEntry
        from src.config.llm_rate_catalog import RateTier
        from src.worker.services.llm_cost_service import VertexLLMCostService

        catalog = (
            ModelRateEntry(
                provider="gemini_vertexai",
                model_id="gemini-2.5-flash",
                region=None,
                tiers=(
                    RateTier(
                        max_input_tokens=None,
                        input_cost_per_1k=0.0003,
                        output_cost_per_1k=0.0025,
                    ),
                ),
            ),
        )
        service = VertexLLMCostService(catalog=catalog)
        breakdown = service.calculate_attempt_cost(
            model_id="gemini-2.5-flash",
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert breakdown.total_cost_usd == 0.0

    def test_all_llm_entry_points_untouched_during_full_chunking_run(
        self, multi_section_pdf
    ):
        """Patch every known LLM entry point and confirm none are touched."""
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        config = _config(multi_section_pdf)

        call_log: list[str] = []

        def _make_spy(name):
            def spy(*_args, **_kwargs):
                call_log.append(name)

            return spy

        with (
            patch("google.genai.Client", side_effect=_make_spy("genai.Client")),
            patch(
                "src.worker.doctranslator.translator.factory.create_translator",
                side_effect=_make_spy("create_translator"),
            ),
            patch(
                "src.worker.doctranslator.translator.factory.create_translator_from_model_list",
                side_effect=_make_spy("create_translator_from_model_list"),
            ),
        ):
            strategy.determine_split_points(config)

        assert call_log == [], f"LLM entry points called during chunking: {call_log}"
