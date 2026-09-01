"""Tests for SplitPoint metadata and chunk validation."""

import pytest
from src.worker.doctranslator.format.pdf.split_manager import MAX_CHUNK_TOKEN_COUNT
from src.worker.doctranslator.format.pdf.split_manager import SplitPoint
from src.worker.doctranslator.format.pdf.split_manager import (
    StructureAwareSplitStrategy,
)
from src.worker.doctranslator.format.pdf.split_manager import validate_chunk_metadata


def _make_chunk(
    chunk_index: int,
    start_page: int = 0,
    end_page: int = 4,
    chapter_title: str | None = None,
    token_count: int = 1200,
) -> SplitPoint:
    return SplitPoint(
        start_page=start_page,
        end_page=end_page,
        chunk_index=chunk_index,
        token_count=token_count,
        chapter_title=chapter_title,
    )


def _multi_section_chunks() -> list[SplitPoint]:
    return [
        _make_chunk(
            0, start_page=0, end_page=4, chapter_title="Introduction", token_count=1500
        ),
        _make_chunk(
            1, start_page=5, end_page=12, chapter_title="Background", token_count=2400
        ),
        _make_chunk(
            2, start_page=13, end_page=20, chapter_title="Methodology", token_count=2400
        ),
        _make_chunk(
            3, start_page=21, end_page=28, chapter_title="Results", token_count=2400
        ),
        _make_chunk(
            4, start_page=29, end_page=34, chapter_title="Conclusion", token_count=1800
        ),
    ]


# ---------------------------------------------------------------------------
# SplitPoint metadata fields
# ---------------------------------------------------------------------------


class TestSplitPointMetadataFields:
    def test_page_number_equals_start_page(self):
        chunk = _make_chunk(0, start_page=5, end_page=10)
        assert chunk.page_number == 5

    def test_section_header_returns_chapter_title_when_set(self):
        chunk = _make_chunk(0, chapter_title="Introduction")
        assert chunk.section_header == "Introduction"

    def test_section_header_falls_back_when_chapter_title_is_none(self):
        chunk = _make_chunk(2, chapter_title=None)
        assert chunk.section_header == "Section 3"

    def test_chunk_index_is_stored(self):
        chunk = _make_chunk(7)
        assert chunk.chunk_index == 7

    def test_token_count_is_stored(self):
        chunk = _make_chunk(0, token_count=3000)
        assert chunk.token_count == 3000

    def test_all_required_fields_populated_for_named_section(self):
        chunk = _make_chunk(
            1, start_page=5, chapter_title="Background", token_count=2400
        )
        assert chunk.page_number is not None
        assert chunk.section_header
        assert chunk.chunk_index is not None
        assert chunk.token_count > 0


# ---------------------------------------------------------------------------
# Multi-section document: all chunks have required metadata
# ---------------------------------------------------------------------------


class TestMultiSectionChunkMetadata:
    def test_every_chunk_has_page_number(self):
        chunks = _multi_section_chunks()
        for chunk in chunks:
            assert chunk.page_number is not None

    def test_every_chunk_has_section_header(self):
        chunks = _multi_section_chunks()
        for chunk in chunks:
            assert chunk.section_header

    def test_every_chunk_has_chunk_index(self):
        chunks = _multi_section_chunks()
        for chunk in chunks:
            assert chunk.chunk_index is not None

    def test_every_chunk_has_token_count(self):
        chunks = _multi_section_chunks()
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_chunk_indices_are_sequential(self):
        chunks = _multi_section_chunks()
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_token_count_within_limit_for_all_chunks(self):
        chunks = _multi_section_chunks()
        for chunk in chunks:
            assert chunk.token_count <= MAX_CHUNK_TOKEN_COUNT


# ---------------------------------------------------------------------------
# validate_chunk_metadata
# ---------------------------------------------------------------------------


class TestValidateChunkMetadata:
    def test_passes_for_valid_chunks(self):
        validate_chunk_metadata(_multi_section_chunks())  # should not raise

    def test_raises_when_token_count_is_zero(self):
        chunks = _multi_section_chunks()
        chunks[2] = _make_chunk(2, token_count=0)
        with pytest.raises(ValueError, match="token_count is not populated"):
            validate_chunk_metadata(chunks)

    def test_raises_when_token_count_exceeds_limit(self):
        chunks = _multi_section_chunks()
        chunks[1] = _make_chunk(1, token_count=MAX_CHUNK_TOKEN_COUNT + 1)
        with pytest.raises(ValueError, match="exceeds"):
            validate_chunk_metadata(chunks)

    def test_raises_when_token_count_is_exactly_at_limit(self):
        chunks = [_make_chunk(0, token_count=MAX_CHUNK_TOKEN_COUNT)]
        validate_chunk_metadata(chunks)  # exactly at limit is allowed

    def test_raises_for_all_offending_chunks_in_one_error(self):
        chunks = [
            _make_chunk(0, token_count=0),
            _make_chunk(1, token_count=MAX_CHUNK_TOKEN_COUNT + 100),
        ]
        with pytest.raises(ValueError) as exc_info:
            validate_chunk_metadata(chunks)
        message = str(exc_info.value)
        assert "Chunk 0" in message
        assert "Chunk 1" in message

    def test_passes_for_single_chunk_document(self):
        validate_chunk_metadata([_make_chunk(0, token_count=1500)])

    def test_section_header_fallback_does_not_affect_validation(self):
        chunks = [
            _make_chunk(i, chapter_title=None, token_count=1000) for i in range(3)
        ]
        validate_chunk_metadata(chunks)
        for i, chunk in enumerate(chunks):
            assert chunk.section_header == f"Section {i + 1}"


# ---------------------------------------------------------------------------
# Boundary provenance, overlap, and section coalescing
# ---------------------------------------------------------------------------


class _FakeDoc:
    """Minimal stand-in for a pymupdf Document with a controllable outline."""

    def __init__(self, page_count: int, toc: list | None = None):
        self.page_count = page_count
        self._toc = toc or []

    def get_toc(self):  # noqa: N802 - mirrors the pymupdf API
        return self._toc


class _FakeConfig:
    def __init__(self, input_file="doc.pdf"):
        self.input_file = input_file


def _determine(strategy, doc, monkeypatch):
    """Run determine_split_points against a fake document."""
    import src.worker.doctranslator.format.pdf.split_manager as sm

    monkeypatch.setattr(sm, "Document", lambda _path: doc, raising=False)
    import pymupdf

    monkeypatch.setattr(pymupdf, "Document", lambda _path: doc)
    return strategy.determine_split_points(_FakeConfig())


class TestBoundaryOverlapProvenance:
    """Overlap pages are fully translated and then discarded by the merger.

    They only earn their cost where a paragraph can actually straddle the
    boundary, which is never true of an author-declared outline section
    start and often true of an arbitrary every-20-pages cut.
    """

    def test_outline_boundaries_get_zero_overlap(self, monkeypatch):
        strategy = StructureAwareSplitStrategy(overlap_pages=2, min_pages_per_part=1)
        toc = [[1, f"Chapter {i}", i * 10 + 1] for i in range(4)]
        points = _determine(strategy, _FakeDoc(40, toc), monkeypatch)
        assert [p.overlap_pages for p in points] == [0, 0, 0, 0]

    def test_fallback_boundaries_keep_overlap(self, monkeypatch):
        strategy = StructureAwareSplitStrategy(overlap_pages=2)
        points = _determine(strategy, _FakeDoc(60, toc=[]), monkeypatch)
        # First part never has overlap (there is nothing before it).
        assert points[0].overlap_pages == 0
        assert all(p.overlap_pages == 2 for p in points[1:])

    def test_outline_split_parses_each_page_exactly_once(self, monkeypatch):
        """The reported failure: 40 slide bookmarks -> 117 pages parsed."""
        strategy = StructureAwareSplitStrategy(overlap_pages=2, min_pages_per_part=1)
        toc = [[1, f"Slide {i + 1}", i + 1] for i in range(40)]
        points = _determine(strategy, _FakeDoc(40, toc), monkeypatch)
        pages_parsed = sum(p.end_page - p.start_page + 1 for p in points)
        assert pages_parsed == 40


class TestSectionCoalescing:
    def test_per_slide_outline_is_coalesced_to_a_handful_of_parts(self, monkeypatch):
        """One bookmark per slide must not mean one pipeline per slide.

        ~90% of a part's wall time is fixed overhead (typesetting, font
        subsetting, PDF write, N-way merge), so part count -- not page
        count -- is what a 40-part run was actually paying for.
        """
        strategy = StructureAwareSplitStrategy(
            min_pages_per_part=10, max_pages_per_part=25
        )
        toc = [[1, f"Slide {i + 1}", i + 1] for i in range(40)]
        points = _determine(strategy, _FakeDoc(40, toc), monkeypatch)
        assert len(points) == 4
        assert [(p.start_page, p.end_page) for p in points] == [
            (0, 9),
            (10, 19),
            (20, 29),
            (30, 39),
        ]
        assert sum(p.end_page - p.start_page + 1 for p in points) == 40

    def test_first_coalesced_section_title_is_preserved(self, monkeypatch):
        strategy = StructureAwareSplitStrategy(min_pages_per_part=10)
        toc = [[1, f"Slide {i + 1}", i + 1] for i in range(40)]
        points = _determine(strategy, _FakeDoc(40, toc), monkeypatch)
        assert points[0].chapter_title == "Slide 1"
        assert points[1].chapter_title == "Slide 11"

    def test_already_large_sections_are_left_alone(self, monkeypatch):
        strategy = StructureAwareSplitStrategy(
            min_pages_per_part=10, max_pages_per_part=25
        )
        toc = [[1, "A", 1], [1, "B", 21], [1, "C", 41]]
        points = _determine(strategy, _FakeDoc(60, toc), monkeypatch)
        assert [(p.start_page, p.end_page) for p in points] == [
            (0, 19),
            (20, 39),
            (40, 59),
        ]

    def test_coalescing_never_exceeds_max_pages_per_part(self, monkeypatch):
        strategy = StructureAwareSplitStrategy(
            min_pages_per_part=20, max_pages_per_part=12
        )
        toc = [[1, f"S{i}", i * 5 + 1] for i in range(8)]
        points = _determine(strategy, _FakeDoc(40, toc), monkeypatch)
        assert all(p.end_page - p.start_page + 1 <= 12 for p in points[:-1])

    def test_fallback_chunks_are_not_coalesced(self, monkeypatch):
        """A 208-page manual with no outline must split exactly as before."""
        strategy = StructureAwareSplitStrategy(overlap_pages=2)
        points = _determine(strategy, _FakeDoc(208, toc=[]), monkeypatch)
        assert len(points) == 11
        pages_parsed = sum(p.end_page - p.start_page + 1 for p in points)
        assert pages_parsed == 228

    def test_small_document_is_still_a_single_chunk(self, monkeypatch):
        strategy = StructureAwareSplitStrategy(min_pages_to_split=10)
        points = _determine(strategy, _FakeDoc(5, toc=[]), monkeypatch)
        assert len(points) == 1
        assert (points[0].start_page, points[0].end_page) == (0, 4)


class TestProductionValidationIsNonFatal:
    def test_oversized_chunk_logs_rather_than_raising(self, monkeypatch, caplog):
        """validate_chunk_metadata now runs on the production path.

        An oversized chunk is a cost and quality signal, not a reason to
        fail a translation a user is waiting on.
        """
        strategy = StructureAwareSplitStrategy(overlap_pages=0)
        with caplog.at_level("WARNING"):
            points = _determine(strategy, _FakeDoc(200, toc=[]), monkeypatch)
        assert points
        # 20 pages x 300 tokens = 6000, under the limit -- no warning expected.
        assert not any("validation failed" in r.message for r in caplog.records)
