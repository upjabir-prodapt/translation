"""Tests for SplitPoint metadata and chunk validation."""

import pytest
from src.worker.doctranslator.format.pdf.split_manager import MAX_CHUNK_TOKEN_COUNT
from src.worker.doctranslator.format.pdf.split_manager import SplitPoint
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
