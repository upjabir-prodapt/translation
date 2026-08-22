import logging
from dataclasses import dataclass
from dataclasses import field

logger = logging.getLogger(__name__)

MAX_CHUNK_TOKEN_COUNT = 8000
_TOKENS_PER_PAGE_ESTIMATE = 300


@dataclass
class SplitPoint:
    """Represents a point where the document should be split"""

    start_page: int
    end_page: int
    estimated_complexity: float = 1.0
    chapter_title: str | None = None
    overlap_pages: int = 0
    chunk_index: int = 0
    token_count: int = field(default=0)

    @property
    def page_number(self) -> int:
        return self.start_page

    @property
    def section_header(self) -> str:
        return self.chapter_title or f"Section {self.chunk_index + 1}"


class BaseSplitStrategy:
    """Base class for split strategies"""

    def determine_split_points(self, config) -> list[SplitPoint]:
        raise NotImplementedError


class PageCountStrategy(BaseSplitStrategy):
    """Split document based on page count"""

    def __init__(self, max_pages_per_part: int = 20):
        self.max_pages_per_part = max_pages_per_part

    def determine_split_points(self, config) -> list[SplitPoint]:
        from pymupdf import Document

        doc = Document(str(config.input_file))
        total_pages = doc.page_count

        split_points = []
        current_page = 0
        chunk_index = 0

        while current_page < total_pages:
            end_page = min(current_page + self.max_pages_per_part, total_pages)
            page_count = end_page - current_page
            split_points.append(
                SplitPoint(
                    start_page=current_page,
                    end_page=end_page - 1,  # end_page is inclusive
                    chunk_index=chunk_index,
                    token_count=page_count * _TOKENS_PER_PAGE_ESTIMATE,
                )
            )
            current_page = end_page
            chunk_index += 1

        return split_points


class StructureAwareSplitStrategy(BaseSplitStrategy):
    """Split document by detected sections/chapters with overlap for translation continuity.

    For PDFs with fewer than min_pages_to_split pages, the entire document is
    returned as a single chunk (no splitting). For larger documents the strategy
    first tries to derive section boundaries from the PDF's own table-of-contents
    (bookmarks). If no usable TOC is found it falls back to fixed-size page
    chunks. Every chunk after the first is extended backwards by overlap_pages
    pages so the translator has context from the previous section; those overlap
    pages are later stripped from the merged output by ResultMerger.
    """

    def __init__(self, min_pages_to_split: int = 10, overlap_pages: int = 2):
        self.min_pages_to_split = min_pages_to_split
        self.overlap_pages = overlap_pages

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def determine_split_points(self, config) -> list[SplitPoint]:
        from pymupdf import Document

        doc = Document(str(config.input_file))
        total_pages = doc.page_count

        # Size check: small documents are processed as a single chunk
        if total_pages < self.min_pages_to_split:
            logger.info(
                f"Document has {total_pages} pages (< {self.min_pages_to_split}), "
                "processing as single document."
            )
            return [
                SplitPoint(
                    start_page=0,
                    end_page=total_pages - 1,
                    chunk_index=0,
                    token_count=total_pages * _TOKENS_PER_PAGE_ESTIMATE,
                )
            ]

        section_starts, titles = self._get_sections_from_toc(doc, total_pages)
        logger.info(
            f"Document has {total_pages} pages; "
            f"detected {len(section_starts)} section(s)."
        )

        return self._build_split_points(section_starts, titles, total_pages)

    # ------------------------------------------------------------------
    # Section detection
    # ------------------------------------------------------------------

    @staticmethod
    def _fixed_chunk_fallback(total_pages: int) -> tuple[list[int], list[str | None]]:
        """Return fixed 20-page chunk boundaries as fallback split points."""
        logger.debug("No usable TOC found; falling back to fixed 20-page chunks.")
        fallback_starts = list(range(0, total_pages, 20))
        fallback_titles: list[str | None] = [None] * len(fallback_starts)
        return fallback_starts, fallback_titles

    @staticmethod
    def _extract_toc_entries(toc: list) -> list:
        """Return the best set of TOC entries (prefer level-1, fall back to level-2)."""
        for target_level in (1, 2):
            entries = [e for e in toc if e[0] == target_level]
            if entries:
                return entries
        return []

    @staticmethod
    def _deduplicate_and_sort_sections(
        entries: list,
    ) -> tuple[list[int], list[str | None]]:
        """Convert TOC entries to deduplicated, sorted 0-based page lists."""
        seen: set[int] = set()
        section_starts: list[int] = []
        titles: list[str | None] = []

        for entry in entries:
            page_0based = max(0, entry[2] - 1)
            if page_0based not in seen:
                seen.add(page_0based)
                section_starts.append(page_0based)
                titles.append(entry[1] if len(entry) > 1 else None)

        order = sorted(range(len(section_starts)), key=lambda i: section_starts[i])
        section_starts = [section_starts[i] for i in order]
        titles = [titles[i] for i in order]

        if section_starts[0] != 0:
            section_starts.insert(0, 0)
            titles.insert(0, None)

        return section_starts, titles

    def _get_sections_from_toc(
        self, doc, total_pages: int
    ) -> tuple[list[int], list[str | None]]:
        """Return (section_start_pages, titles) derived from PDF TOC.

        Falls back to fixed-size chunks when no usable TOC is present.
        Page numbers are 0-based.
        """
        toc = doc.get_toc()
        if not toc:
            return self._fixed_chunk_fallback(total_pages)

        entries = self._extract_toc_entries(toc)
        if not entries:
            return self._fixed_chunk_fallback(total_pages)

        section_starts, titles = self._deduplicate_and_sort_sections(entries)
        if len(section_starts) > 1:
            logger.debug(
                f"Using TOC-based sections: {list(zip(section_starts, titles, strict=False))}"
            )
            return section_starts, titles

        return self._fixed_chunk_fallback(total_pages)

    # ------------------------------------------------------------------
    # SplitPoint construction
    # ------------------------------------------------------------------

    def _section_end_page(
        self, i: int, section_starts: list[int], total_pages: int
    ) -> int:
        """Return the last page (inclusive) of section i."""
        if i + 1 < len(section_starts):
            return section_starts[i + 1] - 1
        return total_pages - 1

    def _section_actual_start_and_overlap(
        self, i: int, section_start: int
    ) -> tuple[int, int]:
        """Return (actual_start, overlap) for a section, applying overlap only after the first."""
        if i == 0:
            return section_start, 0
        actual_start = max(0, section_start - self.overlap_pages)
        return actual_start, section_start - actual_start

    def _build_split_points(
        self,
        section_starts: list[int],
        titles: list[str | None],
        total_pages: int,
    ) -> list[SplitPoint]:
        split_points: list[SplitPoint] = []

        for i, section_start in enumerate(section_starts):
            section_end = self._section_end_page(i, section_starts, total_pages)
            actual_start, overlap = self._section_actual_start_and_overlap(
                i, section_start
            )
            page_count = section_end - actual_start + 1
            split_points.append(
                SplitPoint(
                    start_page=actual_start,
                    end_page=section_end,
                    overlap_pages=overlap,
                    chapter_title=titles[i],
                    chunk_index=i,
                    token_count=page_count * _TOKENS_PER_PAGE_ESTIMATE,
                )
            )
            logger.debug(
                f"Chunk {i}: pages {actual_start}–{section_end} "
                f"(overlap={overlap}, title={titles[i]!r})"
            )

        return split_points


def validate_chunk_metadata(chunks: list[SplitPoint]) -> None:
    """Raise ValueError if any chunk is missing required metadata or exceeds the token limit."""
    errors: list[str] = []
    for chunk in chunks:
        prefix = f"Chunk {chunk.chunk_index}"
        if chunk.token_count <= 0:
            errors.append(f"{prefix}: token_count is not populated")
        elif chunk.token_count > MAX_CHUNK_TOKEN_COUNT:
            errors.append(
                f"{prefix}: token_count {chunk.token_count} exceeds {MAX_CHUNK_TOKEN_COUNT}"
            )
    if errors:
        raise ValueError("Chunk metadata validation failed:\n" + "\n".join(errors))


class SplitManager:
    """Manages document splitting process"""

    def __init__(self, config=None):
        self.strategy = config.split_strategy

    def determine_split_points(self, config) -> list[SplitPoint]:
        """Determine where to split the document"""
        return self.strategy.determine_split_points(config)

    def estimate_part_complexity(self, split_point: SplitPoint) -> float:
        """Estimate the complexity of a document part"""
        # Simple estimation based on page count for now
        return (
            split_point.end_page - split_point.start_page + 1
        ) * split_point.estimated_complexity
