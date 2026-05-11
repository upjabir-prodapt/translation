import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SplitPoint:
    """Represents a point where the document should be split"""

    start_page: int
    end_page: int
    estimated_complexity: float = 1.0
    chapter_title: str | None = None
    overlap_pages: int = 0


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

        while current_page < total_pages:
            end_page = min(current_page + self.max_pages_per_part, total_pages)
            split_points.append(
                SplitPoint(
                    start_page=current_page,
                    end_page=end_page - 1,  # end_page is inclusive
                )
            )
            current_page = end_page

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
            return [SplitPoint(start_page=0, end_page=total_pages - 1)]

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

    def _build_split_points(
        self,
        section_starts: list[int],
        titles: list[str | None],
        total_pages: int,
    ) -> list[SplitPoint]:
        split_points: list[SplitPoint] = []

        for i, section_start in enumerate(section_starts):
            # Section ends one page before the next section starts (or at doc end)
            section_end = (
                section_starts[i + 1] - 1
                if i + 1 < len(section_starts)
                else total_pages - 1
            )

            if i == 0:
                # First chunk: no overlap — the document beginning is its own context
                actual_start = section_start
                overlap = 0
            else:
                # Subsequent chunks: extend backwards for context
                actual_start = max(0, section_start - self.overlap_pages)
                overlap = section_start - actual_start

            split_points.append(
                SplitPoint(
                    start_page=actual_start,
                    end_page=section_end,
                    overlap_pages=overlap,
                    chapter_title=titles[i],
                )
            )
            logger.debug(
                f"Chunk {i}: pages {actual_start}–{section_end} "
                f"(overlap={overlap}, title={titles[i]!r})"
            )

        return split_points


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
