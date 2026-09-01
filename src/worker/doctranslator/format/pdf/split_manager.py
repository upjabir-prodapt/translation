import logging
from dataclasses import dataclass
from dataclasses import field
from enum import Enum

logger = logging.getLogger(__name__)

MAX_CHUNK_TOKEN_COUNT = 8000
_TOKENS_PER_PAGE_ESTIMATE = 300

#: Fixed-size fallback chunk length, used when the document has no usable
#: outline. 20 pages x 300 tokens/page = 6000, comfortably under
#: MAX_CHUNK_TOKEN_COUNT.
_FALLBACK_CHUNK_PAGES = 20


class BoundaryOrigin(Enum):
    """Where a section boundary came from, which decides its overlap.

    This distinction is the whole reason overlap is no longer a single
    global constant. An OUTLINE boundary is a *declared* section start: the
    author (or the exporting tool) asserted that a new section begins on
    that page, so no paragraph straddles it and copying preceding pages into
    the part buys nothing. A FALLBACK boundary is an arbitrary cut every N
    pages that lands mid-sentence as often as not, and there overlap is
    doing real work.

    Getting this wrong is expensive in both directions: a PowerPoint export
    with one bookmark per slide produced 40 parts on a 40-page deck, each
    with 2 overlap pages, so 117 pages were parsed and translated to
    produce 40 -- and every overlap page was then discarded by the merger.
    """

    OUTLINE = "outline"
    FALLBACK = "fallback"


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

    def __init__(
        self,
        min_pages_to_split: int = 10,
        overlap_pages: int = 2,
        min_pages_per_part: int = 10,
        max_pages_per_part: int = 25,
    ):
        self.min_pages_to_split = min_pages_to_split
        self.overlap_pages = overlap_pages
        # Outline sections shorter than min_pages_per_part are coalesced with
        # their neighbours, stopping before max_pages_per_part. Nothing
        # previously placed a floor on section size or a ceiling on section
        # count, so a deck bookmarked per slide became one pipeline per
        # slide. Roughly 90% of a part's wall time is fixed overhead
        # (typesetting, font subsetting, PDF write, and the final N-way
        # merge) rather than translation, so part *count* is the cost driver.
        # 25 pages x 300 tokens/page = 7500, just under MAX_CHUNK_TOKEN_COUNT.
        self.min_pages_per_part = min_pages_per_part
        self.max_pages_per_part = max_pages_per_part

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

        section_starts, titles, origin = self._get_sections_from_toc(doc, total_pages)
        logger.info(
            f"Document has {total_pages} pages; "
            f"detected {len(section_starts)} section(s) via {origin.value} boundaries."
        )

        if origin is BoundaryOrigin.OUTLINE:
            section_starts, titles = self._coalesce_small_sections(
                section_starts, titles, total_pages
            )

        split_points = self._build_split_points(
            section_starts, titles, total_pages, origin
        )
        # Previously referenced only from tests. Run it on the production
        # path too, but log rather than raise: an oversized chunk is a cost
        # and quality signal, not a reason to fail a translation the user is
        # waiting on.
        try:
            validate_chunk_metadata(split_points)
        except ValueError as exc:
            logger.warning(f"[split_manager] {exc}")
        return split_points

    # ------------------------------------------------------------------
    # Section detection
    # ------------------------------------------------------------------

    @staticmethod
    def _fixed_chunk_fallback(
        total_pages: int, reason: str = ""
    ) -> tuple[list[int], list[str | None], BoundaryOrigin]:
        """Return fixed 20-page chunk boundaries as fallback split points."""
        logger.info(
            f"[split_manager] TOC-based splitting unavailable ({reason}); "
            f"falling back to fixed {_FALLBACK_CHUNK_PAGES}-page chunks "
            f"for {total_pages} pages."
        )
        fallback_starts = list(range(0, total_pages, _FALLBACK_CHUNK_PAGES))
        fallback_titles: list[str | None] = [None] * len(fallback_starts)
        return fallback_starts, fallback_titles, BoundaryOrigin.FALLBACK

    @staticmethod
    def _extract_toc_entries(toc: list) -> list:
        """Return the best set of TOC entries (prefer level-1, fall back to level-2)."""
        lv1 = [e for e in toc if e[0] == 1]
        lv2 = [e for e in toc if e[0] == 2]
        logger.info(
            f"[split_manager] TOC inspection: total_raw_entries={len(toc)}, "
            f"level_1_entries={len(lv1)}, level_2_entries={len(lv2)}"
        )
        for target_level in (1, 2):
            entries = [e for e in toc if e[0] == target_level]
            if entries:
                logger.info(
                    f"[split_manager] Using level-{target_level} TOC entries "
                    f"({len(entries)} items)."
                )
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
    ) -> tuple[list[int], list[str | None], BoundaryOrigin]:
        """Return (section_start_pages, titles, origin) derived from PDF TOC.

        Falls back to fixed-size chunks when no usable TOC is present.
        Page numbers are 0-based. The origin is returned rather than inferred
        later because it decides the overlap for every boundary, and once
        only page numbers remain there is no way to tell an author-declared
        section start from an arbitrary every-20-pages cut.
        """
        toc = doc.get_toc()
        if not toc:
            return self._fixed_chunk_fallback(
                total_pages, reason="no outline/TOC found in document"
            )

        entries = self._extract_toc_entries(toc)
        if not entries:
            return self._fixed_chunk_fallback(
                total_pages,
                reason=f"no level-1 or level-2 entries among {len(toc)} raw TOC items",
            )

        section_starts, titles = self._deduplicate_and_sort_sections(entries)
        if len(section_starts) > 1:
            logger.info(
                f"[split_manager] Derived {len(section_starts)} TOC sections: "
                f"section_starts={section_starts[:10]}{'...' if len(section_starts) > 10 else ''} "
                f"sample_titles={titles[:5]}"
            )
            return section_starts, titles, BoundaryOrigin.OUTLINE

        return self._fixed_chunk_fallback(
            total_pages,
            reason=f"only 1 deduplicated section resolved from TOC ({section_starts})",
        )

    # ------------------------------------------------------------------
    # Section coalescing
    # ------------------------------------------------------------------

    def _coalesce_small_sections(
        self,
        section_starts: list[int],
        titles: list[str | None],
        total_pages: int,
    ) -> tuple[list[int], list[str | None]]:
        """Merge adjacent outline sections up to `min_pages_per_part` pages.

        Safe for exactly the reason overlap 0 is safe on these boundaries:
        outline sections are self-contained, so concatenating consecutive
        ones yields a contiguous, coherent page range -- it is simply a
        bigger section. The first section's title is preserved because that
        is the one used for the part's section header.

        Merging stops before a part would exceed `max_pages_per_part`, so a
        document whose sections are already sensibly sized is untouched.
        Fallback chunks never reach this method: they are already exactly
        `_FALLBACK_CHUNK_PAGES` long by construction.
        """
        if len(section_starts) <= 1:
            return section_starts, titles

        merged_starts: list[int] = []
        merged_titles: list[str | None] = []
        i = 0
        n = len(section_starts)
        while i < n:
            start = section_starts[i]
            j = i
            while j + 1 < n:
                current_pages = (
                    self._section_end_page(j, section_starts, total_pages) - start + 1
                )
                if current_pages >= self.min_pages_per_part:
                    break
                # Pages this part would span if the next section joined it.
                prospective_pages = (
                    self._section_end_page(j + 1, section_starts, total_pages)
                    - start
                    + 1
                )
                if prospective_pages > self.max_pages_per_part:
                    break
                j += 1
            merged_starts.append(start)
            merged_titles.append(titles[i])
            i = j + 1

        if len(merged_starts) != len(section_starts):
            logger.info(
                f"[split_manager] Coalesced {len(section_starts)} outline section(s) "
                f"into {len(merged_starts)} part(s) "
                f"(min_pages_per_part={self.min_pages_per_part}, "
                f"max_pages_per_part={self.max_pages_per_part})."
            )
        return merged_starts, merged_titles

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
        self, i: int, section_start: int, origin: BoundaryOrigin
    ) -> tuple[int, int]:
        """Return (actual_start, overlap) for a section.

        Overlap exists so a part's first *kept* page can see its true
        predecessor while cross-page paragraphs are merged. Those pages are
        fully parsed and translated and then thrown away by
        `ResultMerger._merge_pdfs(from_page=overlap)` -- `_build_part_config`
        includes them in `should_translate_pages` -- so they are pure cost
        wherever they are not buying that context.

        On an OUTLINE boundary they buy nothing: the author declared a
        section starts there, so no paragraph crosses it. On a FALLBACK
        boundary -- an arbitrary cut every 20 pages -- they buy exactly what
        they were introduced for.
        """
        if i == 0 or origin is BoundaryOrigin.OUTLINE:
            return section_start, 0
        actual_start = max(0, section_start - self.overlap_pages)
        return actual_start, section_start - actual_start

    def _build_split_points(
        self,
        section_starts: list[int],
        titles: list[str | None],
        total_pages: int,
        origin: BoundaryOrigin = BoundaryOrigin.FALLBACK,
    ) -> list[SplitPoint]:
        split_points: list[SplitPoint] = []

        for i, section_start in enumerate(section_starts):
            section_end = self._section_end_page(i, section_starts, total_pages)
            actual_start, overlap = self._section_actual_start_and_overlap(
                i, section_start, origin
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
            logger.info(
                f"[split_manager] Chunk {i}: pages {actual_start}..{section_end} "
                f"(actual_page_count={page_count}, overlap={overlap}, "
                f"origin={origin.value}, title={titles[i]!r}, "
                f"token_estimate={split_points[-1].token_count})"
            )

        pages_parsed = sum(sp.end_page - sp.start_page + 1 for sp in split_points)
        logger.info(
            f"[split_manager] Split summary: parts={len(split_points)} "
            f"pages_parsed={pages_parsed}/{total_pages} "
            f"(x{pages_parsed / total_pages:.2f}) origin={origin.value}"
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
