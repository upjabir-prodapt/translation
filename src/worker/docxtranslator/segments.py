"""Paragraph segmentation and translated-text write-back for DOCX.

A *segment* is one Word paragraph — wherever it lives, including inside table
cells, text boxes, headers, and footnotes. Within a segment, ``w:t`` nodes are
merged into *groups*: consecutive nodes that share run formatting and are not
separated by an inline element such as a tab, break, or image. Groups are the
unit sent to and returned from the model, so bold or hyperlinked spans survive
translation and nothing structural moves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field

from lxml import etree

from src.worker.docxtranslator.ooxml import FLOW_BREAK_TAGS
from src.worker.docxtranslator.ooxml import SKIP_SUBTREE_TAGS
from src.worker.docxtranslator.ooxml import W_P
from src.worker.docxtranslator.ooxml import W_T
from src.worker.docxtranslator.ooxml import run_format_key
from src.worker.docxtranslator.ooxml import set_text

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TextNode:
    """One ``w:t`` element and the formatting context it sits in."""

    element: etree._Element
    format_key: str
    breaks_before: bool

    @property
    def text(self) -> str:
        return self.element.text or ""


@dataclass(slots=True)
class RunGroup:
    """Consecutive same-formatting text nodes translated as one unit."""

    nodes: list[TextNode]

    @property
    def text(self) -> str:
        return "".join(node.text for node in self.nodes)

    def write(self, value: str) -> None:
        """Write translated text back, keeping every original run in place."""
        set_text(self.nodes[0].element, value)
        for node in self.nodes[1:]:
            set_text(node.element, "")


@dataclass(slots=True)
class Segment:
    """One Word paragraph, decomposed into translatable run groups."""

    part_name: str
    index: int
    groups: list[RunGroup] = field(default_factory=list)

    @property
    def source_text(self) -> str:
        return "".join(group.text for group in self.groups)

    @property
    def group_texts(self) -> list[str]:
        return [group.text for group in self.groups]

    def is_translatable(self) -> bool:
        """True when the paragraph holds letters worth sending to the model.

        Paragraphs made only of digits, punctuation, or whitespace — spacer
        rows, numeric table cells, separators — are left exactly as they are.
        """
        return any(char.isalpha() for char in self.source_text)

    def write(self, group_texts: list[str]) -> None:
        """Write one translated string per group back into the document."""
        if len(group_texts) != len(self.groups):
            raise ValueError(
                f"Expected {len(self.groups)} group texts, got {len(group_texts)}"
            )
        for group, value in zip(self.groups, group_texts, strict=True):
            group.write(value)

    def write_joined(self, text: str) -> None:
        """Write the whole translation into the first group, blanking the rest.

        Fallback for when per-group alignment cannot be recovered: paragraph
        text stays complete and correctly placed, at the cost of intra-paragraph
        formatting collapsing onto the first run's style.
        """
        self.groups[0].write(text)
        for group in self.groups[1:]:
            group.write("")


class _ParagraphWalker:
    """Collect the ``w:t`` nodes a paragraph directly owns, in document order."""

    def __init__(self) -> None:
        self.nodes: list[TextNode] = []
        self._pending_break = False

    def walk(self, element: etree._Element) -> None:
        for child in element:
            tag = child.tag
            if tag == W_P:
                # A nested paragraph (text box content) is its own segment.
                continue
            if tag in SKIP_SUBTREE_TAGS:
                self._pending_break = True
                continue
            if tag in FLOW_BREAK_TAGS:
                self._pending_break = True
                continue
            if tag == W_T:
                self._emit(child)
                continue
            self.walk(child)

    def _emit(self, text_node: etree._Element) -> None:
        self.nodes.append(
            TextNode(
                element=text_node,
                format_key=run_format_key(text_node),
                breaks_before=self._pending_break,
            )
        )
        self._pending_break = False


def _group_nodes(nodes: list[TextNode]) -> list[RunGroup]:
    groups: list[RunGroup] = []
    for node in nodes:
        same_format = (
            groups
            and not node.breaks_before
            and groups[-1].nodes[-1].format_key == node.format_key
        )
        if same_format:
            groups[-1].nodes.append(node)
        else:
            groups.append(RunGroup(nodes=[node]))
    return groups


def collect_segments(parts: list[tuple[str, etree._Element]]) -> list[Segment]:
    """Build the ordered segment list for every body-text part of a package.

    Order is deterministic — part order, then paragraph document order — so
    DLP chunk indices and batch numbering are reproducible across runs.
    """
    segments: list[Segment] = []
    for part_name, root in parts:
        for paragraph in root.iter(W_P):
            walker = _ParagraphWalker()
            walker.walk(paragraph)
            if not walker.nodes:
                continue
            groups = _group_nodes(walker.nodes)
            segments.append(
                Segment(
                    part_name=part_name,
                    index=len(segments),
                    groups=groups,
                )
            )
    return segments


def extract_text(segments: list[Segment], max_chars: int | None = None) -> str:
    """Join segment text for language detection and quality scoring."""
    lines: list[str] = []
    total = 0
    for segment in segments:
        text = segment.source_text.strip()
        if not text:
            continue
        lines.append(text)
        total += len(text)
        if max_chars is not None and total >= max_chars:
            break
    return "\n".join(lines)
