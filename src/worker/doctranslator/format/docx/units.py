"""Extraction of translatable paragraph units from a DOCX document.

A "unit" is one python-docx Paragraph found anywhere in the document
(body, tables -- including nested tables, headers, footers, footnotes,
endnotes, and text boxes) together with a stable integer id and a label
describing where it came from. Units are collected in document reading
order: headers first (by section), then body (including inline tables
and text boxes), then footers, then footnotes, then endnotes.

This module is 100% language-independent -- the same units are produced
regardless of target language, which is why the extraction step can be
shared across a multi-target-language batch (see shared_document_prep.py
for the equivalent optimization on the PDF side).

implementation_plan.md Phase D.2 (EC-04): footnotes, endnotes, and text
boxes (`w:txbxContent`) were previously missed entirely, producing
silently untranslated output. Hyperlink display text (runs inside
`w:hyperlink`) is invisible to both `Paragraph.text` and `Paragraph.runs`
in python-docx, so it was never translated either, and worse, once the
surrounding paragraph *is* translated (paragraph.text folds hyperlink
text in), the untranslated hyperlink run was left behind unmodified,
producing duplicated/mixed-language output. All four gaps are closed
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from docx.document import Document as DocxDocument
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.part import Part
from docx.oxml import OxmlElement
from docx.oxml import parse_xml
from docx.oxml.ns import nsmap
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from lxml import etree

# `w:footnote`/`w:endnote` elements carrying one of these `w:type` values are
# Word-generated separator markup, never user content -- must never be
# extracted or sent to an LLM.
_SKIP_NOTE_TYPES = {"separator", "continuationSeparator", "continuationNotice"}


class _HyperlinkRunsProxy:
    """Duck-typed stand-in for `Paragraph`, scoped to one `w:hyperlink`'s own runs.

    Lets `write_translated_text()` treat a hyperlink's display text exactly
    like a tiny paragraph -- same "first run keeps formatting, extra runs
    removed" logic -- via the `.runs` / `.add_run()` contract, without ever
    touching the `w:hyperlink` element itself or its `r:id` relationship. The
    link target and its clickability therefore survive untouched
    (implementation_plan.md D.2.2/D.2.3).
    """

    def __init__(self, hyperlink_element, parent_paragraph: Paragraph) -> None:
        self._hyperlink = hyperlink_element
        self._parent_paragraph = parent_paragraph

    @property
    def runs(self) -> list[Run]:
        return [
            Run(r, self._parent_paragraph) for r in self._hyperlink.findall(qn("w:r"))
        ]

    def add_run(self, text: str) -> Run:
        r_element = OxmlElement("w:r")
        self._hyperlink.append(r_element)
        run = Run(r_element, self._parent_paragraph)
        run.text = text
        return run


@dataclass(slots=True)
class TranslatableUnit:
    """One paragraph (or hyperlink run) eligible for translation, with a stable id."""

    unit_id: int
    # `Paragraph` for ordinary units; `_HyperlinkRunsProxy` for "hyperlink"
    # units. Both expose the `.runs` / `.add_run()` contract that
    # `write_translated_text()` relies on.
    paragraph: Paragraph | _HyperlinkRunsProxy
    label: str  # "header" | "footer" | "title" | "text" | "table_cell"
    # | "text_box" | "footnote" | "endnote" | "hyperlink"
    text: str


class NotePartRef(NamedTuple):
    """A footnotes.xml/endnotes.xml part plus the detached tree parsed from it.

    `part.blob` is parsed into a fresh, detached lxml tree (python-docx does
    not model footnotes/endnotes natively), so in-place run-text mutations on
    paragraphs inside it are invisible to `Document.save()` until the tree is
    re-serialized back into `part._blob` -- see `flush_note_parts()`.
    """

    part: Part
    root: etree._Element


def _iter_table_paragraphs(table: Table):
    """Yield every paragraph in a table's cells, recursing into nested tables."""
    for row in table.rows:
        for cell in row.cells:
            yield from cell.paragraphs
            for nested_table in cell.tables:
                yield from _iter_table_paragraphs(nested_table)


def _guess_label(paragraph: Paragraph, *, default: str) -> str:
    style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
    if "title" in style_name or "heading" in style_name:
        return "title"
    return default


def _paragraph_text(paragraph: Paragraph) -> str:
    return paragraph.text or ""


def _iter_txbx_paragraphs(container_element):
    """Yield every `w:p` nested inside a `w:txbxContent` under `container_element`.

    Text boxes are anchored inside a run's drawing/shape markup, deeply
    nested below the top-level paragraph/table walk, so they are invisible
    to it entirely -- this is a separate, additive pass. Only paragraphs
    directly inside the text box are extracted (not further-nested tables
    inside a text box), a deliberate, documented scope limit.

    `container_element` may be a `BaseOxmlElement` (already part of the
    main document tree, e.g. the body or a header/footer element) whose
    overridden `.xpath()` takes no `namespaces` kwarg, or a plain lxml
    element from a detached tree parsed via `docx.oxml.parse_xml`
    (footnotes.xml/endnotes.xml, in `extract_units()`), whose `.xpath()`
    requires one explicitly. Calling the unbound `lxml.etree._Element.xpath`
    directly sidesteps that inconsistency uniformly for both.
    """
    yield from etree._Element.xpath(
        container_element, ".//w:txbxContent/w:p", namespaces=nsmap
    )


def extract_units(
    document: DocxDocument,
) -> tuple[list[TranslatableUnit], list[NotePartRef]]:
    """Walk a DOCX document in reading order and return all translatable units.

    Order: for each section, its header paragraphs (incl. text boxes); then
    the body (top-level paragraphs and top-level tables in document order,
    plus text boxes anywhere in the body); then footers (incl. text boxes);
    then footnotes; then endnotes.

    Returns `(units, note_parts)`. `note_parts` holds the footnotes.xml /
    endnotes.xml part(s) that were parsed into a detached tree to extract
    footnote/endnote units from -- callers that mutate unit text (i.e.
    translation, not read-only language detection) MUST call
    `flush_note_parts(note_parts)` after writing all translations back and
    before `document.save()`, or footnote/endnote edits are silently lost.

    Empty/whitespace-only paragraphs are skipped entirely (they never need
    translation and would only waste an LLM-batch slot).
    """
    units: list[TranslatableUnit] = []
    next_id = 0

    def _add(paragraph: Paragraph, label: str) -> None:
        nonlocal next_id
        hyperlinks = paragraph._p.findall(qn("w:hyperlink"))  # noqa: SLF001
        if hyperlinks:
            # D.2.2/D.2.3: split the hyperlink's display text out into its
            # own unit so it can be translated and written back
            # independently, instead of duplicating it (paragraph.text
            # folds hyperlink text in, but paragraph.runs/write-back does
            # not touch the hyperlink element at all).
            own_text = "".join(run.text for run in paragraph.runs)
            if own_text.strip():
                units.append(
                    TranslatableUnit(
                        unit_id=next_id,
                        paragraph=paragraph,
                        label=label,
                        text=own_text,
                    )
                )
                next_id += 1
            for hyperlink in hyperlinks:
                hyperlink_text = "".join(
                    r.text or "" for r in hyperlink.findall(qn("w:r"))
                )
                if not hyperlink_text.strip():
                    continue
                units.append(
                    TranslatableUnit(
                        unit_id=next_id,
                        paragraph=_HyperlinkRunsProxy(hyperlink, paragraph),
                        label="hyperlink",
                        text=hyperlink_text,
                    )
                )
                next_id += 1
            return
        text = _paragraph_text(paragraph)
        if not text.strip():
            return
        units.append(
            TranslatableUnit(
                unit_id=next_id,
                paragraph=paragraph,
                label=label,
                text=text,
            )
        )
        next_id += 1

    def _add_txbx(container_element) -> None:
        for p_el in _iter_txbx_paragraphs(container_element):
            _add(Paragraph(p_el, document), "text_box")

    seen_header_ids: set[int] = set()
    seen_footer_ids: set[int] = set()

    for section in document.sections:
        for header in (
            section.header,
            section.first_page_header,
            section.even_page_header,
        ):
            if header is None or id(header) in seen_header_ids:
                continue
            seen_header_ids.add(id(header))
            if header.is_linked_to_previous:
                continue
            for paragraph in header.paragraphs:
                _add(paragraph, "header")
            for table in header.tables:
                for paragraph in _iter_table_paragraphs(table):
                    _add(paragraph, "table_cell")
            _add_txbx(header._element)  # noqa: SLF001

    # Body: iterate in true document order (paragraphs and tables
    # interleaved), by walking the underlying body XML element children.
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            _add(paragraph, _guess_label(paragraph, default="text"))
        elif tag == "tbl":
            table = Table(child, document)
            for paragraph in _iter_table_paragraphs(table):
                _add(paragraph, "table_cell")
    _add_txbx(body)

    for section in document.sections:
        for footer in (
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        ):
            if footer is None or id(footer) in seen_footer_ids:
                continue
            seen_footer_ids.add(id(footer))
            if footer.is_linked_to_previous:
                continue
            for paragraph in footer.paragraphs:
                _add(paragraph, "footer")
            for table in footer.tables:
                for paragraph in _iter_table_paragraphs(table):
                    _add(paragraph, "table_cell")
            _add_txbx(footer._element)  # noqa: SLF001

    note_parts: list[NotePartRef] = []
    for reltype, note_tag, label in (
        (RT.FOOTNOTES, "w:footnote", "footnote"),
        (RT.ENDNOTES, "w:endnote", "endnote"),
    ):
        try:
            part = document.part.part_related_by(reltype)
        except (KeyError, ValueError):
            # No footnotes.xml/endnotes.xml part -- most documents don't
            # have one. `part_related_by` raises KeyError when absent and
            # (defensively) ValueError if more than one relationship of
            # this type were ever present, which should not happen for a
            # well-formed DOCX.
            continue
        root = parse_xml(part.blob)
        for note in root.findall(qn(note_tag)):
            if note.get(qn("w:type")) in _SKIP_NOTE_TYPES:
                continue
            for p_el in note.findall(qn("w:p")):
                _add(Paragraph(p_el, document), label)
            _add_txbx(note)
        note_parts.append(NotePartRef(part=part, root=root))

    return units, note_parts


def flush_note_parts(note_parts: list[NotePartRef]) -> None:
    """Re-serialize footnote/endnote detached trees back into their parts.

    Must be called after all `write_translated_text()` calls for
    footnote/endnote units and before `Document.save()` -- footnotes.xml
    and endnotes.xml are not modeled by python-docx, so edits to the
    detached tree returned by `extract_units()` are otherwise invisible to
    the saved document (implementation_plan.md D.2.1).
    """
    for part, root in note_parts:
        part._blob = etree.tostring(  # noqa: SLF001
            root, encoding="UTF-8", standalone=True
        )


def write_translated_text(unit: TranslatableUnit, translated_text: str) -> None:
    """Replace a unit's text with the translated string.

    Keeps the unit's FIRST run's formatting (font, bold, italic, size,
    color, language) and removes any additional runs -- this preserves
    paragraph-level fidelity (alignment, list numbering, style, table cell
    membership) but does not preserve character-level mixed formatting
    *within* one paragraph (e.g. a single bold word mid-sentence). This is
    a documented, intentional v1 tradeoff.

    For a "hyperlink" unit (`unit.paragraph` is a `_HyperlinkRunsProxy`),
    only the hyperlink's own inner run(s) are touched -- the `w:hyperlink`
    element and its relationship (i.e. the URL) are never modified, so the
    link keeps working with translated display text
    (implementation_plan.md D.2.2/D.2.3).

    Images, embedded objects, and every other part of the document
    (styles.xml, numbering.xml, section properties, media relationships)
    are never touched -- only run text within this unit is mutated.
    """
    paragraph = unit.paragraph
    runs = paragraph.runs
    if not runs:
        # No runs (rare: e.g. purely field-code paragraph, or an emptied-out
        # hyperlink) -- add a single run with the translated text.
        paragraph.add_run(translated_text)
        return

    first_run = runs[0]
    first_run.text = translated_text
    for extra_run in runs[1:]:
        extra_run.text = ""
        extra_run._element.getparent().remove(extra_run._element)  # noqa: SLF001
