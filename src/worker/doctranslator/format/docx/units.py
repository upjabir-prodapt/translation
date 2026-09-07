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


# Run children that carry document structure rather than translatable text.
# A run holding one of these and no text of its own is a *reference run* --
# a footnote/endnote/comment mark, a field code, an image, a drawing -- and
# must survive translation untouched. See `_is_reference_run()`.
_STRUCTURAL_RUN_CHILDREN = frozenset(
    qn(tag)
    for tag in (
        "w:footnoteReference",
        "w:endnoteReference",
        "w:commentReference",
        "w:footnoteRef",
        "w:endnoteRef",
        "w:annotationRef",
        "w:fldChar",
        "w:instrText",
        "w:drawing",
        "w:pict",
        "w:object",
        "w:separator",
        "w:continuationSeparator",
        "w:sym",
    )
)


def _is_reference_run(run: Run) -> bool:
    """True when `run` carries structure (a reference mark, field code or
    drawing) and no text of its own.

    UAT EC-03 (D-01/D-02): `write_translated_text()` used to delete every run
    after the first, which silently destroyed exactly these runs. A paragraph
    carrying a footnote lost its `w:footnoteReference`, so the translated
    footnote text survived in footnotes.xml with nothing in the body pointing
    at it; a paragraph opening a table-of-contents field lost its
    `w:fldChar`/`w:instrText` runs, so the ToC field was dismantled into flat
    text that Word can no longer refresh or use for navigation.

    A run holding *both* structure and real text (e.g. `w:tab` + `w:t` in a
    ToC entry's page-number run) is deliberately NOT treated as a reference
    run: its text is part of the unit that was sent for translation, so the
    translation already reproduces it and keeping the run would duplicate it.
    """
    element = run._element  # noqa: SLF001
    if any(t.text and t.text.strip() for t in element.findall(qn("w:t"))):
        return False
    return any(child.tag in _STRUCTURAL_RUN_CHILDREN for child in element)


class _ScopedRunsProxy:
    """Duck-typed stand-in for `Paragraph`, scoped to one container's own runs.

    Lets `write_translated_text()` treat the runs inside a `w:hyperlink`,
    `w:ins` or `w:fldSimple` element exactly like a tiny paragraph -- same
    "first text run keeps its formatting, other text runs removed" logic --
    via the `.runs` / `.add_run()` contract, without ever touching the
    container element itself. A hyperlink therefore keeps its `r:id`
    relationship and stays clickable (implementation_plan.md D.2.2/D.2.3), a
    tracked insertion keeps its author/date revision metadata, and a field
    keeps its instruction.
    """

    def __init__(self, container_element, parent_paragraph: Paragraph) -> None:
        self._container = container_element
        self._parent_paragraph = parent_paragraph

    @property
    def runs(self) -> list[Run]:
        return [
            Run(r, self._parent_paragraph) for r in self._container.findall(qn("w:r"))
        ]

    def add_run(self, text: str) -> Run:
        r_element = OxmlElement("w:r")
        self._container.append(r_element)
        run = Run(r_element, self._parent_paragraph)
        run.text = text
        return run


# Retained name: `_HyperlinkRunsProxy` was the original, hyperlink-only
# version of `_ScopedRunsProxy` and is referenced by existing tests.
_HyperlinkRunsProxy = _ScopedRunsProxy


@dataclass(slots=True)
class TranslatableUnit:
    """One paragraph (or scoped run group) eligible for translation, with a stable id."""

    unit_id: int
    # `Paragraph` for ordinary units; `_ScopedRunsProxy` for the runs held
    # inside a hyperlink, tracked insertion or simple field. Both expose the
    # `.runs` / `.add_run()` contract that `write_translated_text()` relies on.
    paragraph: Paragraph | _ScopedRunsProxy
    label: str  # "header" | "footer" | "title" | "text" | "table_cell"
    # | "text_box" | "footnote" | "endnote" | "hyperlink"
    # | "tracked_insert" | "field_text"
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


# Elements that hold their own `w:r` runs inside a paragraph. python-docx
# models none of them, so their text appears in neither `Paragraph.text` nor
# `Paragraph.runs` -- each therefore becomes its own unit with a
# `_ScopedRunsProxy` for write-back.
#
# `w:del` is deliberately absent: its `w:delText` is content the author has
# already deleted, is not part of the readable document, and must not be
# rewritten.
_SCOPED_RUN_CONTAINERS: tuple[tuple[str, str], ...] = (
    ("w:hyperlink", "hyperlink"),
    # UAT EC-03 (D-07): a tracked insertion's text used to be neither
    # translated nor dropped -- it was carried into the output still in the
    # source language, making the delivered document silently bilingual.
    ("w:ins", "tracked_insert"),
    # UAT EC-03 (D-08): the cached result of a simple field (a table of
    # contents, a cross-reference, a STYLEREF header) is real visible text
    # and was previously delivered untranslated. Numeric field results (PAGE,
    # NUMPAGES) are filtered out later by the translator's numeric pre-filter.
    ("w:fldSimple", "field_text"),
)


def _scoped_run_containers(paragraph: Paragraph) -> list[tuple[object, str]]:
    """Return `(element, label)` for every scoped run container in `paragraph`."""
    found: list[tuple[object, str]] = []
    for tag, label in _SCOPED_RUN_CONTAINERS:
        found.extend(
            (element, label)
            for element in paragraph._p.findall(qn(tag))  # noqa: SLF001
        )
    return found


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
        containers = _scoped_run_containers(paragraph)
        if containers:
            # D.2.2/D.2.3: split each scoped container's text out into its
            # own unit so it can be translated and written back
            # independently, instead of being duplicated or missed. None of
            # `w:hyperlink`, `w:ins` or `w:fldSimple` contributes to
            # `paragraph.text` or `paragraph.runs` in python-docx, so text
            # inside them is invisible to the ordinary paragraph path.
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
            for container, container_label in containers:
                container_text = "".join(
                    r.text or "" for r in container.findall(qn("w:r"))
                )
                if not container_text.strip():
                    continue
                units.append(
                    TranslatableUnit(
                        unit_id=next_id,
                        paragraph=_ScopedRunsProxy(container, paragraph),
                        label=container_label,
                        text=container_text,
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

    Keeps the unit's first TEXT run's formatting (font, bold, italic, size,
    color, language) and removes the other text runs -- this preserves
    paragraph-level fidelity (alignment, list numbering, style, table cell
    membership) but does not preserve character-level mixed formatting
    *within* one paragraph (e.g. a single bold word mid-sentence). This is
    a documented, intentional v1 tradeoff.

    Reference runs -- footnote, endnote and comment marks, field codes
    (`w:fldChar`/`w:instrText`), drawings and embedded objects -- are left
    exactly where they are; see `_is_reference_run()` for why deleting them
    corrupted footnotes and tables of contents (UAT EC-03, D-01/D-02).

    For a scoped unit (`unit.paragraph` is a `_ScopedRunsProxy` over a
    hyperlink, tracked insertion or simple field),
    only the hyperlink's own inner run(s) are touched -- the `w:hyperlink`
    element and its relationship (i.e. the URL) are never modified, so the
    link keeps working with translated display text
    (implementation_plan.md D.2.2/D.2.3).

    Images, embedded objects, and every other part of the document
    (styles.xml, numbering.xml, section properties, media relationships)
    are never touched -- only run text within this unit is mutated.
    """
    paragraph = unit.paragraph
    text_runs = [run for run in paragraph.runs if not _is_reference_run(run)]
    if not text_runs:
        # No text-bearing run (rare: e.g. a paragraph that is only a field
        # code or an emptied-out hyperlink) -- add one for the translation.
        paragraph.add_run(translated_text)
        return

    first_run = text_runs[0]
    first_run.text = translated_text
    for extra_run in text_runs[1:]:
        extra_run.text = ""
        extra_run._element.getparent().remove(extra_run._element)  # noqa: SLF001
