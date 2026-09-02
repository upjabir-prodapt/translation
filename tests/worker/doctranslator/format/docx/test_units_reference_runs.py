"""Regression tests for the DOCX structure defects found in UAT EC-03.

D-01  footnote and comment reference marks were deleted from the body
D-02  a Word-generated table-of-contents field was dismantled
D-07  tracked-insertion text was carried through untranslated
D-08  a simple field's cached result text was never translated

All four came from the same place: `write_translated_text()` kept only the
first run of a paragraph, and `extract_units()` only ever looked at runs
that are direct children of `w:p`. Every fixture below is built with
python-docx plus raw XML, then exercised through the real functions.
"""

from __future__ import annotations

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from src.worker.doctranslator.format.docx.units import extract_units
from src.worker.doctranslator.format.docx.units import write_translated_text

_NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
)


def _append(paragraph, xml: str):
    paragraph._p.append(parse_xml(xml.replace("<<NS>>", _NS)))  # noqa: SLF001


def _count(paragraph, tag: str) -> int:
    return len(paragraph._p.findall(f".//{qn(tag)}"))  # noqa: SLF001


class TestReferenceRunsSurviveTranslation:
    def test_footnote_reference_is_not_deleted(self):
        """D-01: the body's `w:footnoteReference` must outlive write-back.

        Without it the translated footnote text still sits in footnotes.xml
        but nothing in the document points at it, so Word renders no
        footnote mark at all.
        """
        document = Document()
        paragraph = document.add_paragraph("Charges are payable within thirty days.")
        _append(
            paragraph,
            '<w:r <<NS>>><w:footnoteReference w:id="1"/></w:r>',
        )
        assert _count(paragraph, "w:footnoteReference") == 1

        units, _ = extract_units(document)
        write_translated_text(units[0], "Les frais sont payables sous trente jours.")

        assert _count(paragraph, "w:footnoteReference") == 1
        assert paragraph.text == "Les frais sont payables sous trente jours."

    def test_comment_reference_is_not_deleted(self):
        """D-01: the same applies to a comment's anchor."""
        document = Document()
        paragraph = document.add_paragraph("This clause is subject to legal review.")
        _append(paragraph, '<w:commentRangeStart <<NS>> w:id="10"/>')
        _append(paragraph, '<w:commentRangeEnd <<NS>> w:id="10"/>')
        _append(paragraph, '<w:r <<NS>>><w:commentReference w:id="10"/></w:r>')

        units, _ = extract_units(document)
        write_translated_text(units[0], "Questa clausola è soggetta a revisione.")

        assert _count(paragraph, "w:commentReference") == 1
        assert _count(paragraph, "w:commentRangeStart") == 1

    def test_field_code_runs_are_not_deleted(self):
        """D-02: a ToC field's begin/instrText/separate runs must survive.

        The field-code runs sit *before* the first text run, so the old
        "keep runs[0], delete the rest" rule wrote the translation into the
        `w:fldChar` run and deleted the instruction, leaving flat text Word
        can no longer refresh or navigate.
        """
        document = Document()
        paragraph = document.add_paragraph()
        _append(paragraph, '<w:r <<NS>>><w:fldChar w:fldCharType="begin"/></w:r>')
        _append(
            paragraph,
            '<w:r <<NS>>><w:instrText xml:space="preserve"> TOC \\o "1-3" '
            "</w:instrText></w:r>",
        )
        _append(paragraph, '<w:r <<NS>>><w:fldChar w:fldCharType="separate"/></w:r>')
        _append(
            paragraph,
            '<w:r <<NS>>><w:t xml:space="preserve">1. Definitions</w:t></w:r>',
        )
        _append(paragraph, '<w:r <<NS>>><w:fldChar w:fldCharType="end"/></w:r>')

        units, _ = extract_units(document)
        assert units[0].text == "1. Definitions"

        write_translated_text(units[0], "1. Definizioni")

        assert _count(paragraph, "w:fldChar") == 3
        assert _count(paragraph, "w:instrText") == 1
        assert paragraph.text == "1. Definizioni"

    def test_drawing_run_is_not_deleted(self):
        """An inline image shares its paragraph with the caption text."""
        document = Document()
        paragraph = document.add_paragraph("Figure 1: network topology")
        _append(paragraph, "<w:r <<NS>>><w:drawing/></w:r>")

        units, _ = extract_units(document)
        write_translated_text(units[0], "Figura 1: topologia di rete")

        assert _count(paragraph, "w:drawing") == 1

    def test_extra_text_runs_are_still_collapsed(self):
        """The v1 tradeoff is unchanged: mixed formatting within one
        paragraph still collapses onto the first text run."""
        document = Document()
        paragraph = document.add_paragraph("The cap is ")
        paragraph.add_run("one hundred").bold = True
        paragraph.add_run(" thousand euro.")

        units, _ = extract_units(document)
        write_translated_text(units[0], "Le plafond est de cent mille euros.")

        assert len(paragraph.runs) == 1
        assert paragraph.text == "Le plafond est de cent mille euros."


class TestScopedRunContainers:
    def test_tracked_insertion_is_extracted_and_translated(self):
        """D-07: `w:ins` text used to reach the output still in the source
        language, so the delivered document was silently bilingual."""
        document = Document()
        paragraph = document.add_paragraph("The liability cap is set at ")
        _append(
            paragraph,
            '<w:ins <<NS>> w:id="9" w:author="Reviewer" w:date="2026-01-05T09:00:00Z">'
            '<w:r><w:t xml:space="preserve">one hundred thousand euro</w:t></w:r>'
            "</w:ins>",
        )
        paragraph.add_run(" per contract year.")

        units, _ = extract_units(document)
        labels = {unit.label: unit.text for unit in units}
        assert "tracked_insert" in labels
        assert labels["tracked_insert"] == "one hundred thousand euro"

        for unit in units:
            if unit.label == "tracked_insert":
                write_translated_text(unit, "centomila euro")

        insert = paragraph._p.find(qn("w:ins"))  # noqa: SLF001
        assert insert is not None, "the revision element itself must survive"
        assert insert.get(qn("w:author")) == "Reviewer"
        assert "centomila euro" in "".join(
            t.text or "" for t in insert.findall(f".//{qn('w:t')}")
        )

    def test_deleted_text_is_never_touched(self):
        """`w:del` holds content the author removed -- not translatable."""
        document = Document()
        paragraph = document.add_paragraph("The cap is ")
        _append(
            paragraph,
            '<w:del <<NS>> w:id="8" w:author="Reviewer" w:date="2026-01-05T09:00:00Z">'
            '<w:r><w:delText xml:space="preserve">fifty thousand euro</w:delText></w:r>'
            "</w:del>",
        )

        units, _ = extract_units(document)
        assert all(unit.label != "tracked_insert" for unit in units)
        assert all("fifty thousand euro" not in unit.text for unit in units)

    def test_simple_field_result_text_is_extracted(self):
        """D-08: a `w:fldSimple` cached result is visible text and was
        delivered untranslated."""
        document = Document()
        paragraph = document.add_paragraph()
        _append(
            paragraph,
            '<w:fldSimple <<NS>> w:instr=" TOC \\o &quot;1-3&quot; ">'
            "<w:r><w:t>2. Service levels</w:t></w:r>"
            "</w:fldSimple>",
        )

        units, _ = extract_units(document)
        field_units = [u for u in units if u.label == "field_text"]
        assert len(field_units) == 1
        assert field_units[0].text == "2. Service levels"

        write_translated_text(field_units[0], "2. Livelli di servizio")

        field = paragraph._p.find(qn("w:fldSimple"))  # noqa: SLF001
        assert field is not None, "the field element and its instruction must survive"
        assert field.get(qn("w:instr")).strip().startswith("TOC")
        assert "2. Livelli di servizio" in "".join(
            t.text or "" for t in field.findall(f".//{qn('w:t')}")
        )

    def test_hyperlink_still_gets_its_own_unit(self):
        """The generalisation must not change hyperlink handling."""
        from docx.opc.constants import RELATIONSHIP_TYPE as RT

        document = Document()
        paragraph = document.add_paragraph("Documentation is published at ")
        r_id = paragraph.part.relate_to(
            "https://www.colt.net", RT.HYPERLINK, is_external=True
        )
        _append(
            paragraph,
            f'<w:hyperlink <<NS>> r:id="{r_id}">'
            "<w:r><w:t>the Colt website</w:t></w:r></w:hyperlink>",
        )

        units, _ = extract_units(document)
        labels = [unit.label for unit in units]
        assert "hyperlink" in labels
        hyperlink_unit = next(u for u in units if u.label == "hyperlink")
        assert hyperlink_unit.text == "the Colt website"

        write_translated_text(hyperlink_unit, "il sito web di Colt")
        hyperlink = paragraph._p.find(qn("w:hyperlink"))  # noqa: SLF001
        assert hyperlink.get(qn("r:id")) == r_id
