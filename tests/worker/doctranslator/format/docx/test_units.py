"""Tests for src/worker/doctranslator/format/docx/units.py.

implementation_plan.md Phase D.2 (EC-04): footnotes, endnotes, text boxes,
and hyperlink display text were all previously invisible to `extract_units`
(or, for hyperlinks, visible-but-broken on write-back). Every fixture here
is a real DOCX built in-process with python-docx plus raw zip/XML surgery
(footnotes.xml, endnotes.xml, and `w:txbxContent` have no python-docx
authoring API), then re-opened with `docx.Document` -- no mocked XML trees.
"""

from __future__ import annotations

import io
import zipfile

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from src.worker.doctranslator.format.docx.units import extract_units
from src.worker.doctranslator.format.docx.units import flush_note_parts
from src.worker.doctranslator.format.docx.units import write_translated_text


def _add_hyperlink(paragraph, url: str, text: str):
    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)  # noqa: SLF001
    return hyperlink


def _make_docx_with_footnote() -> bytes:
    """Build a real DOCX with a footnote via raw zip/XML surgery.

    python-docx has no authoring API for footnotes.xml, so this mirrors
    exactly what Word itself produces: a `w:footnoteReference` in
    document.xml, a `word/footnotes.xml` part with `[Content_Types].xml`
    and relationship entries wiring it in.
    """
    doc = Document()
    doc.add_paragraph("Body text before note")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    zin = zipfile.ZipFile(buf)
    names = zin.namelist()
    content_types = zin.read("[Content_Types].xml").decode()
    doc_xml = zin.read("word/document.xml").decode()
    doc_rels = zin.read("word/_rels/document.xml.rels").decode()

    doc_xml = doc_xml.replace(
        "<w:t>Body text before note</w:t></w:r>",
        "<w:t>Body text before note</w:t></w:r>"
        '<w:r><w:footnoteReference w:id="1"/></w:r>',
    )
    footnotes_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:footnotes xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main">'
        '<w:footnote w:type="separator" w:id="-1"><w:p><w:r>'
        "<w:separator/></w:r></w:p></w:footnote>"
        '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r>'
        "<w:continuationSeparator/></w:r></w:p></w:footnote>"
        '<w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r>'
        '<w:r><w:t xml:space="preserve"> Footnote text needing translation.'
        "</w:t></w:r></w:p></w:footnote>"
        "</w:footnotes>"
    )
    content_types = content_types.replace(
        "</Types>",
        '<Override PartName="/word/footnotes.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.wordprocessingml'
        '.footnotes+xml"/></Types>',
    )
    doc_rels = doc_rels.replace(
        "</Relationships>",
        '<Relationship Id="rIdFootnotes1" Type='
        '"http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/footnotes" Target="footnotes.xml"/></Relationships>',
    )

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in names:
            if name == "[Content_Types].xml":
                zout.writestr(name, content_types)
            elif name == "word/document.xml":
                zout.writestr(name, doc_xml)
            elif name == "word/_rels/document.xml.rels":
                zout.writestr(name, doc_rels)
            else:
                zout.writestr(name, zin.read(name))
        zout.writestr("word/footnotes.xml", footnotes_xml)
    return out.getvalue()


def _make_docx_with_endnote() -> bytes:
    """Same as `_make_docx_with_footnote` but for endnotes.xml."""
    doc = Document()
    doc.add_paragraph("Body text before note")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    zin = zipfile.ZipFile(buf)
    names = zin.namelist()
    content_types = zin.read("[Content_Types].xml").decode()
    doc_xml = zin.read("word/document.xml").decode()
    doc_rels = zin.read("word/_rels/document.xml.rels").decode()

    doc_xml = doc_xml.replace(
        "<w:t>Body text before note</w:t></w:r>",
        "<w:t>Body text before note</w:t></w:r>"
        '<w:r><w:endnoteReference w:id="1"/></w:r>',
    )
    endnotes_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:endnotes xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main">'
        '<w:endnote w:type="separator" w:id="-1"><w:p><w:r>'
        "<w:separator/></w:r></w:p></w:endnote>"
        '<w:endnote w:type="continuationSeparator" w:id="0"><w:p><w:r>'
        "<w:continuationSeparator/></w:r></w:p></w:endnote>"
        '<w:endnote w:id="1"><w:p><w:r><w:endnoteRef/></w:r>'
        '<w:r><w:t xml:space="preserve"> Endnote text needing translation.'
        "</w:t></w:r></w:p></w:endnote>"
        "</w:endnotes>"
    )
    content_types = content_types.replace(
        "</Types>",
        '<Override PartName="/word/endnotes.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.wordprocessingml'
        '.endnotes+xml"/></Types>',
    )
    doc_rels = doc_rels.replace(
        "</Relationships>",
        '<Relationship Id="rIdEndnotes1" Type='
        '"http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/endnotes" Target="endnotes.xml"/></Relationships>',
    )

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in names:
            if name == "[Content_Types].xml":
                zout.writestr(name, content_types)
            elif name == "word/document.xml":
                zout.writestr(name, doc_xml)
            elif name == "word/_rels/document.xml.rels":
                zout.writestr(name, doc_rels)
            else:
                zout.writestr(name, zin.read(name))
        zout.writestr("word/endnotes.xml", endnotes_xml)
    return out.getvalue()


def _make_docx_with_textbox() -> bytes:
    """Build a real DOCX with a `w:txbxContent` text box via zip/XML surgery."""
    doc = Document()
    doc.add_paragraph("Body para one")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    zin = zipfile.ZipFile(buf)
    names = zin.namelist()
    doc_xml = zin.read("word/document.xml").decode()

    txbx_paragraph = (
        "<w:p><w:r><w:pict><v:shape><v:textbox><w:txbxContent><w:p><w:r>"
        "<w:t>Text box content needing translation</w:t></w:r></w:p>"
        "</w:txbxContent></v:textbox></v:shape></w:pict></w:r></w:p>"
    )
    doc_xml = doc_xml.replace("<w:sectPr", txbx_paragraph + "<w:sectPr")

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in names:
            if name == "word/document.xml":
                zout.writestr(name, doc_xml)
            else:
                zout.writestr(name, zin.read(name))
    return out.getvalue()


class TestFootnotes:
    def test_footnote_text_is_extracted(self, tmp_path):
        path = tmp_path / "footnote.docx"
        path.write_bytes(_make_docx_with_footnote())
        document = Document(str(path))
        units, note_parts = extract_units(document)

        labels = [u.label for u in units]
        assert "footnote" in labels
        footnote_unit = next(u for u in units if u.label == "footnote")
        assert "Footnote text needing translation" in footnote_unit.text
        assert len(note_parts) == 1

    def test_footnote_separator_markup_is_never_extracted(self, tmp_path):
        """The `w:type="separator"`/`continuationSeparator` footnotes are
        Word-generated boilerplate (a horizontal rule), never user content."""
        path = tmp_path / "footnote.docx"
        path.write_bytes(_make_docx_with_footnote())
        document = Document(str(path))
        units, _ = extract_units(document)
        assert len([u for u in units if u.label == "footnote"]) == 1

    def test_footnote_translation_round_trips_through_save(self, tmp_path):
        """D.2.1: translated footnote text must survive `document.save()`."""
        path = tmp_path / "footnote.docx"
        out_path = tmp_path / "footnote_out.docx"
        path.write_bytes(_make_docx_with_footnote())
        document = Document(str(path))
        units, note_parts = extract_units(document)

        for unit in units:
            if unit.label == "footnote":
                write_translated_text(unit, "TRANSLATED FOOTNOTE")
        flush_note_parts(note_parts)
        document.save(str(out_path))

        reloaded = Document(str(out_path))
        footnotes_part = reloaded.part.part_related_by(RT.FOOTNOTES)
        assert b"TRANSLATED FOOTNOTE" in footnotes_part.blob
        assert b"Footnote text needing translation" not in footnotes_part.blob


class TestEndnotes:
    def test_endnote_text_is_extracted_and_round_trips(self, tmp_path):
        path = tmp_path / "endnote.docx"
        out_path = tmp_path / "endnote_out.docx"
        path.write_bytes(_make_docx_with_endnote())
        document = Document(str(path))
        units, note_parts = extract_units(document)

        endnote_unit = next(u for u in units if u.label == "endnote")
        assert "Endnote text needing translation" in endnote_unit.text

        write_translated_text(endnote_unit, "TRANSLATED ENDNOTE")
        flush_note_parts(note_parts)
        document.save(str(out_path))

        reloaded = Document(str(out_path))
        endnotes_part = reloaded.part.part_related_by(RT.ENDNOTES)
        assert b"TRANSLATED ENDNOTE" in endnotes_part.blob


class TestTextBoxes:
    def test_text_box_content_is_extracted(self, tmp_path):
        path = tmp_path / "textbox.docx"
        path.write_bytes(_make_docx_with_textbox())
        document = Document(str(path))
        units, _ = extract_units(document)

        labels = [u.label for u in units]
        assert "text_box" in labels
        txbx_unit = next(u for u in units if u.label == "text_box")
        assert txbx_unit.text == "Text box content needing translation"

    def test_text_box_translation_round_trips_through_save(self, tmp_path):
        path = tmp_path / "textbox.docx"
        out_path = tmp_path / "textbox_out.docx"
        path.write_bytes(_make_docx_with_textbox())
        document = Document(str(path))
        units, note_parts = extract_units(document)

        for unit in units:
            if unit.label == "text_box":
                write_translated_text(unit, "TRANSLATED TEXT BOX")
        flush_note_parts(note_parts)
        document.save(str(out_path))

        reloaded = Document(str(out_path))
        body_xml = reloaded.element.body.xml
        assert "TRANSLATED TEXT BOX" in body_xml
        assert "Text box content needing translation" not in body_xml


class TestHyperlinks:
    def test_hyperlink_display_text_is_its_own_unit(self):
        doc = Document()
        paragraph = doc.add_paragraph("Visit ")
        _add_hyperlink(paragraph, "https://example.com", "our website")
        paragraph.add_run(" for more info.")

        units, _ = extract_units(doc)
        assert len(units) == 2
        assert units[0].label == "text"
        assert units[0].text == "Visit  for more info."
        assert units[1].label == "hyperlink"
        assert units[1].text == "our website"

    def test_hyperlink_translation_does_not_duplicate_text(self):
        """D.2.2/D.2.3: translating both units back must not duplicate text
        or leave the original hyperlink text stuck in the paragraph."""
        doc = Document()
        paragraph = doc.add_paragraph("Visit ")
        _add_hyperlink(paragraph, "https://example.com", "our website")
        paragraph.add_run(" for more info.")

        units, note_parts = extract_units(doc)
        for unit in units:
            if unit.label == "text":
                write_translated_text(unit, "Besuchen Sie fuer mehr Informationen.")
            elif unit.label == "hyperlink":
                write_translated_text(unit, "unsere Webseite")
        flush_note_parts(note_parts)

        assert paragraph.text == (
            "Besuchen Sie fuer mehr Informationen.unsere Webseite"
        )
        assert "our website" not in paragraph.text
        assert "Visit" not in paragraph.text

    def test_hyperlink_relationship_survives_translation(self, tmp_path):
        """The `r:id` relationship (i.e. the URL) must be untouched."""
        out_path = tmp_path / "hyperlink_out.docx"
        doc = Document()
        paragraph = doc.add_paragraph("Visit ")
        hyperlink = _add_hyperlink(paragraph, "https://example.com", "our website")
        r_id_before = hyperlink.get(qn("r:id"))

        units, note_parts = extract_units(doc)
        for unit in units:
            if unit.label == "hyperlink":
                write_translated_text(unit, "unsere Webseite")
        flush_note_parts(note_parts)
        doc.save(str(out_path))

        reloaded = Document(str(out_path))
        reloaded_paragraph = reloaded.paragraphs[0]
        reloaded_hyperlink = reloaded_paragraph._p.findall(  # noqa: SLF001
            qn("w:hyperlink")
        )[0]
        r_id_after = reloaded_hyperlink.get(qn("r:id"))
        rel = reloaded_paragraph.part.rels[r_id_after]
        assert rel.target_ref == "https://example.com"
        assert r_id_after == r_id_before

    def test_paragraph_without_hyperlink_is_unaffected(self):
        doc = Document()
        doc.add_paragraph("Plain paragraph, no links here.")
        units, _ = extract_units(doc)
        assert len(units) == 1
        assert units[0].label == "text"
        assert units[0].text == "Plain paragraph, no links here."
