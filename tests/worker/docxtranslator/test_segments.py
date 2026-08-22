from fixtures.docx_builder import W_NS
from fixtures.docx_builder import build_docx
from fixtures.docx_builder import paragraph
from fixtures.docx_builder import run
from fixtures.docx_builder import table
from lxml import etree
from src.worker.docxtranslator.package import DocxPackage
from src.worker.docxtranslator.segments import collect_segments
from src.worker.docxtranslator.segments import extract_text


def segments_for(body: str, tmp_path):
    path = tmp_path / "doc.docx"
    path.write_bytes(build_docx(body))
    package = DocxPackage.open(path)
    return package, collect_segments(package.text_parts())


class TestCollectSegments:
    def test_one_segment_per_paragraph(self, tmp_path):
        body = paragraph(run("First")) + paragraph(run("Second"))
        _, segments = segments_for(body, tmp_path)
        assert [s.source_text for s in segments] == ["First", "Second"]

    def test_indexes_are_sequential(self, tmp_path):
        body = paragraph(run("A")) + paragraph(run("B")) + paragraph(run("C"))
        _, segments = segments_for(body, tmp_path)
        assert [s.index for s in segments] == [0, 1, 2]

    def test_table_cell_paragraphs_are_segments(self, tmp_path):
        body = table(["Item", "Amount"], ["Licence", "1000"])
        _, segments = segments_for(body, tmp_path)
        assert [s.source_text for s in segments] == [
            "Item",
            "Amount",
            "Licence",
            "1000",
        ]

    def test_paragraph_without_text_is_skipped(self, tmp_path):
        body = paragraph() + paragraph(run("Real"))
        _, segments = segments_for(body, tmp_path)
        assert [s.source_text for s in segments] == ["Real"]

    def test_text_box_paragraph_is_its_own_segment(self, tmp_path):
        # A text box nests w:p inside a run of the outer paragraph; the outer
        # paragraph must not absorb the inner text.
        body = (
            "<w:p>"
            + run("Outer ")
            + "<w:r><w:pict><v:shape xmlns:v='urn:schemas-microsoft-com:vml'>"
            "<w:txbxContent xmlns:w='"
            + W_NS
            + "'>"
            + paragraph(run("Inside box"))
            + "</w:txbxContent></v:shape></w:pict></w:r>"
            "</w:p>"
        )
        _, segments = segments_for(body, tmp_path)
        texts = [s.source_text for s in segments]
        assert "Outer " in texts
        assert "Inside box" in texts

    def test_field_code_text_is_ignored(self, tmp_path):
        body = (
            "<w:p>"
            '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            "<w:r><w:instrText>PAGE  \\* MERGEFORMAT</w:instrText></w:r>"
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>' + run("Visible") + "</w:p>"
        )
        _, segments = segments_for(body, tmp_path)
        assert [s.source_text for s in segments] == ["Visible"]

    def test_tracked_deletion_is_ignored(self, tmp_path):
        body = (
            "<w:p>"
            '<w:del w:id="1"><w:r><w:delText>removed</w:delText></w:r></w:del>'
            + run("kept")
            + "</w:p>"
        )
        _, segments = segments_for(body, tmp_path)
        assert [s.source_text for s in segments] == ["kept"]


class TestGrouping:
    def test_same_formatting_runs_merge(self, tmp_path):
        body = paragraph(run("Hello "), run("world"))
        _, segments = segments_for(body, tmp_path)
        assert len(segments[0].groups) == 1
        assert segments[0].source_text == "Hello world"

    def test_different_formatting_splits_groups(self, tmp_path):
        body = paragraph(run("Due "), run("within 30 days", bold=True))
        _, segments = segments_for(body, tmp_path)
        assert segments[0].group_texts == ["Due ", "within 30 days"]

    def test_tab_forces_group_break(self, tmp_path):
        body = "<w:p>" + run("Label") + "<w:r><w:tab/></w:r>" + run("Value") + "</w:p>"
        _, segments = segments_for(body, tmp_path)
        assert segments[0].group_texts == ["Label", "Value"]

    def test_line_break_forces_group_break(self, tmp_path):
        body = "<w:p><w:r><w:t>One</w:t><w:br/><w:t>Two</w:t></w:r></w:p>"
        _, segments = segments_for(body, tmp_path)
        assert segments[0].group_texts == ["One", "Two"]

    def test_hyperlink_does_not_merge_with_plain_text(self, tmp_path):
        body = (
            "<w:p>"
            + run("See ")
            + '<w:hyperlink r:id="rId5" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            + run("the policy")
            + "</w:hyperlink>"
            + "</w:p>"
        )
        _, segments = segments_for(body, tmp_path)
        assert segments[0].group_texts == ["See ", "the policy"]


class TestIsTranslatable:
    def test_letters_are_translatable(self, tmp_path):
        _, segments = segments_for(paragraph(run("Total")), tmp_path)
        assert segments[0].is_translatable() is True

    def test_digits_only_is_not_translatable(self, tmp_path):
        _, segments = segments_for(paragraph(run("1 000,00")), tmp_path)
        assert segments[0].is_translatable() is False

    def test_punctuation_only_is_not_translatable(self, tmp_path):
        _, segments = segments_for(paragraph(run(" --- ")), tmp_path)
        assert segments[0].is_translatable() is False


class TestWriteBack:
    def test_write_replaces_group_text(self, tmp_path):
        body = paragraph(run("Due "), run("within 30 days", bold=True))
        package, segments = segments_for(body, tmp_path)
        segments[0].write(["Vencimiento ", "en 30 días"])
        assert segments[0].source_text == "Vencimiento en 30 días"

    def test_write_keeps_bold_run_properties(self, tmp_path):
        body = paragraph(run("Due "), run("within 30 days", bold=True))
        package, segments = segments_for(body, tmp_path)
        segments[0].write(["Vencimiento ", "en 30 días"])

        out = tmp_path / "out.docx"
        package.save(out)
        xml = etree.tostring(
            DocxPackage.open(out).part_root("word/document.xml"), encoding="unicode"
        )
        assert "en 30 días" in xml
        assert "<w:b/>" in xml

    def test_multi_node_group_blanks_trailing_nodes(self, tmp_path):
        body = paragraph(run("Hello "), run("world"))
        package, segments = segments_for(body, tmp_path)
        segments[0].write(["Hola mundo"])

        root = package.part_root("word/document.xml")
        texts = [node.text or "" for node in root.iter(f"{{{W_NS}}}t")]
        assert texts == ["Hola mundo", ""]

    def test_write_joined_collapses_onto_first_group(self, tmp_path):
        body = paragraph(run("Due "), run("within 30 days", bold=True))
        _, segments = segments_for(body, tmp_path)
        segments[0].write_joined("Vencimiento en 30 días")
        assert segments[0].group_texts == ["Vencimiento en 30 días", ""]

    def test_write_rejects_wrong_group_count(self, tmp_path):
        body = paragraph(run("Due "), run("soon", bold=True))
        _, segments = segments_for(body, tmp_path)
        try:
            segments[0].write(["only one"])
        except ValueError as exc:
            assert "group texts" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_table_structure_survives_write_back(self, tmp_path):
        body = table(["Item", "Amount"], ["Licence", "1000"])
        package, segments = segments_for(body, tmp_path)
        for segment in segments:
            if segment.is_translatable():
                segment.write(["TRADUCIDO"] + [""] * (len(segment.groups) - 1))

        out = tmp_path / "out.docx"
        package.save(out)
        root = DocxPackage.open(out).part_root("word/document.xml")

        # The table is still a real Word table: same grid, rows, and cells.
        assert len(root.findall(f".//{{{W_NS}}}tbl")) == 1
        assert len(root.findall(f".//{{{W_NS}}}tr")) == 2
        assert len(root.findall(f".//{{{W_NS}}}tc")) == 4
        assert len(root.findall(f".//{{{W_NS}}}gridCol")) == 2
        assert root.find(f".//{{{W_NS}}}tblPr") is not None

    def test_whitespace_preserved_attribute_set(self, tmp_path):
        body = paragraph(run("x"))
        package, segments = segments_for(body, tmp_path)
        segments[0].write(["  padded  "])
        node = package.part_root("word/document.xml").find(f".//{{{W_NS}}}t")
        assert node.get("{http://www.w3.org/XML/1998/namespace}space") == "preserve"

    def test_whitespace_attribute_removed_when_not_needed(self, tmp_path):
        body = paragraph(run(" padded "))
        package, segments = segments_for(body, tmp_path)
        segments[0].write(["tight"])
        node = package.part_root("word/document.xml").find(f".//{{{W_NS}}}t")
        assert "{http://www.w3.org/XML/1998/namespace}space" not in node.attrib


class TestExtractText:
    def test_joins_non_empty_segments(self, tmp_path):
        body = paragraph(run("One")) + paragraph(run("   ")) + paragraph(run("Two"))
        _, segments = segments_for(body, tmp_path)
        assert extract_text(segments) == "One\nTwo"

    def test_respects_max_chars(self, tmp_path):
        body = paragraph(run("aaaa")) + paragraph(run("bbbb")) + paragraph(run("cccc"))
        _, segments = segments_for(body, tmp_path)
        assert extract_text(segments, max_chars=5) == "aaaa\nbbbb"
