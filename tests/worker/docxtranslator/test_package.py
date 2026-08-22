import zipfile

import pytest
from fixtures.docx_builder import build_docx
from fixtures.docx_builder import paragraph
from fixtures.docx_builder import run
from fixtures.docx_builder import simple_docx
from src.worker.docxtranslator.package import DocxPackage
from src.worker.docxtranslator.package import InvalidDocxError
from src.worker.docxtranslator.package import is_text_part


class TestIsTextPart:
    @pytest.mark.parametrize(
        "name",
        [
            "word/document.xml",
            "word/header1.xml",
            "word/footer2.xml",
            "word/footnotes.xml",
            "word/endnotes.xml",
            "word/comments.xml",
        ],
    )
    def test_body_text_parts(self, name):
        assert is_text_part(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "word/styles.xml",
            "word/numbering.xml",
            "word/settings.xml",
            "word/theme/theme1.xml",
            "word/media/image1.png",
            "[Content_Types].xml",
            "word/_rels/document.xml.rels",
        ],
    )
    def test_non_text_parts(self, name):
        assert is_text_part(name) is False


class TestDocxPackageOpen:
    def test_missing_file(self, tmp_path):
        with pytest.raises(InvalidDocxError, match="not found"):
            DocxPackage.open(tmp_path / "nope.docx")

    def test_not_a_zip(self, tmp_path):
        path = tmp_path / "fake.docx"
        path.write_bytes(b"this is not a zip file")
        with pytest.raises(InvalidDocxError, match="not a valid .docx"):
            DocxPackage.open(path)

    def test_encrypted_package_rejected(self, tmp_path):
        path = tmp_path / "locked.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("EncryptedPackage", b"\x00\x01")
            archive.writestr("EncryptionInfo", b"\x00\x01")
        with pytest.raises(InvalidDocxError, match="password-protected"):
            DocxPackage.open(path)

    def test_legacy_binary_doc_rejected(self, tmp_path):
        path = tmp_path / "legacy.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.bin", b"\x00")
        with pytest.raises(InvalidDocxError, match="binary Word document"):
            DocxPackage.open(path)

    def test_missing_document_part(self, tmp_path):
        path = tmp_path / "empty.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/styles.xml", "<styles/>")
        with pytest.raises(InvalidDocxError, match="missing word/document.xml"):
            DocxPackage.open(path)


class TestDocxPackageParts:
    def test_document_part_first(self, tmp_path):
        path = tmp_path / "doc.docx"
        path.write_bytes(
            build_docx(
                paragraph(run("Body")),
                extra_parts={
                    "word/header1.xml": (
                        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                        '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                        f"{paragraph(run('Header text'))}</w:hdr>"
                    ),
                    "word/styles.xml": "<styles/>",
                },
            )
        )
        names = DocxPackage.open(path).text_part_names()
        assert names[0] == "word/document.xml"
        assert "word/header1.xml" in names
        assert "word/styles.xml" not in names


class TestDocxPackageSave:
    def test_round_trip_preserves_all_entries(self, tmp_path):
        source = tmp_path / "in.docx"
        source.write_bytes(simple_docx())
        dest = tmp_path / "out.docx"

        DocxPackage.open(source).save(dest)

        with zipfile.ZipFile(source) as before, zipfile.ZipFile(dest) as after:
            assert before.namelist() == after.namelist()
            # Untouched parts come back byte-for-byte identical.
            for name in ("[Content_Types].xml", "_rels/.rels"):
                assert before.read(name) == after.read(name)

    def test_saved_package_reopens(self, tmp_path):
        source = tmp_path / "in.docx"
        source.write_bytes(simple_docx())
        dest = tmp_path / "out.docx"

        package = DocxPackage.open(source)
        package.part_root("word/document.xml")  # force parse + re-serialize
        package.save(dest)

        reopened = DocxPackage.open(dest)
        assert reopened.text_part_names() == ["word/document.xml"]

    def test_creates_missing_output_directory(self, tmp_path):
        source = tmp_path / "in.docx"
        source.write_bytes(simple_docx())
        dest = tmp_path / "nested" / "deeper" / "out.docx"

        DocxPackage.open(source).save(dest)

        assert dest.exists()
