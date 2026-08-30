"""Tests for src.api.utils.document_validator.DocumentValidator.

Covers implementation_plan.md Phase A (password-protected & structurally
invalid DOCX/TXT uploads) checklist items A.5.1, A.5.4-A.5.10.
"""

import io
import zipfile
from unittest.mock import patch

import pytest
from src.api.exceptions import ValidationError
from src.api.utils.document_validator import DocumentValidator
from src.api.utils.document_validator import detect_container


def _make_real_docx_bytes(text: str = "Hello world.") -> bytes:
    from docx import Document as DocxDocument

    document = DocxDocument()
    document.add_paragraph(text)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def _make_password_protected_docx_bytes() -> bytes:
    """Build a real ECMA-376 Agile-Encrypted (password-protected) .docx.

    Encrypted OOXML packages are an OLE/CFB compound file wrapping an
    `EncryptedPackage` stream. We only need the correct magic bytes for
    `DocumentValidator` (which never attempts decryption), so a minimal
    synthetic CFB header is sufficient and avoids adding a
    msoffcrypto/olefile dependency just to build a test fixture.
    """
    return b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64


def _make_zip_bytes_missing_docx_members() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("a.txt", "hello")
    return buf.getvalue()


def _make_image_only_docx_bytes() -> bytes:
    """Real .docx whose body has no non-whitespace <w:t> run text.

    Mirrors what an image-only Word document looks like structurally
    (empty/whitespace-only paragraphs, no real text runs) without needing
    an embedded raster image, since python-docx's add_picture() requires
    a real decodable image file which is unnecessary for this check.
    """
    from docx import Document as DocxDocument

    document = DocxDocument()
    document.add_paragraph("")
    document.add_paragraph("   ")
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


class TestDetectContainer:
    def test_detects_pdf(self):
        assert detect_container(b"%PDF-1.4\n") == "pdf"

    def test_detects_zip(self):
        assert detect_container(b"PK\x03\x04rest") == "zip"

    def test_detects_ole(self):
        assert (
            detect_container(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 8)
            == "ole"
        )

    def test_unknown_container(self):
        assert detect_container(b"random bytes") == "unknown"


class TestValidateDocxBytes:
    def test_valid_docx_accepted(self):
        content = _make_real_docx_bytes()
        metadata = DocumentValidator.validate_docx_bytes(content, "report.docx")
        assert metadata["filename"] == "report.docx"
        assert metadata["size_bytes"] == len(content)
        assert metadata["page_count"] is None
        assert "checksum" in metadata

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError, match="Document is empty"):
            DocumentValidator.validate_docx_bytes(b"", "empty.docx")

    def test_password_protected_docx_rejected(self):
        """A.5.1: real OLE/CFB-container (password-protected) .docx is rejected."""
        content = _make_password_protected_docx_bytes()
        with pytest.raises(ValidationError, match="Password-protected"):
            DocumentValidator.validate_docx_bytes(content, "protected.docx")

    def test_txt_renamed_to_docx_rejected(self):
        """A.5.5: plain text renamed to .docx has no ZIP magic bytes."""
        with pytest.raises(ValidationError, match="not a valid .docx"):
            DocumentValidator.validate_docx_bytes(b"just plain text", "fake.docx")

    def test_zip_renamed_to_docx_rejected(self):
        """A.5.4: a real ZIP archive that is not a Word package."""
        content = _make_zip_bytes_missing_docx_members()
        with pytest.raises(ValidationError, match="not a Word document"):
            DocumentValidator.validate_docx_bytes(content, "fake.docx")

    def test_corrupt_zip_rejected(self):
        """ZIP magic bytes present but the archive itself is malformed."""
        content = b"PK\x03\x04" + b"\x00" * 10
        with pytest.raises(ValidationError, match="not a valid .docx"):
            DocumentValidator.validate_docx_bytes(content, "corrupt.docx")

    def test_oversized_docx_rejected(self):
        with patch("src.api.utils.document_validator.settings") as mock_settings:
            mock_settings.MAX_FILE_SIZE = 10
            content = b"PK\x03\x04" + b"\x00" * 10
            with pytest.raises(ValidationError, match="File size exceeds"):
                DocumentValidator.validate_docx_bytes(content, "big.docx")

    def test_image_only_docx_rejected(self):
        """B.5.2: a real .docx with no non-whitespace <w:t> run text
        (structurally equivalent to an image-only Word document) is
        rejected with a distinct no_text_layer message."""
        content = _make_image_only_docx_bytes()
        with pytest.raises(ValidationError, match="no translatable text"):
            DocumentValidator.validate_docx_bytes(content, "scanned.docx")

    def test_docx_with_real_text_accepted(self):
        """Regression: a normal .docx with actual text still passes B.2."""
        content = _make_real_docx_bytes("Some real translatable content.")
        metadata = DocumentValidator.validate_docx_bytes(content, "real.docx")
        assert metadata["filename"] == "real.docx"


class TestValidateTxtBytes:
    def test_valid_txt_accepted(self):
        content = "Hello, world! Café.".encode()
        metadata = DocumentValidator.validate_txt_bytes(content, "note.txt")
        assert metadata["filename"] == "note.txt"
        assert metadata["size_bytes"] == len(content)
        assert metadata["page_count"] is None

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError, match="Document is empty"):
            DocumentValidator.validate_txt_bytes(b"", "empty.txt")

    def test_non_utf8_content_rejected(self):
        """A.5.8: Latin-1-only bytes that are not valid UTF-8 are rejected
        rather than silently mojibaked."""
        non_utf8 = "café".encode("latin-1")
        with pytest.raises(ValidationError, match="not valid UTF-8"):
            DocumentValidator.validate_txt_bytes(non_utf8, "note.txt")

    def test_oversized_txt_rejected(self):
        with patch("src.api.utils.document_validator.settings") as mock_settings:
            mock_settings.MAX_FILE_SIZE = 5
            with pytest.raises(ValidationError, match="File size exceeds"):
                DocumentValidator.validate_txt_bytes(b"123456", "big.txt")

    def test_whitespace_only_txt_rejected(self):
        """B.5.3: a .txt containing only whitespace has no translatable text."""
        with pytest.raises(ValidationError, match="no translatable text"):
            DocumentValidator.validate_txt_bytes(b"   \n\t  \n", "blank.txt")
