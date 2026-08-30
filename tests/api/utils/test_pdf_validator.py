import io
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi import UploadFile
from src.api.exceptions import ValidationError
from src.api.utils.pdf_validator import PDFValidator


class TestPDFValidator:
    def test_validate_pdf_bytes_too_large(self):
        with patch("src.api.utils.pdf_validator.settings") as mock_settings:
            # Set a threshold large enough to be non-zero in MB formatting (e.g. 2MB)
            mock_settings.MAX_FILE_SIZE = 2 * 1024 * 1024
            with pytest.raises(ValidationError, match="File size exceeds 2MB limit"):
                PDFValidator.validate_pdf_bytes(
                    b"a" * (2 * 1024 * 1024 + 1), "test.pdf"
                )

    @patch("src.api.utils.pdf_validator.PDFValidator.extract_pdf_metadata")
    def test_validate_pdf_bytes_success(self, mock_extract):
        mock_extract.return_value = {"page_count": 1, "encrypted": False}
        content, metadata = PDFValidator.validate_pdf_bytes(b"%PDF-1.4", "test.pdf")
        assert content == b"%PDF-1.4"
        assert metadata["filename"] == "test.pdf"

    @patch("src.api.utils.pdf_validator.PDFValidator.extract_pdf_metadata")
    def test_validate_pdf_bytes_encrypted(self, mock_extract):
        mock_extract.return_value = {"page_count": 1, "encrypted": True}
        with pytest.raises(ValidationError, match="Encrypted"):
            PDFValidator.validate_pdf_bytes(b"%PDF-1.4", "test.pdf")

    def test_validate_pdf_bytes_owner_password_only_rejected(self):
        """An owner-password-only PDF opens freely (is_encrypted=False,
        needs_pass=0) but restricts the COPY permission bit -- verified
        live with real PyMuPDF encryption. Must still be rejected so the
        original restriction is never silently lost."""
        import fitz

        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "hello world")
        owner_password_pdf = doc.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            permissions=fitz.PDF_PERM_ACCESSIBILITY,
        )
        doc.close()
        with pytest.raises(ValidationError, match="restricts text extraction"):
            PDFValidator.validate_pdf_bytes(owner_password_pdf, "restricted.pdf")

    def test_validate_pdf_bytes_user_password_rejected(self):
        """A real user-password-protected PDF is rejected as password_protected."""
        import fitz

        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "hello world")
        user_password_pdf = doc.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner"
        )
        doc.close()
        with pytest.raises(ValidationError, match="Password-protected"):
            PDFValidator.validate_pdf_bytes(user_password_pdf, "protected.pdf")

    def test_validate_pdf_bytes_unrestricted_pdf_accepted(self):
        """A plain, unencrypted PDF with full permissions passes the new check."""
        import fitz

        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "hello world")
        plain_pdf = doc.tobytes()
        doc.close()
        content, metadata = PDFValidator.validate_pdf_bytes(plain_pdf, "plain.pdf")
        assert content == plain_pdf
        assert metadata["filename"] == "plain.pdf"

    def test_validate_pdf_bytes_image_only_rejected(self):
        """B.5.1: an image-only PDF (no extractable text layer on any
        sampled page) is rejected with a distinct no_text_layer message.
        Verified live: page.get_text("text") returns '' for an
        image-only page."""
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 80))
        pix.set_rect(pix.irect, (250, 250, 250))
        page.insert_image(fitz.Rect(50, 50, 250, 130), pixmap=pix)
        image_only_pdf = doc.tobytes()
        doc.close()
        with pytest.raises(ValidationError, match="no extractable text layer"):
            PDFValidator.validate_pdf_bytes(image_only_pdf, "scanned.pdf")

    def test_validate_pdf_bytes_mostly_scanned_but_one_text_page_accepted(self):
        """B.5.5: a multi-page PDF where at least one of the sampled
        (first PDF_TEXT_PROBE_PAGES) pages has real text still passes
        this API-level probe -- the per-page scanned-detection heuristic
        deeper in the worker pipeline decides page-by-page OCR-workaround
        behaviour; this validator only rejects documents with *zero*
        extractable text across the sample."""
        import fitz

        doc = fitz.open()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 80))
        pix.set_rect(pix.irect, (250, 250, 250))
        image_page = doc.new_page()
        image_page.insert_image(fitz.Rect(50, 50, 250, 130), pixmap=pix)
        text_page = doc.new_page()
        text_page.insert_text((72, 72), "hello world")
        mixed_pdf = doc.tobytes()
        doc.close()
        content, metadata = PDFValidator.validate_pdf_bytes(mixed_pdf, "mixed.pdf")
        assert content == mixed_pdf
        assert metadata["has_text_layer"] is True

    def test_extract_pdf_metadata_has_text_layer_true_for_real_text(self):
        import fitz

        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "hello world")
        content = doc.tobytes()
        doc.close()
        meta = PDFValidator.extract_pdf_metadata(content)
        assert meta["has_text_layer"] is True
        assert meta["text_char_count"] > 0

    def test_extract_pdf_metadata_has_text_layer_false_for_image_only(self):
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 80))
        pix.set_rect(pix.irect, (250, 250, 250))
        page.insert_image(fitz.Rect(50, 50, 250, 130), pixmap=pix)
        content = doc.tobytes()
        doc.close()
        meta = PDFValidator.extract_pdf_metadata(content)
        assert meta["has_text_layer"] is False
        assert meta["text_char_count"] == 0

    def test_validate_pdf_bytes_corrupted(self):
        # Content lacking the %PDF- magic bytes is now rejected up front by
        # the container-sniffing precheck, before fitz.open is ever called.
        with pytest.raises(ValidationError, match="File is not a valid PDF"):
            PDFValidator.validate_pdf_bytes(b"not a pdf", "test.pdf")

    def test_validate_pdf_bytes_empty(self):
        with pytest.raises(ValidationError, match="Document is empty"):
            PDFValidator.validate_pdf_bytes(b"", "test.pdf")

    def test_validate_pdf_bytes_wrong_magic_bytes(self):
        """A ZIP renamed to .pdf is rejected with a specific message, not a fitz crash."""
        with pytest.raises(ValidationError, match="File is not a valid PDF"):
            PDFValidator.validate_pdf_bytes(b"PK\x03\x04fake-zip-content", "fake.pdf")

    @patch("src.api.utils.pdf_validator.fitz.open")
    def test_extract_pdf_metadata_success(self, mock_open):
        mock_doc = MagicMock()
        mock_doc.metadata = {"format": "PDF 1.4", "title": "Test"}
        mock_doc.__len__.return_value = 5
        mock_doc.is_encrypted = False
        mock_doc.needs_pass = False
        mock_open.return_value = mock_doc

        metadata = PDFValidator.extract_pdf_metadata(b"fake-content")
        assert metadata["page_count"] == 5
        assert metadata["title"] == "Test"

    @patch("src.api.utils.pdf_validator.fitz.open")
    def test_extract_pdf_metadata_failure(self, mock_open):
        mock_open.side_effect = Exception("Fitz Error")
        with pytest.raises(ValidationError, match="Failed to extract PDF metadata"):
            PDFValidator.extract_pdf_metadata(b"bad-content")

    @pytest.mark.asyncio
    async def test_validate_pdf_file_success(self):
        file_content = b"%PDF-1.4\n%%EOF"
        file = UploadFile(filename="test.pdf", file=io.BytesIO(file_content))
        with patch.object(
            PDFValidator,
            "extract_pdf_metadata",
            return_value={"page_count": 1, "encrypted": False},
        ):
            content, metadata = await PDFValidator.validate_pdf_file(file)
            assert content == file_content
            assert metadata["page_count"] == 1

    @pytest.mark.asyncio
    async def test_validate_pdf_file_empty(self):
        file = UploadFile(filename="empty.pdf", file=io.BytesIO(b""))
        with pytest.raises(ValidationError, match="Document is empty"):
            await PDFValidator.validate_pdf_file(file)

    @pytest.mark.asyncio
    async def test_validate_pdf_file_wrong_magic_bytes(self):
        """A ZIP renamed to .pdf is rejected with a specific message."""
        file = UploadFile(
            filename="fake.pdf", file=io.BytesIO(b"PK\x03\x04fake-zip-content")
        )
        with pytest.raises(ValidationError, match="File is not a valid PDF"):
            await PDFValidator.validate_pdf_file(file)

    def test_validate_pdf_bytes_fitz_file_data_error(self):
        import fitz

        with patch(
            "src.api.utils.pdf_validator.PDFValidator.extract_pdf_metadata"
        ) as mock_extract:
            mock_extract.side_effect = fitz.FileDataError("corrupt")
            with pytest.raises(ValidationError, match="Invalid or corrupted PDF file"):
                PDFValidator.validate_pdf_bytes(b"%PDF-1.4", "test.pdf")

    def test_validate_pdf_bytes_generic_exception(self):
        with patch(
            "src.api.utils.pdf_validator.PDFValidator.extract_pdf_metadata"
        ) as mock_extract:
            mock_extract.side_effect = OSError("unexpected")
            with pytest.raises(ValidationError, match="Failed to validate PDF"):
                PDFValidator.validate_pdf_bytes(b"%PDF-1.4", "test.pdf")

    @pytest.mark.asyncio
    async def test_validate_pdf_file_too_large(self):
        from unittest.mock import AsyncMock

        with patch("src.api.utils.pdf_validator.settings") as mock_settings:
            mock_settings.MAX_FILE_SIZE = 10
            # UploadFile.read(n) returns exactly n bytes when file is >= MAX_FILE_SIZE
            file = MagicMock()
            file.read = AsyncMock(return_value=b"x" * 10)
            file.filename = "big.pdf"
            with pytest.raises(ValidationError, match="File size exceeds"):
                await PDFValidator.validate_pdf_file(file)

    @pytest.mark.asyncio
    async def test_validate_pdf_file_encrypted(self):
        file_content = b"%PDF-1.4\n%%EOF"
        file = UploadFile(filename="enc.pdf", file=io.BytesIO(file_content))
        with patch.object(
            PDFValidator,
            "extract_pdf_metadata",
            return_value={"page_count": 1, "encrypted": True},
        ):
            with pytest.raises(ValidationError, match="Encrypted"):
                await PDFValidator.validate_pdf_file(file)

    @pytest.mark.asyncio
    async def test_validate_pdf_file_fitz_error(self):
        import fitz

        file_content = b"%PDF-1.4\n%%EOF"
        file = UploadFile(filename="bad.pdf", file=io.BytesIO(file_content))
        with patch.object(
            PDFValidator,
            "extract_pdf_metadata",
            side_effect=fitz.FileDataError("corrupt"),
        ):
            with pytest.raises(ValidationError, match="Invalid or corrupted PDF file"):
                await PDFValidator.validate_pdf_file(file)

    @pytest.mark.asyncio
    async def test_validate_pdf_file_generic_exception(self):
        file_content = b"%PDF-1.4\n%%EOF"
        file = UploadFile(filename="bad.pdf", file=io.BytesIO(file_content))
        with patch.object(
            PDFValidator,
            "extract_pdf_metadata",
            side_effect=OSError("unexpected error"),
        ):
            with pytest.raises(ValidationError, match="Failed to validate PDF"):
                await PDFValidator.validate_pdf_file(file)

    @pytest.mark.asyncio
    async def test_validate_pdf_file_no_filename_fallback(self):
        """When UploadFile has no filename, metadata should default to 'unknown.pdf'."""
        file_content = b"%PDF-1.4\n%%EOF"
        file = UploadFile(filename=None, file=io.BytesIO(file_content))
        with patch.object(
            PDFValidator,
            "extract_pdf_metadata",
            return_value={"page_count": 1, "encrypted": False},
        ):
            content, metadata = await PDFValidator.validate_pdf_file(file)
            assert metadata["filename"] == "unknown.pdf"
            assert content == file_content

    @patch("src.api.utils.pdf_validator.fitz.open")
    def test_extract_pdf_metadata_none_metadata(self, mock_open):
        """When doc.metadata is None, the `or {}` fallback is used (no KeyError)."""
        mock_doc = MagicMock()
        mock_doc.metadata = None
        mock_doc.__len__.return_value = 2
        mock_doc.is_encrypted = False
        mock_doc.needs_pass = False
        mock_open.return_value = mock_doc

        metadata = PDFValidator.extract_pdf_metadata(b"fake-content")
        assert metadata["page_count"] == 2
        assert metadata["pdf_version"] == "Unknown"
        assert metadata["title"] == ""

    @patch("src.api.utils.pdf_validator.fitz.open")
    def test_extract_pdf_metadata_full_fields(self, mock_open):
        """All metadata fields are populated from doc.metadata."""
        mock_doc = MagicMock()
        mock_doc.metadata = {
            "format": "PDF 1.7",
            "title": "Report",
            "author": "Alice",
            "subject": "Finance",
            "creator": "LibreOffice",
            "producer": "PDFium",
        }
        mock_doc.__len__.return_value = 10
        mock_doc.is_encrypted = False
        mock_doc.needs_pass = False
        mock_open.return_value = mock_doc

        metadata = PDFValidator.extract_pdf_metadata(b"fake")
        assert metadata["pdf_version"] == "PDF 1.7"
        assert metadata["author"] == "Alice"
        assert metadata["subject"] == "Finance"
        assert metadata["creator"] == "LibreOffice"
        assert metadata["producer"] == "PDFium"
        assert metadata["page_count"] == 10

    def test_validate_pdf_bytes_at_exact_size_limit(self):
        """Content exactly at MAX_FILE_SIZE should raise (>= check)."""
        with patch("src.api.utils.pdf_validator.settings") as mock_settings:
            mock_settings.MAX_FILE_SIZE = 5
            with pytest.raises(ValidationError, match="File size exceeds"):
                PDFValidator.validate_pdf_bytes(b"12345", "test.pdf")
