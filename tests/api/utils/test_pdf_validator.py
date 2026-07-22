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

    def test_validate_pdf_bytes_corrupted(self):
        # fitz.open raises FileDataError for non-PDF, but extract_pdf_metadata wraps it in ValidationError
        with pytest.raises(ValidationError, match="Failed to extract PDF metadata"):
            PDFValidator.validate_pdf_bytes(b"not a pdf", "test.pdf")

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
        with pytest.raises(ValidationError, match="Failed to extract PDF metadata"):
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
