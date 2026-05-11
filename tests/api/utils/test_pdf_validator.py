
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.api.utils.pdf_validator import PDFValidator
from src.api.exceptions import ValidationError
import fitz
from fastapi import UploadFile
import io

class TestPDFValidator:
    def test_validate_pdf_bytes_too_large(self):
        with patch("src.api.utils.pdf_validator.settings") as mock_settings:
            # Set a threshold large enough to be non-zero in MB formatting (e.g. 2MB)
            mock_settings.MAX_FILE_SIZE = 2 * 1024 * 1024
            with pytest.raises(ValidationError, match="File size exceeds 2MB limit"):
                PDFValidator.validate_pdf_bytes(b"a" * (2 * 1024 * 1024 + 1), "test.pdf")

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
        with patch.object(PDFValidator, "extract_pdf_metadata", return_value={"page_count": 1, "encrypted": False}):
            content, metadata = await PDFValidator.validate_pdf_file(file)
            assert content == file_content
            assert metadata["page_count"] == 1

    @pytest.mark.asyncio
    async def test_validate_pdf_file_empty(self):
        file = UploadFile(filename="empty.pdf", file=io.BytesIO(b""))
        with pytest.raises(ValidationError, match="Failed to extract PDF metadata"):
            await PDFValidator.validate_pdf_file(file)
