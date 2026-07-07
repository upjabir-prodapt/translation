"""
Unit tests for api/utils/pdf_validator.py — PDFValidator static methods.

All PDF bytes are generated in-memory using PyMuPDF so no real files are needed.
External services (GCS, Firestore) are not involved.
"""

import hashlib
import io
from unittest.mock import AsyncMock

import fitz
import pytest
from fastapi import UploadFile

from api.exceptions import ValidationError
from api.utils.pdf_validator import PDFValidator
from config.constants import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(
    text: str = "Test content",
    pages: int = 1,
    encrypt: bool = False,
) -> bytes:
    """Build a PDF in-memory, optionally encrypted (owner-only password)."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), text)
    buf = io.BytesIO()
    if encrypt:
        doc.save(
            buf,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
            permissions=0,
        )
    else:
        doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_upload_file(content: bytes, filename: str = "test.pdf") -> UploadFile:
    """Create a FastAPI UploadFile wrapping in-memory bytes."""
    file = AsyncMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=content)
    return file


# ---------------------------------------------------------------------------
# extract_pdf_metadata
# ---------------------------------------------------------------------------


class TestExtractPdfMetadata:
    def test_returns_page_count(self, minimal_pdf_bytes):
        meta = PDFValidator.extract_pdf_metadata(minimal_pdf_bytes)
        assert meta["page_count"] == 1

    def test_multipage_pdf(self):
        content = _make_pdf(pages=5)
        meta = PDFValidator.extract_pdf_metadata(content)
        assert meta["page_count"] == 5

    def test_encrypted_flag_false_for_plain_pdf(self, minimal_pdf_bytes):
        meta = PDFValidator.extract_pdf_metadata(minimal_pdf_bytes)
        assert meta["encrypted"] is False

    def test_encrypted_flag_true_for_encrypted_pdf(self):
        content = _make_pdf(encrypt=True)
        meta = PDFValidator.extract_pdf_metadata(content)
        assert meta["encrypted"] is True

    def test_returns_pdf_version(self, minimal_pdf_bytes):
        meta = PDFValidator.extract_pdf_metadata(minimal_pdf_bytes)
        assert "pdf_version" in meta
        assert meta["pdf_version"]  # non-empty string

    def test_returns_expected_keys(self, minimal_pdf_bytes):
        meta = PDFValidator.extract_pdf_metadata(minimal_pdf_bytes)
        expected_keys = {
            "page_count",
            "pdf_version",
            "title",
            "author",
            "subject",
            "creator",
            "producer",
            "encrypted",
            "needs_pass",
        }
        assert expected_keys.issubset(meta.keys())

    def test_corrupted_bytes_raises_validation_error(self):
        with pytest.raises(ValidationError, match="metadata"):
            PDFValidator.extract_pdf_metadata(b"this is not a pdf")


# ---------------------------------------------------------------------------
# validate_pdf_bytes — positive cases
# ---------------------------------------------------------------------------


class TestValidatePdfBytesPositive:
    def test_returns_tuple(self, minimal_pdf_bytes):
        content, meta = PDFValidator.validate_pdf_bytes(minimal_pdf_bytes, "doc.pdf")
        assert isinstance(content, bytes)
        assert isinstance(meta, dict)

    def test_content_unchanged(self, minimal_pdf_bytes):
        content, _ = PDFValidator.validate_pdf_bytes(minimal_pdf_bytes, "doc.pdf")
        assert content == minimal_pdf_bytes

    def test_metadata_filename(self, minimal_pdf_bytes):
        _, meta = PDFValidator.validate_pdf_bytes(minimal_pdf_bytes, "my_report.pdf")
        assert meta["filename"] == "my_report.pdf"

    def test_metadata_size_bytes(self, minimal_pdf_bytes):
        _, meta = PDFValidator.validate_pdf_bytes(minimal_pdf_bytes, "doc.pdf")
        assert meta["size_bytes"] == len(minimal_pdf_bytes)

    def test_metadata_content_type(self, minimal_pdf_bytes):
        _, meta = PDFValidator.validate_pdf_bytes(minimal_pdf_bytes, "doc.pdf")
        assert meta["content_type"] == "application/pdf"

    def test_metadata_checksum_sha256(self, minimal_pdf_bytes):
        _, meta = PDFValidator.validate_pdf_bytes(minimal_pdf_bytes, "doc.pdf")
        expected = hashlib.sha256(minimal_pdf_bytes).hexdigest()
        assert meta["checksum"] == expected

    def test_metadata_page_count_included(self, minimal_pdf_bytes):
        _, meta = PDFValidator.validate_pdf_bytes(minimal_pdf_bytes, "doc.pdf")
        assert "page_count" in meta
        assert meta["page_count"] >= 1


# ---------------------------------------------------------------------------
# validate_pdf_bytes — negative / edge cases
# ---------------------------------------------------------------------------


class TestValidatePdfBytesNegative:
    def test_oversized_file_raises(self):
        """A file at exactly MAX_FILE_SIZE should be rejected."""
        oversized = b"x" * settings.MAX_FILE_SIZE
        with pytest.raises(ValidationError, match="File size"):
            PDFValidator.validate_pdf_bytes(oversized, "big.pdf")

    def test_encrypted_pdf_raises(self):
        encrypted = _make_pdf(encrypt=True)
        with pytest.raises(ValidationError, match="Encrypted"):
            PDFValidator.validate_pdf_bytes(encrypted, "secure.pdf")

    def test_corrupted_pdf_raises(self):
        with pytest.raises(ValidationError):
            PDFValidator.validate_pdf_bytes(b"GARBAGE_NOT_PDF", "bad.pdf")

    def test_empty_bytes_raises(self):
        with pytest.raises(ValidationError):
            PDFValidator.validate_pdf_bytes(b"", "empty.pdf")

    def test_just_below_size_limit_is_accepted(self, minimal_pdf_bytes):
        """Real PDF below size limit should pass validation."""
        assert len(minimal_pdf_bytes) < settings.MAX_FILE_SIZE
        content, meta = PDFValidator.validate_pdf_bytes(minimal_pdf_bytes, "ok.pdf")
        assert meta["size_bytes"] == len(minimal_pdf_bytes)


# ---------------------------------------------------------------------------
# validate_pdf_file (async) — positive and negative
# ---------------------------------------------------------------------------


class TestValidatePdfFileAsync:
    async def test_valid_file_returns_tuple(self, minimal_pdf_bytes):
        upload = _make_upload_file(minimal_pdf_bytes, "upload.pdf")
        content, meta = await PDFValidator.validate_pdf_file(upload)
        assert content == minimal_pdf_bytes
        assert meta["filename"] == "upload.pdf"

    async def test_file_without_name_uses_unknown(self, minimal_pdf_bytes):
        upload = _make_upload_file(minimal_pdf_bytes)
        upload.filename = None
        content, meta = await PDFValidator.validate_pdf_file(upload)
        assert meta["filename"] == "unknown.pdf"

    async def test_encrypted_file_raises(self):
        encrypted = _make_pdf(encrypt=True)
        upload = _make_upload_file(encrypted, "secure.pdf")
        with pytest.raises(ValidationError, match="Encrypted"):
            await PDFValidator.validate_pdf_file(upload)

    async def test_corrupted_file_raises(self):
        upload = _make_upload_file(b"NOT_A_PDF", "bad.pdf")
        with pytest.raises(ValidationError):
            await PDFValidator.validate_pdf_file(upload)

    async def test_exact_max_size_raises(self):
        """
        validate_pdf_file reads exactly MAX_FILE_SIZE bytes and raises
        if len(content) == MAX_FILE_SIZE (meaning there may be more data).
        """
        upload = _make_upload_file(b"x" * settings.MAX_FILE_SIZE, "big.pdf")
        with pytest.raises(ValidationError, match="File size"):
            await PDFValidator.validate_pdf_file(upload)

    async def test_checksum_computed(self, minimal_pdf_bytes):
        upload = _make_upload_file(minimal_pdf_bytes, "doc.pdf")
        _, meta = await PDFValidator.validate_pdf_file(upload)
        expected = hashlib.sha256(minimal_pdf_bytes).hexdigest()
        assert meta["checksum"] == expected


# ---------------------------------------------------------------------------
# Parametrized edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_bytes",
    [
        b"\x00" * 10,  # null bytes
        b"PDF-1.4 but not real",  # fake header
        b"%PDF\xff\xfe",  # truncated
    ],
    ids=["null-bytes", "fake-header", "truncated"],
)
def test_corrupted_content_raises_validation_error(bad_bytes):
    with pytest.raises(ValidationError):
        PDFValidator.validate_pdf_bytes(bad_bytes, "corrupt.pdf")


# ---------------------------------------------------------------------------
# fitz.FileDataError specific paths
# ---------------------------------------------------------------------------


class TestFitzFileDataError:
    def test_validate_bytes_fitz_file_data_error(self, minimal_pdf_bytes):
        """validate_pdf_bytes re-raises ValidationError when extract_pdf_metadata
        raises fitz.FileDataError directly (covers lines 40-43)."""
        from unittest.mock import patch

        with patch.object(
            PDFValidator,
            "extract_pdf_metadata",
            side_effect=fitz.FileDataError("bad fitz data"),
        ):
            with pytest.raises(ValidationError, match="Invalid or corrupted"):
                PDFValidator.validate_pdf_bytes(minimal_pdf_bytes, "doc.pdf")

    async def test_validate_file_fitz_file_data_error(self, minimal_pdf_bytes):
        """validate_pdf_file re-raises ValidationError when extract_pdf_metadata
        raises fitz.FileDataError directly (covers lines 82-85)."""
        from unittest.mock import patch

        upload = _make_upload_file(minimal_pdf_bytes, "doc.pdf")
        with patch.object(
            PDFValidator,
            "extract_pdf_metadata",
            side_effect=fitz.FileDataError("bad fitz data"),
        ):
            with pytest.raises(ValidationError, match="Invalid or corrupted"):
                await PDFValidator.validate_pdf_file(upload)
