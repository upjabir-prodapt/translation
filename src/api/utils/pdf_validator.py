"""PDF validation utilities."""

import hashlib

import fitz  # PyMuPDF
from fastapi import UploadFile

from src.api.exceptions import ValidationError
from src.config.constants import settings


class PDFValidator:
    """Utility class for PDF validation."""

    @staticmethod
    def validate_pdf_bytes(content: bytes, filename: str) -> tuple[bytes, dict]:
        """Validate raw PDF bytes and return content with metadata."""
        if not content:
            raise ValidationError("Document is empty", "content")

        if len(content) >= settings.MAX_FILE_SIZE:
            max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(f"File size exceeds {max_mb:.0f}MB limit", "size")

        if not content.startswith(b"%PDF-"):
            raise ValidationError("File is not a valid PDF.", "content")

        try:
            pdf_metadata = PDFValidator.extract_pdf_metadata(content)
            PDFValidator._assert_not_protected(pdf_metadata)
            PDFValidator._assert_has_text_layer(pdf_metadata)

            checksum = hashlib.sha256(content).hexdigest()

            metadata = {
                "filename": filename,
                "size_bytes": len(content),
                "content_type": "application/pdf",
                "checksum": checksum,
                **pdf_metadata,
            }

        except ValidationError:
            raise
        except fitz.FileDataError as e:
            raise ValidationError(
                f"Invalid or corrupted PDF file: {str(e)}", "content"
            ) from e
        except Exception as e:
            raise ValidationError(f"Failed to validate PDF: {str(e)}", "content") from e

        return content, metadata

    @staticmethod
    def _assert_not_protected(pdf_metadata: dict) -> None:
        """Reject password-protected, encrypted, or copy-restricted PDFs.

        Consolidates the checks previously duplicated between
        `validate_pdf_bytes` and `validate_pdf_file`. The permissions
        check only fires when `extract_pdf_metadata` actually populated a
        `permissions` int (real PyMuPDF opens always do); tests that mock
        `extract_pdf_metadata` with a bare `{"page_count": ..., "encrypted": ...}`
        dict are unaffected.
        """
        if pdf_metadata.get("needs_pass", False):
            raise ValidationError(
                "Password-protected PDFs are not supported. Please remove the password and resubmit.",
                "password_protected",
            )

        if pdf_metadata.get("encrypted", False):
            raise ValidationError("Encrypted PDFs are not supported", "encrypted")

        permissions = pdf_metadata.get("permissions")
        if isinstance(permissions, int) and not (permissions & fitz.PDF_PERM_COPY):
            # Owner-password-only PDFs report is_encrypted=False/needs_pass=0
            # (they open freely) but restrict the COPY permission bit --
            # verified live with PyMuPDF. Reject so the original
            # restriction is never silently lost in the translated output.
            raise ValidationError(
                "This PDF restricts text extraction (owner-password "
                "protected). Please remove the restriction and resubmit.",
                "permissions",
            )

    @staticmethod
    def _assert_has_text_layer(pdf_metadata: dict) -> None:
        """Reject image-only/scanned PDFs with no extractable text layer.

        implementation_plan.md Phase B: fail fast (~50ms, at the API
        boundary) instead of after a Cloud Tasks round-trip, full IL
        parse, and per-page pixmap rendering deep in the worker pipeline.
        Only fires when `has_text_layer` was actually populated by
        `extract_pdf_metadata` (real PyMuPDF opens always populate it);
        tests that mock `extract_pdf_metadata` with a bare metadata dict
        are unaffected.
        """
        if pdf_metadata.get("has_text_layer") is False:
            raise ValidationError(
                "This PDF has no extractable text layer (scanned or "
                "image-only). OCR is not supported — please supply a "
                "text-based PDF.",
                "no_text_layer",
            )

    @staticmethod
    async def validate_pdf_file(file: UploadFile) -> tuple[bytes, dict]:
        """Validate uploaded PDF file and return content with metadata."""
        # Read file content (limit from settings)
        content = await file.read(settings.MAX_FILE_SIZE)

        if len(content) == settings.MAX_FILE_SIZE:
            max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(f"File size exceeds {max_mb:.0f}MB limit", "size")

        if not content:
            raise ValidationError("Document is empty", "content")

        if not content.startswith(b"%PDF-"):
            raise ValidationError("File is not a valid PDF.", "content")

        # Validate PDF and extract metadata
        try:
            # Extract PDF metadata (this validates the PDF)
            pdf_metadata = PDFValidator.extract_pdf_metadata(content)
            PDFValidator._assert_not_protected(pdf_metadata)
            PDFValidator._assert_has_text_layer(pdf_metadata)

            # Calculate checksum
            checksum = hashlib.sha256(content).hexdigest()

            # Combine file metadata with PDF metadata
            metadata = {
                "filename": file.filename or "unknown.pdf",
                "size_bytes": len(content),
                "content_type": "application/pdf",
                "checksum": checksum,
                **pdf_metadata,  # Include all PDF metadata
            }

        except ValidationError:
            raise  # Re-raise validation errors as-is
        except fitz.FileDataError as e:
            raise ValidationError(
                f"Invalid or corrupted PDF file: {str(e)}", "content"
            ) from e
        except Exception as e:
            raise ValidationError(f"Failed to validate PDF: {str(e)}", "content") from e

        return content, metadata

    @staticmethod
    def extract_pdf_metadata(content: bytes) -> dict:
        """Extract detailed metadata from PDF content using PyMuPDF."""
        metadata = {}

        try:
            doc = fitz.open(stream=content, filetype="pdf")

            # Get metadata, handle None case
            pdf_metadata = doc.metadata or {}

            has_text_layer, text_char_count = PDFValidator._probe_text_layer(doc)

            # Extract comprehensive metadata
            metadata = {
                "page_count": len(doc),
                "pdf_version": pdf_metadata.get("format", "Unknown"),
                "title": pdf_metadata.get("title", ""),
                "author": pdf_metadata.get("author", ""),
                "subject": pdf_metadata.get("subject", ""),
                "creator": pdf_metadata.get("creator", ""),
                "producer": pdf_metadata.get("producer", ""),
                "encrypted": doc.is_encrypted,
                "needs_pass": doc.needs_pass,
                "permissions": doc.permissions,
                "has_text_layer": has_text_layer,
                "text_char_count": text_char_count,
            }

            doc.close()

        except Exception as e:
            raise ValidationError("Failed to extract PDF metadata", "metadata") from e

        return metadata

    @staticmethod
    def _probe_text_layer(doc: fitz.Document) -> tuple[bool, int]:
        """Sample up to `PDF_TEXT_PROBE_PAGES` pages for extractable text.

        Detects image-only/scanned PDFs cheaply at the API boundary
        (implementation_plan.md Phase B) -- verified live that an
        image-only page returns `''` from `page.get_text("text")`.
        Bounded by `settings.PDF_TEXT_PROBE_PAGES` so a large scanned
        document is still cheap to reject.

        Skips the probe entirely when the document still requires a
        password: reading page content of a locked document raises
        inside PyMuPDF, which would otherwise surface as a generic
        "Failed to extract PDF metadata" and mask the real
        password-protected rejection from `_assert_not_protected`.
        Reports `has_text_layer=True` (not applicable) in that case so
        this probe never becomes the reason a password-protected PDF is
        rejected -- that job belongs to `_assert_not_protected`.
        """
        if doc.needs_pass:
            return True, 0

        total_chars = 0
        probe_pages = min(len(doc), max(1, settings.PDF_TEXT_PROBE_PAGES))
        for page_index in range(probe_pages):
            page_text = doc[page_index].get_text("text") or ""
            total_chars += len(page_text.strip())
        return total_chars > 0, total_chars
