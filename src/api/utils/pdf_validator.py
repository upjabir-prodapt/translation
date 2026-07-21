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
        if len(content) >= settings.MAX_FILE_SIZE:
            max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(f"File size exceeds {max_mb:.0f}MB limit", "size")

        try:
            pdf_metadata = PDFValidator.extract_pdf_metadata(content)

            if pdf_metadata.get("needs_pass", False):
                raise ValidationError(
                    "Password-protected PDFs are not supported. Please remove the password and resubmit.",
                    "password_protected",
                )

            if pdf_metadata.get("encrypted", False):
                raise ValidationError("Encrypted PDFs are not supported", "encrypted")

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
    async def validate_pdf_file(file: UploadFile) -> tuple[bytes, dict]:
        """Validate uploaded PDF file and return content with metadata."""
        # Read file content (limit from settings)
        content = await file.read(settings.MAX_FILE_SIZE)

        if len(content) == settings.MAX_FILE_SIZE:
            max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(f"File size exceeds {max_mb:.0f}MB limit", "size")

        # Validate PDF and extract metadata
        try:
            # Extract PDF metadata (this validates the PDF)
            pdf_metadata = PDFValidator.extract_pdf_metadata(content)

            # Check if PDF is password-protected or encrypted - reject both
            if pdf_metadata.get("needs_pass", False):
                raise ValidationError(
                    "Password-protected PDFs are not supported. Please remove the password and resubmit.",
                    "password_protected",
                )

            if pdf_metadata.get("encrypted", False):
                raise ValidationError("Encrypted PDFs are not supported", "encrypted")

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
            }

            doc.close()

        except Exception as e:
            raise ValidationError("Failed to extract PDF metadata", "metadata") from e

        return metadata
