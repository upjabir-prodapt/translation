"""DOCX validation utilities."""

import hashlib
import io
import zipfile

from src.api.exceptions import ValidationError
from src.config.constants import settings

DOCUMENT_PART = "word/document.xml"
_ENCRYPTED_MARKERS = ("EncryptedPackage", "EncryptionInfo")


class DOCXValidator:
    """Validate uploaded Word documents before they enter the pipeline."""

    @staticmethod
    def validate_docx_bytes(content: bytes, filename: str) -> tuple[bytes, dict]:
        """Validate raw .docx bytes and return content with metadata.

        Checks the upload is a real, unencrypted OOXML Word package. The DOCX
        pipeline edits the package in place, so a file that cannot be opened
        here would fail later in the worker with a far less useful message.
        """
        if len(content) >= settings.MAX_FILE_SIZE:
            max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(f"File size exceeds {max_mb:.0f}MB limit", "size")

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile as e:
            raise ValidationError(
                f"Invalid or corrupted Word document: {str(e)}", "content"
            ) from e

        if any(marker in names for marker in _ENCRYPTED_MARKERS):
            raise ValidationError(
                "Password-protected Word documents are not supported. "
                "Remove the password and resubmit.",
                "password_protected",
            )
        if DOCUMENT_PART not in names:
            if "word/document.bin" in names:
                raise ValidationError(
                    "Legacy binary Word documents (.doc) are not supported. "
                    "Save the file as .docx and resubmit.",
                    "content",
                )
            raise ValidationError(
                f"Not a Word document: missing {DOCUMENT_PART}", "content"
            )

        return content, {
            "filename": filename,
            "size_bytes": len(content),
            "content_type": (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            "checksum": hashlib.sha256(content).hexdigest(),
            # Word has no fixed page count until it is laid out by a renderer.
            "page_count": None,
        }
