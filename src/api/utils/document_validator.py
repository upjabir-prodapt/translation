"""Validation utilities for non-PDF document uploads (DOCX, TXT).

Mirrors the style/error-handling conventions of `pdf_validator.py`: every
rejection raises `src.api.exceptions.ValidationError` (-> HTTP 422 via
`api/middleware/exception_handler.py`) so nothing bad ever reaches GCS
upload / BigQuery job creation / Cloud Tasks enqueue.

See `implementation_plan.md` Phase A for the rationale behind each check.
"""

import hashlib
import re
import zipfile
from io import BytesIO

from src.api.exceptions import ValidationError
from src.config.constants import settings

# Magic-byte signatures used to identify a container format regardless of
# the filename extension the caller claims. This is deliberately
# byte-sniffing rather than trusting `filename` -- a ZIP or plain-text
# file renamed to `.docx` must be rejected with a clear message instead
# of reaching python-docx and crashing the worker.
_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"
# ECMA-376 "Agile Encryption" (password-protected OOXML) wraps the whole
# package in a legacy OLE/CFB compound file container with an
# `EncryptedPackage` stream inside. Detecting the CFB signature is
# sufficient to identify a password-protected .docx without ever
# attempting to decrypt it (no msoffcrypto/olefile dependency needed).
_OLE_CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# Required OOXML package members for a file to be considered a genuine
# Word document rather than an arbitrary ZIP archive renamed to .docx.
_DOCX_REQUIRED_MEMBERS = ("[Content_Types].xml", "word/document.xml")

# Matches the text content of every <w:t> run-text element in
# word/document.xml. A plain regex scan is used instead of an XML parser
# because `lxml` is a worker-only dependency (pyproject.toml `worker`
# extra) and is not installed in the lightweight API image -- see
# implementation_plan.md techContext notes on the API/worker dependency
# split. This is a document *content-presence* probe, not a full parse,
# so a regex is an intentional, sufficient tradeoff here.
_WORD_TEXT_RUN_PATTERN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)


def detect_container(content: bytes) -> str:
    """Identify the byte-level container format of `content`.

    Returns one of "pdf", "zip", "ole", "unknown". Used to give accurate,
    specific rejection messages (e.g. distinguishing a ZIP-renamed-.docx
    from a plain-text-renamed-.docx) instead of a generic parse failure.
    """
    if content.startswith(_PDF_MAGIC):
        return "pdf"
    if content.startswith(_ZIP_MAGIC):
        return "zip"
    if content.startswith(_OLE_CFB_MAGIC):
        return "ole"
    return "unknown"


class DocumentValidator:
    """Validation for DOCX and TXT uploads (PDF has its own `PDFValidator`)."""

    @staticmethod
    def _size_check(content: bytes) -> None:
        """Enforce the 10 MB upload limit shared by every document format.

        Mirrors `PDFValidator.validate_pdf_bytes`'s `>=` check so PDF,
        DOCX and TXT all reject an at-the-limit file identically.
        """
        if len(content) >= settings.MAX_FILE_SIZE:
            max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(f"File size exceeds {max_mb:.0f}MB limit", "size")

    @staticmethod
    def validate_docx_bytes(content: bytes, filename: str) -> dict:
        """Validate raw DOCX bytes and return descriptive metadata.

        Rejects: empty content, password-protected/legacy Office files
        (OLE/CFB container), non-ZIP content (e.g. a .txt renamed to
        .docx), corrupt ZIP archives, and ZIP archives that are not
        actually a Word package (e.g. a plain .zip renamed to .docx).
        """
        if not content:
            raise ValidationError("Document is empty", "content")

        DocumentValidator._size_check(content)

        container = detect_container(content)
        if container == "ole":
            raise ValidationError(
                "Password-protected or legacy Office documents are not "
                "supported. Remove the password, save as .docx, and "
                "resubmit.",
                "password_protected",
            )
        if container != "zip":
            raise ValidationError(
                "File is not a valid .docx (expected a Word document).",
                "content",
            )

        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())
                if not all(member in names for member in _DOCX_REQUIRED_MEMBERS):
                    raise ValidationError(
                        "File is a ZIP archive but not a Word document.",
                        "content",
                    )
                document_xml = archive.read("word/document.xml")
        except zipfile.BadZipFile as e:
            raise ValidationError(
                "File is not a valid .docx (expected a Word document).",
                "content",
            ) from e

        DocumentValidator._assert_has_text_layer(document_xml)

        checksum = hashlib.sha256(content).hexdigest()
        return {
            "filename": filename,
            "size_bytes": len(content),
            "content_type": (
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
            "checksum": checksum,
            "page_count": None,
        }

    @staticmethod
    def _assert_has_text_layer(document_xml: bytes) -> None:
        """Reject an image-only DOCX (no non-whitespace `<w:t>` run text).

        implementation_plan.md Phase B: a DOCX built entirely from
        embedded images/drawings has no `<w:t>` run-text elements at all,
        or only whitespace ones -- either way there is nothing to
        translate.
        """
        text_runs = _WORD_TEXT_RUN_PATTERN.findall(document_xml.decode("utf-8", "replace"))
        if not any(run.strip() for run in text_runs):
            raise ValidationError(
                "This document contains no translatable text (images "
                "only).",
                "no_text_layer",
            )

    @staticmethod
    def validate_txt_bytes(content: bytes, filename: str) -> dict:
        """Validate raw TXT bytes and return descriptive metadata.

        Rejects empty content and content that is not valid UTF-8 rather
        than silently mojibaking it (the DOCX bridge's `errors="replace"`
        decode is intentionally not mirrored here -- validation should
        fail loudly instead of masking a bad upload).
        """
        if not content:
            raise ValidationError("Document is empty", "content")

        DocumentValidator._size_check(content)

        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValidationError(
                "Text file is not valid UTF-8. Please save it with UTF-8 "
                "encoding and resubmit.",
                "encoding",
            ) from e

        if not decoded.strip():
            raise ValidationError(
                "This document contains no translatable text (images "
                "only).",
                "no_text_layer",
            )

        checksum = hashlib.sha256(content).hexdigest()
        return {
            "filename": filename,
            "size_bytes": len(content),
            "content_type": "text/plain",
            "checksum": checksum,
            "page_count": None,
        }
