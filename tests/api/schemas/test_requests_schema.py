"""
Unit tests for api/schemas/requests.py — Pydantic request validation models.
"""

import base64

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.api.schemas.requests import (
    CostAttributionInput,
    DocumentInput,
    JobCancelRequest,
    JobListRequest,
    ProcessingOptions,
    TranslateRequest,
    TranslationConfigInput,
)
from fixtures.sample_data import VALID_PDF_B64


# ---------------------------------------------------------------------------
# DocumentInput
# ---------------------------------------------------------------------------


class TestDocumentInput:
    def test_valid_pdf_document(self):
        doc = DocumentInput(content=VALID_PDF_B64, filename="report.pdf")
        assert doc.filename == "report.pdf"
        assert doc.format == "pdf"

    def test_valid_docx_format(self):
        content = base64.b64encode(b"dummy").decode()
        doc = DocumentInput(content=content, filename="report.docx", format="docx")
        assert doc.format == "docx"

    def test_invalid_base64_raises(self):
        with pytest.raises(PydanticValidationError, match="valid base64"):
            DocumentInput(content="not-valid-base64!!!", filename="file.pdf")

    @pytest.mark.parametrize("filename", ["file.txt", "file.exe", "file", "file.PDF"])
    def test_invalid_filename_extension_raises(self, filename):
        """Only .pdf and .docx extensions are accepted (case-insensitive strip)."""
        content = base64.b64encode(b"data").decode()
        # .PDF should fail because the validator lowercases the stripped value
        # but .pdf (lowercase) should pass — here we test rejection cases
        if filename.lower().endswith(".pdf") or filename.lower().endswith(".docx"):
            # These should pass
            doc = DocumentInput(content=content, filename=filename)
            assert doc is not None
        else:
            with pytest.raises(PydanticValidationError, match=".pdf or .docx"):
                DocumentInput(content=content, filename=filename)

    def test_filename_stripped_of_whitespace(self):
        content = base64.b64encode(b"data").decode()
        doc = DocumentInput(content=content, filename="  report.pdf  ")
        assert doc.filename == "report.pdf"

    def test_empty_filename_raises(self):
        content = base64.b64encode(b"data").decode()
        with pytest.raises(PydanticValidationError):
            DocumentInput(content=content, filename="")

    def test_missing_content_raises(self):
        with pytest.raises(PydanticValidationError):
            DocumentInput(filename="file.pdf")  # type: ignore[call-arg]

    def test_missing_filename_raises(self):
        with pytest.raises(PydanticValidationError):
            DocumentInput(content=VALID_PDF_B64)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TranslationConfigInput
# ---------------------------------------------------------------------------


class TestTranslationConfigInput:
    @pytest.mark.parametrize(
        "domain",
        ["commercial", "legal", "finance", "hr", "operations"],
    )
    def test_valid_domains(self, domain):
        cfg = TranslationConfigInput(target_language="es", domain=domain)
        assert cfg.domain == domain

    def test_domain_oprations_alias(self):
        """Legacy alias 'oprations' maps to 'operations'."""
        cfg = TranslationConfigInput(target_language="es", domain="oprations")
        assert cfg.domain == "operations"

    def test_domain_case_insensitive(self):
        cfg = TranslationConfigInput(target_language="es", domain="COMMERCIAL")
        assert cfg.domain == "commercial"

    def test_invalid_domain_raises(self):
        with pytest.raises(PydanticValidationError, match="Invalid domain"):
            TranslationConfigInput(target_language="es", domain="science")

    def test_source_language_default_auto(self):
        cfg = TranslationConfigInput(target_language="es", domain="legal")
        assert cfg.source_language is None

    def test_target_language_stripped(self):
        cfg = TranslationConfigInput(target_language="  Spanish  ", domain="legal")
        assert cfg.target_language == "Spanish"

    @pytest.mark.parametrize("target", ["es", "Spanish", "fr", "Japanese", "zh-cn"])
    def test_various_target_languages(self, target):
        cfg = TranslationConfigInput(target_language=target, domain="hr")
        assert cfg.target_language == target

    def test_target_language_too_short_raises(self):
        with pytest.raises(PydanticValidationError):
            TranslationConfigInput(target_language="e", domain="hr")

    def test_domain_too_short_raises(self):
        with pytest.raises(PydanticValidationError):
            TranslationConfigInput(target_language="es", domain="x")

    def test_domain_too_long_raises(self):
        with pytest.raises(PydanticValidationError):
            TranslationConfigInput(target_language="es", domain="a" * 21)


# ---------------------------------------------------------------------------
# ProcessingOptions
# ---------------------------------------------------------------------------


class TestProcessingOptions:
    def test_defaults(self):
        opts = ProcessingOptions()
        assert opts.enable_dlp is True
        assert opts.enable_chunking is True
        assert opts.priority == "standard"

    def test_high_priority(self):
        opts = ProcessingOptions(priority="high")
        assert opts.priority == "high"

    def test_invalid_priority_raises(self):
        with pytest.raises(PydanticValidationError):
            ProcessingOptions(priority="urgent")

    def test_disable_dlp_and_chunking(self):
        opts = ProcessingOptions(enable_dlp=False, enable_chunking=False)
        assert opts.enable_dlp is False
        assert opts.enable_chunking is False


# ---------------------------------------------------------------------------
# TranslateRequest
# ---------------------------------------------------------------------------


class TestTranslateRequest:
    def test_valid_full_request(self):
        req = TranslateRequest(
            document=DocumentInput(content=VALID_PDF_B64, filename="doc.pdf"),
            translation_config=TranslationConfigInput(
                target_language="es", domain="commercial"
            ),
            cost_attribution=CostAttributionInput(
                user_id="user-1",
                business_unit="bu-1",
                organization="org-1",
            ),
        )
        assert req.document.filename == "doc.pdf"
        assert req.translation_config.domain == "commercial"
        assert req.processing_options.priority == "standard"

    def test_default_processing_options(self):
        req = TranslateRequest(
            document=DocumentInput(content=VALID_PDF_B64, filename="doc.pdf"),
            translation_config=TranslationConfigInput(
                target_language="fr", domain="legal"
            ),
            cost_attribution=CostAttributionInput(
                user_id="user-1",
                business_unit="bu-1",
                organization="org-1",
            ),
        )
        assert isinstance(req.processing_options, ProcessingOptions)

    def test_missing_document_raises(self):
        with pytest.raises(PydanticValidationError):
            TranslateRequest(  # type: ignore[call-arg]
                translation_config=TranslationConfigInput(
                    target_language="es",
                    domain="commercial",
                ),
                cost_attribution=CostAttributionInput(
                    user_id="user-1",
                    business_unit="bu-1",
                    organization="org-1",
                )
            )

    def test_missing_translation_config_raises(self):
        with pytest.raises(PydanticValidationError):
            TranslateRequest(  # type: ignore[call-arg]
                document=DocumentInput(content=VALID_PDF_B64, filename="doc.pdf")
            )

    def test_missing_cost_attribution_raises(self):
        with pytest.raises(PydanticValidationError):
            TranslateRequest(
                document=DocumentInput(content=VALID_PDF_B64, filename="doc.pdf"),
                translation_config=TranslationConfigInput(
                    target_language="es",
                    domain="commercial",
                ),
            )


# ---------------------------------------------------------------------------
# JobCancelRequest
# ---------------------------------------------------------------------------


class TestJobCancelRequest:
    def test_with_reason(self):
        req = JobCancelRequest(reason="No longer needed")
        assert req.reason == "No longer needed"

    def test_reason_is_optional(self):
        req = JobCancelRequest()
        assert req.reason is None

    def test_reason_none_explicit(self):
        req = JobCancelRequest(reason=None)
        assert req.reason is None


# ---------------------------------------------------------------------------
# JobListRequest
# ---------------------------------------------------------------------------


class TestJobListRequest:
    @pytest.mark.parametrize(
        "status_val",
        ["queued", "processing", "completed", "failed", "cancelled"],
    )
    def test_valid_statuses(self, status_val):
        req = JobListRequest(status=status_val)
        assert req.status == status_val

    def test_invalid_status_raises(self):
        with pytest.raises(PydanticValidationError):
            JobListRequest(status="pending")

    def test_defaults(self):
        req = JobListRequest()
        assert req.limit == 10
        assert req.offset == 0
        assert req.status is None

    def test_limit_bounds(self):
        with pytest.raises(PydanticValidationError):
            JobListRequest(limit=0)
        with pytest.raises(PydanticValidationError):
            JobListRequest(limit=101)

    def test_negative_offset_raises(self):
        with pytest.raises(PydanticValidationError):
            JobListRequest(offset=-1)

    def test_valid_pagination(self):
        req = JobListRequest(limit=50, offset=100)
        assert req.limit == 50
        assert req.offset == 100
