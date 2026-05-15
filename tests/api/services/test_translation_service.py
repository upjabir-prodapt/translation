import base64
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.api.exceptions import ValidationError
from src.api.schemas.requests import CostAttributionInput
from src.api.schemas.requests import DocumentInput
from src.api.schemas.requests import TranslateRequest
from src.api.schemas.requests import TranslationConfigInput
from src.api.services.translation_service import TranslationService


@pytest.fixture
def mock_storage():
    return AsyncMock()


@pytest.fixture
def mock_bq():
    return AsyncMock()


@pytest.fixture
def service(mock_storage, mock_bq):
    return TranslationService(storage=mock_storage, bigquery=mock_bq)


@pytest.fixture
def valid_request():
    return TranslateRequest(
        document=DocumentInput(
            content=base64.b64encode(b"%PDF-1.4\n%%EOF").decode("utf-8"),
            filename="test.pdf",
        ),
        translation_config=TranslationConfigInput(
            target_language="French", domain="legal"
        ),
        cost_attribution=CostAttributionInput(
            user_id="user1", business_unit="legal-dept", organization="colt"
        ),
    )


class TestTranslationService:
    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_submit_translation_success(
        self, mock_validate, service, mock_bq, mock_storage, valid_request
    ):
        mock_validate.return_value = (
            b"pdf-content",
            {
                "page_count": 1,
                "filename": "test.pdf",
                "size_bytes": 100,
                "checksum": "abc",
            },
        )
        mock_storage.upload_input_pdf.return_value = "gs://bucket/test.pdf"

        with patch.object(service.orchestrator, "run", new_callable=AsyncMock):
            res = await service.submit_translation(valid_request)
            assert res.status == "queued"
            mock_bq.upsert_translation_job.assert_called_once()

    async def test_submit_translation_decode_error(self, service, valid_request):
        # We can't easily trigger base64.b64decode error with simple strings without validate=True
        # but we can mock it
        with patch(
            "src.api.services.translation_service.base64.b64decode",
            side_effect=Exception("Decode Error"),
        ):
            with pytest.raises(ValidationError, match="Failed to decode"):
                await service.submit_translation(valid_request)

    async def test_submit_translation_docx(self, service, mock_storage, valid_request):
        valid_request.document.format = "docx"
        valid_request.document.filename = "test.docx"
        mock_storage.upload_input_pdf.return_value = "gs://bucket/test.docx"

        with patch.object(service.orchestrator, "run", new_callable=AsyncMock):
            res = await service.submit_translation(valid_request)
            assert res.status == "queued"

    def test_normalize_config_success(self, service, valid_request):
        config = service._normalize_config(valid_request)
        assert config["lang_out"] == "fr"
        assert config["domain"] == "legal"

    def test_normalize_config_source_lang(self, service, valid_request):
        valid_request.translation_config.source_language = "Spanish"
        config = service._normalize_config(valid_request)
        assert config["lang_in"] == "es"

    def test_normalize_config_invalid_domain(self, service, valid_request):
        with patch(
            "src.api.services.translation_service.normalize_domain",
            side_effect=ValueError("bad domain"),
        ):
            with pytest.raises(ValidationError, match="bad domain"):
                service._normalize_config(valid_request)

    async def test_submit_translation_failure_cleanup(
        self, service, mock_storage, valid_request
    ):
        with patch.object(service, "_normalize_config", side_effect=Exception("Oops")):
            with patch(
                "src.api.services.translation_service.PDFValidator.validate_pdf_bytes",
                return_value=(b"", {}),
            ):
                with pytest.raises(Exception, match="Oops"):
                    await service.submit_translation(valid_request)
                mock_storage.delete_job_files.assert_called_once()
