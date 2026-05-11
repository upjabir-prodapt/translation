import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.api.services.translation_service import TranslationService
from src.api.schemas.requests import TranslateRequest, DocumentInput, TranslationConfigInput, CostAttributionInput, ProcessingOptions
import base64

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
            content=base64.b64encode(b"%PDF-1.4\n1 0 obj\n<<\n/Title (Test)\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF").decode("utf-8"),
            filename="test.pdf"
        ),
        translation_config=TranslationConfigInput(
            target_language="French",
            domain="legal"
        ),
        cost_attribution=CostAttributionInput(
            user_id="user1",
            business_unit="legal-dept",
            organization="colt"
        )
    )

class TestTranslationService:
    @patch("src.api.services.translation_service.asyncio.create_task")
    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_submit_translation_success(self, mock_validate, mock_task, service, mock_bq, mock_storage, valid_request):
        # Mock PDF validation
        mock_validate.return_value = (b"pdf-content", {
            "page_count": 1,
            "filename": "test.pdf",
            "size_bytes": 100,
            "checksum": "abc"
        })
        # Mock storage client for file upload
        mock_storage.upload_input_pdf.return_value = "gs://bucket/test.pdf"
        
        res = await service.submit_translation(valid_request)
        
        assert res.status == "queued"
        mock_bq.upsert_translation_job.assert_called_once()
        # It calls _schedule_background_pipeline which calls asyncio.create_task
        mock_task.assert_called()

    def test_normalize_config(self, service, valid_request):
        config = service._normalize_config(valid_request)
        assert config["lang_out"] == "fr"
        assert config["domain"] == "legal"
        assert "lang_in" in config
