import base64
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.api.exceptions import ValidationError
from src.api.schemas.requests import CostAttributionInput
from src.api.schemas.requests import DocumentInput
from src.api.schemas.requests import ProcessingOptions
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
def mock_tasks():
    return MagicMock()


@pytest.fixture
def service(mock_storage, mock_bq, mock_tasks):
    return TranslationService(
        storage=mock_storage, bigquery=mock_bq, cloud_tasks=mock_tasks
    )


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

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_cloud_tasks_enqueue_failure_marks_job_failed(
        self, mock_validate, mock_storage, mock_bq, valid_request, monkeypatch
    ):
        """If Cloud Tasks enqueue fails after BQ queued, job is marked failed."""
        monkeypatch.setattr(
            "src.api.services.translation_service.settings.API_USE_BACKGROUND_PIPELINE",
            False,
        )
        mock_validate.return_value = (
            b"pdf",
            {
                "page_count": 1,
                "filename": "test.pdf",
                "size_bytes": 100,
                "checksum": "abc",
            },
        )
        mock_storage.upload_input_pdf.return_value = "gs://bucket/test.pdf"
        mock_tasks = MagicMock()
        mock_tasks.enqueue_translate.side_effect = RuntimeError("queue down")
        service = TranslationService(
            storage=mock_storage,
            bigquery=mock_bq,
            cloud_tasks=mock_tasks,
        )

        with pytest.raises(RuntimeError, match="Failed to enqueue"):
            await service.submit_translation(valid_request)

        mock_bq.patch_translation_job.assert_awaited()
        patch_args = mock_bq.patch_translation_job.await_args
        assert patch_args.args[1]["status"] == "failed"
        assert "queue down" in patch_args.args[1]["error_message"]

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_submit_translations_shares_input_and_creates_jobs(
        self, mock_validate, mock_storage, mock_bq, monkeypatch
    ):
        monkeypatch.setattr(
            "src.api.services.translation_service.settings.API_USE_BACKGROUND_PIPELINE",
            False,
        )
        mock_validate.return_value = (
            b"pdf",
            {
                "page_count": 1,
                "filename": "test.pdf",
                "size_bytes": 100,
                "checksum": "abc",
            },
        )
        mock_storage.upload_input_pdf.return_value = "gs://bucket/shared/test.pdf"
        mock_tasks = MagicMock()
        service = TranslationService(
            storage=mock_storage, bigquery=mock_bq, cloud_tasks=mock_tasks
        )
        requests = [
            TranslateRequest(
                document=DocumentInput(
                    content=base64.b64encode(b"%PDF-1.4\n%%EOF").decode(),
                    filename="test.pdf",
                ),
                translation_config=TranslationConfigInput(
                    target_language=target, domain="legal"
                ),
                cost_attribution=CostAttributionInput(
                    user_id="user1", business_unit="legal", organization="colt"
                ),
            )
            for target in ("fr", "de")
        ]

        response = await service.submit_translations(requests)

        assert [job.target_language for job in response.jobs] == ["fr", "de"]
        mock_storage.upload_input_pdf.assert_awaited_once()
        assert mock_bq.upsert_translation_job.await_count == 2
        records = [
            call.args[0] for call in mock_bq.upsert_translation_job.await_args_list
        ]
        assert {record["batch_id"] for record in records} == {response.batch_id}
        assert [record["batch_index"] for record in records] == [0, 1]
        assert len({record["source_document"]["gcs_uri"] for record in records}) == 1
        assert mock_tasks.enqueue_translate.call_count == 2
        assert [job.status for job in response.jobs] == ["queued", "queued"]


class TestPriorityRouting:
    """Server-side queue selection. Clients cannot promote their own work."""

    @staticmethod
    def _req(fmt="pdf", client_priority="standard"):
        from tests.conftest import _make_pdf_bytes

        content_map = {
            "pdf": _make_pdf_bytes(),
            "docx": b"PK\x03\x04fake",
            "txt": b"plain text",
        }
        raw = content_map[fmt]
        return TranslateRequest(
            document=DocumentInput(
                content=base64.b64encode(raw).decode(),
                filename=f"test.{fmt}",
                format=fmt,
            ),
            translation_config=TranslationConfigInput(
                target_language="fr", domain="commercial"
            ),
            cost_attribution=CostAttributionInput(
                user_id="user1", business_unit="bu", organization="colt"
            ),
            processing_options=ProcessingOptions(priority=client_priority),
        )

    @pytest.fixture(autouse=True)
    def _use_cloud_tasks_mode(self, monkeypatch):
        """These tests assert Cloud Tasks enqueue routing (IS_LOCAL/pipeline=false)."""
        monkeypatch.setattr(
            "src.api.services.translation_service.settings.API_USE_BACKGROUND_PIPELINE",
            False,
        )

    async def test_txt_is_auto_promoted_to_high_priority(
        self, service, mock_storage, mock_tasks, mock_bq
    ):
        mock_storage.upload_input_pdf.return_value = "gs://b/test.txt"
        await service.submit_translations([self._req(fmt="txt")])
        call = mock_tasks.enqueue_translate.call_args
        assert call[1]["priority"] == "high"
        assert call[1]["doc_format"] == "txt"
        stored = mock_bq.upsert_translation_job.call_args[0][0]
        assert stored["processing_options"]["priority"] == "high"

    async def test_pdf_stays_standard_priority(
        self, service, mock_storage, mock_tasks, mock_bq
    ):
        mock_storage.upload_input_pdf.return_value = "gs://b/test.pdf"
        await service.submit_translations([self._req(fmt="pdf")])
        call = mock_tasks.enqueue_translate.call_args
        assert call[1]["priority"] == "standard"
        assert call[1]["doc_format"] == "pdf"
        stored = mock_bq.upsert_translation_job.call_args[0][0]
        assert stored["processing_options"]["priority"] == "standard"

    async def test_client_high_priority_is_ignored_for_pdf(
        self, service, mock_storage, mock_tasks, mock_bq
    ):
        """Users must not be able to jump the queue by passing priority=high."""
        mock_storage.upload_input_pdf.return_value = "gs://b/test.pdf"
        await service.submit_translations(
            [self._req(fmt="pdf", client_priority="high")]
        )
        call = mock_tasks.enqueue_translate.call_args
        assert call[1]["priority"] == "standard"
        stored = mock_bq.upsert_translation_job.call_args[0][0]
        assert stored["processing_options"]["priority"] == "standard"

    async def test_routing_disabled_forces_standard(
        self, service, mock_storage, mock_tasks, monkeypatch
    ):
        monkeypatch.setattr(
            "src.api.services.translation_service.settings.HIGH_PRIORITY_ROUTING_ENABLED",
            False,
        )
        mock_storage.upload_input_pdf.return_value = "gs://b/test.txt"
        await service.submit_translations([self._req(fmt="txt")])
        assert mock_tasks.enqueue_translate.call_args[1]["priority"] == "standard"

    async def test_configurable_formats_list(
        self, service, mock_storage, mock_tasks, monkeypatch
    ):
        """Format list is configurable without a code deploy."""
        monkeypatch.setattr(
            "src.api.services.translation_service.settings.HIGH_PRIORITY_FORMATS",
            ["docx"],
        )
        mock_storage.upload_input_pdf.return_value = "gs://b/test.docx"
        await service.submit_translations([self._req(fmt="docx")])
        assert mock_tasks.enqueue_translate.call_args[1]["priority"] == "high"

