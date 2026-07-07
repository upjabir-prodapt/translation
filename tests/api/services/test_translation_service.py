"""
Unit tests for api/services/translation_service.py — TranslationService.

Firestore, GCS (APIStorageRepository), and Cloud Tasks are fully mocked.
The real PDFValidator runs against in-memory PDFs, so fitz is a real dep.
"""

import base64
import io
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import fitz
import pytest
from fixtures.sample_data import VALID_PDF_B64
from pydantic import ValidationError as PydanticValidationError

from api.exceptions import ValidationError
from api.schemas.requests import DocumentInput
from api.schemas.requests import ProcessingOptions
from api.schemas.requests import TranslateRequest
from api.schemas.requests import TranslationConfigInput
from api.services.translation_service import TranslationService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    b64: str = VALID_PDF_B64,
    filename: str = "doc.pdf",
    target_lang: str = "Spanish",
    domain: str = "commercial",
) -> TranslateRequest:
    return TranslateRequest(
        document=DocumentInput(content=b64, filename=filename),
        translation_config=TranslationConfigInput(
            source_language="English",
            target_language=target_lang,
            domain=domain,
        ),
        processing_options=ProcessingOptions(),
    )


def _make_service(
    firestore=None, storage=None, tasks_client=None
) -> TranslationService:
    fs = firestore or AsyncMock()
    st = storage or AsyncMock()
    tc = tasks_client or MagicMock()
    return TranslationService(firestore=fs, storage=st, tasks_client=tc)


# ---------------------------------------------------------------------------
# submit_translation — positive cases
# ---------------------------------------------------------------------------


class TestSubmitTranslationPositive:
    async def test_returns_translate_response(self, mock_tasks_client):
        storage = AsyncMock()
        storage.upload_input_pdf.return_value = (
            "gs://bucket/translation/job/input/doc.pdf"
        )
        storage.delete_job_files = AsyncMock()
        firestore = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        response = await service.submit_translation(_make_request())

        assert response.job_id
        assert response.status == "queued"
        assert "/api/v1/translate/" in response.status_url

    async def test_job_id_in_status_url(self, mock_tasks_client):
        storage = AsyncMock()
        storage.upload_input_pdf.return_value = "gs://bucket/path"
        firestore = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        response = await service.submit_translation(_make_request())
        assert response.job_id in response.status_url

    async def test_firestore_create_job_called(self, mock_tasks_client):
        storage = AsyncMock()
        storage.upload_input_pdf.return_value = "gs://bucket/path"
        firestore = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        await service.submit_translation(_make_request())
        firestore.create_job.assert_called_once()

    async def test_storage_upload_called(self, mock_tasks_client):
        storage = AsyncMock()
        storage.upload_input_pdf.return_value = "gs://bucket/path"
        firestore = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        await service.submit_translation(_make_request())
        storage.upload_input_pdf.assert_called_once()

    async def test_cloud_task_created(self, mock_tasks_client):
        storage = AsyncMock()
        storage.upload_input_pdf.return_value = "gs://bucket/path"
        firestore = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        await service.submit_translation(_make_request())
        mock_tasks_client.create_task.assert_called_once()

    async def test_output_filename_contains_target_lang_code(self, mock_tasks_client):
        """Output filename should be: original_name_<lang_code>.pdf"""
        storage = AsyncMock()
        storage.upload_input_pdf.return_value = "gs://bucket/path"
        firestore = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        await service.submit_translation(
            _make_request(filename="report.pdf", target_lang="French")
        )

        job_data = firestore.create_job.call_args[0][1]
        output_fn = job_data["source_document"]["output_filename"]
        assert output_fn.endswith(".pdf")
        assert "fr" in output_fn  # French normalizes to "fr"

    async def test_job_data_has_queued_status(self, mock_tasks_client):
        storage = AsyncMock()
        storage.upload_input_pdf.return_value = "gs://bucket/path"
        firestore = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        await service.submit_translation(_make_request())
        job_data = firestore.create_job.call_args[0][1]
        assert job_data["status"] == "queued"

    async def test_job_data_config_has_lang_out(self, mock_tasks_client):
        storage = AsyncMock()
        storage.upload_input_pdf.return_value = "gs://bucket/path"
        firestore = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        await service.submit_translation(
            _make_request(target_lang="German", domain="legal")
        )
        job_data = firestore.create_job.call_args[0][1]
        assert job_data["config"]["lang_out"] == "de"
        assert job_data["config"]["domain"] == "legal"


# ---------------------------------------------------------------------------
# submit_translation — failure / rollback cases
# ---------------------------------------------------------------------------


class TestSubmitTranslationFailures:
    async def test_invalid_base64_raises_validation_error(self):
        """Invalid base64 content raises ValidationError at request schema level."""
        with pytest.raises(PydanticValidationError):
            # Pydantic will catch this at construction time
            DocumentInput(content="not-valid-base64!!!", filename="doc.pdf")

    async def test_encrypted_pdf_raises_validation_error(self, mock_tasks_client):
        """Encrypted PDFs are rejected by PDFValidator."""
        # Build an encrypted PDF
        doc = fitz.open()
        doc.new_page()
        buf = io.BytesIO()
        doc.save(
            buf,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
            permissions=0,
        )
        doc.close()
        encrypted_b64 = base64.b64encode(buf.getvalue()).decode()

        storage = AsyncMock()
        firestore = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        req = TranslateRequest(
            document=DocumentInput(content=encrypted_b64, filename="secure.pdf"),
            translation_config=TranslationConfigInput(
                target_language="es", domain="commercial"
            ),
        )
        with pytest.raises(ValidationError, match="Encrypted"):
            await service.submit_translation(req)

    async def test_gcs_upload_failure_triggers_cleanup(self, mock_tasks_client):
        """If GCS upload fails, cleanup is attempted (files + Firestore)."""
        storage = AsyncMock()
        storage.upload_input_pdf.side_effect = Exception("GCS unavailable")
        storage.delete_job_files = AsyncMock()
        firestore = AsyncMock()
        firestore.delete_job = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        with pytest.raises(Exception, match="GCS unavailable"):
            await service.submit_translation(_make_request())

        # Cleanup was attempted
        storage.delete_job_files.assert_called_once()

    async def test_firestore_failure_after_upload_triggers_cleanup(
        self, mock_tasks_client
    ):
        """If Firestore create_job fails after GCS upload, cleanup is attempted."""
        storage = AsyncMock()
        storage.upload_input_pdf.return_value = "gs://bucket/path"
        storage.delete_job_files = AsyncMock()
        firestore = AsyncMock()
        firestore.create_job.side_effect = Exception("Firestore down")
        firestore.delete_job = AsyncMock()
        service = _make_service(
            firestore=firestore, storage=storage, tasks_client=mock_tasks_client
        )

        with pytest.raises(Exception, match="Firestore down"):
            await service.submit_translation(_make_request())

        storage.delete_job_files.assert_called_once()


# ---------------------------------------------------------------------------
# _normalize_config (private method tested via submit_translation behaviour)
# ---------------------------------------------------------------------------


class TestNormalizeConfig:
    def test_lang_in_always_auto(self):
        service = _make_service()
        req = _make_request()
        config = service._normalize_config(req)
        assert config["lang_in"] == "auto"

    def test_lang_out_normalized(self):
        service = _make_service()
        req = _make_request(target_lang="Spanish")
        config = service._normalize_config(req)
        assert config["lang_out"] == "es"

    def test_domain_normalized(self):
        service = _make_service()
        req = _make_request(domain="COMMERCIAL")
        config = service._normalize_config(req)
        assert config["domain"] == "commercial"

    def test_invalid_target_language_raises(self):
        service = _make_service()
        # Bypass Pydantic schema validation by building a request with a monkeypatched cfg
        from unittest.mock import MagicMock

        req = MagicMock()
        req.translation_config.domain = "commercial"
        req.translation_config.target_language = "klingon"
        req.translation_config.source_language = "auto"
        with pytest.raises(ValidationError):
            service._normalize_config(req)

    def test_invalid_domain_raises(self):
        service = _make_service()
        from unittest.mock import MagicMock

        req = MagicMock()
        req.translation_config.domain = "science"
        req.translation_config.target_language = "es"
        with pytest.raises(ValidationError):
            service._normalize_config(req)


# ---------------------------------------------------------------------------
# _create_translation_task
# ---------------------------------------------------------------------------


class TestCreateTranslationTask:
    async def test_calls_queue_path(self, mock_tasks_client):
        service = _make_service(tasks_client=mock_tasks_client)
        config = {"job_id": "j1", "lang_in": "auto", "lang_out": "es", "domain": "hr"}

        await service._create_translation_task("j1", config)

        mock_tasks_client.queue_path.assert_called_once()

    async def test_calls_create_task(self, mock_tasks_client):
        service = _make_service(tasks_client=mock_tasks_client)
        config = {"job_id": "j1", "lang_in": "auto", "lang_out": "es", "domain": "hr"}

        await service._create_translation_task("j1", config)

        mock_tasks_client.create_task.assert_called_once()

    async def test_task_payload_contains_job_id(self, mock_tasks_client):
        import json

        service = _make_service(tasks_client=mock_tasks_client)
        config = {"job_id": "j1", "lang_in": "auto", "lang_out": "es", "domain": "hr"}

        await service._create_translation_task("j1", config)

        call_kwargs = mock_tasks_client.create_task.call_args
        request_arg = call_kwargs[1]["request"] if call_kwargs[1] else call_kwargs[0][0]
        body = request_arg["task"]["http_request"]["body"]
        payload = json.loads(body.decode())
        assert payload["job_id"] == "j1"
