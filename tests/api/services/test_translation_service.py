import base64
import hashlib
import uuid
from datetime import UTC
from datetime import datetime
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
    bq = AsyncMock()
    bq.get_completed_job_by_hash.return_value = None  # cache miss by default
    return bq


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

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_submit_translations_handles_cache_hits_per_target(
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
        cached_job = _make_cached_job("hash")
        mock_bq.get_completed_job_by_hash.side_effect = [cached_job, None]
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

        assert [job.status for job in response.jobs] == ["completed", "queued"]
        mock_storage.upload_input_pdf.assert_awaited_once()
        mock_tasks.enqueue_translate.assert_called_once()


# ---------------------------------------------------------------------------
# Cache-hit: submitting the same document twice reuses the prior result
# ---------------------------------------------------------------------------


def _make_cached_job(source_hash: str) -> dict:
    now = datetime.now(UTC)
    jid = str(uuid.uuid4())
    return {
        "job_id": jid,
        "status": "completed",
        "source_document": {
            "gcs_uri": f"gs://bucket/translation/{jid}/input/doc.pdf",
            "format": "pdf",
            "page_count": 2,
            "source_language": "auto",
            "original_filename": "contract.pdf",
            "output_filename": "contract.pdf",
            "file_size_bytes": 1024,
            "checksum": source_hash,
        },
        "translation_config": {
            "source_language": "auto",
            "target_language": "fr",
            "domain": "legal",
        },
        "result": {
            "output_gcs_uri": f"gs://bucket/translation/{jid}/output/contract.pdf",
            "token_count": 3000,
            "cost_usd": 0.90,
            "model_used": "claude-sonnet-4-6",
            "intent": "legal_en_fr",
        },
        "source_hash": source_hash,
        "submitted_at": now,
        "completed_at": now,
    }


_PDF_METADATA = {
    "filename": "contract.pdf",
    "size_bytes": 1024,
    "content_type": "application/pdf",
    "checksum": "placeholder",  # overridden per-fixture
    "page_count": 2,
}

_CACHED_CONTENT = b"%PDF-1.4 cached doc content"


def _make_cache_request() -> TranslateRequest:
    return TranslateRequest(
        document=DocumentInput(
            content=base64.b64encode(_CACHED_CONTENT).decode(),
            filename="contract.pdf",
        ),
        translation_config=TranslationConfigInput(
            target_language="French",
            domain="legal",
        ),
        cost_attribution=CostAttributionInput(
            user_id="user@colt.net",
            business_unit="legal-dept",
            organization="colt",
        ),
        processing_options=ProcessingOptions(),
    )


class TestCachedTranslation:
    """
    Objective: submitting the same document twice returns the cached translation
    without re-processing.

    Pre-conditions:
    - User is authenticated.
    - The same document was previously translated (completed job exists in BigQuery).
    - Cache TTL is still active (completed job is returned by get_completed_job_by_hash).

    Test data: POST request whose SHA-256 matches an existing completed job.

    Expected behaviour:
    - Response status is "completed" (not "queued").
    - A new job record is written to BigQuery immediately as "completed".
    - No GCS upload occurs; no pipeline is scheduled.
    - The result payload is copied verbatim from the cached job.
    """

    @pytest.fixture
    def source_hash(self) -> str:
        return hashlib.sha256(_CACHED_CONTENT).hexdigest()

    @pytest.fixture
    def cached_job(self, source_hash) -> dict:
        return _make_cached_job(source_hash)

    @pytest.fixture
    def cache_bq(self, cached_job) -> AsyncMock:
        bq = AsyncMock()
        bq.get_completed_job_by_hash.return_value = cached_job
        bq.upsert_translation_job.return_value = None
        return bq

    @pytest.fixture
    def cache_storage(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def cache_service(self, cache_storage, cache_bq) -> TranslationService:
        return TranslationService(storage=cache_storage, bigquery=cache_bq)

    @pytest.fixture
    def cache_request(self) -> TranslateRequest:
        return _make_cache_request()

    @pytest.fixture
    def pdf_meta(self, source_hash) -> dict:
        return {**_PDF_METADATA, "checksum": source_hash}

    # ------------------------------------------------------------------
    # Response shape
    # ------------------------------------------------------------------

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_response_status_is_completed(
        self, mock_validate, cache_service, cache_request, pdf_meta
    ):
        mock_validate.return_value = (b"", pdf_meta)
        resp = await cache_service.submit_translation(cache_request)
        assert resp.status == "completed"

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_response_has_job_id(
        self, mock_validate, cache_service, cache_request, pdf_meta
    ):
        mock_validate.return_value = (b"", pdf_meta)
        resp = await cache_service.submit_translation(cache_request)
        assert resp.job_id

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_response_has_status_url(
        self, mock_validate, cache_service, cache_request, pdf_meta
    ):
        mock_validate.return_value = (b"", pdf_meta)
        resp = await cache_service.submit_translation(cache_request)
        assert "/api/v1/translate/" in resp.status_url

    # ------------------------------------------------------------------
    # No re-processing
    # ------------------------------------------------------------------

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_does_not_upload_to_gcs(
        self, mock_validate, cache_service, cache_storage, cache_request, pdf_meta
    ):
        mock_validate.return_value = (b"", pdf_meta)
        await cache_service.submit_translation(cache_request)
        cache_storage.upload_input_pdf.assert_not_called()

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_does_not_schedule_pipeline(
        self, mock_validate, cache_service, cache_request, pdf_meta
    ):
        mock_validate.return_value = (b"", pdf_meta)
        with patch.object(cache_service, "_schedule_background_pipeline") as mock_sched:
            await cache_service.submit_translation(cache_request)
        mock_sched.assert_not_called()

    # ------------------------------------------------------------------
    # BigQuery job is written immediately as completed
    # ------------------------------------------------------------------

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_bq_job_written_with_completed_status(
        self, mock_validate, cache_service, cache_bq, cache_request, pdf_meta
    ):
        mock_validate.return_value = (b"", pdf_meta)
        await cache_service.submit_translation(cache_request)
        cache_bq.upsert_translation_job.assert_called_once()
        job_data = cache_bq.upsert_translation_job.call_args[0][0]
        assert job_data["status"] == "completed"

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_bq_job_has_completed_at(
        self, mock_validate, cache_service, cache_bq, cache_request, pdf_meta
    ):
        mock_validate.return_value = (b"", pdf_meta)
        await cache_service.submit_translation(cache_request)
        job_data = cache_bq.upsert_translation_job.call_args[0][0]
        assert job_data.get("completed_at") is not None

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_bq_job_id_differs_from_cached_job(
        self,
        mock_validate,
        cache_service,
        cache_bq,
        cache_request,
        pdf_meta,
        cached_job,
    ):
        mock_validate.return_value = (b"", pdf_meta)
        resp = await cache_service.submit_translation(cache_request)
        assert resp.job_id != cached_job["job_id"]

    # ------------------------------------------------------------------
    # Result is copied verbatim from the cache
    # ------------------------------------------------------------------

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_result_copied_from_cache(
        self,
        mock_validate,
        cache_service,
        cache_bq,
        cache_request,
        pdf_meta,
        cached_job,
    ):
        mock_validate.return_value = (b"", pdf_meta)
        await cache_service.submit_translation(cache_request)
        job_data = cache_bq.upsert_translation_job.call_args[0][0]
        assert job_data["result"] == cached_job["result"]

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_cache_lookup_uses_correct_hash(
        self,
        mock_validate,
        cache_service,
        cache_bq,
        cache_request,
        pdf_meta,
        source_hash,
    ):
        mock_validate.return_value = (b"", pdf_meta)
        await cache_service.submit_translation(cache_request)
        cache_bq.get_completed_job_by_hash.assert_called_once_with(
            source_hash=source_hash,
            lang_out="fr",
            domain="legal",
        )

    # ------------------------------------------------------------------
    # Cache miss falls through to normal queued flow
    # ------------------------------------------------------------------

    @patch("src.api.services.translation_service.PDFValidator.validate_pdf_bytes")
    async def test_cache_miss_returns_queued(
        self, mock_validate, cache_storage, cache_request, pdf_meta
    ):
        mock_validate.return_value = (b"", pdf_meta)
        bq_miss = AsyncMock()
        bq_miss.get_completed_job_by_hash.return_value = None
        bq_miss.upsert_translation_job.return_value = None
        cache_storage.upload_input_pdf.return_value = "gs://bucket/input.pdf"
        svc = TranslationService(storage=cache_storage, bigquery=bq_miss)
        with patch.object(svc, "_schedule_background_pipeline"):
            resp = await svc.submit_translation(cache_request)
        assert resp.status == "queued"
