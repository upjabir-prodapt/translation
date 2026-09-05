"""Route tests for translation endpoints."""

from datetime import UTC
from datetime import datetime

from src.api.exceptions import JobNotFoundError
from src.api.schemas.responses import JobDetailResponse

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_returns_200(self, api_client):
        resp = api_client.get("/api/v1/healthz")
        assert resp.status_code == 200

    def test_body_has_status_healthy(self, api_client):
        resp = api_client.get("/api/v1/healthz")
        body = resp.json()
        assert body["status"] == "healthy"

    def test_body_has_version(self, api_client):
        resp = api_client.get("/api/v1/healthz")
        body = resp.json()
        assert "version" in body

    def test_body_has_uptime(self, api_client):
        resp = api_client.get("/api/v1/healthz")
        body = resp.json()
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# POST /api/v1/translate
# ---------------------------------------------------------------------------


class TestSubmitTranslation:
    def test_returns_202(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 202

    def test_response_has_job_id(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        body = resp.json()
        assert "jobs" in body
        assert len(body["jobs"]) == 1
        assert body["jobs"][0]["job_id"]

    def test_response_status_is_queued(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        body = resp.json()
        assert body["jobs"][0]["status"] == "queued"

    def test_response_has_status_url(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        body = resp.json()
        assert "status_url" in body["jobs"][0]
        assert "/api/v1/translate/" in body["jobs"][0]["status_url"]

    def test_missing_file_returns_422(self, api_client):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
        )
        assert resp.status_code == 422

    def test_missing_target_language_returns_422(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={"domain": "commercial"},
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_invalid_domain_returns_422(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "science",
            },
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_missing_source_language_returns_422(self, api_client, minimal_pdf_bytes):
        """Source language is mandatory: auto-detection was removed.

        The worker verifies the declared language against the document, so an
        omitted value has nothing to verify and must be rejected up front --
        before any GCS upload, BigQuery row or Cloud Task is created.
        """
        resp = api_client.post(
            "/api/v1/translate",
            data={"target_languages": ["Spanish"], "domain": "commercial"},
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_blank_source_language_returns_422(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "   ",
                "domain": "commercial",
            },
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_missing_domain_returns_422(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={"target_languages": ["Spanish"], "source_language": "English"},
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_missing_target_languages_returns_422(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={"source_language": "English", "domain": "commercial"},
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_source_equal_to_target_returns_422(self, api_client, minimal_pdf_bytes):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "Spanish",
                "domain": "commercial",
            },
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_unsupported_extension_returns_422(self, api_client, minimal_pdf_bytes):
        """A.4.3: unrecognized extensions (e.g. .zip) must be rejected
        explicitly instead of silently defaulting to 'pdf' and failing
        deep inside PDFValidator with a confusing message."""
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
            files={
                "file": ("archive.zip", minimal_pdf_bytes, "application/zip"),
            },
        )
        assert resp.status_code == 422
        assert "Unsupported file type" in resp.text

    def test_empty_file_returns_422(self, api_client):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert resp.status_code == 422

    def test_empty_file_does_not_create_bq_job(
        self, api_client, mock_translation_service
    ):
        mock_translation_service.submit_translations.reset_mock()
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert resp.status_code == 422
        mock_translation_service.submit_translations.assert_not_called()

    def test_oversized_file_returns_422(self, api_client):
        oversized = b"x" * (50 * 1024 * 1024 + 1)
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
            files={"file": ("big.pdf", oversized, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_oversized_file_does_not_create_bq_job(
        self, api_client, mock_translation_service
    ):
        mock_translation_service.submit_translations.reset_mock()
        oversized = b"x" * (50 * 1024 * 1024 + 1)
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "target_languages": ["Spanish"],
                "source_language": "English",
                "domain": "commercial",
            },
            files={"file": ("big.pdf", oversized, "application/pdf")},
        )
        assert resp.status_code == 422
        mock_translation_service.submit_translations.assert_not_called()

    def test_multiple_targets_return_ordered_job_list(
        self, api_client, minimal_pdf_bytes
    ):
        resp = api_client.post(
            "/api/v1/translate",
            data={
                "domain": "commercial",
                "target_languages": ["French", "German"],
                "source_language": "English",
            },
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["batch_id"] == "test-batch-id"
        assert [job["target_language"] for job in body["jobs"]] == ["fr", "de"]


# ---------------------------------------------------------------------------
# GET /api/v1/translate/{job_id}
# ---------------------------------------------------------------------------


class TestGetTranslationStatus:
    def test_returns_200_for_existing_job(self, api_client):
        resp = api_client.get("/api/v1/translate/test-job-id-001")
        assert resp.status_code == 200

    def test_response_has_job_id(self, api_client):
        resp = api_client.get("/api/v1/translate/test-job-id-001")
        body = resp.json()
        assert body["job_id"] == "test-job-id-001"

    def test_passes_authenticated_user_email_for_ownership_check(
        self, api_client, mock_job_service
    ):
        """implementation_plan.md D.1 (Sev-1): the route must thread the
        authenticated caller's identity into the service so ownership can
        be enforced -- without this wiring, any user holding another
        user's job_id could read that job's full detail/result."""
        mock_job_service.get_translation_status.reset_mock()
        api_client.get("/api/v1/translate/test-job-id-001")
        mock_job_service.get_translation_status.assert_called_once_with(
            "test-job-id-001", "user@colt.net"
        )

    def test_response_has_status(self, api_client):
        resp = api_client.get("/api/v1/translate/test-job-id-001")
        body = resp.json()
        assert body["status"] in (
            "queued",
            "processing",
            "completed",
            "failed",
            "cancelled",
        )

    def test_returns_404_for_missing_job(self, api_client, mock_job_service):
        mock_job_service.get_translation_status.side_effect = JobNotFoundError(
            "ghost-job"
        )
        resp = api_client.get("/api/v1/translate/ghost-job")
        assert resp.status_code == 404
        # Reset
        now = datetime.now(UTC)
        mock_job_service.get_translation_status.side_effect = None
        mock_job_service.get_translation_status.return_value = JobDetailResponse(
            job_id="test-job-id-001",
            status="queued",
            submitted_at=now,
            completed_at=None,
            result=None,
        )

    def test_completed_job_has_result_field(self, api_client, mock_job_service):
        """When job is completed, result field is populated."""
        from src.api.schemas.responses import TranslatedDocumentResult
        from src.api.schemas.responses import TranslationLabels
        from src.api.schemas.responses import TranslationMetadata
        from src.api.schemas.responses import TranslationResult

        now = datetime.now(UTC)
        mock_job_service.get_translation_status.return_value = JobDetailResponse(
            job_id="test-job-id-001",
            status="completed",
            submitted_at=now,
            completed_at=now,
            result=TranslationResult(
                translated_document=TranslatedDocumentResult(
                    content=None,
                    format="pdf",
                    filename="doc_es.pdf",
                    download_url="https://signed.url/doc_es.pdf",
                ),
                metadata=TranslationMetadata(
                    source_language="en",
                    target_language="es",
                    domain="commercial",
                ),
                labels=TranslationLabels(),
            ),
        )
        resp = api_client.get("/api/v1/translate/test-job-id-001")
        body = resp.json()
        assert body["status"] == "completed"
        assert body["result"] is not None
        # Restore
        mock_job_service.get_translation_status.return_value = JobDetailResponse(
            job_id="test-job-id-001",
            status="queued",
            submitted_at=now,
            completed_at=None,
            result=None,
        )
