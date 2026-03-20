"""
Integration tests for translation endpoints:
  POST /api/v1/translate
  GET  /api/v1/translate/{job_id}
  GET  /api/v1/healthz

Uses the FastAPI TestClient with mocked service dependencies
(no real GCP calls). See integration/conftest.py for fixtures.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_job_service, get_translation_service
from api.exceptions import JobNotFoundError
from api.main import app
from api.schemas.responses import JobDetailResponse, TranslateResponse
from fixtures.sample_data import TRANSLATE_REQUEST_VALID


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
    def test_returns_200(self, api_client):
        resp = api_client.post("/api/v1/translate", json=TRANSLATE_REQUEST_VALID)
        assert resp.status_code == 200

    def test_response_has_job_id(self, api_client):
        resp = api_client.post("/api/v1/translate", json=TRANSLATE_REQUEST_VALID)
        body = resp.json()
        assert "job_id" in body
        assert body["job_id"]

    def test_response_status_is_queued(self, api_client):
        resp = api_client.post("/api/v1/translate", json=TRANSLATE_REQUEST_VALID)
        body = resp.json()
        assert body["status"] == "queued"

    def test_response_has_status_url(self, api_client):
        resp = api_client.post("/api/v1/translate", json=TRANSLATE_REQUEST_VALID)
        body = resp.json()
        assert "status_url" in body
        assert "/api/v1/translate/" in body["status_url"]

    def test_missing_document_returns_422(self, api_client):
        payload = {
            "translation_config": {
                "target_language": "es",
                "domain": "commercial",
            }
        }
        resp = api_client.post("/api/v1/translate", json=payload)
        assert resp.status_code == 422

    def test_missing_translation_config_returns_422(self, api_client):
        payload = {
            "document": {
                "content": "dGVzdA==",  # valid base64 of "test"
                "filename": "file.pdf",
            }
        }
        resp = api_client.post("/api/v1/translate", json=payload)
        assert resp.status_code == 422

    def test_invalid_base64_content_returns_422(self, api_client):
        payload = {
            "document": {
                "content": "!!!NOT_BASE64!!!",
                "filename": "file.pdf",
            },
            "translation_config": {
                "target_language": "es",
                "domain": "commercial",
            },
        }
        resp = api_client.post("/api/v1/translate", json=payload)
        assert resp.status_code == 422

    def test_invalid_domain_returns_422(self, api_client):
        from fixtures.sample_data import VALID_PDF_B64

        payload = {
            "document": {"content": VALID_PDF_B64, "filename": "doc.pdf"},
            "translation_config": {
                "target_language": "es",
                "domain": "science",  # invalid
            },
        }
        resp = api_client.post("/api/v1/translate", json=payload)
        assert resp.status_code == 422

    def test_invalid_filename_extension_returns_422(self, api_client):
        from fixtures.sample_data import VALID_PDF_B64

        payload = {
            "document": {"content": VALID_PDF_B64, "filename": "doc.exe"},
            "translation_config": {
                "target_language": "es",
                "domain": "commercial",
            },
        }
        resp = api_client.post("/api/v1/translate", json=payload)
        assert resp.status_code == 422

    def test_service_exception_returns_error_response(self, api_client, mock_translation_service):
        """If the service raises, the middleware converts it to an error response."""
        from api.exceptions import ValidationError as APIValidationError

        mock_translation_service.submit_translation.side_effect = APIValidationError(
            "encrypted PDF not supported"
        )
        resp = api_client.post("/api/v1/translate", json=TRANSLATE_REQUEST_VALID)
        assert resp.status_code in (400, 422, 500)
        # Reset side effect for subsequent tests
        mock_translation_service.submit_translation.side_effect = None
        mock_translation_service.submit_translation.return_value = TranslateResponse(
            job_id="test-job-id-001",
            status="queued",
            status_url="/api/v1/translate/test-job-id-001",
        )


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

    def test_response_has_status(self, api_client):
        resp = api_client.get("/api/v1/translate/test-job-id-001")
        body = resp.json()
        assert body["status"] in ("queued", "processing", "completed", "failed", "cancelled")

    def test_returns_404_for_missing_job(self, api_client, mock_job_service):
        mock_job_service.get_translation_status.side_effect = JobNotFoundError("ghost-job")
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
        """When job is completed, result field should be populated."""
        from api.schemas.responses import (
            TranslatedDocumentResult,
            TranslationLabels,
            TranslationMetadata,
            TranslationResult,
        )

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
