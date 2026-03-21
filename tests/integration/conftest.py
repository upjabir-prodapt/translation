"""
Integration test fixtures — FastAPI TestClient with mocked service dependencies.

The real FastAPI app (api.main.app) is used, but the service-layer dependencies
(get_translation_service, get_job_service) are overridden so no GCP calls occur.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_job_service, get_translation_service
from api.main import app
from api.schemas.responses import (
    JobDetailResponse,
    JobListResponse,
    JobStatusResponse,
    TranslateResponse,
)
from conftest import make_job_doc


# ---------------------------------------------------------------------------
# Service mocks
# ---------------------------------------------------------------------------


def _make_translate_response(job_id: str | None = None) -> TranslateResponse:
    jid = job_id or str(uuid.uuid4())
    return TranslateResponse(
        job_id=jid,
        status="queued",
        status_url=f"/api/v1/translate/{jid}",
    )


def _make_job_status(job_id: str, status: str = "queued") -> JobStatusResponse:
    now = datetime.now(UTC)
    return JobStatusResponse(
        job_id=job_id,
        status=status,
        progress=0.0,
        current_stage=None,
        user="",
        department="",
        created_at=now,
        updated_at=now,
        completed_at=None,
        error_message=None,
    )


def _make_job_detail(job_id: str, status: str = "queued") -> JobDetailResponse:
    now = datetime.now(UTC)
    return JobDetailResponse(
        job_id=job_id,
        status=status,
        submitted_at=now,
        completed_at=None,
        result=None,
    )


# ---------------------------------------------------------------------------
# Module-scoped mocked service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_translation_service():
    """A mock TranslationService that returns predictable responses."""
    service = AsyncMock()
    job_id = "test-job-id-001"
    service.submit_translation.return_value = _make_translate_response(job_id)
    return service


@pytest.fixture(scope="module")
def mock_job_service():
    """A mock JobService that returns predictable responses."""
    service = AsyncMock()
    job_id = "test-job-id-001"
    service.get_job_status.return_value = _make_job_status(job_id, "queued")
    service.get_translation_status.return_value = _make_job_detail(job_id, "queued")
    service.list_jobs.return_value = JobListResponse(
        jobs=[_make_job_status(job_id)],
        total=1,
        limit=10,
        offset=0,
    )
    service.cancel_job.return_value = None
    service.get_download_url = AsyncMock(
        side_effect=Exception("not configured")  # default: raise for download tests
    )
    return service


@pytest.fixture(scope="module")
def api_client(mock_translation_service, mock_job_service):
    """
    TestClient with dependency overrides for TranslationService and JobService.
    Scoped to module so the same app instance is reused across tests in a module.
    """
    app.dependency_overrides[get_translation_service] = lambda: mock_translation_service
    app.dependency_overrides[get_job_service] = lambda: mock_job_service

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()
