"""
Shared fixtures for the Translation API Service test suite.

Requirements (add to dev dependencies):
    pytest-asyncio>=0.24.0
    pytest-mock>=3.14.0
"""

import os

# Set dummy environment variables to prevent Pydantic validation errors
# when loading application settings in the test suite without a .env file.
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "mock-project-id")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GCS_BUCKET_NAME", "mock-bucket")
os.environ.setdefault("BIGQUERY_DATASET", "mock-dataset")
os.environ.setdefault("JWT_SECRET_KEY", "mock-secret-key-for-testing-only-12345")
os.environ.setdefault("TRACE_ENABLED", "false")

import base64
import io
import uuid
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import fitz  # PyMuPDF
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf_bytes(text: str = "Test document for translation.") -> bytes:
    """Create a minimal, valid, non-encrypted 1-page PDF using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def make_job_doc(
    job_id: str | None = None,
    status: str = "queued",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a representative translation job document dict."""
    now = datetime.now(UTC)
    jid = job_id or str(uuid.uuid4())
    doc: dict[str, Any] = {
        "job_id": jid,
        "status": status,
        "progress": 0.0,
        "current_stage": None,
        "error_message": None,
        "source_document": {
            "gcs_uri": f"gs://test-bucket/translation/{jid}/input/doc.pdf",
            "format": "pdf",
            "page_count": 3,
            "source_language": "auto",
            "original_filename": "doc.pdf",
            "output_filename": "doc_es.pdf",
            "file_size_bytes": 4096,
            "checksum": "deadbeef",
        },
        "translation_config": {
            "source_language": "English",
            "target_language": "Spanish",
            "domain": "commercial",
        },
        "processing_options": {
            "enable_dlp": True,
            "enable_chunking": True,
            "priority": "standard",
        },
        "result": None,
        "timestamps": {"submitted_at": now, "completed_at": None},
        "config": {
            "job_id": jid,
            "lang_in": "auto",
            "lang_out": "es",
            "domain": "commercial",
        },
        "cost_attribution": {},
        "created_at": now,
        "updated_at": now,
        "expire_at": now + timedelta(hours=24),
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# Session-scoped: heavy shared objects created once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def minimal_pdf_bytes() -> bytes:
    """Return valid, non-encrypted PDF bytes (created once per session)."""
    return _make_pdf_bytes()


@pytest.fixture(scope="session")
def minimal_pdf_b64(minimal_pdf_bytes: bytes) -> str:
    """Return base64-encoded minimal PDF string."""
    return base64.b64encode(minimal_pdf_bytes).decode()


# ---------------------------------------------------------------------------
# Function-scoped: GCP mock clients (fresh per test)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_gcs_bucket():
    """A mock GCS bucket with blob support."""
    bucket = MagicMock()
    blob = MagicMock()
    blob.name = "test/path/file.pdf"
    blob.size = 4096
    blob.content_type = "application/pdf"
    blob.updated = datetime.now(UTC)
    blob.time_created = datetime.now(UTC)
    blob.md5_hash = "abc123=="
    blob.metadata = {}
    blob.exists.return_value = True
    bucket.blob.return_value = blob
    bucket.list_blobs.return_value = [blob]
    bucket.delete_blobs.return_value = None
    return bucket, blob


@pytest.fixture()
def mock_gcs_client(mock_gcs_bucket):
    """A mock GCS storage.Client, wired to a mock bucket."""
    bucket, blob = mock_gcs_bucket
    client = MagicMock()
    client.bucket.return_value = bucket
    client.project = "test-project"
    return client, bucket, blob


# ---------------------------------------------------------------------------
# Shared job document fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_job_data() -> dict[str, Any]:
    """A queued translation job document."""
    return make_job_doc(status="queued")


@pytest.fixture()
def completed_job_data() -> dict[str, Any]:
    """A completed translation job document with result populated."""
    now = datetime.now(UTC)
    jid = str(uuid.uuid4())
    return make_job_doc(
        job_id=jid,
        status="completed",
        progress=1.0,
        current_stage="completed",
        result={
            "output_gcs_uri": f"gs://test-bucket/translation/{jid}/output/doc_es.pdf",
            "confidence_score": 0.92,
            "token_count": 5000,
            "cost_usd": 1.50,
            "model_used": "gpt-4o-mini",
            "model_version": "1.0",
            "chunks": 3,
            "retry_count": 1,
            "ab_variant": "A",
            "intent": "legal_en_es",
        },
        timestamps={
            "submitted_at": now,
            "completed_at": now,
        },
    )
