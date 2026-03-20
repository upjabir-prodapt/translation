"""
Shared fixtures for the Translation API Service test suite.

Requirements (add to dev dependencies):
    pytest-asyncio>=0.24.0
    pytest-mock>=3.14.0
"""

import base64
import io
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
    """Build a representative Firestore job document dict."""
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
        "processing": {
            "model_used": None,
            "model_version": None,
            "chunks": None,
            "chunking_applied": False,
            "retry_count": 0,
            "ab_variant": None,
            "dlp_applied": True,
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
def mock_firestore_doc_ref():
    """A mock Firestore document reference supporting async operations."""
    doc_ref = AsyncMock()
    # snapshot returned by doc_ref.get()
    snapshot = AsyncMock()
    snapshot.exists = True
    snapshot.to_dict.return_value = {}
    doc_ref.get.return_value = snapshot
    return doc_ref, snapshot


@pytest.fixture()
def mock_firestore_collection(mock_firestore_doc_ref):
    """A mock Firestore collection reference."""
    doc_ref, snapshot = mock_firestore_doc_ref
    collection_ref = MagicMock()
    collection_ref.document.return_value = doc_ref
    # query chain
    mock_query = MagicMock()
    mock_query.where.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.offset.return_value = mock_query

    async def _empty_stream():
        return
        yield  # make it an async generator

    mock_query.stream.return_value = _empty_stream()
    collection_ref.order_by.return_value = mock_query
    collection_ref.where.return_value = mock_query
    return collection_ref, mock_query, doc_ref, snapshot


@pytest.fixture()
def mock_firestore_client(mock_firestore_collection):
    """Full mock AsyncClient for Firestore, wired up with collection."""
    collection_ref, mock_query, doc_ref, snapshot = mock_firestore_collection
    client = AsyncMock()
    client.collection.return_value = collection_ref
    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=False)
    client.transaction.return_value = mock_transaction
    return client, collection_ref, mock_query, doc_ref, snapshot


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


@pytest.fixture()
def mock_tasks_client():
    """A mock Cloud Tasks CloudTasksClient."""
    client = MagicMock()
    client.queue_path.return_value = (
        "projects/test-proj/locations/us-central1/queues/test-queue"
    )
    task_resp = MagicMock()
    task_resp.name = (
        "projects/test-proj/locations/us-central1/queues/test-queue/tasks/abc"
    )
    client.create_task.return_value = task_resp
    return client


# ---------------------------------------------------------------------------
# Shared job document fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_job_data() -> dict[str, Any]:
    """A queued job Firestore document."""
    return make_job_doc(status="queued")


@pytest.fixture()
def completed_job_data() -> dict[str, Any]:
    """A completed job Firestore document with result populated."""
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
        },
        timestamps={
            "submitted_at": now,
            "completed_at": now,
        },
        processing={
            "model_used": "gpt-4o-mini",
            "model_version": "1.0",
            "chunks": 3,
            "chunking_applied": True,
            "retry_count": 1,
            "ab_variant": "A",
            "dlp_applied": True,
        },
    )
