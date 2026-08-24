"""
Shared fixtures for the Translation API Service test suite.

Requirements (add to dev dependencies):
    pytest-asyncio>=0.24.0
    pytest-mock>=3.14.0
"""

import os
from pathlib import Path

# Load tests/test.env before any module imports Settings (constants.py loads at import time).
os.environ.setdefault("DOTENV_PATH", str(Path(__file__).resolve().parent / "test.env"))

import base64
import io
import json
import uuid
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import fitz  # PyMuPDF
import pytest

# Seed a minimal pricing_catalog.json into the test asset cache root so that
# real (non-mocked) construction of VertexLLMCostService/DocxJobProcessor/
# TranslationAttemptRunner in unit tests doesn't hit the FileNotFoundError
# that build_rate_catalog_from_settings() now raises when the file is
# missing. pricing_catalog.json is the sole source of LLM pricing (see
# src/config/llm_rate_catalog.py) -- there is no env-var fallback.
from src.config.constants import settings as _settings  # noqa: E402

_test_pricing_catalog_path = _settings.assets_root_path / _settings.PRICING_CATALOG_FILENAME
if not _test_pricing_catalog_path.is_file():
    _test_pricing_catalog_path.write_text(
        json.dumps(
            [
                {
                    "provider": "gemini_vertexai",
                    "model_id": "gemini-2.5-pro",
                    "region": None,
                    "tiers": [
                        {
                            "max_input_tokens": 200000,
                            "input_cost_per_1k": 0.00125,
                            "output_cost_per_1k": 0.01,
                            "cache_hit_cost_per_1k": 0.00013,
                        },
                        {
                            "max_input_tokens": None,
                            "input_cost_per_1k": 0.0025,
                            "output_cost_per_1k": 0.015,
                            "cache_hit_cost_per_1k": 0.00025,
                        },
                    ],
                },
                {
                    "provider": "gemini_vertexai",
                    "model_id": "gemini-2.5-flash",
                    "region": None,
                    "tiers": [
                        {
                            "max_input_tokens": None,
                            "input_cost_per_1k": 0.0003,
                            "output_cost_per_1k": 0.0025,
                            "cache_hit_cost_per_1k": 0.00003,
                        }
                    ],
                },
                {
                    "provider": "gemini_vertexai",
                    "model_id": "gemini-2.5-flash-lite",
                    "region": None,
                    "tiers": [
                        {
                            "max_input_tokens": None,
                            "input_cost_per_1k": 0.0001,
                            "output_cost_per_1k": 0.0004,
                            "cache_hit_cost_per_1k": 0.00001,
                        }
                    ],
                },
                {
                    "provider": "gemini_vertexai",
                    "model_id": "gemini",
                    "region": None,
                    "tiers": [
                        {
                            "max_input_tokens": None,
                            "input_cost_per_1k": 0.0003,
                            "output_cost_per_1k": 0.0025,
                        }
                    ],
                },
                {
                    "provider": "claude",
                    "model_id": "claude-sonnet-4-6",
                    "region": "europe-west1",
                    "tiers": [
                        {
                            "max_input_tokens": None,
                            "input_cost_per_1k": 0.0033,
                            "output_cost_per_1k": 0.0165,
                            "cache_hit_cost_per_1k": 0.00033,
                            "cache_write_5m_cost_per_1k": 0.00413,
                            "cache_write_1h_cost_per_1k": 0.0066,
                        }
                    ],
                },
                {
                    "provider": "claude",
                    "model_id": "claude",
                    "region": None,
                    "tiers": [
                        {
                            "max_input_tokens": None,
                            "input_cost_per_1k": 0.0033,
                            "output_cost_per_1k": 0.0165,
                            "cache_hit_cost_per_1k": 0.00033,
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

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
