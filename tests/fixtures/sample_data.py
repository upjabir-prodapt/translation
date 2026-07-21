"""
Static sample data shared across multiple test modules.

Import these constants instead of duplicating inline payloads.
"""

import base64
import io
from datetime import UTC
from datetime import datetime
from typing import Any

import fitz

# ---------------------------------------------------------------------------
# Minimal valid PDF bytes (created once at import time)
# ---------------------------------------------------------------------------


def _make_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Hello world. This is a sample PDF for testing.")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


VALID_PDF_BYTES: bytes = _make_pdf()
VALID_PDF_B64: str = base64.b64encode(VALID_PDF_BYTES).decode()

# ---------------------------------------------------------------------------
# Sample translate request payloads
# ---------------------------------------------------------------------------

TRANSLATE_REQUEST_VALID: dict[str, Any] = {
    "document": {
        "content": VALID_PDF_B64,
        "format": "pdf",
        "filename": "sample.pdf",
    },
    "translation_config": {
        "source_language": "English",
        "target_language": "Spanish",
        "domain": "commercial",
    },
    "cost_attribution": {
        "user_id": "user-12345",
        "business_unit": "Legal-EMEA",
        "organization": "Colt-Group",
    },
    "processing_options": {
        "enable_dlp": True,
        "enable_chunking": True,
        "priority": "standard",
    },
}

TRANSLATE_REQUEST_NO_OPTIONS: dict[str, Any] = {
    "document": {
        "content": VALID_PDF_B64,
        "format": "pdf",
        "filename": "sample.pdf",
    },
    "translation_config": {
        "source_language": "auto",
        "target_language": "fr",
        "domain": "legal",
    },
    "cost_attribution": {
        "user_id": "user-12345",
        "business_unit": "Legal-EMEA",
        "organization": "Colt-Group",
    },
}

# ---------------------------------------------------------------------------
# Sample model_selection.json entry (for translation_routing tests)
# ---------------------------------------------------------------------------

MODEL_SELECTION_ENTRY: dict[str, Any] = {
    "source_language": "english",
    "target_language": "spanish",
    "domain": "commercial",
    "model_chain": [
        {"model_id": "gpt-4o-mini", "priority": 1},
        {"model_id": "gpt-4o", "priority": 2},
    ],
}

MODEL_SELECTION_LIST: list[dict[str, Any]] = [MODEL_SELECTION_ENTRY]

# ---------------------------------------------------------------------------
# Sample translate_tracking.json for quality judge tests
# ---------------------------------------------------------------------------

TRACKING_JSON_VALID: dict[str, Any] = {
    "page": [
        {
            "paragraph": [
                {
                    "input": "Hello world. This is a test.",
                    "output": "Hola mundo. Esto es una prueba.",
                },
                {
                    "input": "Another sentence for testing.",
                    "output": "Otra oración para probar.",
                },
            ]
        },
        {
            "paragraph": [
                {
                    "pdf_unicode": "Page two content here.",
                    "output": "Contenido de la página dos aquí.",
                }
            ]
        },
    ]
}

TRACKING_JSON_EMPTY: dict[str, Any] = {"page": []}

# ---------------------------------------------------------------------------
# Sample BigQuery job analytics row
# ---------------------------------------------------------------------------

BIGQUERY_JOB_DATA: dict[str, Any] = {
    "job_id": "test-job-001",
    "status": "completed",
    "domain": "commercial",
    "lang_in": "en",
    "lang_out": "es",
    "user": "user@example.com",
    "department": "engineering",
    "file_size_bytes": 204800,
    "processing_seconds": 45,
    "pages_processed": 10,
    "created_at": datetime.now(UTC),
    "completed_at": datetime.now(UTC),
    "output_gs_uris": {"mono": "gs://bucket/output.pdf"},
    "quality_report": {"final_score": 0.92},
    "token_usage": 5000,
    "total_cost_usd": 1.50,
    "iteration_details": [],
    "selected_model": "gpt-4o-mini",
    "attempt_count": 1,
}
