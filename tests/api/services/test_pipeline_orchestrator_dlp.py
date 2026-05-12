"""Tests for DLP wiring inside PipelineOrchestrator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from src.api.services.pipeline_orchestrator import PipelineOrchestrator
from src.api.services.temp_workspace_service import TempWorkspaceService


@pytest.mark.asyncio
async def test_pipeline_orchestrator_passes_enable_dlp_and_persists_il_tokens(
    monkeypatch, tmp_path: Path
):
    captured: dict = {}

    class FakeProcessor:
        def __init__(self, progress_tracker):
            del progress_tracker

        async def translate(self, config):
            captured["processor_config"] = config
            return {
                "attempt_index": 1,
                "model_id": "model-a",
                "human_review_required": False,
                "confidence_score": 0.99,
                "verification": {"passed": True, "final_score": 0.95},
                "attempts": [
                    {
                        "attempt_index": 1,
                        "quality": {"final_score": 0.95},
                        "working_dir": str(tmp_path),
                        "model_id": "model-a",
                    }
                ],
                "token_usage": {
                    "total_tokens": 12,
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "estimated_cost_usd": 0.03,
                },
                "dlp_provider": "google_cloud_dlp",
                "dlp_chunk_mode": "il_paragraph",
                "dlp_token_rows": [
                    {
                        "job_id": "job-123",
                        "chunk_index": 0,
                        "token": "__DLP_TOKEN_0001__",
                        "original_value": "alice@example.com",
                        "info_type": "EMAIL_ADDRESS",
                    }
                ],
            }

    monkeypatch.setattr(
        "src.api.services.pipeline_orchestrator.JobProcessor",
        FakeProcessor,
    )

    bigquery = SimpleNamespace(
        patch_translation_job=AsyncMock(),
        write_dlp_tokens=AsyncMock(),
        write_cost_attribution=AsyncMock(),
    )
    storage = SimpleNamespace(
        download_file=AsyncMock(
            side_effect=lambda _blob, path: path.write_bytes(b"%PDF")
        )
    )

    orchestrator = PipelineOrchestrator(
        bigquery=bigquery,
        storage=storage,
        temp_workspace_service=TempWorkspaceService(base_dir=tmp_path / "work"),
    )
    orchestrator.intent_router = SimpleNamespace(
        build_intent=lambda *_args, **_kwargs: "legal_en_es",
        get_model_chain=lambda *_args, **_kwargs: ["model-a"],
    )
    orchestrator.glossary_service = SimpleNamespace(
        load_domain_glossary=lambda **_kwargs: []
    )
    orchestrator.cover_page_service = SimpleNamespace(
        build=lambda **_kwargs: {"cover": True}
    )
    orchestrator.assembly_service = SimpleNamespace(
        upload_outputs=AsyncMock(
            return_value={"mono_pdf_path": "gs://bucket/output.pdf"}
        )
    )

    await orchestrator.run(
        job_id="job-123",
        job_data={
            "source_document": {
                "gcs_uri": "gs://bucket/input.pdf",
                "original_filename": "input.pdf",
            },
            "translation_config": {
                "source_language": "en",
                "target_language": "es",
                "domain": "legal",
            },
            "processing_options": {
                "enable_dlp": True,
            },
            "cost_attribution": {
                "user_id": "u1",
                "business_unit": "bu",
                "organization": "org",
            },
        },
    )

    assert captured["processor_config"]["enable_dlp"] is True
    bigquery.write_dlp_tokens.assert_awaited_once_with(
        [
            {
                "job_id": "job-123",
                "chunk_index": 0,
                "token": "__DLP_TOKEN_0001__",
                "original_value": "alice@example.com",
                "info_type": "EMAIL_ADDRESS",
            }
        ]
    )
