"""Pipeline BigQuery fail-fast behaviour tests."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.repository.repository_exception import BigQueryError
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator
from src.worker.services.temp_workspace_service import TempWorkspaceService


def _job_data(job_id: str = "test-job-123") -> dict:
    return {
        "source_document": {
            "gcs_uri": f"gs://bucket/input/{job_id}/file.pdf",
            "original_filename": "input.pdf",
        },
        "translation_config": {
            "source_language": "en",
            "target_language": "es",
            "domain": "general",
        },
        "cost_attribution": {"user_id": "user-1"},
        "processing_options": {"enable_dlp": False},
    }


def _attempt_result(tmp_path: Path) -> dict:
    out_pdf = tmp_path / "mono.pdf"
    out_pdf.write_bytes(b"%PDF-1.4\n")
    return {
        "attempt_index": 1,
        "model_id": "gemini-2.5-flash",
        "mono_pdf_path": str(out_pdf),
        "token_usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 400,
            "estimated_cost_usd": 0.05,
        },
        "quality_report": {"final_score": 0.9, "pass_fail": True},
        "chunks_processed": 3,
        "dlp_token_rows": [],
    }


class FakeProcessor:
    def __init__(self, progress_tracker, result: dict):
        del progress_tracker
        self._result = result

    async def translate(self, config):
        del config
        return self._result


@contextmanager
def _patched_orchestrator(bigquery, storage, tmp_path: Path, attempt_result: dict):
    orchestrator = PipelineOrchestrator(
        bigquery=bigquery,
        storage=storage,
        temp_workspace_service=TempWorkspaceService(base_dir=tmp_path / "work"),
    )
    orchestrator.intent_router = MagicMock()
    orchestrator.intent_router.build_intent.return_value = "general_en_es"
    orchestrator.intent_router.get_model_chain.return_value = ["gemini-2.5-flash"]
    orchestrator.glossary_service = MagicMock()
    orchestrator.glossary_service.load_domain_glossary.return_value = []
    orchestrator.assembly_service = MagicMock()
    orchestrator.assembly_service.upload_output = AsyncMock(
        return_value="gs://bucket/output.pdf"
    )
    with (
        patch(
            "src.worker.services.pipeline_orchestrator.JobProcessor",
            lambda progress_tracker: FakeProcessor(progress_tracker, attempt_result),
        ),
        patch.object(
            orchestrator,
            "_compute_accumulated_chunk_costs",
            new=AsyncMock(
                return_value={
                    "input_tokens": 1000,
                    "output_tokens": 400,
                    "cost_usd": 0.05,
                    "chunk_count": 3,
                }
            ),
        ),
    ):
        yield orchestrator


@pytest.fixture
def pipeline_mocks(tmp_path: Path):
    bigquery = AsyncMock()
    bigquery.patch_translation_job = AsyncMock()
    storage = AsyncMock()
    storage.download_file = AsyncMock(
        side_effect=lambda _blob, path: path.write_bytes(b"%PDF-1.4\n")
    )
    return bigquery, storage, tmp_path


class TestMarkJobFailedErrorMessageMapping:
    """B.3.2: `_mark_job_failed`'s BigQuery-persisted, user-facing
    `error_message` maps ScannedPDFError to the same wording as
    PDFValidator's API-side rejection, instead of leaking the internal
    'Scanned PDF detected.' string."""

    @pytest.mark.asyncio
    async def test_scanned_pdf_error_maps_to_user_facing_message(self, pipeline_mocks):
        from src.worker.doctranslator.doctranslator_exception.DocTranslatorException import (
            ScannedPDFError,
        )

        bigquery, storage, tmp_path = pipeline_mocks
        orchestrator = PipelineOrchestrator(
            bigquery=bigquery,
            storage=storage,
            temp_workspace_service=TempWorkspaceService(base_dir=tmp_path / "work"),
        )
        workspace = orchestrator.temp_workspace_service.create("job-scanned")
        await orchestrator.session_manager.start("job-scanned", workspace)
        pipeline_span = MagicMock()
        await orchestrator._mark_job_failed(
            "job-scanned",
            ScannedPDFError("Scanned PDF detected."),
            pipeline_span,
            stage="translate",
        )

        failed_calls = [
            c
            for c in bigquery.patch_translation_job.call_args_list
            if c[0][1].get("status") == "failed"
        ]
        assert len(failed_calls) == 1
        persisted_message = failed_calls[0][0][1]["error_message"]
        assert "no extractable text layer" in persisted_message
        assert "Scanned PDF detected" not in persisted_message

    @pytest.mark.asyncio
    async def test_other_exceptions_use_str_unchanged(self, pipeline_mocks):
        """Regression: non-document-shape exceptions keep their original
        `str(exc)` wording -- only ScannedPDFError is remapped."""
        bigquery, storage, tmp_path = pipeline_mocks
        orchestrator = PipelineOrchestrator(
            bigquery=bigquery,
            storage=storage,
            temp_workspace_service=TempWorkspaceService(base_dir=tmp_path / "work"),
        )
        workspace = orchestrator.temp_workspace_service.create("job-other")
        await orchestrator.session_manager.start("job-other", workspace)
        pipeline_span = MagicMock()
        await orchestrator._mark_job_failed(
            "job-other",
            RuntimeError("some transient LLM error"),
            pipeline_span,
            stage="translate",
        )

        failed_calls = [
            c
            for c in bigquery.patch_translation_job.call_args_list
            if c[0][1].get("status") == "failed"
        ]
        assert failed_calls[0][0][1]["error_message"] == "some transient LLM error"


class TestPipelineBigQueryFailure:
    @pytest.mark.asyncio
    async def test_write_cost_attribution_failure_marks_job_failed(
        self, pipeline_mocks
    ):
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)
        bq_error = BigQueryError(
            "BigQuery insert errors",
            dataset="test_dataset",
            table="translation_costs",
        )
        bigquery.write_cost_attribution.side_effect = bq_error

        with _patched_orchestrator(
            bigquery, storage, tmp_path, attempt_result
        ) as orchestrator:
            pipeline_span = MagicMock()
            await orchestrator._execute_pipeline(
                "job-cost-fail", _job_data("job-cost-fail"), pipeline_span
            )

        patch_calls = bigquery.patch_translation_job.call_args_list
        failed_calls = [c for c in patch_calls if c[0][1].get("status") == "failed"]
        completed_calls = [
            c for c in patch_calls if c[0][1].get("status") == "completed"
        ]
        assert len(failed_calls) >= 1
        assert len(completed_calls) == 0
        assert "BigQuery insert errors" in failed_calls[0][0][1]["error_message"]

    @pytest.mark.asyncio
    async def test_write_dlp_tokens_failure_marks_job_failed(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)
        attempt_result["dlp_token_rows"] = [
            {
                "job_id": "job-dlp-fail",
                "chunk_index": 0,
                "token": "tok",
                "original_value": "secret",
            }
        ]
        bigquery.write_dlp_tokens.side_effect = BigQueryError(
            "DLP insert failed",
            dataset="test_dataset",
            table="dlp_mappings",
        )

        with _patched_orchestrator(
            bigquery, storage, tmp_path, attempt_result
        ) as orchestrator:
            pipeline_span = MagicMock()
            await orchestrator._execute_pipeline(
                "job-dlp-fail", _job_data("job-dlp-fail"), pipeline_span
            )

        bigquery.write_cost_attribution.assert_not_awaited()
        failed_calls = [
            c
            for c in bigquery.patch_translation_job.call_args_list
            if c[0][1].get("status") == "failed"
        ]
        assert len(failed_calls) >= 1

    @pytest.mark.asyncio
    async def test_chunk_cost_failure_skips_cost_attribution(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)

        orchestrator = PipelineOrchestrator(
            bigquery=bigquery,
            storage=storage,
            temp_workspace_service=TempWorkspaceService(base_dir=tmp_path / "work"),
        )
        orchestrator.intent_router = MagicMock()
        orchestrator.intent_router.build_intent.return_value = "general_en_es"
        orchestrator.intent_router.get_model_chain.return_value = ["gemini-2.5-flash"]
        orchestrator.glossary_service = MagicMock()
        orchestrator.glossary_service.load_domain_glossary.return_value = []
        orchestrator.assembly_service = MagicMock()
        orchestrator.assembly_service.upload_output = AsyncMock(
            return_value="gs://bucket/output.pdf"
        )

        with (
            patch(
                "src.worker.services.pipeline_orchestrator.JobProcessor",
                lambda progress_tracker: FakeProcessor(
                    progress_tracker, attempt_result
                ),
            ),
            patch.object(
                orchestrator,
                "_compute_accumulated_chunk_costs",
                new=AsyncMock(side_effect=RuntimeError("chunk cost failed")),
            ),
        ):
            pipeline_span = MagicMock()
            await orchestrator._execute_pipeline(
                "job-chunk-fail", _job_data("job-chunk-fail"), pipeline_span
            )

        bigquery.write_cost_attribution.assert_not_awaited()
        failed_calls = [
            c
            for c in bigquery.patch_translation_job.call_args_list
            if c[0][1].get("status") == "failed"
        ]
        assert len(failed_calls) >= 1
        assert "chunk cost failed" in failed_calls[0][0][1]["error_message"]

    @pytest.mark.asyncio
    async def test_failed_status_patch_reraises_when_bq_down_twice(
        self, pipeline_mocks
    ):
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)
        bq_error = BigQueryError("cost insert failed", table="translation_costs")
        bigquery.write_cost_attribution.side_effect = bq_error
        bigquery.patch_translation_job.side_effect = [
            None,
            None,
            bq_error,
        ]

        with _patched_orchestrator(
            bigquery, storage, tmp_path, attempt_result
        ) as orchestrator:
            pipeline_span = MagicMock()
            with pytest.raises(BigQueryError, match="cost insert failed"):
                await orchestrator._execute_pipeline(
                    "job-double-fail", _job_data("job-double-fail"), pipeline_span
                )

    @pytest.mark.asyncio
    async def test_routing_metadata_persisted_immediately(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)

        with _patched_orchestrator(
            bigquery, storage, tmp_path, attempt_result
        ) as orchestrator:
            pipeline_span = MagicMock()
            await orchestrator._execute_pipeline(
                "job-routing-check", _job_data("job-routing-check"), pipeline_span
            )

        patch_calls = bigquery.patch_translation_job.call_args_list
        routing_patches = [
            c
            for c in patch_calls
            if c[0][1].get("result") == {"intent": "general_en_es"}
        ]
        assert len(routing_patches) >= 1
        cfg = routing_patches[0][0][1]["translation_config"]
        assert cfg["source_language"] == "en"
        assert cfg["target_language"] == "es"

    @pytest.mark.asyncio
    async def test_session_discarded_after_failure(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)
        bigquery.write_cost_attribution.side_effect = BigQueryError(
            "insert failed", table="translation_costs"
        )

        with _patched_orchestrator(
            bigquery, storage, tmp_path, attempt_result
        ) as orchestrator:
            session_manager = orchestrator.session_manager
            pipeline_span = MagicMock()
            await orchestrator._execute_pipeline(
                "job-session-fail", _job_data("job-session-fail"), pipeline_span
            )
            assert await session_manager.active_count() == 0
