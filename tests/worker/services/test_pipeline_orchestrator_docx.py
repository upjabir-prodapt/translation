"""Pipeline behaviour for Word sources: no PDF conversion, .docx deliverable."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator


@pytest.fixture
def mock_bq():
    return AsyncMock()


@pytest.fixture
def mock_storage():
    return AsyncMock()


@pytest.fixture
def orchestrator(mock_bq, mock_storage):
    return PipelineOrchestrator(bigquery=mock_bq, storage=mock_storage)


def docx_job_data(**overrides):
    job_data = {
        "source_document": {
            "gcs_uri": "gs://b/contract.docx",
            "format": "docx",
            "original_filename": "contract.docx",
            "output_filename": "contract.docx",
        },
        "translation_config": {
            "source_language": "auto",
            "target_language": "Spanish",
            "domain": "legal",
        },
        "cost_attribution": {"user_id": "u1"},
        "processing_options": {"enable_dlp": False},
    }
    job_data.update(overrides)
    return job_data


ATTEMPT_RESULT = {
    "output_path": "workspace/attempts/iter_1/contract.docx",
    "attempt_index": 1,
    "model_id": "gemini-2.5-flash",
    "quality_report": {"final_score": 0.92, "pass_fail": True},
    "token_usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 400,
        "cache_hit_prompt_tokens": 0,
    },
    "chunks_processed": 2,
    "batch_token_counts": [600, 400],
    "paragraph_count": 12,
    "dlp_token_rows": [],
    "dlp_provider": None,
    "dlp_chunk_mode": None,
}


async def run_docx_pipeline(orchestrator, job_data, attempt_result=None):
    """Drive the pipeline with the DOCX processor and external calls stubbed."""
    processor = MagicMock()
    processor.translate = AsyncMock(return_value=attempt_result or dict(ATTEMPT_RESULT))

    with (
        patch(
            "src.worker.services.pipeline_orchestrator.DocxJobProcessor",
            return_value=processor,
        ) as docx_cls,
        patch("src.worker.services.pipeline_orchestrator.JobProcessor") as pdf_cls,
        patch.object(
            orchestrator.language_detector, "detect_docx", return_value="English"
        ) as detect_docx,
        patch.object(orchestrator.language_detector, "detect") as detect_pdf,
        patch.object(
            orchestrator.intent_router,
            "get_model_chain",
            return_value=["gemini-2.5-flash"],
        ),
        patch.object(
            orchestrator.intent_router, "build_intent", return_value="intent1"
        ),
        patch.object(
            orchestrator.glossary_service, "load_domain_glossary", return_value=[]
        ),
        patch.object(
            orchestrator.assembly_service,
            "upload_output",
            new=AsyncMock(return_value="gs://b/out/contract.docx"),
        ) as upload,
    ):
        await orchestrator.run("job1", job_data)

    return {
        "processor": processor,
        "docx_cls": docx_cls,
        "pdf_cls": pdf_cls,
        "detect_docx": detect_docx,
        "detect_pdf": detect_pdf,
        "upload": upload,
    }


class TestDocxPipeline:
    async def test_uses_docx_processor_not_pdf_processor(self, orchestrator):
        calls = await run_docx_pipeline(orchestrator, docx_job_data())
        calls["docx_cls"].assert_called_once()
        calls["pdf_cls"].assert_not_called()

    async def test_no_pdf_conversion_step_exists(self, orchestrator):
        # The LibreOffice converter is gone; importing it should fail.
        with pytest.raises(ImportError):
            from src.worker.utils.docx_converter import (  # noqa: F401
                convert_docx_to_pdf,
            )

    async def test_input_downloaded_as_docx(self, orchestrator, mock_storage):
        await run_docx_pipeline(orchestrator, docx_job_data())
        blob_path, local_path = mock_storage.download_file.call_args[0]
        assert blob_path.endswith("contract.docx")
        assert local_path.name == "contract.docx"

    async def test_language_detected_from_docx(self, orchestrator):
        calls = await run_docx_pipeline(orchestrator, docx_job_data())
        calls["detect_docx"].assert_called_once()
        calls["detect_pdf"].assert_not_called()

    async def test_declared_source_language_skips_detection(self, orchestrator):
        job_data = docx_job_data()
        job_data["translation_config"]["source_language"] = "English"
        calls = await run_docx_pipeline(orchestrator, job_data)
        calls["detect_docx"].assert_not_called()

    async def test_output_uploaded_with_docx_filename(self, orchestrator):
        calls = await run_docx_pipeline(orchestrator, docx_job_data())
        kwargs = calls["upload"].call_args.kwargs
        assert kwargs["preferred_filename"] == "contract.docx"
        assert kwargs["local_path"].name == "contract.docx"

    async def test_cover_page_disabled_for_docx(self, orchestrator):
        calls = await run_docx_pipeline(orchestrator, docx_job_data())
        config = calls["processor"].translate.call_args[0][0]
        assert config["add_cover_page"] is False
        assert config["input_file"].endswith("contract.docx")

    async def test_job_completes_with_docx_output_uri(self, orchestrator, mock_bq):
        await run_docx_pipeline(orchestrator, docx_job_data())
        patches = [call[0][1] for call in mock_bq.patch_translation_job.call_args_list]
        completed = [p for p in patches if p.get("status") == "completed"]
        assert completed, f"job did not complete: {patches}"
        result = completed[-1]["result"]
        assert result["output_gcs_uri"] == "gs://b/out/contract.docx"
        assert result["model_used"] == "gemini-2.5-flash"
        assert result["confidence_score"] == 0.92

    async def test_cost_attributed_across_translation_batches(
        self, orchestrator, mock_bq
    ):
        await run_docx_pipeline(orchestrator, docx_job_data())
        row = mock_bq.write_cost_attribution.call_args[0][0]
        assert row["input_tokens"] == 1000
        assert row["output_tokens"] == 400
        assert row["cost_usd"] > 0

    async def test_missing_batch_counts_fails_the_job(self, orchestrator, mock_bq):
        attempt = dict(ATTEMPT_RESULT)
        attempt["batch_token_counts"] = []
        await run_docx_pipeline(orchestrator, docx_job_data(), attempt_result=attempt)
        patches = [call[0][1] for call in mock_bq.patch_translation_job.call_args_list]
        failed = [p for p in patches if p.get("status") == "failed"]
        assert failed
        assert "cannot attribute cost" in failed[-1]["error_message"]
