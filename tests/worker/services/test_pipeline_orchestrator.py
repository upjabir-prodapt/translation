from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator
from src.worker.services.pipeline_orchestrator import _PipelineProgressTracker


@pytest.fixture
def mock_bq():
    return AsyncMock()


@pytest.fixture
def mock_storage():
    return AsyncMock()


@pytest.fixture
def orchestrator(mock_bq, mock_storage):
    return PipelineOrchestrator(bigquery=mock_bq, storage=mock_storage)


class TestPipelineProgressTracker:
    async def test_update(self):
        tracker = _PipelineProgressTracker(job_id="job1")
        assert await tracker.update(0.5, "Translating") is True


class TestPipelineOrchestrator:
    def test_extract_blob_path(self, orchestrator):
        with patch(
            "src.worker.services.pipeline_orchestrator.settings"
        ) as mock_settings:
            mock_settings.GCS_BUCKET_NAME = "test-bucket"
            assert (
                orchestrator._extract_blob_path("gs://test-bucket/path/to/blob")
                == "path/to/blob"
            )
            assert orchestrator._extract_blob_path("path/to/blob") == "path/to/blob"

    def test_convert_txt_to_docx_produces_docx_sibling_file(self, orchestrator, tmp_path):
        txt_path = tmp_path / "input.txt"
        txt_path.write_text("Hello world.\nSecond line.", encoding="utf-8")

        docx_path = orchestrator._convert_txt_to_docx(txt_path)

        assert docx_path.suffix == ".docx"
        assert docx_path.exists()

        from docx import Document as open_docx

        document = open_docx(str(docx_path))
        paragraphs = [p.text for p in document.paragraphs]
        assert paragraphs == ["Hello world.", "Second line."]

    async def test_update_status(self, orchestrator, mock_bq):
        await orchestrator._update_status(
            "job1", status="completed", error_message="none"
        )
        mock_bq.patch_translation_job.assert_called_once()
        args, kwargs = mock_bq.patch_translation_job.call_args
        assert args[0] == "job1"
        assert args[1]["status"] == "completed"
        assert args[1]["error_message"] == "none"

    @patch("src.worker.services.pipeline_orchestrator.JobProcessor")
    async def test_run_success(
        self, mock_processor_cls, orchestrator, mock_bq, mock_storage
    ):
        mock_processor = AsyncMock()
        mock_processor.translate.return_value = {"status": "ok", "page_count": 1}
        mock_processor_cls.return_value = mock_processor

        job_data = {
            "source_document": {"gcs_uri": "gs://b/doc.pdf"},
            "translation_config": {"target_language": "fr", "domain": "legal"},
            "cost_attribution": {"user_id": "u1"},
        }

        # We need to bypass or mock lots of internal calls
        with patch.object(orchestrator.temp_workspace_service, "create"):
            with patch.object(
                orchestrator.language_detector, "detect", return_value="en"
            ):
                with patch.object(
                    orchestrator.intent_router, "get_model_chain", return_value=["m1"]
                ):
                    with patch.object(
                        orchestrator.intent_router,
                        "build_intent",
                        return_value="intent1",
                    ):
                        with patch.object(
                            orchestrator.glossary_service,
                            "load_domain_glossary",
                            return_value=[],
                        ):
                            with patch.object(orchestrator, "_update_status"):
                                await orchestrator.run("job1", job_data)
                                orchestrator._update_status.assert_called()
