from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fixtures.docx_builder import build_docx
from fixtures.docx_builder import paragraph
from fixtures.docx_builder import run
from fixtures.docx_builder import simple_docx
from src.worker.services.docx_processor_service import DLP_CHUNK_MODE
from src.worker.services.docx_processor_service import DocxJobProcessor
from src.worker.services.docx_processor_service import extract_docx_text
from src.worker.services.docx_translation_service import DocxTranslationResult
from src.worker.services.quality_judge_service import QualityJudgeResult


@pytest.fixture
def tracker():
    tracker = AsyncMock()
    tracker.update = AsyncMock(return_value=True)
    return tracker


def quality(score: float, passed: bool) -> QualityJudgeResult:
    return QualityJudgeResult(
        alignment_score=score,
        omission_score=score,
        hallucination_score=score,
        final_score=score,
        pass_fail=passed,
        reasons=[],
        model="judge-model",
    )


def make_result(output_path: Path, **overrides) -> DocxTranslationResult:
    defaults = {
        "output_path": output_path,
        "source_text": "source text",
        "translated_text": "translated text",
        "segment_count": 4,
        "translated_segment_count": 4,
        "batch_token_counts": [120, 80],
    }
    defaults.update(overrides)
    return DocxTranslationResult(**defaults)


def stub_translation_service(results):
    """Return a service whose translate_document yields ``results`` in order."""
    service = MagicMock()
    calls = list(results)

    async def translate_document(**kwargs):
        result = calls.pop(0)
        if isinstance(result, Exception):
            raise result
        return make_result(Path(kwargs["output_path"]), **result)

    service.translate_document = translate_document
    return service


class TestExtractDocxText:
    def test_reads_paragraph_text(self, tmp_path):
        path = tmp_path / "doc.docx"
        path.write_bytes(build_docx(paragraph(run("Hello")) + paragraph(run("World"))))
        assert extract_docx_text(path) == "Hello\nWorld"

    def test_respects_max_chars(self, tmp_path):
        path = tmp_path / "doc.docx"
        path.write_bytes(build_docx(paragraph(run("aaaa")) + paragraph(run("bbbb"))))
        assert extract_docx_text(path, max_chars=2) == "aaaa"


class TestDocxJobProcessorTranslate:
    def base_config(self, tmp_path, **overrides):
        source = tmp_path / "in.docx"
        source.write_bytes(simple_docx())
        config = {
            "job_id": "job1",
            "input_file": str(source),
            "output_dir": str(tmp_path / "attempts"),
            "lang_in": "English",
            "lang_out": "Spanish",
            "domain": "legal",
            "model_list": ["gemini-2.5-flash"],
            "max_model_attempts": 2,
            "glossaries": [],
            "enable_dlp": False,
        }
        config.update(overrides)
        return config

    async def test_requires_model_list(self, tracker, tmp_path):
        processor = DocxJobProcessor(progress_tracker=tracker)
        with pytest.raises(ValueError, match="model_list is required"):
            await processor.translate(self.base_config(tmp_path, model_list=[]))

    @patch("src.worker.services.docx_processor_service.create_translator")
    @patch("src.worker.services.docx_processor_service.GoogleADKJudgeAgent")
    async def test_returns_attempt_result_shape(
        self, mock_judge_cls, mock_create, tracker, tmp_path
    ):
        mock_judge_cls.return_value.evaluate_async = AsyncMock(
            return_value=quality(0.9, True)
        )
        mock_create.return_value = MagicMock(
            prompt_token_count=1000,
            completion_token_count=500,
            token_count=1500,
            cache_hit_prompt_token_count=0,
        )
        processor = DocxJobProcessor(
            progress_tracker=tracker,
            translation_service=stub_translation_service([{}]),
        )

        result = await processor.translate(self.base_config(tmp_path))

        assert result["attempt_index"] == 1
        assert result["model_id"] == "gemini-2.5-flash"
        assert result["output_path"].endswith("in.docx")
        assert result["quality_report"]["final_score"] == 0.9
        assert result["chunks_processed"] == 2
        assert result["batch_token_counts"] == [120, 80]
        assert result["paragraph_count"] == 4
        assert result["token_usage"]["prompt_tokens"] == 1000
        assert result["token_usage"]["completion_tokens"] == 500
        assert "quality_passed" not in result

    @patch("src.worker.services.docx_processor_service.create_translator")
    @patch("src.worker.services.docx_processor_service.GoogleADKJudgeAgent")
    async def test_output_keeps_docx_extension(
        self, mock_judge_cls, mock_create, tracker, tmp_path
    ):
        mock_judge_cls.return_value.evaluate_async = AsyncMock(
            return_value=quality(0.9, True)
        )
        mock_create.return_value = MagicMock()
        processor = DocxJobProcessor(
            progress_tracker=tracker,
            translation_service=stub_translation_service([{}]),
        )
        result = await processor.translate(self.base_config(tmp_path))
        assert Path(result["output_path"]).suffix == ".docx"

    @patch("src.worker.services.docx_processor_service.create_translator")
    @patch("src.worker.services.docx_processor_service.GoogleADKJudgeAgent")
    async def test_stops_on_first_passing_attempt(
        self, mock_judge_cls, mock_create, tracker, tmp_path
    ):
        mock_judge_cls.return_value.evaluate_async = AsyncMock(
            return_value=quality(0.95, True)
        )
        mock_create.return_value = MagicMock()
        processor = DocxJobProcessor(
            progress_tracker=tracker,
            translation_service=stub_translation_service([{}, {}]),
        )
        config = self.base_config(
            tmp_path,
            model_list=["gemini-2.5-flash", "claude-sonnet-4"],
            max_model_attempts=2,
        )
        result = await processor.translate(config)
        assert result["model_id"] == "gemini-2.5-flash"
        assert mock_create.call_count == 1

    @patch("src.worker.services.docx_processor_service.create_translator")
    @patch("src.worker.services.docx_processor_service.GoogleADKJudgeAgent")
    async def test_keeps_highest_scoring_attempt(
        self, mock_judge_cls, mock_create, tracker, tmp_path
    ):
        mock_judge_cls.return_value.evaluate_async = AsyncMock(
            side_effect=[quality(0.4, False), quality(0.8, False)]
        )
        mock_create.return_value = MagicMock()
        processor = DocxJobProcessor(
            progress_tracker=tracker,
            translation_service=stub_translation_service([{}, {}]),
        )
        config = self.base_config(
            tmp_path,
            model_list=["gemini-2.5-flash", "claude-sonnet-4"],
            max_model_attempts=2,
        )
        result = await processor.translate(config)
        assert result["model_id"] == "claude-sonnet-4"
        assert result["quality_report"]["final_score"] == 0.8

    @patch("src.worker.services.docx_processor_service.create_translator")
    @patch("src.worker.services.docx_processor_service.GoogleADKJudgeAgent")
    async def test_failed_attempt_falls_through_to_next_model(
        self, mock_judge_cls, mock_create, tracker, tmp_path
    ):
        mock_judge_cls.return_value.evaluate_async = AsyncMock(
            return_value=quality(0.7, True)
        )
        mock_create.return_value = MagicMock()
        processor = DocxJobProcessor(
            progress_tracker=tracker,
            translation_service=stub_translation_service(
                [RuntimeError("gemini attempt exploded"), {}]
            ),
        )
        config = self.base_config(
            tmp_path,
            model_list=["gemini-2.5-flash", "claude-sonnet-4"],
            max_model_attempts=2,
        )
        result = await processor.translate(config)
        assert result["model_id"] == "claude-sonnet-4"

    @patch("src.worker.services.docx_processor_service.create_translator")
    @patch("src.worker.services.docx_processor_service.GoogleADKJudgeAgent")
    async def test_last_attempt_failure_propagates(
        self, mock_judge_cls, mock_create, tracker, tmp_path
    ):
        mock_judge_cls.return_value.evaluate_async = AsyncMock(
            return_value=quality(0.7, True)
        )
        mock_create.return_value = MagicMock()
        processor = DocxJobProcessor(
            progress_tracker=tracker,
            translation_service=stub_translation_service(
                [RuntimeError("single model exploded")]
            ),
        )
        with pytest.raises(RuntimeError, match="single model exploded"):
            await processor.translate(self.base_config(tmp_path))

    @patch("src.worker.services.docx_processor_service.create_translator")
    @patch("src.worker.services.docx_processor_service.GoogleADKJudgeAgent")
    async def test_empty_translation_scores_zero_without_calling_judge(
        self, mock_judge_cls, mock_create, tracker, tmp_path
    ):
        judge = mock_judge_cls.return_value
        judge.model = "judge-model"
        judge.evaluate_async = AsyncMock()
        mock_create.return_value = MagicMock()
        processor = DocxJobProcessor(
            progress_tracker=tracker,
            translation_service=stub_translation_service([{"translated_text": "  "}]),
        )
        result = await processor.translate(self.base_config(tmp_path))
        assert result["quality_report"]["final_score"] == 0.0
        judge.evaluate_async.assert_not_called()

    @patch("src.worker.services.docx_processor_service.create_translator")
    @patch("src.worker.services.docx_processor_service.GoogleADKJudgeAgent")
    async def test_dlp_metadata_reported_when_tokens_present(
        self, mock_judge_cls, mock_create, tracker, tmp_path
    ):
        mock_judge_cls.return_value.evaluate_async = AsyncMock(
            return_value=quality(0.9, True)
        )
        mock_create.return_value = MagicMock()
        rows = [{"token": "__DLP_TOKEN_0001__", "original_value": "Jane"}]
        processor = DocxJobProcessor(
            progress_tracker=tracker,
            translation_service=stub_translation_service([{"dlp_token_rows": rows}]),
        )
        result = await processor.translate(self.base_config(tmp_path))
        assert result["dlp_token_rows"] == rows
        assert result["dlp_chunk_mode"] == DLP_CHUNK_MODE

    @patch("src.worker.services.docx_processor_service.create_translator")
    @patch("src.worker.services.docx_processor_service.GoogleADKJudgeAgent")
    async def test_no_dlp_chunk_mode_without_tokens(
        self, mock_judge_cls, mock_create, tracker, tmp_path
    ):
        mock_judge_cls.return_value.evaluate_async = AsyncMock(
            return_value=quality(0.9, True)
        )
        mock_create.return_value = MagicMock()
        processor = DocxJobProcessor(
            progress_tracker=tracker,
            translation_service=stub_translation_service([{}]),
        )
        result = await processor.translate(self.base_config(tmp_path))
        assert result["dlp_token_rows"] == []
        assert result["dlp_chunk_mode"] is None
