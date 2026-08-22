"""Translation attempt runner tests."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig
from src.worker.services.processor_service import JobProcessor
from src.worker.services.quality_judge_service import QualityJudgeResult
from src.worker.services.translation_attempt_runner import TranslationAttemptRunner


@pytest.fixture
def runner(processor):
    return TranslationAttemptRunner(processor)


@pytest.fixture
def processor():
    return JobProcessor(progress_tracker=AsyncMock())


class TestTranslationAttemptRunner:
    async def test_evaluate_attempt_quality(self, runner):
        mock_judge = MagicMock()
        mock_judge.model = "judge-model"
        mock_judge.evaluate_async = AsyncMock(return_value=MagicMock(final_score=0.9))

        res = await runner._evaluate_attempt_quality(
            judge=mock_judge, source_text="s", translated_text="t"
        )
        assert res.final_score == 0.9

        res = await runner._evaluate_attempt_quality(
            judge=mock_judge, source_text="", translated_text=""
        )
        assert res.final_score == 0.0

    def test_write_quality_report(self, runner, tmp_path):
        runner._write_quality_report(tmp_path, {"quality": {"score": 0.9}})
        report_file = tmp_path / "quality_report.json"
        assert report_file.exists()
        assert "score" in report_file.read_text()

    @patch.object(JobProcessor, "_run_single_attempt", new_callable=AsyncMock)
    @patch.object(JobProcessor, "_build_translation_config")
    async def test_run_attempt_success(self, mock_build, mock_run, runner, processor):
        mock_trans_config = MagicMock(spec=TranslationConfig)
        mock_trans_config.working_dir = Path(tempfile.gettempdir()) / "work"
        mock_trans_config.translator = MagicMock(
            prompt_token_count=MagicMock(value=0),
            completion_token_count=MagicMock(value=0),
            token_count=MagicMock(value=0),
            cache_hit_prompt_token_count=MagicMock(value=0),
        )
        mock_build.return_value = mock_trans_config
        mock_run.return_value = {"status": "success"}

        mock_judge = MagicMock()
        mock_judge.model = "judge"
        mock_judge.evaluate_async = AsyncMock(
            return_value=QualityJudgeResult(
                alignment_score=0.9,
                omission_score=0.9,
                hallucination_score=0.9,
                final_score=0.9,
                pass_fail=True,
                reasons=[],
                model="judge",
            )
        )

        with patch(
            "src.worker.services.translation_attempt_runner.extract_attempt_text",
            return_value=("source", "target"),
        ):
            result, _, _, quality, report = await runner.run_attempt(
                model_index=0,
                model_list=["gemini-2.5-flash"],
                config={"job_id": "1", "output_dir": tempfile.gettempdir()},
                output_base_dir=Path(tempfile.gettempdir()),
                max_attempts=1,
                judge=mock_judge,
            )

        assert result == {"status": "success"}
        assert quality.pass_fail is True
        assert report["model_id"] == "gemini-2.5-flash"
