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
        mock_judge.evaluate_segments_async = AsyncMock(
            return_value=MagicMock(final_score=0.9)
        )

        res = await runner._evaluate_attempt_quality(
            judge=mock_judge, segments=[("s", "t")], job_id="job-1"
        )
        assert res.final_score == 0.9
        assert mock_judge.evaluate_segments_async.await_args.kwargs["job_id"] == "job-1"

    async def test_evaluate_attempt_quality_without_segments_is_inconclusive(
        self, runner
    ):
        """No segments means the judge could not measure anything.

        This must not be reported as a score of 0.0: on every split PDF the
        parent working_dir held no tracking file, so this branch fired on
        every attempt, scored below QUALITY_THRESHOLD, and drove the loop
        into re-translating the whole document on every remaining model.
        """
        mock_judge = MagicMock()
        mock_judge.model = "judge-model"
        mock_judge.evaluate_segments_async = AsyncMock()

        res = await runner._evaluate_attempt_quality(
            judge=mock_judge, segments=[], job_id="job-1"
        )
        assert res.inconclusive is True
        assert res.coverage_ratio == 0.0
        mock_judge.evaluate_segments_async.assert_not_awaited()

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
        mock_judge.evaluate_segments_async = AsyncMock(
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
            "src.worker.services.translation_attempt_runner.extract_attempt_segments",
            return_value=[("source", "target")],
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

    @patch.object(JobProcessor, "_run_single_attempt", new_callable=AsyncMock)
    @patch.object(JobProcessor, "_build_translation_config")
    async def test_run_attempt_passes_shared_context(
        self, mock_build, mock_run, runner, processor
    ):
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
        mock_judge.evaluate_segments_async = AsyncMock(
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

        mock_shared_context = MagicMock()
        with patch(
            "src.worker.services.translation_attempt_runner.extract_attempt_segments",
            return_value=[("source", "target")],
        ):
            await runner.run_attempt(
                model_index=0,
                model_list=["gemini-2.5-flash"],
                config={"job_id": "1", "output_dir": tempfile.gettempdir()},
                output_base_dir=Path(tempfile.gettempdir()),
                max_attempts=1,
                judge=mock_judge,
                shared_context=mock_shared_context,
            )

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["shared_context"] == mock_shared_context

    @patch.object(JobProcessor, "_run_single_attempt", new_callable=AsyncMock)
    @patch.object(JobProcessor, "_build_translation_config")
    async def test_run_attempt_reraises_scanned_pdf_error_immediately(
        self, mock_build, mock_run, runner, processor
    ):
        """B.3.1/B.5.4: a ScannedPDFError must propagate immediately even
        when this is NOT the last model in the chain (attempt_index !=
        max_attempts) -- it is a document-shape failure, not a
        model-specific one, so retrying with a different model wastes a
        full re-parse for a guaranteed-identical failure."""
        from src.worker.doctranslator.doctranslator_exception.DocTranslatorException import (
            ScannedPDFError,
        )

        mock_trans_config = MagicMock(spec=TranslationConfig)
        mock_trans_config.working_dir = Path(tempfile.gettempdir()) / "work"
        mock_build.return_value = mock_trans_config
        mock_run.side_effect = ScannedPDFError("Scanned PDF detected.")

        with pytest.raises(ScannedPDFError):
            await runner.run_attempt(
                model_index=0,
                model_list=["gemini-2.5-flash", "gemini-2.5-pro"],
                config={"job_id": "1", "output_dir": tempfile.gettempdir()},
                output_base_dir=Path(tempfile.gettempdir()),
                max_attempts=2,  # NOT the last attempt -- would otherwise retry
                judge=MagicMock(),
            )
        # Only the one, failing attempt should ever have run.
        mock_run.assert_awaited_once()

    @patch.object(JobProcessor, "_run_single_attempt", new_callable=AsyncMock)
    @patch.object(JobProcessor, "_build_translation_config")
    async def test_run_attempt_reraises_input_file_generated_error_immediately(
        self, mock_build, mock_run, runner, processor
    ):
        from src.worker.doctranslator.doctranslator_exception.DocTranslatorException import (
            InputFileGeneratedByDocTranslatorError,
        )

        mock_trans_config = MagicMock(spec=TranslationConfig)
        mock_trans_config.working_dir = Path(tempfile.gettempdir()) / "work"
        mock_build.return_value = mock_trans_config
        mock_run.side_effect = InputFileGeneratedByDocTranslatorError(
            "already translated"
        )

        with pytest.raises(InputFileGeneratedByDocTranslatorError):
            await runner.run_attempt(
                model_index=0,
                model_list=["gemini-2.5-flash", "gemini-2.5-pro"],
                config={"job_id": "1", "output_dir": tempfile.gettempdir()},
                output_base_dir=Path(tempfile.gettempdir()),
                max_attempts=2,
                judge=MagicMock(),
            )
        mock_run.assert_awaited_once()

    @patch.object(JobProcessor, "_run_single_attempt", new_callable=AsyncMock)
    @patch.object(JobProcessor, "_build_translation_config")
    async def test_run_attempt_retries_other_exceptions_when_not_last_attempt(
        self, mock_build, mock_run, runner, processor
    ):
        """Regression: ordinary (retryable) exceptions still follow the
        original behaviour -- return None to signal "try the next model"
        instead of raising, when this is not the final attempt."""
        mock_trans_config = MagicMock(spec=TranslationConfig)
        mock_trans_config.working_dir = Path(tempfile.gettempdir()) / "work"
        mock_build.return_value = mock_trans_config
        mock_run.side_effect = RuntimeError("transient LLM error")

        (
            result,
            attempt_config,
            translation_config,
            quality_result,
            report,
        ) = await runner.run_attempt(
            model_index=0,
            model_list=["gemini-2.5-flash", "gemini-2.5-pro"],
            config={"job_id": "1", "output_dir": tempfile.gettempdir()},
            output_base_dir=Path(tempfile.gettempdir()),
            max_attempts=2,
            judge=MagicMock(),
        )
        assert result is None
        assert translation_config is None
        assert quality_result is None
        assert report is None
