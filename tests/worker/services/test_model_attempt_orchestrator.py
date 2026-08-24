"""Model attempt orchestrator tests: cross-attempt IL / shared-context reuse."""

from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.doctranslator.format.pdf.translation_config import (
    SharedContextCrossSplitPart,
)
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig
from src.worker.services.model_attempt_orchestrator import ModelAttemptOrchestrator
from src.worker.services.processor_service import JobProcessor
from src.worker.services.quality_judge_service import QualityJudgeResult


class TestModelAttemptOrchestratorIlReuse:
    @patch("src.worker.services.model_attempt_orchestrator.GoogleADKJudgeAgent")
    async def test_run_model_chain_propagates_shared_context_across_attempts(
        self, mock_judge_cls, tmp_path: Path
    ):
        """C1: ModelAttemptOrchestrator captures shared_context from attempt 1 and passes to attempt 2+."""
        processor = JobProcessor(progress_tracker=AsyncMock())
        orchestrator = ModelAttemptOrchestrator(processor)

        shared_context = SharedContextCrossSplitPart()
        mock_trans_config_1 = MagicMock(spec=TranslationConfig)
        mock_trans_config_1.shared_context_cross_split_part = shared_context
        mock_trans_config_1.working_dir = tmp_path / "iter_1"
        mock_trans_config_1.input_file = tmp_path / "input.pdf"
        mock_trans_config_1.lang_in = "en"
        mock_trans_config_1.lang_out = "fr"
        mock_trans_config_1.dlp_provider = None
        mock_trans_config_1.dlp_chunk_mode = None
        mock_trans_config_1.dlp_token_rows = []
        mock_trans_config_1.get_translated_sections_summary.return_value = "All"

        mock_trans_config_2 = MagicMock(spec=TranslationConfig)
        mock_trans_config_2.shared_context_cross_split_part = shared_context
        mock_trans_config_2.working_dir = tmp_path / "iter_2"
        mock_trans_config_2.input_file = tmp_path / "input.pdf"
        mock_trans_config_2.lang_in = "en"
        mock_trans_config_2.lang_out = "fr"
        mock_trans_config_2.dlp_provider = None
        mock_trans_config_2.dlp_chunk_mode = None
        mock_trans_config_2.dlp_token_rows = []
        mock_trans_config_2.get_translated_sections_summary.return_value = "All"

        attempt_1_quality = QualityJudgeResult(
            alignment_score=0.5,
            omission_score=0.5,
            hallucination_score=0.5,
            final_score=0.5,
            pass_fail=False,
            reasons=["Score below threshold"],
            model="judge",
        )
        attempt_2_quality = QualityJudgeResult(
            alignment_score=0.95,
            omission_score=0.95,
            hallucination_score=0.95,
            final_score=0.95,
            pass_fail=True,
            reasons=[],
            model="judge",
        )

        attempt_1_report = {
            "attempt_index": 1,
            "model_id": "gemini-2.5-flash",
            "quality": attempt_1_quality.to_dict(),
            "token_usage": {"total_tokens": 100},
        }
        attempt_2_report = {
            "attempt_index": 2,
            "model_id": "gemini-3.5-flash",
            "quality": attempt_2_quality.to_dict(),
            "token_usage": {"total_tokens": 120},
        }

        # Mock orchestrator._attempt_runner.run_attempt
        orchestrator._attempt_runner.run_attempt = AsyncMock(
            side_effect=[
                (
                    {"mono_pdf_path": str(tmp_path / "out1.pdf")},
                    {"attempt_index": 1, "selected_model": "gemini-2.5-flash"},
                    mock_trans_config_1,
                    attempt_1_quality,
                    attempt_1_report,
                ),
                (
                    {"mono_pdf_path": str(tmp_path / "out2.pdf")},
                    {"attempt_index": 2, "selected_model": "gemini-3.5-flash"},
                    mock_trans_config_2,
                    attempt_2_quality,
                    attempt_2_report,
                ),
            ]
        )

        processor._apply_cover_pages = MagicMock()
        processor._get_total_pdf_pages = MagicMock(return_value=1)

        config = {
            "job_id": "test-pdf-c1",
            "input_file": str(tmp_path / "input.pdf"),
            "output_dir": str(tmp_path / "out"),
            "model_list": ["gemini-2.5-flash", "gemini-3.5-flash"],
            "max_model_attempts": 2,
        }

        result = await orchestrator.run_model_chain(config)

        assert orchestrator._attempt_runner.run_attempt.call_count == 2

        # Check call args
        calls = orchestrator._attempt_runner.run_attempt.call_args_list
        # First call: shared_context is None
        assert calls[0].kwargs["shared_context"] is None
        # Second call: shared_context is the shared_context object captured from attempt 1
        assert calls[1].kwargs["shared_context"] is shared_context

        assert result["attempt_index"] == 2
        assert result["model_id"] == "gemini-3.5-flash"
