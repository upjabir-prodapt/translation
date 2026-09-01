"""The stop/continue rule, and the guarantee that both loops obey it.

`ModelAttemptOrchestrator` (PDF) and `DocxJobProcessor` (DOCX) were
hand-mirrored -- the DOCX loop still carried a "Mirror
ModelAttemptOrchestrator" comment -- and had already drifted: both of its
exit conditions were guarded on `quality_result is not None`, so a judge
failure fell straight through and ran the whole model chain.

The parametrised class at the bottom is the regression guard against that
class of drift: the same behavioural assertion is made against both
pipelines through their real public entry points.
"""

from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.doctranslator.format.docx.docx_translator import DocxTranslationResult
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig
from src.worker.services.attempt_decision import AttemptDecision
from src.worker.services.attempt_decision import decide_after_attempt
from src.worker.services.docx_job_processor import DocxJobProcessor
from src.worker.services.model_attempt_orchestrator import ModelAttemptOrchestrator
from src.worker.services.processor_service import JobProcessor
from src.worker.services.quality_judge_service import QualityJudgeResult


def _result(final_score, *, pass_fail=False, inconclusive=False):
    return QualityJudgeResult(
        alignment_score=final_score,
        omission_score=final_score,
        hallucination_score=final_score,
        final_score=final_score,
        pass_fail=pass_fail,
        reasons=["r"],
        model="judge",
        inconclusive=inconclusive,
    )


class TestDecideAfterAttempt:
    def test_pass_stops(self):
        decision = decide_after_attempt(_result(0.9, pass_fail=True), attempt_index=1)
        assert decision is AttemptDecision.PASS
        assert decision.should_stop is True

    def test_low_score_continues(self):
        decision = decide_after_attempt(_result(0.1), attempt_index=1)
        assert decision is AttemptDecision.CONTINUE
        assert decision.should_stop is False

    def test_near_miss_is_early_accepted(self, monkeypatch):
        from src.worker.services import attempt_decision

        monkeypatch.setattr(
            attempt_decision.settings, "QUALITY_EARLY_ACCEPT_THRESHOLD", 0.9
        )
        decision = decide_after_attempt(_result(0.92), attempt_index=1)
        assert decision is AttemptDecision.EARLY_ACCEPT

    def test_inconclusive_stops_without_retrying(self):
        """A judge that did not answer is a measurement failure.

        Re-translating on a different model cannot fix it, and doing so is
        what turned a 4-minute job into a 25-minute one.
        """
        decision = decide_after_attempt(
            _result(0.0, inconclusive=True), attempt_index=1
        )
        assert decision is AttemptDecision.INCONCLUSIVE_ACCEPT
        assert decision.should_stop is True

    def test_judge_disabled_stops(self):
        assert (
            decide_after_attempt(None, attempt_index=1)
            is AttemptDecision.INCONCLUSIVE_ACCEPT
        )

    def test_inconclusive_beats_a_would_be_continue_score(self):
        """`inconclusive` is checked before the numeric score.

        The scores on an inconclusive verdict are zeroed for schema
        stability; reading them as a real verdict is precisely the bug.
        """
        decision = decide_after_attempt(
            _result(0.0, inconclusive=True), attempt_index=2
        )
        assert decision is not AttemptDecision.CONTINUE


async def _run_pdf_chain(tmp_path: Path, quality_results):
    """Drive ModelAttemptOrchestrator over canned per-attempt verdicts."""
    processor = JobProcessor(progress_tracker=AsyncMock())
    orchestrator = ModelAttemptOrchestrator(processor)

    def _config(index):
        cfg = MagicMock(spec=TranslationConfig)
        cfg.shared_context_cross_split_part = MagicMock()
        cfg.working_dir = tmp_path / f"iter_{index}"
        cfg.input_file = tmp_path / "input.pdf"
        cfg.lang_in = "en"
        cfg.lang_out = "fr"
        cfg.dlp_provider = None
        cfg.dlp_chunk_mode = None
        cfg.dlp_token_rows = []
        cfg.get_translated_sections_summary.return_value = "All"
        return cfg

    orchestrator._attempt_runner.run_attempt = AsyncMock(
        side_effect=[
            (
                {"mono_pdf_path": str(tmp_path / f"out{i}.pdf")},
                {"attempt_index": i, "selected_model": f"model-{i}"},
                _config(i),
                quality,
                {
                    "attempt_index": i,
                    "model_id": f"model-{i}",
                    "quality": quality.to_dict(),
                    "token_usage": {"total_tokens": 1},
                },
            )
            for i, quality in enumerate(quality_results, start=1)
        ]
    )
    processor._apply_cover_pages = MagicMock()
    processor._get_total_pdf_pages = MagicMock(return_value=1)

    with patch("src.worker.services.model_attempt_orchestrator.GoogleADKJudgeAgent"):
        result = await orchestrator.run_model_chain(
            {
                "job_id": "job-decision",
                "input_file": str(tmp_path / "input.pdf"),
                "output_dir": str(tmp_path / "out"),
                "model_list": [f"model-{i}" for i in range(1, 4)],
                "max_model_attempts": len(quality_results),
            }
        )
    return orchestrator._attempt_runner.run_attempt.call_count, result


async def _run_docx_chain(tmp_path: Path, quality_results):
    """Drive DocxJobProcessor over the same canned per-attempt verdicts."""
    input_file = tmp_path / "in.docx"
    input_file.write_text("dummy")

    with (
        patch(
            "src.worker.services.docx_job_processor.translate_docx"
        ) as mock_translate,
        patch("src.worker.services.docx_job_processor.create_translator"),
        patch("src.worker.services.docx_job_processor.GoogleADKJudgeAgent") as mock_cls,
    ):
        mock_translate.side_effect = lambda **kwargs: DocxTranslationResult(
            output_path=kwargs["output_path"],
            source_text="hello",
            translated_text="bonjour",
            dlp_provider=None,
            dlp_token_rows=[],
            extracted_terms=[],
            segments=[("hello", "bonjour")],
        )
        judge = MagicMock()
        judge.model = "judge"
        judge.evaluate_segments_async = AsyncMock(side_effect=list(quality_results))
        mock_cls.return_value = judge

        processor = DocxJobProcessor(glossary_service=MagicMock())
        processor._cost_service = MagicMock()
        processor._cost_service.calculate_attempt_cost.return_value = MagicMock(
            provider="gemini_vertexai", total_cost_usd=0.0, to_dict=lambda: {}
        )
        result = await processor.translate(
            {
                "job_id": "job-decision",
                "input_file": str(input_file),
                "output_dir": str(tmp_path / "out"),
                "lang_in": "en",
                "lang_out": "fr",
                "domain": "commercial",
                "model_list": [f"model-{i}" for i in range(1, 4)],
                "max_model_attempts": len(quality_results),
                "add_cover_page": False,
            }
        )
        return judge.evaluate_segments_async.await_count, result


@pytest.mark.parametrize(
    "run_chain", [_run_pdf_chain, _run_docx_chain], ids=["pdf", "docx"]
)
class TestBothLoopsObeyTheSameRule:
    """One assertion, both pipelines. This is the drift guard."""

    async def test_inconclusive_verdict_stops_after_attempt_one(
        self, run_chain, tmp_path
    ):
        attempts, result = await run_chain(
            tmp_path,
            [
                _result(0.0, inconclusive=True),
                _result(0.99, pass_fail=True),
                _result(0.99, pass_fail=True),
            ],
        )
        assert attempts == 1, (
            "An inconclusive judge verdict must not trigger a full "
            "re-translation on the next model in the chain"
        )
        # The verdict is still reported rather than silently dropped.
        assert result["quality_report"]["inconclusive"] is True

    async def test_passing_verdict_stops_after_attempt_one(self, run_chain, tmp_path):
        attempts, _ = await run_chain(
            tmp_path,
            [_result(0.99, pass_fail=True), _result(0.99, pass_fail=True)],
        )
        assert attempts == 1

    async def test_failing_verdict_advances_to_the_next_model(
        self, run_chain, tmp_path
    ):
        attempts, _ = await run_chain(
            tmp_path,
            [_result(0.1), _result(0.99, pass_fail=True)],
        )
        assert attempts == 2
