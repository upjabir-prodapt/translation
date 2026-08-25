from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from docx import Document
from src.worker.doctranslator.format.docx.docx_translator import DocxTranslationResult
from src.worker.services.dlp_service import DlpProvider
from src.worker.services.dlp_service import DlpResult
from src.worker.services.docx_job_processor import DocxJobProcessor


class TestEventLoopIsNotBlocked:
    """translate_docx() must run off the asyncio event loop.

    In the 2026-08-24 baseline it was awaited-less and fully synchronous, so
    a running DOCX job blocked uvicorn entirely: a second job uploaded at
    09:21:43 was not even acknowledged by the worker until 09:25:06 (203s).
    """

    @patch("src.worker.services.docx_job_processor.GoogleADKJudgeAgent")
    @patch("src.worker.services.docx_job_processor.create_translator")
    @patch("src.worker.services.docx_job_processor.translate_docx")
    async def test_translate_docx_runs_in_a_worker_thread(
        self,
        mock_translate_docx,
        mock_create_translator,
        mock_judge_cls,
        tmp_path: Path,
    ):
        import asyncio
        import threading

        main_thread = threading.get_ident()
        seen_threads: list[int] = []

        def _record_thread(**kwargs):
            seen_threads.append(threading.get_ident())
            return DocxTranslationResult(
                output_path=kwargs["output_path"],
                source_text="Hello world",
                translated_text="Bonjour le monde",
                dlp_provider=None,
                dlp_token_rows=[],
                extracted_terms=[],
            )

        mock_translate_docx.side_effect = _record_thread
        mock_create_translator.return_value = MagicMock()
        judge = MagicMock()
        judge.evaluate_async = AsyncMock(
            return_value=MagicMock(
                final_score=0.95, pass_fail=True, to_dict=lambda: {"final_score": 0.95}
            )
        )
        mock_judge_cls.return_value = judge

        processor = DocxJobProcessor(glossary_service=MagicMock())
        processor._cost_service = MagicMock()
        processor._cost_service.calculate_attempt_cost.return_value = MagicMock(
            provider="gemini_vertexai", total_cost_usd=0.0, to_dict=lambda: {}
        )

        # A sentinel task proves the loop stayed responsive during the call.
        loop_was_responsive = False

        async def _heartbeat():
            nonlocal loop_was_responsive
            await asyncio.sleep(0)
            loop_was_responsive = True

        heartbeat = asyncio.create_task(_heartbeat())
        await processor.translate(
            {
                "job_id": "job-1",
                "input_file": str(tmp_path / "in.docx"),
                "output_dir": str(tmp_path / "out"),
                "lang_in": "en",
                "lang_out": "fr",
                "domain": "commercial",
                "model_list": ["gemini-3.5-flash"],
                "max_model_attempts": 1,
                "add_cover_page": False,
            }
        )
        await heartbeat

        assert seen_threads, "translate_docx was never invoked"
        assert seen_threads[0] != main_thread, (
            "translate_docx ran on the event-loop thread; it must be offloaded "
            "via asyncio.to_thread so the worker can still accept requests"
        )
        assert loop_was_responsive


class TestApplyCoverPage:
    def test_apply_cover_page_prepends_cover_and_disclaimer(self, tmp_path: Path):
        output_path = tmp_path / "translated.docx"
        document = Document()
        document.add_paragraph("Bonjour le monde")
        document.save(str(output_path))

        best_result = DocxTranslationResult(
            output_path=output_path,
            source_text="Hello world",
            translated_text="Bonjour le monde",
            dlp_provider=None,
            dlp_token_rows=[],
            extracted_terms=[],
        )

        processor = DocxJobProcessor.__new__(DocxJobProcessor)
        processor._apply_cover_page(
            best_result=best_result,
            lang_in="en",
            lang_out="fr",
            domain="commercial",
            model_used="gemini-3.5-flash",
            best_quality={"final_score": 0.9, "model": "gemini-3.5-flash"},
        )

        reopened = Document(str(output_path))
        texts = [p.text for p in reopened.paragraphs]
        assert texts[0] == "AI Translated Document"
        assert any("Original language: English" in t for t in texts)
        assert any("Target language: French" in t for t in texts)
        assert not any("Confidence score" in t for t in texts)
        assert any(
            "AI generated translation that may have mistakes" in t for t in texts
        )
        assert "Bonjour le monde" in texts
        assert texts.index("Bonjour le monde") > texts.index("AI Translated Document")

    def test_apply_cover_page_failure_is_swallowed(self, tmp_path: Path):
        missing_path = tmp_path / "missing.docx"
        best_result = DocxTranslationResult(
            output_path=missing_path,
            source_text="Hello",
            translated_text="Bonjour",
            dlp_provider=None,
            dlp_token_rows=[],
            extracted_terms=[],
        )
        processor = DocxJobProcessor.__new__(DocxJobProcessor)
        # Should not raise even though the output file does not exist.
        processor._apply_cover_page(
            best_result=best_result,
            lang_in="en",
            lang_out="fr",
            domain="commercial",
            model_used="gemini-3.5-flash",
            best_quality=None,
        )


class TestDocxJobProcessorAttemptReuse:
    @patch("src.worker.services.docx_job_processor.translate_docx")
    @patch("src.worker.services.docx_job_processor.create_translator")
    @patch("src.worker.services.docx_job_processor.GoogleADKJudgeAgent")
    async def test_translate_reuses_extracted_terms_and_dlp_across_attempts(
        self,
        mock_judge_cls,
        mock_create_translator,
        mock_translate_docx,
        tmp_path: Path,
    ):
        """C2: Subsequent attempts receive cached extracted_terms and dlp_result from attempt 1."""
        input_file = tmp_path / "input.docx"
        input_file.write_text("dummy")

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge
        # Attempt 1 scores 0.5 (fail), Attempt 2 scores 0.95 (pass)
        mock_judge.evaluate_async = AsyncMock(
            side_effect=[
                MagicMock(
                    final_score=0.5,
                    pass_fail=False,
                    to_dict=lambda: {"final_score": 0.5},
                ),
                MagicMock(
                    final_score=0.95,
                    pass_fail=True,
                    to_dict=lambda: {"final_score": 0.95},
                ),
            ]
        )

        dlp_res = DlpResult(
            masked_chunks=["Masked chunk"],
            token_rows=[{"token": "[EMAIL_1]", "original_value": "user@test.com"}],
            dlp_provider=DlpProvider.REGEX_FALLBACK,
        )
        extracted_terms = [("source_term", "target_term")]

        res1 = DocxTranslationResult(
            output_path=tmp_path / "iter_1" / "input.docx",
            source_text="source",
            translated_text="target 1",
            dlp_provider=DlpProvider.REGEX_FALLBACK,
            dlp_token_rows=dlp_res.token_rows,
            extracted_terms=extracted_terms,
            dlp_result=dlp_res,
        )
        res2 = DocxTranslationResult(
            output_path=tmp_path / "iter_2" / "input.docx",
            source_text="source",
            translated_text="target 2",
            dlp_provider=DlpProvider.REGEX_FALLBACK,
            dlp_token_rows=dlp_res.token_rows,
            extracted_terms=extracted_terms,
            dlp_result=dlp_res,
        )
        mock_translate_docx.side_effect = [res1, res2]

        processor = DocxJobProcessor()
        processor._apply_cover_page = MagicMock()

        config = {
            "job_id": "job-test-c2",
            "input_file": str(input_file),
            "output_dir": str(tmp_path / "out"),
            "lang_in": "en",
            "lang_out": "fr",
            "domain": "commercial",
            "model_list": ["gemini-2.5-flash", "gemini-3.5-flash"],
            "max_model_attempts": 2,
            "enable_dlp": True,
            "auto_extract_glossary": True,
        }

        result = await processor.translate(config)

        assert mock_translate_docx.call_count == 2
        # First attempt: extracted_terms=None, dlp_result=None
        first_call_kwargs = mock_translate_docx.call_args_list[0].kwargs
        assert first_call_kwargs["extracted_terms"] is None
        assert first_call_kwargs["dlp_result"] is None

        # Second attempt: received cached extracted_terms and dlp_result from attempt 1
        second_call_kwargs = mock_translate_docx.call_args_list[1].kwargs
        assert second_call_kwargs["extracted_terms"] == extracted_terms
        assert second_call_kwargs["dlp_result"] == dlp_res

        assert result["attempt_index"] == 2
        assert result["model_id"] == "gemini-3.5-flash"
        assert "attempts" in result
        assert len(result["attempts"]) == 2
        assert result["attempts"][0]["attempt_number"] == 1
        assert result["attempts"][0]["is_selected"] is False
        assert result["attempts"][1]["attempt_number"] == 2
        assert result["attempts"][1]["is_selected"] is True
        assert result["attempts"][0]["docx_path"] is not None
