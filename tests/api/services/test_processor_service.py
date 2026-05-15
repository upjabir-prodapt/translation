import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.api.services.processor_service import JobProcessor
from src.api.services.processor_service import _job_runtime_root
from src.api.services.quality_judge_service import QualityJudgeResult
from src.doctranslator.format.pdf.translation_config import TranslationConfig


@pytest.fixture(autouse=True)
def mock_global_assets():
    temp_dir = Path(tempfile.gettempdir())
    with (
        patch(
            "src.api.services.processor_service.get_doclayout_onnx_model_path",
            return_value=temp_dir / "fake.onnx",
        ),
        patch(
            "src.loaders.assets.get_font_and_metadata",
            return_value=(temp_dir / "font.ttf", MagicMock()),
        ),
        patch("src.api.services.processor_service.OnnxModel"),
    ):
        yield


@pytest.fixture
def processor():
    return JobProcessor(progress_tracker=AsyncMock())


@pytest.fixture
def mock_config():
    return {
        "job_id": "test-job-id",
        "input_file": "dummy.pdf",
        "output_dir": "output",
        "lang_in": "en",
        "lang_out": "fr",
        "model_list": ["gemini-pro"],
        "max_model_attempts": 1,
    }


def test_job_runtime_root():
    root = _job_runtime_root("test-id")
    assert "test-id" in str(root)
    assert "jobs" in str(root)


class TestJobProcessorCore:
    def test_counter_value(self, processor):
        mock_val = MagicMock()
        mock_val.value = 10
        assert processor._counter_value(mock_val) == 10
        assert processor._counter_value(5) == 5
        assert processor._counter_value(None) == 0

    def test_estimate_cost(self, processor):
        cost = processor._estimate_cost(
            model_id="gemini-1.5-pro", prompt_tokens=1000, completion_tokens=1000
        )
        assert isinstance(cost, float)

    @patch("src.api.services.processor_service.get_doclayout_onnx_model_path")
    @patch("src.api.services.processor_service.OnnxModel")
    def test_get_doc_layout_model(self, mock_onnx, mock_path, processor):
        mock_path.return_value = Path(tempfile.gettempdir()) / "model.onnx"
        model = processor._get_doc_layout_model()
        assert model is not None
        mock_onnx.assert_called_once()
        # Test singleton
        model2 = processor._get_doc_layout_model()
        assert model is model2

    @patch("src.api.services.processor_service.get_translation_storage_repository")
    @patch("src.api.services.processor_service.Glossary.from_csv")
    @patch("src.api.services.processor_service.Path.mkdir")
    async def test_load_glossary_from_gcs_success(
        self, mock_mkdir, mock_glossary, mock_repo, processor
    ):
        mock_storage = AsyncMock()
        mock_repo.return_value = mock_storage

        result = await processor._load_glossary_from_gcs("job1", "gloss.csv", "fr")

        assert len(result) == 1
        mock_storage.download_glossary.assert_called_once()
        mock_glossary.assert_called_once()

    async def test_handle_progress_update(self, processor):
        event = {"overall_progress": 50, "stage": "Translating"}
        await processor._handle_progress_update(event)
        processor.progress_tracker.update.assert_called()

    async def test_handle_finish_event(self, processor):
        mock_result = {"page_count": 5, "mono_pdf_path": "mono.pdf"}

        with patch("pathlib.Path.exists", return_value=True):
            event = {"type": "finish", "translate_result": mock_result}
            res = await processor._handle_finish_event(event)
            assert res["page_count"] == 5
            assert "mono_pdf_path" in res

    @patch(
        "src.api.services.processor_service.extract_attempt_text",
        return_value=("s", "t"),
    )
    async def test_evaluate_attempt_quality(self, mock_extract, processor):
        mock_judge = MagicMock()
        mock_judge.evaluate_async = AsyncMock(return_value=MagicMock(final_score=0.9))

        # Test success
        res = await processor._evaluate_attempt_quality(
            judge=mock_judge, source_text="s", translated_text="t"
        )
        assert res.final_score == 0.9

        # Test missing text
        res = await processor._evaluate_attempt_quality(
            judge=mock_judge, source_text="", translated_text=""
        )
        assert res.final_score == 0.0

    def test_write_quality_report(self, processor, tmp_path):
        mock_qual_dict = {"quality": {"score": 0.9}}
        p = tmp_path
        processor._write_quality_report(p, mock_qual_dict)
        report_file = p / "quality_report.json"
        assert report_file.exists()
        assert "score" in report_file.read_text()

    def test_build_cover_page_metadata(self, processor):
        config = {
            "job_id": "j1",
            "lang_in": "en",
            "lang_out": "fr",
            "domain": "legal",
            "selected_model": "m1",
        }
        mock_trans_config = MagicMock(spec=TranslationConfig)
        mock_trans_config.input_file = "in.pdf"
        mock_trans_config.lang_in = "en"
        mock_trans_config.lang_out = "fr"
        mock_trans_config.get_translated_sections_summary.return_value = "All"

        mock_qual = MagicMock(final_score=0.9, model="m1")

        with patch.object(processor, "_get_total_pdf_pages", return_value=10):
            res = processor._build_cover_page_metadata(
                mock_trans_config, config, mock_qual
            )
            assert res.original_language == "en"
            assert res.target_language == "fr"
            assert res.confidence_score == 0.9

    @patch.object(JobProcessor, "_execute_attempt", new_callable=AsyncMock)
    @patch("src.api.services.processor_service.Path.mkdir")
    async def test_translate_max_retries_reached(
        self, mock_mkdir, mock_exec, processor, mock_config
    ):
        mock_exec.return_value = (None, {"attempt_index": 1}, None, None, None)
        mock_config["max_model_attempts"] = 1
        with pytest.raises(RuntimeError, match="All translation attempts failed"):
            await processor.translate(mock_config)

    def test_detect_language_for_text_low_confidence(self, processor):
        mock_candidate = MagicMock()
        mock_candidate.lang = "en"
        mock_candidate.prob = 0.1  # Below default threshold
        with patch(
            "src.api.services.processor_service.detect_langs",
            return_value=[mock_candidate],
        ):
            assert processor._detect_language_for_text("some text") is None

    def test_detect_language_for_text_exception(self, processor):
        from langdetect import LangDetectException

        with patch(
            "src.api.services.processor_service.detect_langs",
            side_effect=LangDetectException(0, "Error"),
        ):
            assert processor._detect_language_for_text("some text") is None

    def test_detect_source_language_no_langs(self, processor):
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = []
        with patch(
            "src.api.services.processor_service.pymupdf.open",
            return_value=MagicMock(__enter__=lambda _: mock_doc),
        ):
            with pytest.raises(ValueError, match="Unable to detect"):
                processor.detect_source_language("dummy.pdf")


class TestJobProcessorLanguageDetection:
    @patch("src.api.services.processor_service.pymupdf.open")
    def test_detect_source_language_success(self, mock_open, processor):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__iter__.return_value = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_doc
        with patch.object(processor, "_detect_page_languages") as mock_detect_page:
            mock_detect_page.return_value = (Counter({"en": 1}), 100)
            assert processor.detect_source_language("dummy.pdf") == "en"

    def test_iter_page_text_chunks(self, processor):
        mock_page = MagicMock()
        mock_page.get_text.return_value = [(0, 0, 0, 0, "Valid", 0, 0)]
        with patch.object(processor, "_is_detectable_text", return_value=True):
            chunks = list(processor._iter_page_text_chunks(mock_page))
            assert chunks == ["Valid"]

    def test_normalize_detection_text(self, processor):
        assert processor._normalize_detection_text("  abc  ") == "abc"

    def test_is_detectable_text(self, processor):
        assert processor._is_detectable_text("Valid text that is long enough") is True
        assert processor._is_detectable_text("123") is False

    def test_normalize_detected_language(self, processor):
        assert processor._normalize_detected_language("zh-cn") == "zh"

    def test_build_translation_config(self, processor):
        config = {
            "input_file": "gs://in.pdf",
            "lang_in": "en",
            "lang_out": "fr",
            "domain": "legal",
            "glossary_path": "gs://gloss.csv",
            "model_list": ["gemini-pro"],
        }
        with patch(
            "src.api.services.processor_service.TranslationConfig"
        ) as mock_cfg_cls:
            mock_cfg = MagicMock()
            mock_cfg.source_language = "en"
            mock_cfg.target_language = "fr"
            mock_cfg.working_dir = Path(tempfile.gettempdir()) / "work"
            mock_cfg_cls.return_value = mock_cfg

            res = processor._build_translation_config(
                config, Path(tempfile.gettempdir()) / "work"
            )
            assert res.source_language == "en"
            assert res.target_language == "fr"
            assert res.working_dir == Path(tempfile.gettempdir()) / "work"

    async def test_handle_translation_event_progress(self, processor):
        event = {"type": "progress_update", "overall_progress": 50, "stage": "testing"}
        with patch.object(
            processor, "_handle_progress_update", new_callable=AsyncMock
        ) as mock_update:
            res = await processor._handle_translation_event(event, {})
            assert res is None
            mock_update.assert_called_once_with(event)

    async def test_handle_translation_event_finish(self, processor):
        event = {"type": "finish", "translate_result": {"status": "ok"}}
        with patch.object(
            processor, "_handle_finish_event", new_callable=AsyncMock
        ) as mock_finish:
            mock_finish.return_value = {"status": "done"}
            res = await processor._handle_translation_event(event, {})
            assert res == {"status": "done"}

    async def test_handle_translation_event_unknown(self, processor):
        res = await processor._handle_translation_event({"type": "unknown"}, {})
        assert res is None

    @patch.object(JobProcessor, "_run_single_attempt", new_callable=AsyncMock)
    @patch.object(JobProcessor, "_evaluate_attempt_quality", new_callable=AsyncMock)
    @patch(
        "src.api.services.processor_service.extract_attempt_text",
        return_value=("source", "target"),
    )
    @patch.object(JobProcessor, "_build_translation_config")
    @patch.object(JobProcessor, "_collect_token_usage", return_value={})
    @patch.object(JobProcessor, "_write_quality_report")
    async def test_execute_attempt_success(
        self,
        mock_write,
        mock_tokens,
        mock_build,
        mock_extract,
        mock_eval,
        mock_run,
        processor,
    ):
        mock_trans_config = MagicMock(spec=TranslationConfig)
        mock_trans_config.working_dir = Path(tempfile.gettempdir()) / "work"
        mock_build.return_value = mock_trans_config
        mock_qual_result = MagicMock(spec=QualityJudgeResult)
        mock_qual_result.to_dict.return_value = {}
        mock_qual_result.final_score = 0.9
        mock_eval.return_value = mock_qual_result
        mock_run.return_value = {"status": "success"}

        result, _, _, _, _ = await processor._execute_attempt(
            model_index=0,
            model_list=["m1"],
            config={"job_id": "1"},
            output_base_dir=Path(tempfile.gettempdir()),
            max_attempts=1,
            judge=MagicMock(),
        )
        assert result == {"status": "success"}

    @patch("src.api.services.processor_service.async_translate")
    async def test_run_single_attempt_success(self, mock_translate, processor):
        async def mock_gen(_cfg):
            yield {"type": "finish", "translate_result": {"status": "done"}}

        mock_translate.side_effect = mock_gen
        with patch.object(
            processor, "_handle_translation_event", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = {"status": "done"}
            res = await processor._run_single_attempt(MagicMock(), {})
            assert res == {"status": "done"}

    @patch.object(JobProcessor, "_execute_attempt", new_callable=AsyncMock)
    @patch.object(JobProcessor, "_build_cover_page_metadata")
    @patch.object(JobProcessor, "_apply_cover_pages")
    @patch("src.api.services.processor_service.Path.mkdir")
    async def test_translate_retry_on_failure(
        self, mock_mkdir, mock_apply, mock_metadata, mock_exec, processor, mock_config
    ):
        # First attempt fails (returns None result)
        # Second attempt succeeds
        mock_qual = MagicMock(final_score=0.9, pass_fail=True)
        mock_exec.side_effect = [
            (None, {"attempt_index": 1}, None, None, None),
            (
                {"status": "ok"},
                {"attempt_index": 2, "selected_model": "m2"},
                MagicMock(),
                mock_qual,
                {"token_usage": {}},
            ),
        ]
        mock_config["max_model_attempts"] = 2
        mock_config["model_list"] = ["m1", "m2"]

        res = await processor.translate(mock_config)
        assert res["status"] == "ok"
        assert mock_exec.call_count == 2

    @patch.object(JobProcessor, "_execute_attempt", new_callable=AsyncMock)
    @patch.object(JobProcessor, "_build_cover_page_metadata")
    @patch.object(JobProcessor, "_apply_cover_pages")
    async def test_translate_full_flow(
        self, mock_apply, mock_metadata, mock_exec, processor, mock_config
    ):
        mock_qual = MagicMock()
        mock_qual.final_score = 0.9
        mock_qual.pass_fail = True
        mock_exec.return_value = (
            {"status": "ok"},
            {"attempt_index": 1, "selected_model": "m1"},
            MagicMock(),
            mock_qual,
            {"token_usage": {}},
        )

        with patch("src.api.services.processor_service.Path.mkdir"):
            res = await processor.translate(mock_config)
            assert res["status"] == "ok"
            mock_apply.assert_called_once()

    def test_collect_token_usage(self, processor):
        mock_config = MagicMock(spec=TranslationConfig)
        mock_config.translator = MagicMock()
        mock_config.translator.prompt_token_count = 100
        mock_config.translator.completion_token_count = 50
        mock_config.translator.token_count = 150

        with patch.object(processor, "_estimate_cost", return_value=0.01):
            usage = processor._collect_token_usage(mock_config, "m1")
            assert usage["prompt_tokens"] == 100
            assert usage["estimated_cost_usd"] == 0.01

    @patch("src.api.services.processor_service.pymupdf.open")
    def test_apply_cover_pages(self, mock_open, processor):
        mock_doc = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_doc

        attempt_result = {"mono_pdf_path": "mono.pdf", "dual_pdf_path": "dual.pdf"}
        metadata = MagicMock()

        with patch.object(processor, "_prepend_cover_page") as mock_prepend:
            processor._apply_cover_pages(MagicMock(), attempt_result, metadata)
            assert mock_prepend.call_count == 2

    def test_draw_cover_page(self, processor):
        mock_page = MagicMock()
        mock_page.rect.width = 600
        mock_page.rect.height = 800
        metadata = MagicMock()
        metadata.iter_rows.return_value = [("Label", "Value")]

        processor._draw_cover_page(mock_page, metadata)
        assert mock_page.draw_line.called
        assert mock_page.insert_textbox.called

    @patch("src.api.services.processor_service.pymupdf.open")
    def test_prepend_cover_page_success(self, mock_open, processor, tmp_path):
        mock_orig = MagicMock()
        mock_orig.page_count = 1
        mock_new = MagicMock()
        # original_doc opened first, then new_doc
        mock_open.side_effect = [mock_orig, mock_new]

        p = tmp_path / "test.pdf"
        p.write_text("c")

        with (
            patch.object(processor, "_draw_cover_page") as mock_draw,
            patch("pathlib.Path.replace") as mock_replace,
        ):
            processor._prepend_cover_page(p, MagicMock())
            assert mock_draw.called
            assert mock_new.insert_pdf.called
            assert mock_replace.called

    @patch("src.api.services.processor_service.pymupdf.open")
    def test_prepend_cover_page_failure(self, mock_open, processor):
        mock_open.side_effect = Exception("pymupdf fail")
        processor._prepend_cover_page(Path("none.pdf"), MagicMock())
        # Should catch exception and log warning
