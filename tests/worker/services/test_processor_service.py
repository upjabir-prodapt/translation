import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.services.processor_service import JobProcessor
from src.worker.services.processor_service import _job_runtime_root


@pytest.fixture(autouse=True)
def mock_global_assets():
    temp_dir = Path(tempfile.gettempdir())
    with (
        patch(
            "src.worker.services.processor_service.get_doclayout_onnx_model_path",
            return_value=temp_dir / "fake.onnx",
        ),
        patch(
            "src.worker.loaders.assets.get_font_and_metadata",
            return_value=(temp_dir / "font.ttf", MagicMock()),
        ),
        patch("src.worker.services.processor_service.OnnxModel"),
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

    def test_collect_token_usage(self, processor):
        from unittest.mock import MagicMock

        from src.worker.doctranslator.translator.providers.gemini import (
            GeminiVertexAITranslator,
        )
        from src.worker.services.translation_attempt_runner import (
            TranslationAttemptRunner,
        )

        runner = TranslationAttemptRunner(processor)
        translator = MagicMock(spec=GeminiVertexAITranslator)
        translator.prompt_token_count = MagicMock(value=1000)
        translator.completion_token_count = MagicMock(value=500)
        translator.token_count = MagicMock(value=1500)
        translator.cache_hit_prompt_token_count = MagicMock(value=0)
        config = MagicMock()
        config.translator = translator

        usage = runner.collect_token_usage(config, "gemini-2.5-flash")
        assert usage["model_id"] == "gemini-2.5-flash"
        assert usage["prompt_tokens"] == 1000
        assert usage["completion_tokens"] == 500
        assert isinstance(usage["estimated_cost_usd"], float)
        assert usage["estimated_cost_usd"] > 0

    @patch("src.worker.services.processor_service.get_doclayout_onnx_model_path")
    @patch("src.worker.services.processor_service.OnnxModel")
    def test_get_doc_layout_model(self, mock_onnx, mock_path, processor):
        mock_path.return_value = Path(tempfile.gettempdir()) / "model.onnx"
        model = processor._get_doc_layout_model()
        assert model is not None
        mock_onnx.assert_called_once()
        # Test singleton
        model2 = processor._get_doc_layout_model()
        assert model is model2

    @patch("src.worker.services.processor_service.get_translation_storage_repository")
    @patch("src.worker.services.processor_service.Glossary.from_csv")
    @patch("src.worker.services.processor_service.Path.mkdir")
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
        "src.worker.services.model_attempt_orchestrator.ModelAttemptOrchestrator.run_model_chain",
        new_callable=AsyncMock,
    )
    @patch("src.worker.services.processor_service.Path.mkdir")
    async def test_translate_max_retries_reached(
        self, mock_mkdir, mock_run_chain, processor, mock_config
    ):
        mock_run_chain.side_effect = RuntimeError("All translation attempts failed")
        mock_config["max_model_attempts"] = 1
        with pytest.raises(RuntimeError, match="All translation attempts failed"):
            await processor.translate(mock_config)

    def test_detect_language_for_text_low_confidence(self, processor):
        """C.1.1: `_detect_language_for_text` now delegates to the shared
        `language_detection_core.detect_language_for_text`, so the
        underlying `detect_langs` call is patched there instead of on
        `processor_service` directly."""
        mock_candidate = MagicMock()
        mock_candidate.lang = "en"
        mock_candidate.prob = 0.1  # Below default threshold
        with patch(
            "src.worker.services.language_detection_core.detect_langs",
            return_value=[mock_candidate],
        ):
            assert processor._detect_language_for_text("some text") is None

    def test_detect_language_for_text_exception(self, processor):
        from langdetect import LangDetectException

        with patch(
            "src.worker.services.language_detection_core.detect_langs",
            side_effect=LangDetectException(0, "Error"),
        ):
            assert processor._detect_language_for_text("some text") is None

    def test_detect_source_language_no_langs(self, processor):
        """B.3.3: this branch now raises the same user-facing
        no-text-layer wording as PDFValidator's API-side rejection,
        since it fires under the same underlying condition (no
        detectable text on any sampled page)."""
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = []
        with patch(
            "src.worker.services.processor_service.pymupdf.open",
            return_value=MagicMock(__enter__=lambda _: mock_doc),
        ):
            with pytest.raises(ValueError, match="no extractable text layer"):
                processor.detect_source_language("dummy.pdf")

    def test_max_distinct_languages_per_page_message_matches_constant(self, processor):
        """C.3.1/C.6.5: the guard message must render the real constant
        value, not a hardcoded '2' (the bug: message said "more than 2
        languages" while the constant was actually 10)."""
        many_languages = {
            f"lang{i}": 100
            for i in range(processor.MAX_DISTINCT_LANGUAGES_PER_PAGE + 1)
        }
        mock_page = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [mock_page]
        with (
            patch(
                "src.worker.services.processor_service.pymupdf.open",
                return_value=MagicMock(__enter__=lambda _: mock_doc),
            ),
            patch.object(
                processor,
                "_detect_page_languages",
                return_value=(Counter(many_languages), 1000),
            ),
        ):
            with pytest.raises(
                ValueError,
                match=(
                    f"more than {processor.MAX_DISTINCT_LANGUAGES_PER_PAGE} languages"
                ),
            ):
                processor.detect_source_language("dummy.pdf")

    def test_detect_source_language_with_distribution_returns_full_counter(
        self, processor
    ):
        """C.5.1 prerequisite: the full per-language Counter must be
        returned alongside the winner, not discarded."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__iter__.return_value = [mock_page]
        with (
            patch(
                "src.worker.services.processor_service.pymupdf.open",
                return_value=MagicMock(__enter__=lambda _: mock_doc),
            ),
            patch.object(
                processor,
                "_detect_page_languages",
                return_value=(Counter({"de": 700, "en": 300}), 1000),
            ),
        ):
            winner, distribution = processor.detect_source_language_with_distribution(
                "dummy.pdf"
            )
            assert winner == "de"
            assert distribution == Counter({"de": 700, "en": 300})


class TestJobProcessorLanguageDetection:
    @patch("src.worker.services.processor_service.pymupdf.open")
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
            "src.worker.services.processor_service.TranslationConfig"
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

    async def test_handle_translation_event_error_preserves_exception_type(
        self, processor
    ):
        """B.3.1 prerequisite: the original exception object (e.g.
        ScannedPDFError) set by ProgressMonitor.translate_error() must
        survive as-is, not be flattened into a generic RuntimeError, so
        TranslationAttemptRunner's non-retryable-exception check can
        recognize it."""
        from src.worker.doctranslator.doctranslator_exception.DocTranslatorException import (
            ScannedPDFError,
        )

        original = ScannedPDFError("Scanned PDF detected.")
        event = {"type": "error", "error": original}
        with pytest.raises(ScannedPDFError) as exc_info:
            await processor._handle_translation_event(event, {})
        assert exc_info.value is original

    async def test_handle_translation_event_error_string_wrapped_in_runtime_error(
        self, processor
    ):
        """Backward compatibility: a plain string error still raises a
        descriptive RuntimeError (unchanged behaviour)."""
        event = {"type": "error", "error": "boom"}
        with pytest.raises(RuntimeError, match="Translation failed: boom"):
            await processor._handle_translation_event(event, {})

    @patch("src.worker.services.processor_service.async_translate")
    async def test_run_single_attempt_success(self, mock_translate, processor):
        async def mock_gen(_cfg):
            yield {"type": "finish", "translate_result": {"status": "done"}}

        mock_translate.side_effect = mock_gen
        with patch.object(
            processor, "_handle_translation_event", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = {"status": "done"}
            config = MagicMock()
            config.shared_context_cross_split_part.split_page_ranges = [(0, 9, 3000)]
            res = await processor._run_single_attempt(config, {})
            # The real split is carried forward so per-chunk cost attribution
            # does not re-derive a different one.
            assert res == {"status": "done", "split_page_ranges": [(0, 9, 3000)]}

    @patch(
        "src.worker.services.model_attempt_orchestrator.ModelAttemptOrchestrator.run_model_chain",
        new_callable=AsyncMock,
    )
    @patch("src.worker.services.processor_service.Path.mkdir")
    async def test_translate_retry_on_failure(
        self, mock_mkdir, mock_run_chain, processor, mock_config
    ):
        mock_run_chain.return_value = {"status": "ok"}
        mock_config["max_model_attempts"] = 2
        mock_config["model_list"] = ["m1", "m2"]

        res = await processor.translate(mock_config)
        assert res["status"] == "ok"

    @patch(
        "src.worker.services.model_attempt_orchestrator.ModelAttemptOrchestrator.run_model_chain",
        new_callable=AsyncMock,
    )
    @patch("src.worker.services.processor_service.Path.mkdir")
    async def test_translate_full_flow(
        self, mock_mkdir, mock_run_chain, processor, mock_config
    ):
        mock_run_chain.return_value = {"status": "ok", "model_id": "m1"}

        res = await processor.translate(mock_config)
        assert res["status"] == "ok"
        mock_run_chain.assert_called_once()

    @patch("src.worker.services.processor_service.pymupdf.open")
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

    @patch("src.worker.services.processor_service.pymupdf.open")
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

    @patch("src.worker.services.processor_service.pymupdf.open")
    def test_prepend_cover_page_failure(self, mock_open, processor):
        mock_open.side_effect = Exception("pymupdf fail")
        processor._prepend_cover_page(Path("none.pdf"), MagicMock())
        # Should catch exception and log warning
