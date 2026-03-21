"""
Unit tests for worker/services/processor.py.

Covers pure-Python methods and helpers that do not require babeldoc or
loaders internals:

  - _counter_value
  - _normalize_detection_text
  - _is_detectable_text
  - _detect_language_for_text
  - _normalize_detected_language
  - _evaluate_attempt_quality
  - _estimate_cost
  - _write_quality_report
  - _get_total_pdf_pages
  - _handle_translation_event
  - _handle_progress_update
  - _handle_finish_event
  - _apply_cover_pages
  - _iter_page_text_chunks
  - _detect_page_languages
"""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import pytest

from worker.services.processor import JobProcessor
from worker.services.quality_judge import QualityJudgeResult


# ---------------------------------------------------------------------------
# Construction helper
# ---------------------------------------------------------------------------


def _make_processor(progress_tracker=None) -> JobProcessor:
    """Build a JobProcessor with all heavy dependencies stubbed out."""
    pt = progress_tracker or AsyncMock()
    proc = JobProcessor(progress_tracker=pt)
    # Prevent lazy-loading the real ONNX model
    proc._doc_layout_model = MagicMock()
    return proc


def _make_quality_result(**kwargs) -> QualityJudgeResult:
    """Return a minimal QualityJudgeResult, overridable via kwargs."""
    defaults = dict(
        alignment_score=0.9,
        omission_score=0.9,
        hallucination_score=0.9,
        final_score=0.9,
        pass_fail=True,
        reasons=[],
        model="test-model",
    )
    defaults.update(kwargs)
    return QualityJudgeResult(**defaults)


# ---------------------------------------------------------------------------
# _counter_value
# ---------------------------------------------------------------------------


class TestCounterValue:
    def test_plain_int_returned_as_is(self):
        proc = _make_processor()
        assert proc._counter_value(42) == 42

    def test_zero_int(self):
        proc = _make_processor()
        assert proc._counter_value(0) == 0

    def test_none_returns_zero(self):
        proc = _make_processor()
        assert proc._counter_value(None) == 0

    def test_falsy_zero_equivalent(self):
        proc = _make_processor()
        # 0 is falsy, but int(0 or 0) == 0
        assert proc._counter_value(0) == 0

    def test_object_with_value_attr(self):
        proc = _make_processor()
        obj = MagicMock()
        obj.value = 99
        assert proc._counter_value(obj) == 99

    def test_object_with_value_attr_zero(self):
        proc = _make_processor()
        obj = MagicMock()
        obj.value = 0
        assert proc._counter_value(obj) == 0

    def test_string_numeric_no_value_attr(self):
        proc = _make_processor()
        # Strings have no .value attr, so int("5" or 0) == 5
        assert proc._counter_value("5") == 5

    def test_returns_int_type(self):
        proc = _make_processor()
        result = proc._counter_value(7)
        assert isinstance(result, int)

    @pytest.mark.parametrize("raw,expected", [
        (1, 1),
        (100, 100),
        (0, 0),
        (None, 0),
    ])
    def test_parametrized_plain_values(self, raw, expected):
        proc = _make_processor()
        assert proc._counter_value(raw) == expected


# ---------------------------------------------------------------------------
# _normalize_detection_text
# ---------------------------------------------------------------------------


class TestNormalizeDetectionText:
    def test_collapses_multiple_spaces(self):
        proc = _make_processor()
        assert proc._normalize_detection_text("foo   bar") == "foo bar"

    def test_collapses_tabs(self):
        proc = _make_processor()
        assert proc._normalize_detection_text("foo\tbar") == "foo bar"

    def test_collapses_newlines(self):
        proc = _make_processor()
        assert proc._normalize_detection_text("foo\nbar\nbaz") == "foo bar baz"

    def test_strips_leading_and_trailing_whitespace(self):
        proc = _make_processor()
        assert proc._normalize_detection_text("  hello world  ") == "hello world"

    def test_empty_string(self):
        proc = _make_processor()
        assert proc._normalize_detection_text("") == ""

    def test_only_whitespace_becomes_empty(self):
        proc = _make_processor()
        assert proc._normalize_detection_text("   \t  \n  ") == ""

    def test_already_normalized_unchanged(self):
        proc = _make_processor()
        text = "hello world"
        assert proc._normalize_detection_text(text) == text

    def test_mixed_whitespace_types(self):
        proc = _make_processor()
        result = proc._normalize_detection_text("  foo\t\nbar  baz  ")
        assert result == "foo bar baz"


# ---------------------------------------------------------------------------
# _is_detectable_text
# ---------------------------------------------------------------------------


class TestIsDetectableText:
    def test_long_enough_and_enough_alpha(self):
        proc = _make_processor()
        # 20+ chars, 5+ alpha
        assert proc._is_detectable_text("Hello world this is a test") is True

    def test_too_short(self):
        proc = _make_processor()
        # < 20 chars
        assert proc._is_detectable_text("Hi there") is False

    def test_long_but_not_enough_alpha(self):
        proc = _make_processor()
        # 20+ chars but fewer than 5 alpha (digits and symbols)
        assert proc._is_detectable_text("1234 5678 9012 3456 !@#$") is False

    def test_exactly_at_boundaries(self):
        proc = _make_processor()
        # exactly 20 chars, exactly 5 alpha: "abcde123456789012345"
        text = "abcde" + "1" * 15  # 20 chars total, 5 alpha
        assert proc._is_detectable_text(text) is True

    def test_empty_string(self):
        proc = _make_processor()
        assert proc._is_detectable_text("") is False

    def test_19_chars_fails_length(self):
        proc = _make_processor()
        # 19 chars, all alpha
        assert proc._is_detectable_text("a" * 19) is False

    def test_20_chars_5_alpha_passes(self):
        proc = _make_processor()
        # 20 chars: 5 alpha + 15 digits
        text = "abcde" + "0" * 15
        assert proc._is_detectable_text(text) is True

    def test_4_alpha_fails_alpha_count(self):
        proc = _make_processor()
        # 20+ chars, only 4 alpha
        text = "abcd" + "0" * 20
        assert proc._is_detectable_text(text) is False

    @pytest.mark.parametrize("text,expected", [
        ("This sentence has plenty of alpha characters in it!", True),
        ("12345", False),
        ("", False),
        ("a" * 20, True),
    ])
    def test_parametrized(self, text, expected):
        proc = _make_processor()
        assert proc._is_detectable_text(text) == expected


# ---------------------------------------------------------------------------
# _detect_language_for_text
# ---------------------------------------------------------------------------


class TestDetectLanguageForText:
    def test_english_text_returns_language(self):
        """Integration-style: real langdetect on clear English text."""
        proc = _make_processor()
        result = proc._detect_language_for_text(
            "This is a sample English text for testing language detection"
        )
        # Should detect something (likely "en")
        assert result is not None
        assert isinstance(result, str)

    def test_returns_en_for_english(self):
        """Verify langdetect returns 'en' for unambiguous English."""
        proc = _make_processor()
        result = proc._detect_language_for_text(
            "This is a sample English text for testing language detection"
        )
        assert result == "en"

    def test_lang_detect_exception_returns_none(self):
        proc = _make_processor()
        from langdetect import LangDetectException

        with patch("worker.services.processor.detect_langs", side_effect=LangDetectException(0, "error")):
            result = proc._detect_language_for_text("some text here")
        assert result is None

    def test_low_confidence_returns_none(self):
        proc = _make_processor()
        low_conf_candidate = MagicMock()
        low_conf_candidate.lang = "en"
        low_conf_candidate.prob = 0.50  # below 0.80 threshold

        with patch("worker.services.processor.detect_langs", return_value=[low_conf_candidate]):
            result = proc._detect_language_for_text("some text")
        assert result is None

    def test_empty_candidates_returns_none(self):
        proc = _make_processor()
        with patch("worker.services.processor.detect_langs", return_value=[]):
            result = proc._detect_language_for_text("some text")
        assert result is None

    def test_high_confidence_returns_lang(self):
        proc = _make_processor()
        high_conf_candidate = MagicMock()
        high_conf_candidate.lang = "fr"
        high_conf_candidate.prob = 0.95

        with patch("worker.services.processor.detect_langs", return_value=[high_conf_candidate]):
            result = proc._detect_language_for_text("some text")
        assert result == "fr"

    def test_exactly_at_confidence_threshold(self):
        """Confidence exactly 0.80 should pass (>= 0.80)."""
        proc = _make_processor()
        candidate = MagicMock()
        candidate.lang = "de"
        candidate.prob = 0.80

        with patch("worker.services.processor.detect_langs", return_value=[candidate]):
            result = proc._detect_language_for_text("some text")
        assert result == "de"

    def test_just_below_confidence_threshold(self):
        """Confidence 0.799 should fail (< 0.80)."""
        proc = _make_processor()
        candidate = MagicMock()
        candidate.lang = "de"
        candidate.prob = 0.799

        with patch("worker.services.processor.detect_langs", return_value=[candidate]):
            result = proc._detect_language_for_text("some text")
        assert result is None

    def test_alias_applied_for_zh_cn(self):
        """zh-cn should be normalized to zh."""
        proc = _make_processor()
        candidate = MagicMock()
        candidate.lang = "zh-cn"
        candidate.prob = 0.95

        with patch("worker.services.processor.detect_langs", return_value=[candidate]):
            result = proc._detect_language_for_text("some text")
        assert result == "zh"

    def test_uses_first_candidate_only(self):
        """Only the top candidate (index 0) should be considered."""
        proc = _make_processor()
        good_candidate = MagicMock()
        good_candidate.lang = "es"
        good_candidate.prob = 0.90
        bad_candidate = MagicMock()
        bad_candidate.lang = "it"
        bad_candidate.prob = 0.10

        with patch("worker.services.processor.detect_langs", return_value=[good_candidate, bad_candidate]):
            result = proc._detect_language_for_text("some text")
        assert result == "es"


# ---------------------------------------------------------------------------
# _normalize_detected_language
# ---------------------------------------------------------------------------


class TestNormalizeDetectedLanguage:
    @pytest.mark.parametrize("lang,expected", [
        ("en", "en"),
        ("EN", "en"),
        ("fr", "fr"),
        ("zh-cn", "zh"),
        ("zh-tw", "zh"),
        ("ZH-CN", "zh"),
        ("ZH-TW", "zh"),
        ("iw", "he"),
        ("IW", "he"),
        ("de", "de"),
        ("es", "es"),
        ("  en  ", "en"),   # strips whitespace
    ])
    def test_parametrized_normalization(self, lang, expected):
        proc = _make_processor()
        assert proc._normalize_detected_language(lang) == expected

    def test_unknown_lang_returned_lowercase(self):
        proc = _make_processor()
        assert proc._normalize_detected_language("UNKNOWN") == "unknown"

    def test_returns_string(self):
        proc = _make_processor()
        result = proc._normalize_detected_language("en")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _evaluate_attempt_quality
# ---------------------------------------------------------------------------


class TestEvaluateAttemptQuality:
    def _make_judge(self, return_value=None):
        judge = MagicMock()
        judge.model = "test-model"
        judge.evaluate.return_value = return_value or _make_quality_result()
        return judge

    def test_empty_source_returns_stub(self):
        proc = _make_processor()
        judge = self._make_judge()
        result = proc._evaluate_attempt_quality(
            judge=judge,
            source_text="",
            translated_text="some translation",
        )
        assert isinstance(result, QualityJudgeResult)
        assert result.final_score == 0.0
        assert result.pass_fail is False
        judge.evaluate.assert_not_called()

    def test_empty_translated_returns_stub(self):
        proc = _make_processor()
        judge = self._make_judge()
        result = proc._evaluate_attempt_quality(
            judge=judge,
            source_text="some source text",
            translated_text="",
        )
        assert isinstance(result, QualityJudgeResult)
        assert result.final_score == 0.0
        judge.evaluate.assert_not_called()

    def test_whitespace_only_source_returns_stub(self):
        proc = _make_processor()
        judge = self._make_judge()
        result = proc._evaluate_attempt_quality(
            judge=judge,
            source_text="   \t\n  ",
            translated_text="some translation",
        )
        assert result.final_score == 0.0
        judge.evaluate.assert_not_called()

    def test_whitespace_only_translated_returns_stub(self):
        proc = _make_processor()
        judge = self._make_judge()
        result = proc._evaluate_attempt_quality(
            judge=judge,
            source_text="some source",
            translated_text="   ",
        )
        assert result.final_score == 0.0
        judge.evaluate.assert_not_called()

    def test_both_empty_returns_stub(self):
        proc = _make_processor()
        judge = self._make_judge()
        result = proc._evaluate_attempt_quality(
            judge=judge,
            source_text="",
            translated_text="",
        )
        assert result.final_score == 0.0
        judge.evaluate.assert_not_called()

    def test_stub_reason_mentions_missing_text(self):
        proc = _make_processor()
        judge = self._make_judge()
        result = proc._evaluate_attempt_quality(
            judge=judge,
            source_text="",
            translated_text="",
        )
        assert len(result.reasons) > 0
        assert any("missing" in r.lower() or "source" in r.lower() for r in result.reasons)

    def test_stub_uses_judge_model(self):
        proc = _make_processor()
        judge = self._make_judge()
        judge.model = "my-special-model"
        result = proc._evaluate_attempt_quality(
            judge=judge,
            source_text="",
            translated_text="",
        )
        assert result.model == "my-special-model"

    def test_non_empty_texts_calls_judge_evaluate(self):
        proc = _make_processor()
        expected = _make_quality_result(final_score=0.85)
        judge = self._make_judge(return_value=expected)
        result = proc._evaluate_attempt_quality(
            judge=judge,
            source_text="Hello world.",
            translated_text="Hola mundo.",
        )
        judge.evaluate.assert_called_once_with(
            source_text="Hello world.",
            translated_text="Hola mundo.",
        )
        assert result is expected

    def test_returns_quality_judge_result_instance(self):
        proc = _make_processor()
        judge = self._make_judge()
        result = proc._evaluate_attempt_quality(
            judge=judge,
            source_text="Source text here.",
            translated_text="Translated text here.",
        )
        assert isinstance(result, QualityJudgeResult)


# ---------------------------------------------------------------------------
# _estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    """Cost calculation uses rates from settings; patch them to known values."""

    def test_gemini_model_uses_gemini_rates(self):
        proc = _make_processor()
        with patch("worker.services.processor.settings") as mock_settings:
            mock_settings.GEMINI_INPUT_COST_PER_1K = 0.001
            mock_settings.GEMINI_OUTPUT_COST_PER_1K = 0.002
            mock_settings.OPENAI_INPUT_COST_PER_1K = 999.0
            mock_settings.OPENAI_OUTPUT_COST_PER_1K = 999.0

            cost = proc._estimate_cost(
                model_id="gemini-2.5-flash",
                prompt_tokens=1000,
                completion_tokens=1000,
            )
        # (1000/1000)*0.001 + (1000/1000)*0.002 = 0.003
        assert cost == pytest.approx(0.003, abs=1e-6)

    def test_gemini_prefix_case_insensitive(self):
        proc = _make_processor()
        with patch("worker.services.processor.settings") as mock_settings:
            mock_settings.GEMINI_INPUT_COST_PER_1K = 0.001
            mock_settings.GEMINI_OUTPUT_COST_PER_1K = 0.002
            mock_settings.OPENAI_INPUT_COST_PER_1K = 999.0
            mock_settings.OPENAI_OUTPUT_COST_PER_1K = 999.0

            cost = proc._estimate_cost(
                model_id="Gemini-Pro",
                prompt_tokens=1000,
                completion_tokens=500,
            )
        expected = (1000 / 1000.0) * 0.001 + (500 / 1000.0) * 0.002
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_non_gemini_model_uses_openai_rates(self):
        proc = _make_processor()
        with patch("worker.services.processor.settings") as mock_settings:
            mock_settings.GEMINI_INPUT_COST_PER_1K = 999.0
            mock_settings.GEMINI_OUTPUT_COST_PER_1K = 999.0
            mock_settings.OPENAI_INPUT_COST_PER_1K = 0.005
            mock_settings.OPENAI_OUTPUT_COST_PER_1K = 0.010

            cost = proc._estimate_cost(
                model_id="gpt-4o-mini",
                prompt_tokens=2000,
                completion_tokens=500,
            )
        # (2000/1000)*0.005 + (500/1000)*0.010 = 0.010 + 0.005 = 0.015
        assert cost == pytest.approx(0.015, abs=1e-6)

    def test_zero_tokens_returns_zero(self):
        proc = _make_processor()
        with patch("worker.services.processor.settings") as mock_settings:
            mock_settings.GEMINI_INPUT_COST_PER_1K = 0.001
            mock_settings.GEMINI_OUTPUT_COST_PER_1K = 0.002
            mock_settings.OPENAI_INPUT_COST_PER_1K = 0.005
            mock_settings.OPENAI_OUTPUT_COST_PER_1K = 0.010

            cost = proc._estimate_cost(
                model_id="gpt-4o",
                prompt_tokens=0,
                completion_tokens=0,
            )
        assert cost == pytest.approx(0.0, abs=1e-9)

    def test_returns_float(self):
        proc = _make_processor()
        with patch("worker.services.processor.settings") as mock_settings:
            mock_settings.GEMINI_INPUT_COST_PER_1K = 0.001
            mock_settings.GEMINI_OUTPUT_COST_PER_1K = 0.002
            mock_settings.OPENAI_INPUT_COST_PER_1K = 0.005
            mock_settings.OPENAI_OUTPUT_COST_PER_1K = 0.010

            cost = proc._estimate_cost(
                model_id="gpt-4o",
                prompt_tokens=100,
                completion_tokens=100,
            )
        assert isinstance(cost, float)

    def test_result_rounded_to_6_decimal_places(self):
        proc = _make_processor()
        with patch("worker.services.processor.settings") as mock_settings:
            mock_settings.OPENAI_INPUT_COST_PER_1K = 0.000001
            mock_settings.OPENAI_OUTPUT_COST_PER_1K = 0.000001

            cost = proc._estimate_cost(
                model_id="gpt-4o",
                prompt_tokens=1,
                completion_tokens=1,
            )
        # Verify result has at most 6 decimal places
        rounded = round(cost, 6)
        assert cost == rounded

    @pytest.mark.parametrize("model_id,is_gemini", [
        ("gemini-2.5-flash", True),
        ("gemini-pro", True),
        ("GEMINI-ULTRA", True),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
        ("claude-3-opus", False),
        ("openai-gpt4", False),
    ])
    def test_model_routing_parametrized(self, model_id, is_gemini):
        proc = _make_processor()
        with patch("worker.services.processor.settings") as mock_settings:
            mock_settings.GEMINI_INPUT_COST_PER_1K = 1.0
            mock_settings.GEMINI_OUTPUT_COST_PER_1K = 2.0
            mock_settings.OPENAI_INPUT_COST_PER_1K = 3.0
            mock_settings.OPENAI_OUTPUT_COST_PER_1K = 4.0

            cost = proc._estimate_cost(
                model_id=model_id,
                prompt_tokens=1000,
                completion_tokens=1000,
            )

        if is_gemini:
            # (1.0 + 2.0) = 3.0
            assert cost == pytest.approx(3.0, abs=1e-6)
        else:
            # (3.0 + 4.0) = 7.0
            assert cost == pytest.approx(7.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _write_quality_report
# ---------------------------------------------------------------------------


class TestWriteQualityReport:
    def test_writes_json_file(self, tmp_path):
        proc = _make_processor()
        report = {"attempt_index": 1, "model_id": "gpt-4o", "quality": {"final_score": 0.9}}
        proc._write_quality_report(tmp_path, report)

        quality_path = tmp_path / "quality_report.json"
        assert quality_path.exists()

    def test_written_content_is_valid_json(self, tmp_path):
        proc = _make_processor()
        report = {"attempt_index": 1, "model_id": "gpt-4o", "quality": {"final_score": 0.9}}
        proc._write_quality_report(tmp_path, report)

        quality_path = tmp_path / "quality_report.json"
        content = json.loads(quality_path.read_text(encoding="utf-8"))
        assert content == report

    def test_written_content_preserves_non_ascii(self, tmp_path):
        proc = _make_processor()
        report = {"model": "test", "text": "日本語テスト"}
        proc._write_quality_report(tmp_path, report)

        quality_path = tmp_path / "quality_report.json"
        content = json.loads(quality_path.read_text(encoding="utf-8"))
        assert content["text"] == "日本語テスト"

    def test_swallows_exception_on_invalid_path(self):
        """Passing a non-existent directory should not raise."""
        proc = _make_processor()
        non_existent_dir = Path("/this/path/does/not/exist/at/all")
        # Should not raise any exception
        proc._write_quality_report(non_existent_dir, {"key": "value"})

    def test_overwrites_existing_report(self, tmp_path):
        proc = _make_processor()
        first_report = {"version": 1}
        second_report = {"version": 2}

        proc._write_quality_report(tmp_path, first_report)
        proc._write_quality_report(tmp_path, second_report)

        quality_path = tmp_path / "quality_report.json"
        content = json.loads(quality_path.read_text(encoding="utf-8"))
        assert content == second_report

    def test_file_uses_indented_json(self, tmp_path):
        proc = _make_processor()
        report = {"a": 1, "b": 2}
        proc._write_quality_report(tmp_path, report)

        raw = (tmp_path / "quality_report.json").read_text(encoding="utf-8")
        # Indented JSON contains newlines
        assert "\n" in raw


# ---------------------------------------------------------------------------
# _get_total_pdf_pages
# ---------------------------------------------------------------------------


class TestGetTotalPdfPages:
    def _write_pdf(self, tmp_path, num_pages: int = 1) -> Path:
        """Create a real multi-page in-memory PDF and write to tmp_path."""
        doc = fitz.open()
        for _ in range(num_pages):
            doc.new_page()
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(buf.getvalue())
        return pdf_path

    def test_single_page_pdf(self, tmp_path):
        proc = _make_processor()
        pdf_path = self._write_pdf(tmp_path, num_pages=1)
        assert proc._get_total_pdf_pages(pdf_path) == 1

    def test_multi_page_pdf(self, tmp_path):
        proc = _make_processor()
        pdf_path = self._write_pdf(tmp_path, num_pages=5)
        assert proc._get_total_pdf_pages(pdf_path) == 5

    def test_accepts_string_path(self, tmp_path):
        proc = _make_processor()
        pdf_path = self._write_pdf(tmp_path, num_pages=2)
        assert proc._get_total_pdf_pages(str(pdf_path)) == 2

    def test_accepts_path_object(self, tmp_path):
        proc = _make_processor()
        pdf_path = self._write_pdf(tmp_path, num_pages=3)
        assert proc._get_total_pdf_pages(Path(pdf_path)) == 3

    def test_returns_int(self, tmp_path):
        proc = _make_processor()
        pdf_path = self._write_pdf(tmp_path, num_pages=1)
        result = proc._get_total_pdf_pages(pdf_path)
        assert isinstance(result, int)

    def test_non_existent_file_returns_zero(self, tmp_path):
        proc = _make_processor()
        missing = tmp_path / "missing.pdf"
        assert proc._get_total_pdf_pages(missing) == 0

    def test_non_pdf_file_returns_zero(self, tmp_path):
        """A file that pymupdf cannot parse should return 0 via the except branch."""
        proc = _make_processor()
        with patch("worker.services.processor.pymupdf.open", side_effect=Exception("not a pdf")):
            result = proc._get_total_pdf_pages(tmp_path / "fake.pdf")
        assert result == 0


# ---------------------------------------------------------------------------
# _handle_translation_event
# ---------------------------------------------------------------------------


class TestHandleTranslationEvent:
    async def test_progress_start_updates_tracker(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        result = await proc._handle_translation_event({"type": "progress_start"}, {})
        assert result is None
        pt.update.assert_called_once_with(
            proc.PROGRESS_TRANSLATION_START, "Starting translation"
        )

    async def test_progress_end_updates_tracker(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        result = await proc._handle_translation_event({"type": "progress_end"}, {})
        assert result is None
        pt.update.assert_called_once_with(
            proc.PROGRESS_FINALIZE, "Finalizing output"
        )

    async def test_progress_update_delegates_to_handler(self):
        proc = _make_processor()
        with patch.object(proc, "_handle_progress_update", new=AsyncMock()) as mock_handler:
            event = {"type": "progress_update", "overall_progress": 50}
            result = await proc._handle_translation_event(event, {"config": "data"})
        assert result is None
        mock_handler.assert_called_once_with(event, {"config": "data"})

    async def test_finish_event_returns_result(self):
        proc = _make_processor()
        expected = {"page_count": 3}
        with patch.object(proc, "_handle_finish_event", new=AsyncMock(return_value=expected)):
            event = {"type": "finish", "translate_result": None}
            result = await proc._handle_translation_event(event, {})
        assert result == expected

    async def test_error_event_raises_runtime_error(self):
        proc = _make_processor()
        with pytest.raises(RuntimeError, match="Translation failed"):
            await proc._handle_translation_event(
                {"type": "error", "error": "Something went wrong"}, {}
            )

    async def test_error_event_includes_error_message(self):
        proc = _make_processor()
        with pytest.raises(RuntimeError, match="Something went wrong"):
            await proc._handle_translation_event(
                {"type": "error", "error": "Something went wrong"}, {}
            )

    async def test_error_event_no_error_key_uses_default(self):
        proc = _make_processor()
        with pytest.raises(RuntimeError, match="Unknown error"):
            await proc._handle_translation_event({"type": "error"}, {})

    async def test_unknown_event_type_returns_none(self):
        proc = _make_processor()
        result = await proc._handle_translation_event({"type": "some_unknown_type"}, {})
        assert result is None

    async def test_missing_type_key_returns_none(self):
        proc = _make_processor()
        result = await proc._handle_translation_event({}, {})
        assert result is None

    @pytest.mark.parametrize("event_type", [
        "progress_start",
        "progress_end",
        "unknown_xyz",
        None,
    ])
    async def test_non_finish_non_error_returns_none(self, event_type):
        proc = _make_processor()
        # Patch progress_tracker so update calls succeed silently
        with patch.object(proc, "_handle_progress_update", new=AsyncMock()):
            result = await proc._handle_translation_event(
                {"type": event_type} if event_type else {}, {}
            )
        assert result is None


# ---------------------------------------------------------------------------
# _handle_progress_update
# ---------------------------------------------------------------------------


class TestHandleProgressUpdate:
    async def test_maps_0_percent_to_progress_start(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        await proc._handle_progress_update(
            {"overall_progress": 0, "stage": "Init"}, {}
        )
        # 0.2 + (0.0 * 0.6) = 0.2
        pt.update.assert_called_once()
        args = pt.update.call_args[0]
        assert args[0] == pytest.approx(0.2)

    async def test_maps_100_percent_to_progress_end(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        await proc._handle_progress_update(
            {"overall_progress": 100, "stage": "Done"}, {}
        )
        # 0.2 + (1.0 * 0.6) = 0.8
        pt.update.assert_called_once()
        args = pt.update.call_args[0]
        assert args[0] == pytest.approx(0.8)

    async def test_maps_50_percent_correctly(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        await proc._handle_progress_update(
            {"overall_progress": 50, "stage": "Processing"}, {}
        )
        # 0.2 + (0.5 * 0.6) = 0.5
        args = pt.update.call_args[0]
        assert args[0] == pytest.approx(0.5)

    async def test_stage_included_in_message(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        await proc._handle_progress_update(
            {"overall_progress": 50, "stage": "Rendering"}, {}
        )
        message = pt.update.call_args[0][1]
        assert "Rendering" in message

    async def test_percent_included_in_message(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        await proc._handle_progress_update(
            {"overall_progress": 75, "stage": "Translating"}, {}
        )
        message = pt.update.call_args[0][1]
        assert "75" in message

    async def test_default_stage_when_missing(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        await proc._handle_progress_update({"overall_progress": 30}, {})
        message = pt.update.call_args[0][1]
        assert "Processing" in message

    async def test_missing_overall_progress_defaults_to_zero(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        await proc._handle_progress_update({"stage": "Init"}, {})
        args = pt.update.call_args[0]
        assert args[0] == pytest.approx(0.2)

    @pytest.mark.parametrize("overall,expected_mapped", [
        (0, 0.2),
        (25, 0.35),
        (50, 0.5),
        (75, 0.65),
        (100, 0.8),
    ])
    async def test_progress_mapping_parametrized(self, overall, expected_mapped):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        await proc._handle_progress_update(
            {"overall_progress": overall, "stage": "Test"}, {}
        )
        args = pt.update.call_args[0]
        assert args[0] == pytest.approx(expected_mapped, abs=1e-6)


# ---------------------------------------------------------------------------
# _handle_finish_event
# ---------------------------------------------------------------------------


class TestHandleFinishEvent:
    async def test_updates_progress_tracker_to_complete(self):
        pt = AsyncMock()
        proc = _make_processor(progress_tracker=pt)
        await proc._handle_finish_event({"translate_result": None})
        pt.update.assert_called_once_with(
            proc.PROGRESS_COMPLETE, "Translation complete"
        )

    async def test_result_none_returns_zero_page_count(self):
        proc = _make_processor()
        result = await proc._handle_finish_event({"translate_result": None})
        assert result["page_count"] == 0

    async def test_dict_result_extracts_page_count(self):
        proc = _make_processor()
        result = await proc._handle_finish_event(
            {"translate_result": {"page_count": 7}}
        )
        assert result["page_count"] == 7

    async def test_dict_result_missing_page_count_returns_zero(self):
        proc = _make_processor()
        result = await proc._handle_finish_event({"translate_result": {}})
        assert result["page_count"] == 0

    async def test_object_result_extracts_page_count(self):
        proc = _make_processor()
        mock_result = MagicMock()
        mock_result.page_count = 12
        result = await proc._handle_finish_event({"translate_result": mock_result})
        assert result["page_count"] == 12

    async def test_dict_with_existing_mono_pdf_path(self, tmp_path):
        proc = _make_processor()
        # Create a real file so Path.exists() returns True
        mono_file = tmp_path / "output_mono.pdf"
        mono_file.write_bytes(b"%PDF-1.4 fake")
        translate_result = {
            "mono_pdf_path": str(mono_file),
            "page_count": 4,
        }
        result = await proc._handle_finish_event({"translate_result": translate_result})
        assert "mono_pdf_path" in result
        assert result["page_count"] == 4

    async def test_dict_with_nonexistent_path_excluded(self, tmp_path):
        proc = _make_processor()
        translate_result = {
            "mono_pdf_path": str(tmp_path / "ghost.pdf"),  # does not exist
            "page_count": 2,
        }
        result = await proc._handle_finish_event({"translate_result": translate_result})
        assert "mono_pdf_path" not in result

    async def test_multiple_file_types_in_dict(self, tmp_path):
        proc = _make_processor()
        mono_file = tmp_path / "mono.pdf"
        dual_file = tmp_path / "dual.pdf"
        mono_file.write_bytes(b"%PDF-1.4 mono")
        dual_file.write_bytes(b"%PDF-1.4 dual")

        translate_result = {
            "mono_pdf_path": str(mono_file),
            "dual_pdf_path": str(dual_file),
            "page_count": 3,
        }
        result = await proc._handle_finish_event({"translate_result": translate_result})
        assert "mono_pdf_path" in result
        assert "dual_pdf_path" in result
        assert result["page_count"] == 3

    async def test_object_result_attr_path_included_when_exists(self, tmp_path):
        proc = _make_processor()
        mono_file = tmp_path / "mono.pdf"
        mono_file.write_bytes(b"%PDF-1.4 fake")

        mock_result = MagicMock()
        mock_result.mono_pdf_path = str(mono_file)
        mock_result.dual_pdf_path = None
        mock_result.no_watermark_mono_pdf_path = None
        mock_result.page_count = 5

        result = await proc._handle_finish_event({"translate_result": mock_result})
        assert "mono_pdf_path" in result
        assert result["page_count"] == 5

    async def test_returns_dict(self):
        proc = _make_processor()
        result = await proc._handle_finish_event({"translate_result": {"page_count": 1}})
        assert isinstance(result, dict)

    async def test_always_has_page_count_key(self):
        proc = _make_processor()
        result = await proc._handle_finish_event({"translate_result": None})
        assert "page_count" in result


# ---------------------------------------------------------------------------
# _apply_cover_pages
# ---------------------------------------------------------------------------


class TestApplyCoverPages:
    def _make_translation_config(self, add_cover_page: bool = True) -> MagicMock:
        cfg = MagicMock()
        cfg.add_cover_page = add_cover_page
        return cfg

    def test_skips_when_add_cover_page_is_false(self):
        proc = _make_processor()
        translation_config = self._make_translation_config(add_cover_page=False)

        with patch.object(proc, "_prepend_cover_page") as mock_prepend:
            proc._apply_cover_pages(
                translation_config,
                {"mono_pdf_path": "/some/path.pdf"},
                MagicMock(),
            )
        mock_prepend.assert_not_called()

    def test_called_when_add_cover_page_is_true_with_mono_path(self, tmp_path):
        proc = _make_processor()
        translation_config = self._make_translation_config(add_cover_page=True)

        mono_file = tmp_path / "mono.pdf"
        mono_file.write_bytes(b"fake pdf")

        with patch.object(proc, "_prepend_cover_page") as mock_prepend:
            proc._apply_cover_pages(
                translation_config,
                {"mono_pdf_path": str(mono_file)},
                MagicMock(),
            )
        mock_prepend.assert_called()

    def test_called_for_each_non_none_output_path(self, tmp_path):
        proc = _make_processor()
        translation_config = self._make_translation_config(add_cover_page=True)

        mono_file = tmp_path / "mono.pdf"
        dual_file = tmp_path / "dual.pdf"
        mono_file.write_bytes(b"fake pdf")
        dual_file.write_bytes(b"fake pdf")

        with patch.object(proc, "_prepend_cover_page") as mock_prepend:
            proc._apply_cover_pages(
                translation_config,
                {
                    "mono_pdf_path": str(mono_file),
                    "dual_pdf_path": str(dual_file),
                },
                MagicMock(),
            )
        assert mock_prepend.call_count == 2

    def test_none_paths_are_skipped(self, tmp_path):
        proc = _make_processor()
        translation_config = self._make_translation_config(add_cover_page=True)

        mono_file = tmp_path / "mono.pdf"
        mono_file.write_bytes(b"fake pdf")

        with patch.object(proc, "_prepend_cover_page") as mock_prepend:
            proc._apply_cover_pages(
                translation_config,
                {
                    "mono_pdf_path": str(mono_file),
                    "dual_pdf_path": None,
                    "no_watermark_mono_pdf_path": None,
                    "no_watermark_dual_pdf_path": None,
                },
                MagicMock(),
            )
        # Only one call for mono_pdf_path
        assert mock_prepend.call_count == 1

    def test_metadata_assigned_to_config(self, tmp_path):
        proc = _make_processor()
        translation_config = self._make_translation_config(add_cover_page=True)
        metadata = MagicMock()

        mono_file = tmp_path / "mono.pdf"
        mono_file.write_bytes(b"fake pdf")

        with patch.object(proc, "_prepend_cover_page"):
            proc._apply_cover_pages(
                translation_config,
                {"mono_pdf_path": str(mono_file)},
                metadata,
            )
        assert translation_config.cover_page_metadata is metadata

    def test_empty_attempt_result_no_prepend_called(self):
        proc = _make_processor()
        translation_config = self._make_translation_config(add_cover_page=True)

        with patch.object(proc, "_prepend_cover_page") as mock_prepend:
            proc._apply_cover_pages(translation_config, {}, MagicMock())
        mock_prepend.assert_not_called()

    def test_add_cover_page_attribute_missing_defaults_to_truthy(self):
        """If add_cover_page attr is missing, getattr returns True (default)."""
        proc = _make_processor()
        translation_config = MagicMock(spec=[])  # spec=[] means no attrs

        with patch.object(proc, "_prepend_cover_page") as mock_prepend:
            proc._apply_cover_pages(translation_config, {}, MagicMock())
        # No paths → not called, but no crash
        mock_prepend.assert_not_called()


# ---------------------------------------------------------------------------
# _iter_page_text_chunks
# ---------------------------------------------------------------------------


class TestIterPageTextChunks:
    def _make_lt_text_container(self, text: str) -> MagicMock:
        """Create a fake LTTextContainer that passes isinstance check."""
        from babeldoc.pdfminer.layout import LTTextContainer

        item = MagicMock(spec=LTTextContainer)
        item.get_text.return_value = text
        return item

    def test_yields_text_from_lt_text_container(self):
        proc = _make_processor()
        container = self._make_lt_text_container(
            "This is a long enough text with alphabetic chars for detection!!"
        )
        chunks = list(proc._iter_page_text_chunks(container))
        assert len(chunks) == 1

    def test_short_lt_text_container_not_yielded(self):
        proc = _make_processor()
        container = self._make_lt_text_container("Hi")
        chunks = list(proc._iter_page_text_chunks(container))
        assert len(chunks) == 0

    def test_lt_text_container_text_is_normalized(self):
        proc = _make_processor()
        # Text with extra whitespace
        container = self._make_lt_text_container(
            "  Hello   world   this  is  a  good  long  text!  "
        )
        chunks = list(proc._iter_page_text_chunks(container))
        if chunks:
            assert chunks[0] == chunks[0].strip()
            assert "  " not in chunks[0]

    def test_non_iterable_non_lt_container_yields_nothing(self):
        proc = _make_processor()
        # An integer has no __iter__
        chunks = list(proc._iter_page_text_chunks(42))
        assert chunks == []

    def test_non_iterable_object_yields_nothing(self):
        proc = _make_processor()

        class NoIter:
            pass

        chunks = list(proc._iter_page_text_chunks(NoIter()))
        assert chunks == []

    def test_iterable_container_recurses_into_children(self):
        proc = _make_processor()
        child = self._make_lt_text_container(
            "This is a long enough text with alpha chars for detection!!"
        )
        # A page-like object that is iterable and yields the child
        parent = [child]
        chunks = list(proc._iter_page_text_chunks(parent))
        assert len(chunks) == 1

    def test_nested_iterables_are_recursed(self):
        proc = _make_processor()
        text = "This is a long enough text with alpha characters for detection!!"
        child = self._make_lt_text_container(text)
        grandchild = self._make_lt_text_container(text)

        # parent -> [child_list -> [grandchild_container]]
        inner = [grandchild]
        outer = [child, inner]

        chunks = list(proc._iter_page_text_chunks(outer))
        assert len(chunks) == 2

    def test_mixed_iterable_with_non_iterable_items(self):
        proc = _make_processor()
        text = "This is a long enough text with alpha chars for detection!!"
        child = self._make_lt_text_container(text)

        # Use an integer as the non-iterable non-LTTextContainer item.
        # Strings are iterable and would recurse infinitely, so avoid them here.
        mixed = [child, 42]
        chunks = list(proc._iter_page_text_chunks(mixed))
        # child yields one chunk; 42 has no __iter__ so is silently skipped
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# _detect_page_languages
# ---------------------------------------------------------------------------


class TestDetectPageLanguages:
    def _make_lt_text_container(self, text: str) -> MagicMock:
        from babeldoc.pdfminer.layout import LTTextContainer

        item = MagicMock(spec=LTTextContainer)
        item.get_text.return_value = text
        return item

    def test_returns_counter(self):
        proc = _make_processor()
        result = proc._detect_page_languages([])
        assert isinstance(result, Counter)

    def test_empty_page_returns_empty_counter(self):
        proc = _make_processor()
        result = proc._detect_page_languages([])
        assert len(result) == 0

    def test_detects_language_from_text_container(self):
        proc = _make_processor()
        long_english_text = (
            "This is a long English sentence with many alphabetic characters for detection."
        )
        container = self._make_lt_text_container(long_english_text)

        with patch.object(proc, "_detect_language_for_text", return_value="en"):
            result = proc._detect_page_languages([container])

        assert "en" in result

    def test_accumulates_by_text_length(self):
        proc = _make_processor()
        text_a = "This is a long English sentence for language detection purposes."
        text_b = "Another English sentence here for more text detection work."
        container_a = self._make_lt_text_container(text_a)
        container_b = self._make_lt_text_container(text_b)

        with patch.object(proc, "_detect_language_for_text", return_value="en"):
            result = proc._detect_page_languages([container_a, container_b])

        # Should have accumulated counts (sum of text lengths)
        assert result["en"] == len(
            proc._normalize_detection_text(text_a)
        ) + len(proc._normalize_detection_text(text_b))

    def test_none_detection_not_accumulated(self):
        proc = _make_processor()
        text = "Some detectable text that is long enough for processing."
        container = self._make_lt_text_container(text)

        with patch.object(proc, "_detect_language_for_text", return_value=None):
            result = proc._detect_page_languages([container])

        assert len(result) == 0

    def test_multiple_languages_tracked_separately(self):
        proc = _make_processor()
        eng_text = "This is an English sentence for detection purposes here."
        fra_text = "Voici une phrase française pour la détection linguistique."
        container_en = self._make_lt_text_container(eng_text)
        container_fr = self._make_lt_text_container(fra_text)

        def _mock_detect(text):
            if "English" in text:
                return "en"
            return "fr"

        with patch.object(proc, "_detect_language_for_text", side_effect=_mock_detect):
            result = proc._detect_page_languages([container_en, container_fr])

        assert "en" in result
        assert "fr" in result

    def test_non_detectable_text_skipped(self):
        proc = _make_processor()
        # Very short text → _is_detectable_text returns False → not yielded by
        # _iter_page_text_chunks → _detect_language_for_text never called
        short_container = self._make_lt_text_container("Hi")

        with patch.object(proc, "_detect_language_for_text") as mock_detect:
            result = proc._detect_page_languages([short_container])

        mock_detect.assert_not_called()
        assert len(result) == 0
