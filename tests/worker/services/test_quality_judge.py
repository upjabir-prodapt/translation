"""
Unit tests for worker/services/quality_judge.py.

Covers:
  - _split_sentences helper
  - _compute_alignment_score helper
  - QualityJudgeResult dataclass
  - GoogleADKJudgeAgent.evaluate (LLM client present / absent)
  - GoogleADKJudgeAgent._judge_with_llm (successful parse / parse failure)
  - extract_attempt_text (valid tracking file / missing file / malformed JSON)
"""

import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fixtures.sample_data import TRACKING_JSON_EMPTY
from fixtures.sample_data import TRACKING_JSON_VALID

from worker.services.quality_judge import GoogleADKJudgeAgent
from worker.services.quality_judge import QualityJudgeResult
from worker.services.quality_judge import _compute_alignment_score
from worker.services.quality_judge import _split_sentences
from worker.services.quality_judge import extract_attempt_text

# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_splits_on_period(self):
        parts = _split_sentences("Hello world. Goodbye world.")
        assert len(parts) == 2

    def test_splits_on_exclamation(self):
        parts = _split_sentences("Wow! Amazing!")
        assert len(parts) == 2

    def test_splits_on_question_mark(self):
        parts = _split_sentences("How are you? I am fine.")
        assert len(parts) == 2

    def test_splits_on_cjk_punctuation(self):
        parts = _split_sentences("你好。再见。")
        assert len(parts) == 2

    def test_empty_string_returns_empty_list(self):
        parts = _split_sentences("")
        assert parts == []

    def test_single_sentence_no_punct_returns_one(self):
        parts = _split_sentences("no punctuation here")
        assert len(parts) == 1

    def test_strips_whitespace_from_parts(self):
        parts = _split_sentences("  Hello.  World.  ")
        assert all(p == p.strip() for p in parts)

    def test_consecutive_punctuation_not_empty_parts(self):
        parts = _split_sentences("Wow!!! Great.")
        # Should not have empty strings
        assert all(len(p) > 0 for p in parts)


# ---------------------------------------------------------------------------
# _compute_alignment_score
# ---------------------------------------------------------------------------


class TestComputeAlignmentScore:
    def test_identical_sentence_count_is_one(self):
        src = "Hello. World. Foo."
        tgt = "Hola. Mundo. Foo."
        score = _compute_alignment_score(src, tgt)
        assert score == pytest.approx(1.0)

    def test_score_bounded_zero_to_one(self):
        score = _compute_alignment_score("One sentence.", "A. B. C. D. E.")
        assert 0.0 <= score <= 1.0

    def test_empty_source_returns_valid_score(self):
        score = _compute_alignment_score("", "Something here.")
        # max(0, min(1, ratio)) → ratio = min(1,1)/max(1,1) = 1.0 (both treated as 1)
        assert 0.0 <= score <= 1.0

    def test_empty_both_returns_one(self):
        # Both treated as 1 sentence
        score = _compute_alignment_score("", "")
        assert score == pytest.approx(1.0)

    def test_target_has_more_sentences(self):
        src = "One."
        tgt = "One. Two. Three."
        score = _compute_alignment_score(src, tgt)
        assert score == pytest.approx(1 / 3)

    def test_source_has_more_sentences(self):
        src = "One. Two. Three."
        tgt = "One."
        score = _compute_alignment_score(src, tgt)
        assert score == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# QualityJudgeResult
# ---------------------------------------------------------------------------


class TestQualityJudgeResult:
    def test_creation(self):
        result = QualityJudgeResult(
            alignment_score=0.9,
            omission_score=0.85,
            hallucination_score=0.8,
            final_score=0.855,
            pass_fail=True,
            reasons=["Good alignment"],
            model="gemini-2.5-flash",
        )
        assert result.alignment_score == 0.9
        assert result.pass_fail is True

    def test_to_dict_returns_dict(self):
        result = QualityJudgeResult(
            alignment_score=0.9,
            omission_score=0.85,
            hallucination_score=0.8,
            final_score=0.855,
            pass_fail=True,
            reasons=[],
            model="gemini-2.5-flash",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["alignment_score"] == 0.9
        assert d["model"] == "gemini-2.5-flash"

    def test_to_dict_includes_all_fields(self):
        result = QualityJudgeResult(
            alignment_score=0.5,
            omission_score=0.6,
            hallucination_score=0.7,
            final_score=0.6,
            pass_fail=False,
            reasons=["Missing paragraphs"],
            model="test-model",
        )
        d = result.to_dict()
        expected_keys = {
            "alignment_score",
            "omission_score",
            "hallucination_score",
            "final_score",
            "pass_fail",
            "reasons",
            "model",
        }
        assert expected_keys == set(d.keys())


# ---------------------------------------------------------------------------
# GoogleADKJudgeAgent — evaluate (no LLM client)
# ---------------------------------------------------------------------------


class TestGoogleADKJudgeAgentNoLLM:
    """When genai is unavailable, the agent falls back to heuristics."""

    def _make_agent_no_client(self) -> GoogleADKJudgeAgent:
        """Construct judge with LLM client forcibly set to None."""
        with (
            patch("worker.services.quality_judge.genai", None),
            patch("worker.services.quality_judge.Agent", None),
        ):
            agent = GoogleADKJudgeAgent(model="test-model")
        agent._client = None  # ensure client is None
        return agent

    def test_returns_quality_judge_result(self):
        agent = self._make_agent_no_client()
        result = agent.evaluate(
            source_text="Hello. How are you?",
            translated_text="Hola. ¿Cómo estás?",
        )
        assert isinstance(result, QualityJudgeResult)

    def test_final_score_in_range(self):
        agent = self._make_agent_no_client()
        result = agent.evaluate(
            source_text="One sentence.",
            translated_text="Una oración.",
        )
        assert 0.0 <= result.final_score <= 1.0

    def test_fallback_reason_present(self):
        agent = self._make_agent_no_client()
        result = agent.evaluate(
            source_text="Hello.",
            translated_text="Hola.",
        )
        assert any(
            "unavailable" in r.lower() or "heuristic" in r.lower()
            for r in result.reasons
        )

    def test_pass_fail_determined_by_threshold(self):
        from config.constants import settings

        agent = self._make_agent_no_client()
        result = agent.evaluate(
            source_text="Hello. World. Test.",
            translated_text="Hola. Mundo. Prueba.",
        )
        assert result.pass_fail == (result.final_score >= settings.QUALITY_THRESHOLD)

    def test_scores_clamped_to_valid_range(self):
        agent = self._make_agent_no_client()
        result = agent.evaluate(source_text="", translated_text="")
        assert 0.0 <= result.alignment_score <= 1.0
        assert 0.0 <= result.omission_score <= 1.0
        assert 0.0 <= result.hallucination_score <= 1.0

    def test_final_score_formula(self):
        """final = 0.4 * alignment + 0.35 * omission + 0.25 * hallucination"""
        agent = self._make_agent_no_client()
        result = agent.evaluate(
            source_text="A. B. C.",
            translated_text="X. Y. Z.",
        )
        expected = (
            0.4 * result.alignment_score
            + 0.35 * result.omission_score
            + 0.25 * result.hallucination_score
        )
        assert result.final_score == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# GoogleADKJudgeAgent — evaluate (with mocked LLM client)
# ---------------------------------------------------------------------------


class TestGoogleADKJudgeAgentWithLLM:
    def _make_agent_with_mock_llm(self, llm_response: dict) -> GoogleADKJudgeAgent:
        """Build agent with a mocked genai client returning specific scores."""
        with (
            patch("worker.services.quality_judge.genai", None),
            patch("worker.services.quality_judge.Agent", None),
        ):
            agent = GoogleADKJudgeAgent(model="gemini-test")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(llm_response)
        mock_client.models.generate_content.return_value = mock_response
        agent._client = mock_client
        return agent

    def test_uses_llm_scores(self):
        agent = self._make_agent_with_mock_llm(
            {"omission_score": 0.9, "hallucination_score": 0.95, "reasons": ["Good"]}
        )
        result = agent.evaluate(
            source_text="Hello. World.",
            translated_text="Hola. Mundo.",
        )
        assert result.omission_score == pytest.approx(0.9)
        assert result.hallucination_score == pytest.approx(0.95)

    def test_reasons_populated_from_llm(self):
        agent = self._make_agent_with_mock_llm(
            {
                "omission_score": 0.8,
                "hallucination_score": 0.85,
                "reasons": ["No omissions"],
            }
        )
        result = agent.evaluate(source_text="Hi.", translated_text="Hola.")
        assert "No omissions" in result.reasons

    def test_scores_clamped_from_llm(self):
        """LLM scores out of range [0,1] should be clamped."""
        agent = self._make_agent_with_mock_llm(
            {"omission_score": 2.5, "hallucination_score": -0.3, "reasons": []}
        )
        result = agent.evaluate(source_text="Hi.", translated_text="Hola.")
        assert 0.0 <= result.omission_score <= 1.0
        assert 0.0 <= result.hallucination_score <= 1.0

    def test_parse_failure_falls_back_to_heuristic(self):
        """If LLM returns unparseable JSON, fallback heuristic is used."""
        with (
            patch("worker.services.quality_judge.genai", None),
            patch("worker.services.quality_judge.Agent", None),
        ):
            agent = GoogleADKJudgeAgent(model="gemini-test")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "INVALID JSON {"
        mock_client.models.generate_content.return_value = mock_response
        agent._client = mock_client

        result = agent.evaluate(source_text="Hello.", translated_text="Hola.")
        assert isinstance(result, QualityJudgeResult)
        assert any(
            "fallback" in r.lower() or "parse" in r.lower() for r in result.reasons
        )


# ---------------------------------------------------------------------------
# extract_attempt_text
# ---------------------------------------------------------------------------


class TestExtractAttemptText:
    def test_extracts_source_and_translated(self, tmp_path):
        tracking_file = tmp_path / "translate_tracking.json"
        tracking_file.write_text(json.dumps(TRACKING_JSON_VALID), encoding="utf-8")
        source, translated = extract_attempt_text(tmp_path)
        assert "Hello world" in source
        assert "Hola mundo" in translated

    def test_missing_tracking_file_returns_empty(self, tmp_path):
        source, translated = extract_attempt_text(tmp_path)
        assert source == ""
        assert translated == ""

    def test_empty_pages_returns_empty_strings(self, tmp_path):
        tracking_file = tmp_path / "translate_tracking.json"
        tracking_file.write_text(json.dumps(TRACKING_JSON_EMPTY), encoding="utf-8")
        source, translated = extract_attempt_text(tmp_path)
        assert source == ""
        assert translated == ""

    def test_malformed_json_returns_empty(self, tmp_path):
        tracking_file = tmp_path / "translate_tracking.json"
        tracking_file.write_text("{ invalid json }", encoding="utf-8")
        source, translated = extract_attempt_text(tmp_path)
        assert source == ""
        assert translated == ""

    def test_uses_pdf_unicode_fallback(self, tmp_path):
        """If 'input' key is missing, 'pdf_unicode' should be used."""
        data = {
            "page": [
                {
                    "paragraph": [
                        {"pdf_unicode": "Source text.", "output": "Texto fuente."}
                    ]
                }
            ]
        }
        tracking_file = tmp_path / "translate_tracking.json"
        tracking_file.write_text(json.dumps(data), encoding="utf-8")
        source, translated = extract_attempt_text(tmp_path)
        assert "Source text." in source

    def test_missing_output_skipped(self, tmp_path):
        """Paragraphs without 'output' contribute only to source."""
        data = {
            "page": [
                {
                    "paragraph": [
                        {"input": "Source sentence.", "output": ""},
                    ]
                }
            ]
        }
        tracking_file = tmp_path / "translate_tracking.json"
        tracking_file.write_text(json.dumps(data), encoding="utf-8")
        source, translated = extract_attempt_text(tmp_path)
        assert "Source sentence." in source
        assert translated == ""

    def test_multiple_pages_concatenated(self, tmp_path):
        data = {
            "page": [
                {"paragraph": [{"input": "Page one.", "output": "Página uno."}]},
                {"paragraph": [{"input": "Page two.", "output": "Página dos."}]},
            ]
        }
        tracking_file = tmp_path / "translate_tracking.json"
        tracking_file.write_text(json.dumps(data), encoding="utf-8")
        source, translated = extract_attempt_text(tmp_path)
        assert "Page one." in source and "Page two." in source
        assert "Página uno." in translated and "Página dos." in translated
