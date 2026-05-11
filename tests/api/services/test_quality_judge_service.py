import pytest
from src.api.services.quality_judge_service import (
    GoogleADKJudgeAgent, QualityJudgeResult, QualityJudgeLLMScores,
    _compute_alignment_score, _split_sentences, extract_attempt_text
)
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json

class TestQualityJudgeService:
    def test_split_sentences(self):
        assert _split_sentences("Hello. World!") == ["Hello", "World"]
        assert _split_sentences("你好。世界！") == ["你好", "世界"]
        assert _split_sentences("") == []

    def test_compute_alignment_score(self):
        assert _compute_alignment_score("A. B.", "C. D.") == 1.0
        assert _compute_alignment_score("A.", "C. D.") == 0.5
        assert _compute_alignment_score("", "") == 1.0

    @patch("src.api.services.quality_judge_service.genai.Client")
    def test_judge_init(self, mock_client):
        agent = GoogleADKJudgeAgent(model="m1")
        assert agent.model == "m1"
        mock_client.assert_called_once()

    @patch.object(GoogleADKJudgeAgent, "_judge_with_llm")
    def test_evaluate_success(self, mock_judge, mock_settings):
        mock_judge.return_value = {
            "alignment_score": 1.0, "omission_score": 1.0, "hallucination_score": 1.0, "reasons": ["R1"]
        }
        agent = GoogleADKJudgeAgent()
        res = agent.evaluate(source_text="s", translated_text="t")
        assert isinstance(res, QualityJudgeResult)
        assert res.final_score == pytest.approx(1.0)
        assert res.pass_fail is True
        assert res.reasons == ["R1"]

    def test_evaluate_async(self, mock_settings):
        agent = GoogleADKJudgeAgent()
        with patch.object(agent, "evaluate", return_value=MagicMock()) as mock_eval:
            import asyncio
            asyncio.run(agent.evaluate_async(source_text="s", translated_text="t"))
            mock_eval.assert_called_once()

    @patch("src.api.services.quality_judge_service.genai")
    def test_judge_with_llm_client_none_fallback(self, mock_genai):
        with patch("src.api.services.quality_judge_service.genai", None):
            agent = GoogleADKJudgeAgent()
            res = agent._judge_with_llm("s", "t")
            assert isinstance(res, QualityJudgeLLMScores)
            assert "heuristic" in res.reasons[0]

    def test_extract_attempt_text_success(self, tmp_path):
        working_dir = tmp_path
        tracking_file = working_dir / "translate_tracking.json"
        tracking_data = {
            "page": [
                {
                    "paragraph": [
                        {"input": "Hello", "output": "Bonjour"},
                        {"pdf_unicode": "World", "output": "Monde"}
                    ]
                }
            ]
        }
        tracking_file.write_text(json.dumps(tracking_data))
        
        src, tgt = extract_attempt_text(working_dir)
        assert src == "Hello\nWorld"
        assert tgt == "Bonjour\nMonde"

    def test_extract_attempt_text_not_exists(self):
        src, tgt = extract_attempt_text(Path("/nonexistent"))
        assert src == ""
        assert tgt == ""

    def test_extract_attempt_text_invalid_json(self, tmp_path):
        working_dir = tmp_path
        tracking_file = working_dir / "translate_tracking.json"
        tracking_file.write_text("invalid json")
        src, tgt = extract_attempt_text(working_dir)
        assert src == ""
        assert tgt == ""

@pytest.fixture
def mock_settings():
    with patch("src.api.services.quality_judge_service.settings") as s:
        s.QUALITY_THRESHOLD = 0.8
        s.JUDGE_MODEL = "gemini-pro"
        yield s
