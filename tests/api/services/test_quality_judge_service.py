import pytest
from src.api.services.quality_judge_service import GoogleADKJudgeAgent, QualityJudgeResult, _compute_alignment_score, _split_sentences
from unittest.mock import MagicMock, patch

class TestQualityJudgeService:
    def test_split_sentences(self):
        assert _split_sentences("Hello. World!") == ["Hello", "World"]
        assert _split_sentences("你好。世界！") == ["你好", "世界"]

    def test_compute_alignment_score(self):
        assert _compute_alignment_score("A. B.", "C. D.") == 1.0
        assert _compute_alignment_score("A.", "C. D.") == 0.5

    @patch("src.api.services.quality_judge_service.genai.Client")
    def test_judge_init(self, mock_client):
        agent = GoogleADKJudgeAgent(model="m1")
        assert agent.model == "m1"
        mock_client.assert_called_once()

    @patch.object(GoogleADKJudgeAgent, "_judge_with_llm")
    def test_evaluate_success(self, mock_judge, mock_settings):
        mock_judge.return_value = {
            "alignment_score": 1.0, "omission_score": 1.0, "hallucination_score": 1.0, "reasons": []
        }
        agent = GoogleADKJudgeAgent()
        res = agent.evaluate(source_text="s", translated_text="t")
        assert isinstance(res, QualityJudgeResult)
        assert res.final_score == pytest.approx(1.0)

@pytest.fixture
def mock_settings():
    with patch("src.api.services.quality_judge_service.settings") as s:
        s.QUALITY_THRESHOLD = 0.8
        s.JUDGE_MODEL = "gemini-pro"
        yield s
