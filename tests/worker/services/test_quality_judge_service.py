import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.services.quality_judge_service import GoogleADKJudgeAgent
from src.worker.services.quality_judge_service import QualityJudgeLLMScores
from src.worker.services.quality_judge_service import QualityJudgeResult
from src.worker.services.quality_judge_service import _compute_alignment_score
from src.worker.services.quality_judge_service import _split_sentences
from src.worker.services.quality_judge_service import extract_attempt_text


class TestQualityJudgeService:
    def test_split_sentences(self):
        assert _split_sentences("Hello. World!") == ["Hello", "World"]
        assert _split_sentences("你好。世界！") == ["你好", "世界"]
        assert _split_sentences("") == []

    def test_compute_alignment_score(self):
        assert _compute_alignment_score("A. B.", "C. D.") == 1.0
        assert _compute_alignment_score("A.", "C. D.") == 0.5
        assert _compute_alignment_score("", "") == 1.0

    @patch("src.worker.services.quality_judge_service.genai.Client")
    def test_judge_init(self, mock_client):
        agent = GoogleADKJudgeAgent(model="m1", region="europe-west3")
        assert agent.model == "m1"
        assert agent.region == "europe-west3"
        mock_client.assert_called_once()
        _, kwargs = mock_client.call_args
        assert kwargs.get("location") == "europe-west3"

    @patch("src.worker.services.quality_judge_service.genai.Client")
    def test_judge_init_region_fallback(self, mock_client, monkeypatch):
        from src.worker.services.quality_judge_service import settings
        monkeypatch.setattr(settings, "JUDGE_MODEL_REGION", "")
        monkeypatch.setattr(settings, "GOOGLE_CLOUD_LOCATION", "europe-west1")
        agent = GoogleADKJudgeAgent()
        assert agent.region == "europe-west1"
        _, kwargs = mock_client.call_args
        assert kwargs.get("location") == "europe-west1"

    @patch.object(GoogleADKJudgeAgent, "_judge_with_llm")
    def test_evaluate_success(self, mock_judge, mock_settings):
        mock_judge.return_value = {
            "alignment_score": 1.0,
            "omission_score": 1.0,
            "hallucination_score": 1.0,
            "reasons": ["R1"],
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

    @patch("src.worker.services.quality_judge_service.genai")
    def test_judge_with_llm_client_none_fallback(self, mock_genai):
        with patch("src.worker.services.quality_judge_service.genai", None):
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
                        {"pdf_unicode": "World", "output": "Monde"},
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

    def test_quality_judge_result_to_dict(self, mock_settings):
        result = QualityJudgeResult(
            alignment_score=0.9,
            omission_score=0.85,
            hallucination_score=0.95,
            final_score=0.9,
            pass_fail=True,
            reasons=["good"],
            model="gemini-pro",
        )
        d = result.to_dict()
        assert d["alignment_score"] == 0.9
        assert d["pass_fail"] is True

    @patch.object(GoogleADKJudgeAgent, "_generate_judge_content_with_retry")
    def test_judge_with_llm_success(self, mock_generate, mock_settings):
        mock_response = MagicMock()
        mock_response.parsed = QualityJudgeLLMScores(
            alignment_score=0.9,
            omission_score=0.85,
            hallucination_score=0.92,
            reasons=["good translation"],
        )
        mock_generate.return_value = mock_response
        agent = GoogleADKJudgeAgent(model="gemini-pro")
        agent._client = MagicMock()
        result = agent._judge_with_llm("Hello", "Bonjour")
        assert isinstance(result, QualityJudgeLLMScores)
        assert result.alignment_score == 0.9

    @patch.object(GoogleADKJudgeAgent, "_generate_judge_content_with_retry")
    def test_judge_with_llm_parse_exception_fallback(
        self, mock_generate, mock_settings
    ):
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = '{"alignment_score": 0.7, "omission_score": 0.7, "hallucination_score": 0.8, "reasons": ["ok"]}'
        mock_generate.return_value = mock_response
        agent = GoogleADKJudgeAgent(model="gemini-pro")
        agent._client = MagicMock()
        # Make response.parsed raise when accessed
        type(mock_response).parsed = property(
            lambda _: (_ for _ in ()).throw(Exception("parse error"))
        )
        result = agent._judge_with_llm("Hello", "Bonjour")
        assert isinstance(result, QualityJudgeLLMScores)

    @patch.object(GoogleADKJudgeAgent, "_generate_judge_content_with_retry")
    def test_judge_with_llm_full_fallback(self, mock_generate, mock_settings):
        mock_response = MagicMock()
        mock_generate.return_value = mock_response
        agent = GoogleADKJudgeAgent(model="gemini-pro")
        agent._client = MagicMock()
        # Make parsed raise and text return unparseable JSON
        type(mock_response).parsed = property(
            lambda _: (_ for _ in ()).throw(Exception("parse error"))
        )
        mock_response.text = "not json at all !!!"
        result = agent._judge_with_llm("Hello", "Bonjour")
        assert isinstance(result, QualityJudgeLLMScores)
        assert result.alignment_score == 0.0

    def test_evaluate_with_dict_scores(self, mock_settings):
        agent = GoogleADKJudgeAgent()
        with patch.object(agent, "_judge_with_llm") as mock_judge:
            # Return something that is neither dict nor QualityJudgeLLMScores (triggers dict() path)
            # Must be iterable of 2-tuples for dict() to work
            class FakeScores:
                def __iter__(self):
                    return iter(
                        [
                            ("alignment_score", 0.8),
                            ("omission_score", 0.9),
                            ("hallucination_score", 0.85),
                            ("reasons", ["r1"]),
                        ]
                    )

            mock_judge.return_value = FakeScores()
            result = agent.evaluate(source_text="s", translated_text="t")
            assert isinstance(result, QualityJudgeResult)


@pytest.fixture
def mock_settings():
    with patch("src.worker.services.quality_judge_service.settings") as s:
        s.QUALITY_THRESHOLD = 0.8
        s.JUDGE_MODEL = "gemini-pro"
        yield s
