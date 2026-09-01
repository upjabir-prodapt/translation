import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.services.quality_judge_service import GoogleADKJudgeAgent
from src.worker.services.quality_judge_service import JudgeResponseParseError
from src.worker.services.quality_judge_service import QualityJudgeLLMScores
from src.worker.services.quality_judge_service import QualityJudgeResult
from src.worker.services.quality_judge_service import _compute_alignment_score
from src.worker.services.quality_judge_service import _parse_judge_response
from src.worker.services.quality_judge_service import _split_sentences
from src.worker.services.quality_judge_service import build_judge_chunks
from src.worker.services.quality_judge_service import extract_attempt_segments
from src.worker.services.quality_judge_service import extract_attempt_text
from src.worker.services.quality_judge_service import select_judge_chunks


def _scores(alignment=0.9, omission=0.9, hallucination=0.9, reasons=None):
    return QualityJudgeLLMScores(
        alignment_score=alignment,
        omission_score=omission,
        hallucination_score=hallucination,
        reasons=reasons or ["ok"],
    )


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
        mock_judge.return_value = _scores(1.0, 1.0, 1.0, ["R1"])
        agent = GoogleADKJudgeAgent()
        res = agent.evaluate(source_text="s", translated_text="t")
        assert isinstance(res, QualityJudgeResult)
        assert res.final_score == pytest.approx(1.0)
        assert res.pass_fail is True
        assert res.inconclusive is False
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
            assert res.is_fallback is True

    def test_quality_judge_result_to_dict(self, mock_settings):
        result = QualityJudgeResult(
            alignment_score=0.9,
            omission_score=0.85,
            hallucination_score=0.95,
            final_score=0.9,
            pass_fail=True,
            reasons=["good"],
            model="gemini-pro",
            is_fallback=False,
        )
        d = result.to_dict()
        assert d["alignment_score"] == 0.9
        assert d["pass_fail"] is True
        assert d["is_fallback"] is False
        assert d["inconclusive"] is False
        assert d["coverage_ratio"] == 1.0

    @patch.object(GoogleADKJudgeAgent, "_generate_judge_content_with_retry")
    def test_judge_with_llm_success(self, mock_generate, mock_settings):
        mock_generate.return_value = _scores(0.9, 0.85, 0.92, ["good translation"])
        agent = GoogleADKJudgeAgent(model="gemini-pro")
        agent._client = MagicMock()
        result = agent._judge_with_llm("Hello", "Bonjour")
        assert isinstance(result, QualityJudgeLLMScores)
        assert result.alignment_score == 0.9
        assert result.is_fallback is False


class TestJudgeResponseParsing:
    """The old code caught a parse failure inline and returned
    alignment=0.0/omission=0.0/hallucination=0.8 -- a final score of 0.28,
    below every threshold, which then burned the whole model chain
    re-translating a document that may well have been fine."""

    def test_schema_parsed_response_is_used_directly(self):
        response = MagicMock()
        response.parsed = _scores(0.8, 0.8, 0.8)
        parsed = _parse_judge_response(response)
        assert parsed.alignment_score == 0.8
        assert parsed.is_fallback is False

    def test_raw_json_body_is_recovered(self):
        response = MagicMock()
        type(response).parsed = property(
            lambda _: (_ for _ in ()).throw(ValueError("schema mismatch"))
        )
        response.text = (
            '```json\n{"alignment_score": 0.7, "omission_score": 0.7, '
            '"hallucination_score": 0.8, "reasons": ["ok"]}\n```'
        )
        parsed = _parse_judge_response(response)
        assert parsed.omission_score == 0.7
        assert parsed.is_fallback is False

    def test_unparseable_body_raises_instead_of_fabricating_a_score(self):
        response = MagicMock()
        type(response).parsed = property(
            lambda _: (_ for _ in ()).throw(ValueError("schema mismatch"))
        )
        response.text = "not json at all !!!"
        with pytest.raises(JudgeResponseParseError):
            _parse_judge_response(response)

    def test_parse_failure_reaches_tenacity(self, mock_settings, monkeypatch):
        """The parse must happen inside the retried callable.

        Previously `@llm_retry` wrapped only the network call while the
        schema parse sat in the caller's `try/except`, so a transient
        unparseable response was never re-sampled.
        """
        from src.config import retry as retry_module
        from src.worker.doctranslator.translator import invoke as invoke_module

        monkeypatch.setattr(retry_module.settings, "LLM_RETRY_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(retry_module.settings, "LLM_RETRY_MULTIPLIER", 0)
        monkeypatch.setattr(retry_module.settings, "LLM_RETRY_MIN_SECONDS", 0)
        monkeypatch.setattr(retry_module.settings, "LLM_RETRY_MAX_SECONDS", 0)

        bad = MagicMock()
        type(bad).parsed = property(
            lambda _: (_ for _ in ()).throw(ValueError("schema mismatch"))
        )
        bad.text = "garbage"
        good = MagicMock()
        good.parsed = _scores(0.6, 0.6, 0.6)

        agent = GoogleADKJudgeAgent(model="gemini-pro")
        agent._client = MagicMock()
        agent._client.models.generate_content.side_effect = [bad, good]

        with patch.object(invoke_module, "llm_call_slot"):
            result = agent._generate_judge_content_with_retry(
                model="gemini-pro", contents="prompt", config=MagicMock(temperature=0.1)
            )
        assert result.alignment_score == 0.6
        assert agent._client.models.generate_content.call_count == 2


class TestJudgeChunking:
    def test_small_document_yields_exactly_one_chunk(self):
        pairs = [("a" * 100, "b" * 100), ("c" * 100, "d" * 100)]
        chunks = build_judge_chunks(pairs, 8000)
        assert len(chunks) == 1
        assert chunks[0].pairs == tuple(pairs)
        # Payload shape is identical to the pre-chunking single call.
        source, target = chunks[0].rendered()
        assert source == "\n".join(p[0] for p in pairs)
        assert target == "\n".join(p[1] for p in pairs)

    def test_chunker_never_splits_a_pair(self):
        pairs = [("x" * 60, "y" * 60) for _ in range(10)]
        chunks = build_judge_chunks(pairs, 100)
        assert sum(len(c.pairs) for c in chunks) == len(pairs)
        for chunk in chunks:
            for source, target in chunk.pairs:
                assert (source, target) in pairs

    def test_oversized_single_pair_becomes_its_own_chunk(self):
        pairs = [("s" * 50, "t"), ("h" * 500, "u"), ("s" * 50, "t")]
        chunks = build_judge_chunks(pairs, 100)
        assert any(c.pairs == (("h" * 500, "u"),) for c in chunks)

    def test_source_chars_are_tracked_for_weighting(self):
        chunks = build_judge_chunks([("abc", "xyz"), ("de", "f")], 8000)
        assert chunks[0].source_chars == 5


class TestJudgeChunkSampling:
    @staticmethod
    def _chunks(n):
        return build_judge_chunks([("x" * 10, "y") for _ in range(n)], 10)

    def test_under_the_cap_every_chunk_is_judged(self):
        chunks = self._chunks(5)
        assert select_judge_chunks(chunks, 16, "job") == chunks

    def test_over_the_cap_selects_exactly_the_cap(self):
        chunks = self._chunks(65)
        selected = select_judge_chunks(chunks, 16, "job")
        assert len(selected) == 16

    def test_one_chunk_per_stratum_spanning_the_document(self):
        chunks = self._chunks(64)
        selected = select_judge_chunks(chunks, 16, "job")
        indices = [c.index for c in selected]
        assert indices == sorted(indices)
        assert len(set(indices)) == 16
        for stratum, index in enumerate(indices):
            assert stratum * 4 <= index < (stratum + 1) * 4

    def test_same_job_id_selects_the_same_chunks_across_attempts(self):
        """The orchestrator compares attempts against each other.

        A fresh draw per attempt would score each attempt on a different
        subset, letting a worse model win by being handed easier chunks.
        """
        chunks = self._chunks(65)
        first = select_judge_chunks(chunks, 16, "job-abc")
        second = select_judge_chunks(chunks, 16, "job-abc")
        assert [c.index for c in first] == [c.index for c in second]

    def test_different_job_ids_select_different_chunks(self):
        chunks = self._chunks(200)
        a = [c.index for c in select_judge_chunks(chunks, 16, "job-a")]
        b = [c.index for c in select_judge_chunks(chunks, 16, "job-b")]
        assert a != b


class TestEvaluateSegments:
    @staticmethod
    def _agent():
        with patch("src.worker.services.quality_judge_service.genai", None):
            agent = GoogleADKJudgeAgent(model="judge-model")
        agent._client = MagicMock()
        return agent

    def test_no_pairs_is_inconclusive_not_zero(self, mock_settings):
        result = self._agent().evaluate_segments([], job_id="j")
        assert result.inconclusive is True
        assert result.pass_fail is False
        assert result.coverage_ratio == 0.0

    def test_every_chunk_failing_is_inconclusive(self, mock_settings):
        agent = self._agent()
        with patch.object(
            agent, "_judge_with_llm", side_effect=RuntimeError("judge down")
        ):
            result = agent.evaluate_segments([("hello", "bonjour")], job_id="j")
        assert result.inconclusive is True
        assert result.coverage_ratio == 0.0

    def test_failed_chunks_are_retried_once(self, mock_settings):
        agent = self._agent()
        calls = {"n": 0}

        def _flaky(_source, _target):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return _scores(0.9, 0.9, 0.9)

        with patch.object(agent, "_judge_with_llm", side_effect=_flaky):
            result = agent.evaluate_segments([("hello", "bonjour")], job_id="j")
        assert calls["n"] == 2
        assert result.inconclusive is False
        assert result.coverage_ratio == pytest.approx(1.0)

    def test_aggregation_is_source_length_weighted(self, mock_settings, monkeypatch):
        from src.worker.services import quality_judge_service as qjs

        monkeypatch.setattr(qjs.settings, "QUALITY_JUDGE_CHUNK_CHARS", 10)
        agent = self._agent()
        # 90 source chars scored 1.0 against 10 source chars scored 0.0:
        # an unweighted mean would say 0.5, the weighted mean says 0.9.
        pairs = [("a" * 90, "x"), ("b" * 10, "y")]

        def _by_length(source, _target):
            return (
                _scores(1.0, 1.0, 1.0) if len(source) == 90 else _scores(0.0, 0.0, 0.0)
            )

        with patch.object(agent, "_judge_with_llm", side_effect=_by_length):
            result = agent.evaluate_segments(pairs, job_id="j")
        assert result.alignment_score == pytest.approx(0.9)
        assert result.final_score == pytest.approx(0.9)

    def test_partial_failure_reports_coverage_below_one(
        self, mock_settings, monkeypatch
    ):
        from src.worker.services import quality_judge_service as qjs

        monkeypatch.setattr(qjs.settings, "QUALITY_JUDGE_CHUNK_CHARS", 10)
        agent = self._agent()
        pairs = [("a" * 10, "x"), ("b" * 10, "y")]

        def _one_fails(source, _target):
            if source.startswith("b"):
                raise RuntimeError("chunk down")
            return _scores(0.9, 0.9, 0.9)

        with patch.object(agent, "_judge_with_llm", side_effect=_one_fails):
            result = agent.evaluate_segments(pairs, job_id="j")
        assert result.inconclusive is False
        assert result.coverage_ratio == pytest.approx(0.5)

    def test_low_coverage_is_flagged_as_fallback(self, mock_settings, monkeypatch):
        from src.worker.services import quality_judge_service as qjs

        monkeypatch.setattr(qjs.settings, "QUALITY_JUDGE_CHUNK_CHARS", 10)
        monkeypatch.setattr(qjs.settings, "QUALITY_JUDGE_MAX_CHUNKS", 1)
        agent = self._agent()
        pairs = [("a" * 10, "x") for _ in range(10)]
        with patch.object(agent, "_judge_with_llm", return_value=_scores()):
            result = agent.evaluate_segments(pairs, job_id="j")
        assert result.coverage_ratio == pytest.approx(0.1)
        assert result.is_fallback is True


class TestExtractAttemptSegments:
    def test_reads_all_three_buckets(self, tmp_path):
        """cross_page/cross_column paragraphs are excluded from `page`, so
        reading only `page` meant merged paragraphs were never judged."""
        (tmp_path / "translate_tracking.json").write_text(
            json.dumps(
                {
                    "page": [{"paragraph": [{"input": "Hello", "output": "Bonjour"}]}],
                    "cross_page": [
                        {"paragraph": [{"input": "Spans", "output": "Traverse"}]}
                    ],
                    "cross_column": [
                        {"paragraph": [{"pdf_unicode": "Cols", "output": "Colonnes"}]}
                    ],
                }
            )
        )
        pairs = extract_attempt_segments(tmp_path)
        assert pairs == [
            ("Hello", "Bonjour"),
            ("Spans", "Traverse"),
            ("Cols", "Colonnes"),
        ]

    def test_missing_output_keeps_pairs_aligned(self, tmp_path):
        """The old extractor appended to two independent lists under two
        independent guards, so a paragraph with input but no output shifted
        every later translation against the wrong source."""
        (tmp_path / "translate_tracking.json").write_text(
            json.dumps(
                {
                    "page": [
                        {
                            "paragraph": [
                                {"input": "One", "output": "Un"},
                                {"input": "Two", "output": ""},
                                {"input": "Three", "output": "Trois"},
                            ]
                        }
                    ]
                }
            )
        )
        pairs = extract_attempt_segments(tmp_path)
        assert pairs == [("One", "Un"), ("Two", ""), ("Three", "Trois")]

    def test_not_exists(self):
        assert extract_attempt_segments(Path("/nonexistent")) == []

    def test_invalid_json(self, tmp_path):
        (tmp_path / "translate_tracking.json").write_text("invalid json")
        assert extract_attempt_segments(tmp_path) == []

    def test_extract_attempt_text_still_works(self, tmp_path):
        (tmp_path / "translate_tracking.json").write_text(
            json.dumps(
                {
                    "page": [
                        {
                            "paragraph": [
                                {"input": "Hello", "output": "Bonjour"},
                                {"pdf_unicode": "World", "output": "Monde"},
                            ]
                        }
                    ]
                }
            )
        )
        src, tgt = extract_attempt_text(tmp_path)
        assert src == "Hello\nWorld"
        assert tgt == "Bonjour\nMonde"

    def test_extract_attempt_text_not_exists(self):
        assert extract_attempt_text(Path("/nonexistent")) == ("", "")


@pytest.fixture
def mock_settings():
    with patch("src.worker.services.quality_judge_service.settings") as s:
        s.QUALITY_THRESHOLD = 0.8
        s.JUDGE_MODEL = "gemini-pro"
        s.QUALITY_JUDGE_CHUNK_CHARS = 8000
        s.QUALITY_JUDGE_MAX_CHUNKS = 16
        s.QUALITY_JUDGE_TOTAL_BUDGET_SECONDS = 420.0
        s.QUALITY_JUDGE_MIN_COVERAGE_RATIO = 0.5
        s.LLM_JUDGE_TIMEOUT_SECONDS = 120.0
        yield s


class TestJudgeTotalBudget:
    """The wall-clock safety net must actually stop work being issued.

    LLM_JUDGE_TIMEOUT_SECONDS x LLM_RETRY_MAX_ATTEMPTS x two waves is a
    ~12 min theoretical worst case, which on its own would push a job past
    the 1800 s Cloud Tasks dispatch deadline.
    """

    def test_expired_budget_stops_issuing_chunks(self, mock_settings, monkeypatch):
        from src.worker.services import quality_judge_service as qjs

        monkeypatch.setattr(qjs.settings, "QUALITY_JUDGE_CHUNK_CHARS", 10)
        monkeypatch.setattr(qjs.settings, "QUALITY_JUDGE_TOTAL_BUDGET_SECONDS", 0.0)
        with patch("src.worker.services.quality_judge_service.genai", None):
            agent = GoogleADKJudgeAgent(model="judge-model")
        agent._client = MagicMock()

        judged = []

        def _record(source, _target):
            judged.append(source)
            return _scores()

        with patch.object(agent, "_judge_with_llm", side_effect=_record):
            result = agent.evaluate_segments(
                [("a" * 10, "x"), ("b" * 10, "y")], job_id="j"
            )

        assert judged == []
        assert result.inconclusive is True

    def test_budget_within_bounds_judges_everything(self, mock_settings, monkeypatch):
        from src.worker.services import quality_judge_service as qjs

        monkeypatch.setattr(qjs.settings, "QUALITY_JUDGE_CHUNK_CHARS", 10)
        with patch("src.worker.services.quality_judge_service.genai", None):
            agent = GoogleADKJudgeAgent(model="judge-model")
        agent._client = MagicMock()

        with patch.object(agent, "_judge_with_llm", return_value=_scores()) as mock:
            result = agent.evaluate_segments(
                [("a" * 10, "x"), ("b" * 10, "y")], job_id="j"
            )
        assert mock.call_count == 2
        assert result.coverage_ratio == pytest.approx(1.0)
