from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.doctranslator.glossary import Glossary
from src.worker.doctranslator.glossary import GlossaryEntry
from src.worker.services.quality_judge_service import QualityJudgeResult
from src.worker.services.verification_service import VerificationService


@pytest.fixture
def service():
    with patch("src.worker.services.verification_service.GoogleADKJudgeAgent"):
        return VerificationService()


class TestVerificationService:
    def test_terminology_score_empty(self, service):
        assert service._terminology_score("text", []) == 1.0

    def test_terminology_score_success(self, service):
        mock_entry = MagicMock(spec=GlossaryEntry)
        mock_entry.target = "T1"
        mock_glossary = MagicMock(spec=Glossary)
        mock_glossary.entries = [mock_entry]

        assert service._terminology_score("This contains T1", [mock_glossary]) == 1.0
        assert service._terminology_score("Missing", [mock_glossary]) == 0.0

    def test_verify_success(self, service):
        mock_judge = service._judge
        mock_judge.evaluate.return_value = QualityJudgeResult(
            alignment_score=0.9,
            omission_score=0.9,
            hallucination_score=0.9,
            final_score=0.9,
            pass_fail=True,
            reasons=[],
            model="m1",
        )

        res = service.verify(source_text="s", translated_text="t", glossaries=[])
        assert res.passed is True
        assert res.completeness_score == 0.9

    def test_verify_failure_hallucination(self, service):
        mock_judge = service._judge
        mock_judge.evaluate.return_value = QualityJudgeResult(
            alignment_score=0.9,
            omission_score=0.9,
            hallucination_score=0.5,  # FAIL
            final_score=0.7,
            pass_fail=False,
            reasons=[],
            model="m1",
        )
        res = service.verify(source_text="s", translated_text="t", glossaries=[])
        assert res.passed is False
        assert "Hallucination risk" in res.reasons[-1]
