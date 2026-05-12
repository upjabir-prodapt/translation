import pytest
from src.api.services.confidence_service import ConfidenceService
from src.api.services.verification_service import VerificationResult


@pytest.fixture
def service():
    return ConfidenceService()


class TestConfidenceService:
    def test_compute_no_retry(self, service):
        res = VerificationResult(
            completeness_score=1.0,
            omission_score=1.0,
            hallucination_score=1.0,
            terminology_score=1.0,
            final_score=0.95,
            passed=True,
            reasons=[],
            judge_model="m1",
        )
        assert service.compute(res, 0) == 0.95

    def test_compute_with_retry_penalty(self, service):
        res = VerificationResult(
            completeness_score=1.0,
            omission_score=1.0,
            hallucination_score=1.0,
            terminology_score=1.0,
            final_score=0.95,
            passed=True,
            reasons=[],
            judge_model="m1",
        )
        # penalty 0.03
        assert service.compute(res, 1) == 0.92
        # penalty 0.06
        assert service.compute(res, 2) == 0.89

    def test_compute_max_penalty(self, service):
        res = VerificationResult(
            completeness_score=1.0,
            omission_score=1.0,
            hallucination_score=1.0,
            terminology_score=1.0,
            final_score=0.95,
            passed=True,
            reasons=[],
            judge_model="m1",
        )
        # penalty capped at 0.09
        assert service.compute(res, 3) == 0.86
        assert service.compute(res, 4) == 0.86

    def test_compute_zero_lower_bound(self, service):
        res = VerificationResult(
            completeness_score=1.0,
            omission_score=1.0,
            hallucination_score=1.0,
            terminology_score=1.0,
            final_score=0.05,
            passed=False,
            reasons=[],
            judge_model="m1",
        )
        assert service.compute(res, 3) == 0.0  # 0.05 - 0.09 = -0.04 -> 0.0
