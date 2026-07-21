"""Confidence score engine."""

from src.worker.services.verification_service import VerificationResult


class ConfidenceService:
    """Compute document-level confidence score."""

    def compute(self, verification: VerificationResult, retry_count: int) -> float:
        retry_penalty = min(retry_count * 0.03, 0.09)
        confidence = max(0.0, min(1.0, verification.final_score - retry_penalty))
        return round(confidence, 4)
