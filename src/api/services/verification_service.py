"""Output verification service."""

from __future__ import annotations

from dataclasses import dataclass

from src.api.services.quality_judge_service import GoogleADKJudgeAgent
from src.api.services.quality_judge_service import QualityJudgeResult
from src.doctranslator.glossary import Glossary


@dataclass(slots=True)
class VerificationResult:
    """Combined verification metrics."""

    completeness_score: float
    omission_score: float
    hallucination_score: float
    terminology_score: float
    final_score: float
    passed: bool
    reasons: list[str]
    judge_model: str

    def to_dict(self) -> dict:
        return {
            "completeness_score": self.completeness_score,
            "omission_score": self.omission_score,
            "hallucination_score": self.hallucination_score,
            "terminology_score": self.terminology_score,
            "final_score": self.final_score,
            "passed": self.passed,
            "reasons": self.reasons,
            "judge_model": self.judge_model,
        }


class VerificationService:
    """Verify translation quality using LLM judge + terminology checks."""

    def __init__(self):
        self._judge = GoogleADKJudgeAgent()

    def _terminology_score(
        self, translated_text: str, glossaries: list[Glossary]
    ) -> float:
        expected_terms = 0
        matched_terms = 0
        lowered_translation = translated_text.lower()
        for glossary in glossaries:
            for entry in glossary.entries:
                expected_terms += 1
                if entry.target.lower() in lowered_translation:
                    matched_terms += 1
        if expected_terms == 0:
            return 1.0
        return matched_terms / expected_terms

    def verify(
        self,
        *,
        source_text: str,
        translated_text: str,
        glossaries: list[Glossary],
    ) -> VerificationResult:
        judge_result: QualityJudgeResult = self._judge.evaluate(
            source_text=source_text,
            translated_text=translated_text,
        )
        terminology_score = self._terminology_score(translated_text, glossaries)
        completeness_score = judge_result.alignment_score
        final_score = (
            0.35 * completeness_score
            + 0.30 * judge_result.omission_score
            + 0.20 * judge_result.hallucination_score
            + 0.15 * terminology_score
        )
        passed = bool(
            completeness_score >= 0.85
            and terminology_score >= 0.95
            and judge_result.hallucination_score >= 0.85
            and final_score >= 0.80
        )
        reasons = list(judge_result.reasons)
        if terminology_score < 0.95:
            reasons.append("Terminology accuracy below 95%.")
        if judge_result.hallucination_score < 0.85:
            reasons.append("Hallucination risk above threshold.")
        return VerificationResult(
            completeness_score=completeness_score,
            omission_score=judge_result.omission_score,
            hallucination_score=judge_result.hallucination_score,
            terminology_score=terminology_score,
            final_score=final_score,
            passed=passed,
            reasons=reasons,
            judge_model=judge_result.model,
        )
