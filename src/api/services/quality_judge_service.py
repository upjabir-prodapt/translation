"""Quality gate helpers for iterative translation retries."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.constants import settings
from src.config.logging import logger

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None

try:
    from google.adk.agents import Agent
except ImportError:  # pragma: no cover - optional dependency
    Agent = None


@dataclass(slots=True)
class QualityJudgeResult:
    """Normalized quality gate output per attempt."""

    alignment_score: float
    omission_score: float
    hallucination_score: float
    final_score: float
    pass_fail: bool
    reasons: list[str]
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?。！？]+", text)
    return [part.strip() for part in parts if part.strip()]


def _compute_alignment_score(source: str, translated: str) -> float:
    src_sentences = max(len(_split_sentences(source)), 1)
    tgt_sentences = max(len(_split_sentences(translated)), 1)
    ratio = min(src_sentences, tgt_sentences) / max(src_sentences, tgt_sentences)
    return max(0.0, min(1.0, ratio))


class GoogleADKJudgeAgent:
    """LLM judge facade for omission/hallucination checks."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.JUDGE_MODEL
        self._client = None
        self._agent = None
        if Agent is not None:
            self._agent = Agent(
                name="quality_judge_agent",
                model=self.model,
                description=(
                    "Judges translation quality with omission and hallucination scoring."
                ),
            )
        if genai is not None:
            self._client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT_ID,
                location=settings.GOOGLE_CLOUD_LOCATION,
            )

    def _judge_with_llm(self, source_text: str, translated_text: str) -> dict[str, Any]:
        if self._client is None:
            return {
                "omission_score": _compute_alignment_score(
                    source_text, translated_text
                ),
                "hallucination_score": 0.9,
                "reasons": ["LLM judge unavailable, heuristic fallback used."],
            }

        prompt = (
            "You are a strict translation quality judge.\n"
            "Score the translation quality between 0 and 1 for omission and hallucination.\n"
            "Return valid JSON only with keys: omission_score, hallucination_score, reasons.\n"
            "reasons must be a list of concise strings.\n\n"
            f"Source:\n{source_text}\n\n"
            f"Translation:\n{translated_text}"
        )
        response = self._client.models.generate_content(model=self.model, contents=prompt)
        raw = getattr(response, "text", "") or ""
        try:
            parsed = json.loads(raw)
            return {
                "omission_score": float(parsed.get("omission_score", 0.0)),
                "hallucination_score": float(parsed.get("hallucination_score", 0.0)),
                "reasons": parsed.get("reasons", []),
            }
        except Exception:
            logger.warning("Judge response parse failed; fallback heuristic will be used.")
            return {
                "omission_score": _compute_alignment_score(
                    source_text, translated_text
                ),
                "hallucination_score": 0.8,
                "reasons": ["Judge parse failed, fallback heuristic used."],
            }

    def evaluate(self, *, source_text: str, translated_text: str) -> QualityJudgeResult:
        alignment = _compute_alignment_score(source_text, translated_text)
        llm_scores = self._judge_with_llm(source_text, translated_text)
        omission = max(0.0, min(1.0, float(llm_scores.get("omission_score", 0.0))))
        hallucination = max(
            0.0, min(1.0, float(llm_scores.get("hallucination_score", 0.0)))
        )
        final = (0.4 * alignment) + (0.35 * omission) + (0.25 * hallucination)
        return QualityJudgeResult(
            alignment_score=alignment,
            omission_score=omission,
            hallucination_score=hallucination,
            final_score=final,
            pass_fail=final >= settings.QUALITY_THRESHOLD,
            reasons=list(llm_scores.get("reasons", [])),
            model=self.model,
        )


def extract_attempt_text(working_dir: Path) -> tuple[str, str]:
    """Extract concatenated source/translated text from translate_tracking.json."""
    tracking_path = working_dir / "translate_tracking.json"
    if not tracking_path.exists():
        return "", ""

    try:
        data = json.loads(tracking_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning(f"Failed to parse tracking file at {tracking_path}")
        return "", ""

    source: list[str] = []
    translated: list[str] = []
    for page in data.get("page", []):
        for paragraph in page.get("paragraph", []):
            input_text = paragraph.get("input") or paragraph.get("pdf_unicode") or ""
            output_text = paragraph.get("output") or ""
            if input_text:
                source.append(str(input_text))
            if output_text:
                translated.append(str(output_text))
    return "\n".join(source), "\n".join(translated)

