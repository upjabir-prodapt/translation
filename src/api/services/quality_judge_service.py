"""Quality gate helpers for iterative translation retries."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src.config.constants import settings
from src.config.logging import logger

from google import genai
from google.genai import types as genai_types


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


class QualityJudgeLLMScores(BaseModel):
    """Structured judge output from the model (before composite final_score)."""

    model_config = ConfigDict(extra="ignore")

    alignment_score: float = Field(
        description=(
            "Structural and segment correspondence between SOURCE and TRANSLATION: "
            "parallel sentences/bullets, sensible splits/merges, ordering, and length balance; "
            "0.0 worst, 1.0 best. Not the same as omission (missing meaning) or hallucination (invented content)."
        )
    )
    omission_score: float = Field(
        description="Coverage of source meaning; 0.0 worst, 1.0 best."
    )
    hallucination_score: float = Field(
        description="Faithfulness without invented content; 0.0 worst, 1.0 best."
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="3–8 concise English strings citing concrete evidence.",
    )


class GoogleADKJudgeAgent:
    """LLM judge via google-genai (Vertex); omission/hallucination scoring."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.JUDGE_MODEL
        self._client = None
        if genai is not None:
            self._client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT_ID,
                location=settings.GOOGLE_CLOUD_LOCATION,
            )

    def _judge_with_llm(self, source_text: str, translated_text: str) -> QualityJudgeLLMScores:
        if self._client is None:
            h = _compute_alignment_score(source_text, translated_text)
            return QualityJudgeLLMScores(
                alignment_score=h,
                omission_score=h,
                hallucination_score=0.9,
                reasons=["LLM judge unavailable, heuristic fallback used."],
            )

        prompt = (
            "# Role\n"
            "You are an expert translation quality judge. You compare SOURCE to TRANSLATION and return "
            "three calibrated numeric scores plus short English rationales for automation and review.\n\n"
            "# Task\n"
            "1. Read SOURCE and TRANSLATION (any language pair; judge faithfully).\n"
            "2. Assign alignment_score, omission_score, and hallucination_score as floats from 0.0 (worst) to 1.0 (best).\n"
            "3. List 3–8 concise reasons in English only, each citing concrete evidence.\n"
            "4. Output a single JSON object only—no markdown fences, preamble, or trailing text.\n\n"
            "# Inputs\n"
            "- SOURCE: original text.\n"
            "- TRANSLATION: candidate output scored against SOURCE.\n\n"
            "# What to ignore when scoring\n"
            "- Glossary-style product or company names kept in the target language.\n"
            "- HTML-like or markup tags; do not treat them as errors.\n"
            "Be conservative: reserve scores near 1.0 only when the case for that dimension is clearly strong.\n\n"
            "## alignment_score (structure and segment correspondence)\n"
            "How well TRANSLATION mirrors the shape of SOURCE at the paragraph/list/sentence level—not word-for-word identity.\n"
            "- 1.0: Paragraph breaks, list items, steps, and major clauses stay in sensible one-to-one correspondence; "
            "splits or merges are justified by target-language norms and do not scramble order of ideas.\n"
            "- Lower the score for: collapsed lists, merged unrelated bullets, reordered steps that change procedure logic, "
            "one sentence in SOURCE ballooning into many unrelated sentences in TRANSLATION (or the reverse), "
            "or obviously skewed length with no linguistic excuse.\n"
            "- Do not conflate with omission_score: alignment is about correspondence of segments and flow, "
            "not whether every fact was translated.\n\n"
            "## omission_score (coverage of source meaning)\n"
            "- 1.0: All substantive meaning, facts, and obligations in SOURCE appear in TRANSLATION "
            "(order and wording may differ).\n"
            "- Do not penalize unavoidable target-language grammar (e.g. articles, auxiliaries) with no one-to-one source token.\n\n"
            "## hallucination_score (faithfulness; no invented content)\n"
            "- 1.0: TRANSLATION does not assert facts, opinions, or details not supported by SOURCE. "
            "Idiomatic phrasing that preserves meaning is fine.\n"
            "- Lower the score for: added explanations, disclaimers, or editorial text; invented numbers, dates, or names; "
            "contradictions of SOURCE; elaboration not implied by SOURCE.\n"
            "- Acceptable: register shifts, cultural adaptation, splitting or merging sentences when meaning is preserved.\n"
            "- For clearly spurious short phrases not licensed by SOURCE, reduce hallucination_score proportionally "
            "(rough guide: about 0.05–0.15 per spurious phrase).\n\n"
            "## reasons (language requirement)\n"
            "- Every string in reasons must be English, even if SOURCE and TRANSLATION are not.\n"
            "- Mention evidence for alignment, omission, and hallucination where relevant; avoid generic praise.\n\n"
            "# Output format\n"
            "One JSON object matching the enforced schema (same keys as the example). No markdown, no commentary.\n\n"
            "## Example response (illustrative only)\n"
            "{\n"
            '  "alignment_score": 0.90,\n'
            '  "omission_score": 0.88,\n'
            '  "hallucination_score": 0.92,\n'
            '  "reasons": [\n'
            '    "List length and ordering match SOURCE; no unrelated merge of bullet points.",\n'
            '    "All main obligations in SOURCE appear in TRANSLATION.",\n'
            '    "No facts, dates, or names appear in TRANSLATION that are absent from SOURCE.",\n'
            '    "Glossary names and tags were not scored as errors."\n'
            "  ]\n"
            "}\n\n"
            f"SOURCE:\n{source_text}\n\n"
            f"TRANSLATION:\n{translated_text}"
        )
        try:
            judge_config = genai_types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_json_schema=QualityJudgeLLMScores.model_json_schema(),
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=judge_config,
            )
            parsed = response.parsed
            return parsed
        except Exception as exc:
            logger.warning(f"Judge response parse failed; fallback heuristic will be used. Error: {exc}")
            parsed = response.text.strip()
            try:
                parsed = json.loads(parsed.replace("```json", "").replace("```", "").strip())
                return parsed
            except Exception:
                return {
                    "alignment_score":0.0,
                    "omission_score":0.0,
                    "hallucination_score":0.8,
                    "reasons":["Judge parse failed, fallback heuristic used."],
                }


    def evaluate(self, *, source_text: str, translated_text: str) -> QualityJudgeResult:
        llm_scores = self._judge_with_llm(source_text, translated_text)
        alignment = max(0.0, min(1.0, llm_scores.alignment_score))
        omission = max(0.0, min(1.0, llm_scores.omission_score))
        hallucination = max(0.0, min(1.0, llm_scores.hallucination_score))
        final = (0.30 * alignment) + (0.35 * omission) + (0.35 * hallucination)
        return QualityJudgeResult(
            alignment_score=alignment,
            omission_score=omission,
            hallucination_score=hallucination,
            final_score=final,
            pass_fail=final >= settings.QUALITY_THRESHOLD,
            reasons=list(llm_scores.reasons),
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

