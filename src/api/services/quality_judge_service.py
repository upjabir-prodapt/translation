"""Quality gate helpers for iterative translation retries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types
from opentelemetry.trace import SpanKind
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src.config.constants import settings
from src.config.retry import llm_retry
from src.config.tracing import set_root_span_attributes
from src.config.tracing import tracer_llm

logger = logging.getLogger(__name__)


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

    @llm_retry(logger=logger)
    def _generate_judge_content_with_retry(
        self, *, model: str, contents: str, config: genai_types.GenerateContentConfig
    ):
        from opentelemetry.trace import Status
        from opentelemetry.trace import StatusCode

        prompt_chars = len(contents)
        prompt_hash = hashlib.sha256(
            contents.encode("utf-8", errors="replace")
        ).hexdigest()[:12]
        prompt_preview = contents[:300].replace("\n", "\\n")
        temperature = float(getattr(config, "temperature", 0.0) or 0.0)
        logger.debug(
            f"Judge generate_content: model={model} "
            f"prompt_chars={prompt_chars} prompt_hash={prompt_hash} "
            f"temperature={temperature} "
            f"prompt_preview={prompt_preview!r}",
        )
        t0 = time.monotonic()
        with tracer_llm.start_as_current_span(
            "llm.judge.generate_content",
            kind=SpanKind.CLIENT,
            attributes={
                "llm.name": "judge",
                "llm.model": model,
                "llm.provider": "google_vertexai",
                "llm.temperature": temperature,
                "llm.prompt_chars": prompt_chars,
                "llm.prompt_hash": prompt_hash,
                "llm.prompt_preview": prompt_preview,
            },
        ) as span:
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                    output_tokens = int(
                        getattr(usage, "candidates_token_count", 0) or 0
                    )
                    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
                    cached_tokens = int(
                        getattr(usage, "cached_content_token_count", 0) or 0
                    )
                    span.set_attribute("llm.input_tokens", input_tokens)
                    span.set_attribute("llm.output_tokens", output_tokens)
                    span.set_attribute("llm.total_tokens", total_tokens)
                    span.set_attribute("llm.cached_tokens", cached_tokens)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                elapsed = time.monotonic() - t0
                span.set_attribute("llm.latency_s", round(elapsed, 3))
                logger.debug(
                    f"Judge generate_content finished: model={model} "
                    f"latency_s={elapsed:.3f} prompt_hash={prompt_hash}",
                )
        return response

    def _judge_with_llm(
        self, source_text: str, translated_text: str
    ) -> QualityJudgeLLMScores:
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
        response = None
        try:
            judge_config = genai_types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=QualityJudgeLLMScores,
            )
            response = self._generate_judge_content_with_retry(
                model=self.model,
                contents=prompt,
                config=judge_config,
            )
            parsed = response.parsed
            return parsed
        except Exception as exc:
            logger.warning(
                f"Judge response parse failed; fallback heuristic will be used. Error: {exc}"
            )
            parsed = (getattr(response, "text", "") or "").strip()
            try:
                parsed = json.loads(
                    parsed.replace("```json", "").replace("```", "").strip()
                )
                return QualityJudgeLLMScores(**parsed)
            except Exception as parse_error:
                logger.warning(f"Judge fallback JSON parse failed: {parse_error}")
                return QualityJudgeLLMScores(
                    alignment_score=0.0,
                    omission_score=0.0,
                    hallucination_score=0.8,
                    reasons=["Judge parse failed, fallback heuristic used."],
                )

    def evaluate(self, *, source_text: str, translated_text: str) -> QualityJudgeResult:
        source_chars = len(source_text)
        translated_chars = len(translated_text)
        logger.debug(
            f"Judge evaluate: model={self.model} "
            f"source_chars={source_chars} translated_chars={translated_chars}",
        )
        t0 = time.monotonic()
        with tracer_llm.start_as_current_span(
            "llm.judge",
            kind=SpanKind.CLIENT,
            attributes={
                "llm.model": self.model,
                "llm.name": "judge",
                "llm.provider": "google_vertexai",
                "judge.source_chars": source_chars,
                "judge.translated_chars": translated_chars,
            },
        ) as span:
            llm_scores = self._judge_with_llm(source_text, translated_text)
            if not isinstance(llm_scores, dict):
                llm_scores = (
                    llm_scores.model_dump()
                    if isinstance(llm_scores, QualityJudgeLLMScores)
                    else dict(llm_scores)
                )
            alignment = max(0.0, min(1.0, llm_scores.get("alignment_score", 0.0)))
            omission = max(0.0, min(1.0, llm_scores.get("omission_score", 0.0)))
            hallucination = max(
                0.0, min(1.0, llm_scores.get("hallucination_score", 0.0))
            )
            final = (0.30 * alignment) + (0.35 * omission) + (0.35 * hallucination)
            passed = final >= settings.QUALITY_THRESHOLD
            elapsed = time.monotonic() - t0
            span.set_attribute("judge.alignment_score", alignment)
            span.set_attribute("judge.omission_score", omission)
            span.set_attribute("judge.hallucination_score", hallucination)
            span.set_attribute("judge.final_score", round(final, 4))
            span.set_attribute("judge.passed", passed)
            span.set_attribute("llm.latency_s", round(elapsed, 3))
        logger.info(
            f"Judge evaluate done: model={self.model} "
            f"latency_s={elapsed:.3f} "
            f"source_chars={source_chars} translated_chars={translated_chars} "
            f"alignment={alignment:.3f} omission={omission:.3f} "
            f"hallucination={hallucination:.3f} final={final:.3f} passed={passed}",
        )
        # ── Bubble judge summary up to the pipeline.run root span ──────────
        set_root_span_attributes(
            {
                "judge.model": self.model,
                "judge.final_score": round(final, 4),
                "judge.passed": passed,
            }
        )
        return QualityJudgeResult(
            alignment_score=alignment,
            omission_score=omission,
            hallucination_score=hallucination,
            final_score=final,
            pass_fail=passed,
            reasons=list(llm_scores.get("reasons", [])),
            model=self.model,
        )

    async def evaluate_async(
        self, *, source_text: str, translated_text: str
    ) -> QualityJudgeResult:
        """Run synchronous judge + LLM in a worker thread (non-blocking for asyncio)."""
        return await asyncio.to_thread(
            self.evaluate,
            source_text=source_text,
            translated_text=translated_text,
        )


def extract_attempt_text(working_dir: Path) -> tuple[str, str]:
    """Extract concatenated source/translated text from translate_tracking.json."""
    tracking_path = working_dir / "translate_tracking.json"
    if not tracking_path.exists():
        return "", ""

    try:
        data = json.loads(tracking_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception(f"Failed to parse tracking file at {tracking_path}")
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
