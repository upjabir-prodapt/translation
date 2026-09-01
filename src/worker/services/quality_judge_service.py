"""Quality gate helpers for iterative translation retries."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
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
    is_fallback: bool = False
    # True when the judge could not form an opinion at all (no segments to
    # judge, or every chunk failed). Distinct from a low score: a numeric
    # 0.0 is a verdict ("this translation is bad") and drives the attempt
    # loop into re-translating the whole document, which is exactly the
    # failure mode that made split PDFs run all MAX_MODEL_ATTEMPTS.
    # It is also distinct from `quality_result is None`, which already means
    # "the judge is disabled" on the DOCX path.
    inconclusive: bool = False
    # Fraction of source characters actually judged (sampling/partial
    # failure). 1.0 when the whole document was covered.
    coverage_ratio: float = 1.0

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
    is_fallback: bool = Field(
        default=False,
        description="True if score was generated via fallback due to LLM outage or parse failure.",
    )


class JudgeResponseParseError(RuntimeError):
    """The judge replied, but not with a usable score object.

    Raised from inside the retried call so tenacity can re-sample the model
    instead of the caller silently substituting a fabricated score. The old
    code caught this inline and returned
    `alignment=0.0, omission=0.0, hallucination=0.8` -- a `final_score` of
    0.28, below every threshold, which then burned the entire model chain
    re-translating a document that may well have been fine.
    """


@dataclass(frozen=True, slots=True)
class JudgeChunk:
    """A contiguous run of aligned segment pairs judged in one LLM call."""

    index: int
    pairs: tuple[tuple[str, str], ...]
    source_chars: int

    def rendered(self) -> tuple[str, str]:
        """Return (source_text, translation_text) for the prompt payload.

        Plain newline joins, exactly the shape the single-call judge used to
        send, so a document small enough to fit in one chunk produces a
        byte-identical payload to the previous behaviour.
        """
        return (
            "\n".join(source for source, _ in self.pairs),
            "\n".join(target for _, target in self.pairs),
        )


def build_judge_chunks(
    pairs: list[tuple[str, str]], max_source_chars: int
) -> list[JudgeChunk]:
    """Group aligned pairs into chunks of at most `max_source_chars` source chars.

    A pair is never split across chunks: source and target must stay
    together or omission/hallucination scoring is meaningless. A single pair
    larger than the budget therefore forms an oversized chunk of its own
    rather than being cut.

    Grouping is on *source* length because the source is the one thing that
    does not change between model attempts. Identical chunk boundaries
    across attempts is what makes comparing their scores a like-for-like
    comparison; grouping on output length would hand each model a different
    exam.
    """
    budget = max(1, int(max_source_chars))
    chunks: list[JudgeChunk] = []
    current: list[tuple[str, str]] = []
    current_chars = 0

    for pair in pairs:
        pair_chars = len(pair[0])
        if current and current_chars + pair_chars > budget:
            chunks.append(
                JudgeChunk(
                    index=len(chunks),
                    pairs=tuple(current),
                    source_chars=current_chars,
                )
            )
            current = []
            current_chars = 0
        current.append(pair)
        current_chars += pair_chars

    if current:
        chunks.append(
            JudgeChunk(
                index=len(chunks), pairs=tuple(current), source_chars=current_chars
            )
        )
    return chunks


def select_judge_chunks(
    chunks: list[JudgeChunk], max_chunks: int, job_id: str
) -> list[JudgeChunk]:
    """Return at most `max_chunks` chunks, stratified across the document.

    Under the cap every chunk is judged and this is a no-op. Above it, the
    chunk list is cut into `max_chunks` contiguous strata spanning the whole
    document and one chunk is drawn from each, so the sample cannot cluster
    in the opening pages -- which matters because translation quality drifts
    *later* in a document as glossary and cross-part context degrade.

    The RNG is seeded on `job_id`, not left to global randomness, because
    the attempt loop compares attempts against each other. A fresh draw per
    attempt would score each attempt on a different subset, letting a worse
    model win by being handed easier chunks. Seeding on the job keeps the
    sample random across documents, identical across that job's attempts,
    and reproducible when investigating a score.
    """
    cap = max(1, int(max_chunks))
    if len(chunks) <= cap:
        return list(chunks)

    # A reproducibility seed for stratified sampling, not a security
    # primitive: determinism per job_id is the entire point, and a CSPRNG
    # would defeat it.
    rng = random.Random(str(job_id))  # noqa: S311
    total = len(chunks)
    selected: list[JudgeChunk] = []
    for stratum in range(cap):
        start = stratum * total // cap
        end = (stratum + 1) * total // cap
        # Integer-proportional boundaries can only collapse when cap > total,
        # which the early return above already excludes; guard anyway.
        end = max(end, start + 1)
        selected.append(chunks[rng.randrange(start, min(end, total))])
    return selected


class GoogleADKJudgeAgent:
    """LLM judge via google-genai (Vertex); omission/hallucination scoring."""

    def __init__(self, model: str | None = None, region: str | None = None):
        self.model = model or settings.JUDGE_MODEL
        self.region = (
            region or settings.JUDGE_MODEL_REGION or settings.GOOGLE_CLOUD_LOCATION
        )
        self._client = None
        if genai is not None:
            # Judge calls get their own (shorter) deadline. Thinking is
            # deliberately NOT constrained here -- the judge runs once per
            # attempt at ~12-19s and its reasoning quality gates the whole
            # pipeline, so it is not worth trading accuracy for.
            self._client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT,
                location=self.region,
                http_options=genai_types.HttpOptions(
                    timeout=int(float(settings.LLM_JUDGE_TIMEOUT_SECONDS) * 1000)
                ),
            )

    def _generate_judge_content_with_retry(
        self, *, model: str, contents: str, config: genai_types.GenerateContentConfig
    ) -> QualityJudgeLLMScores:
        """Issue one judge call and parse it, as a single retried unit.

        Parsing happens *inside* the retried callable on purpose. Previously
        the retry decorator wrapped only the network call while the schema
        parse sat in a `try/except` in the caller, so an unparseable response
        -- which is transient, the same prompt usually parses on the next
        sample -- never reached tenacity and was converted straight into a
        fabricated 0.28 score.

        Slot acquisition, retry and tracing are composed by `invoke_llm`. The
        judge previously retried and traced but never took a slot; that was
        harmless at one call per attempt and is not harmless now that it
        fans out over chunks alongside the translator.
        """
        from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_MODEL
        from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_NAME
        from src.worker.doctranslator.translator.instrumentation import (
            ATTR_LLM_PROMPT_CHARS,
        )
        from src.worker.doctranslator.translator.instrumentation import (
            ATTR_LLM_PROMPT_HASH,
        )
        from src.worker.doctranslator.translator.instrumentation import (
            ATTR_LLM_PROMPT_PREVIEW,
        )
        from src.worker.doctranslator.translator.instrumentation import (
            ATTR_LLM_TEMPERATURE,
        )
        from src.worker.doctranslator.translator.instrumentation import (
            prompt_fingerprint,
        )
        from src.worker.doctranslator.translator.invoke import invoke_llm
        from src.worker.doctranslator.translator.usage import TokenUsage

        prompt_chars, prompt_hash, prompt_preview = prompt_fingerprint(contents)
        temperature = float(getattr(config, "temperature", 0.0) or 0.0)
        logger.debug(
            f"Judge generate_content: model={model} "
            f"prompt_chars={prompt_chars} prompt_hash={prompt_hash} "
            f"temperature={temperature} "
            f"prompt_preview={prompt_preview!r}",
        )

        # The parsed scores are carried out of the retried callable in this
        # holder so `usage_fn` still receives the raw SDK response and token
        # accounting on the span is preserved.
        parsed_holder: list[QualityJudgeLLMScores] = []

        def _call():
            parsed_holder.clear()
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            parsed_holder.append(_parse_judge_response(response))
            return response

        invoke_llm(
            _call,
            span_name="llm.judge.generate_content",
            attributes={
                ATTR_LLM_NAME: self.model,
                ATTR_LLM_MODEL: model,
                "llm.provider": "google_vertexai",
                ATTR_LLM_TEMPERATURE: temperature,
                ATTR_LLM_PROMPT_CHARS: prompt_chars,
                ATTR_LLM_PROMPT_HASH: prompt_hash,
                ATTR_LLM_PROMPT_PREVIEW: prompt_preview,
            },
            usage_fn=lambda resp: TokenUsage.from_gemini_usage(
                getattr(resp, "usage_metadata", None)
            ),
            log_prefix=f"Judge generate_content model={model}",
            logger=logger,
            retry_on=(JudgeResponseParseError,),
        )
        return parsed_holder[0]

    def _judge_with_llm(
        self, source_text: str, translated_text: str
    ) -> QualityJudgeLLMScores:
        """Score one chunk. Raises when the judge could not be made to answer.

        Deliberately no longer returns a fabricated low score on failure: a
        judge that did not answer is a *measurement* failure, and turning it
        into `final=0.28` is what convinced the attempt loop to re-translate
        healthy documents on every model in the chain. Callers treat a raise
        as lost coverage.
        """
        if self._client is None:
            h = _compute_alignment_score(source_text, translated_text)
            return QualityJudgeLLMScores(
                alignment_score=h,
                omission_score=h,
                hallucination_score=0.9,
                reasons=["LLM judge unavailable, heuristic fallback used."],
                is_fallback=True,
            )

        judge_config = genai_types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=QualityJudgeLLMScores,
        )
        return self._generate_judge_content_with_retry(
            model=self.model,
            contents=_build_judge_prompt(source_text, translated_text),
            config=judge_config,
        )

    # ------------------------------------------------------------------
    # Chunked evaluation
    # ------------------------------------------------------------------

    def _judge_chunk(self, chunk: JudgeChunk) -> QualityJudgeLLMScores:
        source_text, translated_text = chunk.rendered()
        return self._judge_with_llm(source_text, translated_text)

    def _run_chunks(
        self, chunks: list[JudgeChunk], deadline: float
    ) -> tuple[dict[int, QualityJudgeLLMScores], list[JudgeChunk]]:
        """Judge chunks concurrently; return (scores by index, failed chunks).

        Concurrency is bounded twice over: this pool, and the process-wide
        `LLM_MAX_INFLIGHT_CALLS` semaphore that `invoke_llm` acquires. The
        latter is the one that matters -- it is shared with the translator,
        so judge fan-out cannot exceed the budget when both are active.
        """
        if not chunks:
            return {}, []
        scores: dict[int, QualityJudgeLLMScores] = {}
        failed: list[JudgeChunk] = []
        skipped = 0
        workers = min(len(chunks), max(1, int(settings.QUALITY_JUDGE_MAX_CHUNKS)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for chunk in chunks:
                # Checked per submission, not once up front: the budget must
                # actually stop work being issued. LLM_JUDGE_TIMEOUT_SECONDS x
                # LLM_RETRY_MAX_ATTEMPTS x two waves is a ~12 min theoretical
                # worst case, which on its own would push a job past the
                # Cloud Tasks dispatch deadline.
                if time.monotonic() >= deadline:
                    skipped += 1
                    failed.append(chunk)
                    continue
                futures[executor.submit(self._judge_chunk, chunk)] = chunk
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    scores[chunk.index] = future.result()
                except Exception as exc:
                    logger.warning(
                        f"Judge chunk {chunk.index} failed "
                        f"({chunk.source_chars} source chars): {exc}"
                    )
                    failed.append(chunk)
        if skipped:
            logger.warning(
                "Judge total budget "
                f"({settings.QUALITY_JUDGE_TOTAL_BUDGET_SECONDS}s) exhausted; "
                f"{skipped} chunk(s) not issued, aggregating what completed"
            )
        return scores, failed

    def evaluate_segments(
        self, pairs: list[tuple[str, str]], *, job_id: str = ""
    ) -> QualityJudgeResult:
        """Score aligned (source, translation) pairs, chunked and sampled.

        Judging pairs rather than two concatenated blobs is what keeps
        source and target aligned: target text runs 10-20% longer than
        source in many language pairs, so chunking the two independently
        would compare paragraph N of the source against paragraph N-3 of the
        translation and report the difference as omission plus hallucination.
        """
        pairs = [(s, t) for s, t in pairs if s and s.strip()]
        total_chars = sum(len(source) for source, _ in pairs)
        if not pairs or total_chars == 0:
            return self._inconclusive(
                ["No source/translation segments were available to judge."],
                coverage_ratio=0.0,
            )

        chunks = build_judge_chunks(pairs, settings.QUALITY_JUDGE_CHUNK_CHARS)
        selected = select_judge_chunks(
            chunks, settings.QUALITY_JUDGE_MAX_CHUNKS, job_id
        )
        t0 = time.monotonic()
        deadline = t0 + float(settings.QUALITY_JUDGE_TOTAL_BUDGET_SECONDS)

        with tracer_llm.start_as_current_span(
            "llm.judge",
            kind=SpanKind.CLIENT,
            attributes={
                "llm.model": self.model,
                "llm.name": self.model,
                "llm.provider": "google_vertexai",
                "judge.segment_pairs": len(pairs),
                "judge.source_chars": total_chars,
                "judge.chunks_total": len(chunks),
                "judge.chunks_selected": len(selected),
            },
        ) as span:
            scores, failed = self._run_chunks(selected, deadline)
            # One retry of the failed subset only. A chunk that fails twice
            # costs coverage; it must never cost the verdict.
            if failed and time.monotonic() < deadline:
                logger.info(f"Retrying {len(failed)} failed judge chunk(s) once")
                retried, still_failed = self._run_chunks(failed, deadline)
                scores.update(retried)
                failed = still_failed

            result = self._aggregate(
                selected=selected,
                scores=scores,
                total_chars=total_chars,
            )
            elapsed = time.monotonic() - t0
            span.set_attribute("judge.alignment_score", result.alignment_score)
            span.set_attribute("judge.omission_score", result.omission_score)
            span.set_attribute("judge.hallucination_score", result.hallucination_score)
            span.set_attribute("judge.final_score", round(result.final_score, 4))
            span.set_attribute("judge.passed", result.pass_fail)
            span.set_attribute("judge.is_fallback", result.is_fallback)
            span.set_attribute("judge.inconclusive", result.inconclusive)
            span.set_attribute("judge.coverage_ratio", round(result.coverage_ratio, 4))
            span.set_attribute("judge.chunks_failed", len(failed))
            span.set_attribute("llm.latency_s", round(elapsed, 3))

        logger.info(
            f"Judge evaluate done: model={self.model} latency_s={elapsed:.3f} "
            f"pairs={len(pairs)} source_chars={total_chars} "
            f"chunks={len(chunks)} judged={len(scores)}/{len(selected)} "
            f"coverage={result.coverage_ratio:.3f} "
            f"alignment={result.alignment_score:.3f} "
            f"omission={result.omission_score:.3f} "
            f"hallucination={result.hallucination_score:.3f} "
            f"final={result.final_score:.3f} passed={result.pass_fail} "
            f"is_fallback={result.is_fallback} inconclusive={result.inconclusive}"
        )
        # ── Bubble judge summary up to the pipeline.run root span ──────────
        set_root_span_attributes(
            {
                "judge.model": self.model,
                "judge.final_score": round(result.final_score, 4),
                "judge.passed": result.pass_fail,
                "judge.is_fallback": result.is_fallback,
                "judge.inconclusive": result.inconclusive,
                "judge.coverage_ratio": round(result.coverage_ratio, 4),
            }
        )
        return result

    def _aggregate(
        self,
        *,
        selected: list[JudgeChunk],
        scores: dict[int, QualityJudgeLLMScores],
        total_chars: int,
    ) -> QualityJudgeResult:
        """Combine surviving chunk scores into one verdict.

        Sub-scores are averaged with source-length weights so a 200-char
        chunk cannot outvote a 8000-char one, and `final_score` is
        recomputed from the aggregated sub-scores rather than averaged from
        per-chunk finals -- averaging finals and re-deriving them are only
        equal when the weights match exactly, and there is no reason to make
        that assumption hold silently.
        """
        survivors = [c for c in selected if c.index in scores]
        if not survivors:
            return self._inconclusive(
                [
                    "Every judge chunk failed after retry; no quality signal "
                    "could be obtained for this attempt."
                ],
                coverage_ratio=0.0,
            )

        weight_total = sum(c.source_chars for c in survivors) or 1

        def _weighted(attr: str) -> float:
            acc = sum(
                max(0.0, min(1.0, float(getattr(scores[c.index], attr))))
                * c.source_chars
                for c in survivors
            )
            return max(0.0, min(1.0, acc / weight_total))

        alignment = _weighted("alignment_score")
        omission = _weighted("omission_score")
        hallucination = _weighted("hallucination_score")
        final = (0.30 * alignment) + (0.35 * omission) + (0.35 * hallucination)
        passed = final >= settings.QUALITY_THRESHOLD

        judged_chars = sum(c.source_chars for c in survivors)
        coverage_ratio = min(1.0, judged_chars / total_chars) if total_chars else 0.0
        is_fallback = any(
            bool(getattr(scores[c.index], "is_fallback", False)) for c in survivors
        ) or coverage_ratio < float(settings.QUALITY_JUDGE_MIN_COVERAGE_RATIO)

        reasons: list[str] = []
        for chunk in survivors:
            reasons.extend(str(r) for r in (scores[chunk.index].reasons or []))
        # The rubric asks for 3-8 reasons per chunk; at 16 chunks that is up
        # to 128 strings, which is noise in a report and in BigQuery.
        if len(reasons) > 8:
            reasons = reasons[:8]
        if coverage_ratio < 1.0:
            reasons.append(
                f"Judged {coverage_ratio:.0%} of the document "
                f"({len(survivors)}/{len(selected)} sampled chunks)."
            )

        return QualityJudgeResult(
            alignment_score=alignment,
            omission_score=omission,
            hallucination_score=hallucination,
            final_score=final,
            pass_fail=passed,
            reasons=reasons,
            model=self.model,
            is_fallback=is_fallback,
            inconclusive=False,
            coverage_ratio=coverage_ratio,
        )

    def _inconclusive(
        self, reasons: list[str], *, coverage_ratio: float
    ) -> QualityJudgeResult:
        """A verdict of "no verdict".

        Scores are zeroed for schema stability, but `inconclusive=True` is
        what callers must branch on -- a numeric 0.0 read as a real score is
        precisely the bug this replaces.
        """
        logger.warning(f"Quality judge inconclusive: {'; '.join(reasons)}")
        return QualityJudgeResult(
            alignment_score=0.0,
            omission_score=0.0,
            hallucination_score=0.0,
            final_score=0.0,
            pass_fail=False,
            reasons=reasons,
            model=self.model,
            is_fallback=True,
            inconclusive=True,
            coverage_ratio=coverage_ratio,
        )

    async def evaluate_segments_async(
        self, pairs: list[tuple[str, str]], *, job_id: str = ""
    ) -> QualityJudgeResult:
        """Run the chunked judge off the event loop."""
        return await asyncio.to_thread(self.evaluate_segments, pairs, job_id=job_id)

    def evaluate(self, *, source_text: str, translated_text: str) -> QualityJudgeResult:
        """Single-pair convenience wrapper over `evaluate_segments`.

        Kept so callers holding two blobs rather than pairs still work; the
        blobs become one pair, hence one chunk unless they exceed
        QUALITY_JUDGE_CHUNK_CHARS.
        """
        return self.evaluate_segments([(source_text, translated_text)])

    async def evaluate_async(
        self, *, source_text: str, translated_text: str
    ) -> QualityJudgeResult:
        """Run synchronous judge + LLM in a worker thread (non-blocking for asyncio)."""
        return await asyncio.to_thread(
            self.evaluate,
            source_text=source_text,
            translated_text=translated_text,
        )


def _parse_judge_response(response: Any) -> QualityJudgeLLMScores:
    """Turn an SDK response into scores, or raise `JudgeResponseParseError`.

    Called from inside the retried callable, so raising here re-samples the
    model rather than fabricating a score.
    """
    parsed = None
    try:
        parsed = response.parsed
    except Exception as exc:  # SDK raises when the body is not schema-valid
        logger.debug(f"Judge response.parsed raised: {exc}")
    if isinstance(parsed, QualityJudgeLLMScores):
        parsed.is_fallback = False
        return parsed

    raw = (getattr(response, "text", "") or "").strip()
    try:
        payload = json.loads(raw.replace("```json", "").replace("```", "").strip())
        if not isinstance(payload, dict):
            raise TypeError(f"judge JSON is {type(payload).__name__}, not an object")
        payload.pop("is_fallback", None)
        return QualityJudgeLLMScores(**payload, is_fallback=False)
    except Exception as exc:
        raise JudgeResponseParseError(
            f"Judge response could not be parsed as QualityJudgeLLMScores: {exc}"
        ) from exc


def _build_judge_prompt(source_text: str, translated_text: str) -> str:
    """Render the judging rubric around one chunk's payload.

    The rubric itself is unchanged from the single-call judge; only the
    SOURCE/TRANSLATION payload is now one chunk rather than the whole
    document.
    """
    return (
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


#: Buckets in translate_tracking.json that hold judgeable paragraphs.
#: `page` holds ordinary per-page paragraphs; `cross_page` and `cross_column`
#: hold paragraphs that were merged across a page or column boundary and are
#: therefore *excluded* from `page` (see `translated_ids` in
#: il_translator_llm_only). The old extractor read only `page`, so merged
#: paragraphs -- exactly the ones most likely to be mistranslated -- were
#: never judged on any document, split or not.
_TRACKING_BUCKETS = ("page", "cross_page", "cross_column")


def extract_attempt_segments(working_dir: Path) -> list[tuple[str, str]]:
    """Return aligned (source, translation) pairs from translate_tracking.json.

    Pairs, not two concatenated blobs. The previous implementation appended
    to two independent lists under two independent `if` guards, so a single
    paragraph with an input but no output shifted every subsequent
    translation one slot relative to its source: from that point on the
    judge was comparing unrelated paragraphs. Emitting pairs makes that
    structurally impossible, and a missing output becomes `""` -- which the
    judge correctly scores as an omission instead of it silently vanishing.
    """
    tracking_path = working_dir / "translate_tracking.json"
    if not tracking_path.exists():
        return []

    try:
        data = json.loads(tracking_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception(f"Failed to parse tracking file at {tracking_path}")
        return []

    pairs: list[tuple[str, str]] = []
    for bucket in _TRACKING_BUCKETS:
        for page in data.get(bucket) or []:
            for paragraph in page.get("paragraph") or []:
                source = str(
                    paragraph.get("input") or paragraph.get("pdf_unicode") or ""
                ).strip()
                if not source:
                    # No source means nothing to judge against; an output
                    # with no input is not an omission or a hallucination
                    # we can attribute.
                    continue
                target = str(paragraph.get("output") or "").strip()
                pairs.append((source, target))
    return pairs


def extract_attempt_text(working_dir: Path) -> tuple[str, str]:
    """Concatenated source/translated text, kept for non-judge callers.

    Still used by the protected-token verifier, which works on whole-document
    text rather than pairs. Built from `extract_attempt_segments` so the two
    views cannot drift.
    """
    pairs = extract_attempt_segments(working_dir)
    return (
        "\n".join(source for source, _ in pairs),
        "\n".join(target for _, target in pairs if target),
    )
