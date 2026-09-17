#!/usr/bin/env python3
"""Benchmark several language-detection approaches on one document.

Standalone script (deliberately does NOT import `src.config.constants`,
whose `Settings()` requires the full worker `.env` to validate) that
compares, on the same extracted PDF text:

  1. langdetect (raw)       -- a single whole-text `detect_langs()` call.
  2. langdetect (production) -- the per-block, confidence-floored,
     char-weighted aggregation this repo used to run in production
     (`src/worker/services/language_detection_core.py`, before it was
     replaced by lingua), reimplemented inline here so the script has no
     dependency on `Settings()`. Kept purely for side-by-side comparison.
  3. lingua                 -- `lingua-language-detector`, this repo's
     current production engine (see `language_detection_core.py`).
  4. Gemini (Vertex AI)     -- one LLM call asking for the ISO 639-1 code.
  5. Claude (Vertex AI)     -- same prompt via Anthropic's Vertex client.

Each method reports the detected language, a confidence/score where
available, and wall-clock latency, so the trade-offs between local
statistical detectors and LLM calls are visible side by side.

NOTE: `langdetect` is no longer a project dependency (removed in favour of
lingua) -- the two `langdetect (*)` rows require `uv pip install langdetect`
into your venv to run; they degrade to an error row otherwise.

Usage:
    uv run python scripts/lang_detect_benchmark.py <path-to-pdf-or-text> \\
        [--project GOOGLE_CLOUD_PROJECT] [--location REGION] \\
        [--gemini-model MODEL] [--claude-model MODEL] \\
        [--claude-region REGION] [--sample-chars N] [--skip-llm]

Requires `lingua-language-detector` (not a project dependency; install
with `uv pip install lingua-language-detector`) and Application Default
Credentials with Vertex AI access for the Gemini/Claude steps.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults mirror .env.worker.local so the script works out of the box in
# this environment; override via CLI flags for a different project.
# ---------------------------------------------------------------------------
DEFAULT_PROJECT = "gclt-aicoe-dev-st"
DEFAULT_LOCATION = "europe-west3"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CLAUDE_REGION = "europe-west3"
DEFAULT_SAMPLE_CHARS = 4000

# Same thresholds as language_detection_core.py, duplicated here so this
# script stays import-independent from src.config.*.
MIN_DETECTION_TEXT_LENGTH = 20
MIN_DETECTION_ALPHA_CHARS = 5
MIN_DETECTION_CONFIDENCE = 0.80

LLM_PROMPT = (
    "Identify the single dominant natural language of the text below. "
    "Respond with ONLY the ISO 639-1 two-letter code (e.g. en, fr, de, "
    "zh, ja) and nothing else -- no punctuation, no explanation.\n\n"
    "--- BEGIN TEXT ---\n{sample}\n--- END TEXT ---"
)


@dataclass
class Result:
    method: str
    language: str
    confidence: str
    elapsed_s: float
    note: str = ""


def extract_pdf_text(path: Path) -> tuple[str, list[str]]:
    import pymupdf

    pages: list[str] = []
    with pymupdf.open(str(path)) as document:
        for page in document:
            pages.append(page.get_text())
    return "\n".join(pages), pages


def load_text(path: Path) -> tuple[str, list[str]]:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, text.splitlines() or [text]


def normalize(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# 1. langdetect, raw single call over the whole text
# ---------------------------------------------------------------------------
def run_langdetect_raw(full_text: str) -> Result:
    try:
        from langdetect import DetectorFactory
        from langdetect import LangDetectException
        from langdetect import detect_langs
    except ImportError:
        return Result(
            "langdetect (raw)",
            "?",
            "-",
            0.0,
            note="not installed (removed as a project dependency): uv pip install langdetect",
        )

    DetectorFactory.seed = 0
    text = normalize(full_text)
    start = time.perf_counter()
    try:
        candidates = detect_langs(text)
    except LangDetectException as exc:
        elapsed = time.perf_counter() - start
        return Result("langdetect (raw)", "?", "-", elapsed, note=str(exc))
    elapsed = time.perf_counter() - start
    if not candidates:
        return Result("langdetect (raw)", "?", "-", elapsed, note="no candidates")
    top = candidates[0]
    others = ", ".join(f"{c.lang}:{c.prob:.2f}" for c in candidates[1:4])
    return Result(
        "langdetect (raw)",
        top.lang,
        f"{top.prob:.4f}",
        elapsed,
        note=f"runner-up: {others}" if others else "",
    )


# ---------------------------------------------------------------------------
# 2. langdetect, this repo's production algorithm (per-block, confidence
#    floor, char-weighted aggregation) -- reimplemented inline to avoid
#    importing src.config.constants.settings.
# ---------------------------------------------------------------------------
def run_langdetect_production(blocks: list[str]) -> Result:
    try:
        from langdetect import DetectorFactory
        from langdetect import LangDetectException
        from langdetect import detect_langs
    except ImportError:
        return Result(
            "langdetect (production algo)",
            "?",
            "-",
            0.0,
            note="not installed (removed as a project dependency): uv pip install langdetect",
        )

    DetectorFactory.seed = 0
    counter: Counter[str] = Counter()
    skipped_short = 0
    skipped_low_conf = 0
    start = time.perf_counter()
    for block in blocks:
        text = normalize(block)
        alpha_count = sum(1 for ch in text if ch.isalpha())
        if (
            len(text) < MIN_DETECTION_TEXT_LENGTH
            or alpha_count < MIN_DETECTION_ALPHA_CHARS
        ):
            skipped_short += 1
            continue
        try:
            candidates = detect_langs(text)
        except LangDetectException:
            continue
        if not candidates or candidates[0].prob < MIN_DETECTION_CONFIDENCE:
            skipped_low_conf += 1
            continue
        counter[candidates[0].lang] += len(text)
    elapsed = time.perf_counter() - start
    if not counter:
        return Result(
            "langdetect (production algo)",
            "?",
            "-",
            elapsed,
            note=f"no confident block (skipped short={skipped_short}, "
            f"low-conf={skipped_low_conf})",
        )
    winner, chars = counter.most_common(1)[0]
    dist = ", ".join(f"{lang}:{n}ch" for lang, n in counter.most_common(4))
    return Result(
        "langdetect (production algo)",
        winner,
        f"{chars} chars won",
        elapsed,
        note=f"distribution: {dist}",
    )


# ---------------------------------------------------------------------------
# 3. lingua
# ---------------------------------------------------------------------------
def run_lingua(full_text: str) -> Result:
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        return Result(
            "lingua",
            "?",
            "-",
            0.0,
            note="not installed: uv pip install lingua-language-detector",
        )

    text = normalize(full_text)
    build_start = time.perf_counter()
    detector = (
        LanguageDetectorBuilder.from_all_languages()
        .with_preloaded_language_models()
        .build()
    )
    build_elapsed = time.perf_counter() - build_start

    start = time.perf_counter()
    language = detector.detect_language_of(text)
    values = detector.compute_language_confidence_values(text)
    elapsed = time.perf_counter() - start

    if language is None:
        return Result(
            "lingua",
            "?",
            "-",
            elapsed,
            note=f"no confident result (model build took {build_elapsed:.2f}s)",
        )
    top_value = next((v.value for v in values if v.language == language), None)
    others = ", ".join(
        f"{v.language.iso_code_639_1.name.lower()}:{v.value:.2f}" for v in values[1:4]
    )
    return Result(
        "lingua",
        language.iso_code_639_1.name.lower(),
        f"{top_value:.4f}" if top_value is not None else "-",
        elapsed,
        note=f"model build: {build_elapsed:.2f}s (one-time) | runner-up: {others}",
    )


# ---------------------------------------------------------------------------
# 4. Gemini via Vertex AI
# ---------------------------------------------------------------------------
def run_gemini(sample: str, *, project: str, location: str, model: str) -> Result:
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return Result(
            "Gemini (Vertex)", "?", "-", 0.0, note="google-genai not installed"
        )

    start = time.perf_counter()
    try:
        client = genai.Client(vertexai=True, project=project, location=location)
        response = client.models.generate_content(
            model=model,
            contents=LLM_PROMPT.format(sample=sample),
            config=genai_types.GenerateContentConfig(temperature=0.0),
        )
        text = (response.text or "").strip()
    except Exception as exc:  # noqa: BLE001 - surfaced as a benchmark row
        elapsed = time.perf_counter() - start
        return Result("Gemini (Vertex)", "?", "-", elapsed, note=f"error: {exc}")
    elapsed = time.perf_counter() - start
    code = re.sub(r"[^a-zA-Z-]", "", text).lower()[:5] or text
    usage = getattr(response, "usage_metadata", None)
    tokens = ""
    if usage is not None:
        tokens = (
            f"in={getattr(usage, 'prompt_token_count', '?')} "
            f"out={getattr(usage, 'candidates_token_count', '?')}"
        )
    return Result(
        f"Gemini (Vertex, {model})",
        code,
        "-",
        elapsed,
        note=f"raw reply: {text!r} | tokens: {tokens}",
    )


# ---------------------------------------------------------------------------
# 5. Claude via Vertex AI
# ---------------------------------------------------------------------------
def run_claude(sample: str, *, project: str, region: str, model: str) -> Result:
    try:
        from anthropic import AnthropicVertex
    except ImportError:
        return Result("Claude (Vertex)", "?", "-", 0.0, note="anthropic not installed")

    start = time.perf_counter()
    try:
        client = AnthropicVertex(project_id=project, region=region)
        response = client.messages.create(
            model=model,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": LLM_PROMPT.format(sample=sample)}],
        )
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()
    except Exception as exc:  # noqa: BLE001 - surfaced as a benchmark row
        elapsed = time.perf_counter() - start
        return Result("Claude (Vertex)", "?", "-", elapsed, note=f"error: {exc}")
    elapsed = time.perf_counter() - start
    code = re.sub(r"[^a-zA-Z-]", "", text).lower()[:5] or text
    usage = getattr(response, "usage", None)
    tokens = ""
    if usage is not None:
        tokens = f"in={getattr(usage, 'input_tokens', '?')} out={getattr(usage, 'output_tokens', '?')}"
    return Result(
        f"Claude (Vertex, {model})",
        code,
        "-",
        elapsed,
        note=f"raw reply: {text!r} | tokens: {tokens}",
    )


def print_table(results: list[Result]) -> None:
    headers = ["Method", "Detected", "Confidence", "Time (s)", "Notes"]
    rows = [
        [r.method, r.language, r.confidence, f"{r.elapsed_s:.3f}", r.note]
        for r in results
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        if rows
        else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path", type=Path, help="PDF or text file to detect the language of"
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--location", default=DEFAULT_LOCATION, help="Gemini Vertex region"
    )
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--claude-region", default=DEFAULT_CLAUDE_REGION)
    parser.add_argument("--sample-chars", type=int, default=DEFAULT_SAMPLE_CHARS)
    parser.add_argument(
        "--skip-llm", action="store_true", help="Only run local detectors"
    )
    args = parser.parse_args()

    if not args.path.is_file():
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1

    full_text, blocks = load_text(args.path)
    print(f"Source: {args.path}")
    print(f"Extracted {len(full_text)} chars across {len(blocks)} block(s)/page(s).\n")

    if not full_text.strip():
        print("No extractable text found.", file=sys.stderr)
        return 1

    sample = normalize(full_text)[: args.sample_chars]

    results: list[Result] = []
    results.append(run_langdetect_raw(full_text))
    results.append(run_langdetect_production(blocks))
    results.append(run_lingua(full_text))

    if not args.skip_llm:
        results.append(
            run_gemini(
                sample,
                project=args.project,
                location=args.location,
                model=args.gemini_model,
            )
        )
        results.append(
            run_claude(
                sample,
                project=args.project,
                region=args.claude_region,
                model=args.claude_model,
            )
        )

    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
