"""Batch-translate DOCX paragraph segments with an LLM translator."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from dataclasses import field

from src.config.constants import settings
from src.config.domain_prompts import build_domain_role_block
from src.worker.doctranslator.glossary import Glossary
from src.worker.doctranslator.translator.base import BaseTranslator
from src.worker.doctranslator.translator.schemas import BatchTranslationResponse
from src.worker.docxtranslator.prompts import DOCX_PROMPT_TEMPLATE
from src.worker.docxtranslator.segments import Segment

logger = logging.getLogger(__name__)

# Tag matching is case-insensitive: models occasionally echo run tags as
# <G0>…</G0>, and that alone should not cost the paragraph its formatting.
_TAG_RE = re.compile(r"<g(\d+)>(.*?)</g\1>", re.DOTALL | re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"</?g\d+>", re.IGNORECASE)
_CHARS_PER_TOKEN_FALLBACK = 4


@dataclass(slots=True)
class SegmentBatch:
    """One LLM request: a slice of segments plus their tagged payloads."""

    batch_index: int
    segments: list[Segment]
    payloads: list[str]
    token_count: int


@dataclass(slots=True)
class TranslationOutcome:
    """Per-segment translation results and the failures worth reporting."""

    translations: dict[int, list[str]] = field(default_factory=dict)
    batches: list[SegmentBatch] = field(default_factory=list)
    failed_batches: int = 0
    untranslated_segments: int = 0
    tag_fallback_segments: int = 0


class _TokenCounter:
    """tiktoken-backed length estimate with a character-ratio fallback."""

    def __init__(self) -> None:
        self._encoder = None
        try:
            import tiktoken

            self._encoder = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            logger.warning(
                "tiktoken unavailable; estimating DOCX batch sizes by length"
            )

    def count(self, text: str) -> int:
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text, disallowed_special=()))
            except Exception:
                pass
        return max(1, len(text) // _CHARS_PER_TOKEN_FALLBACK)


def build_payload(segment: Segment) -> str:
    """Render a segment as plain text, or as tagged spans when it has several."""
    texts = segment.group_texts
    if len(texts) == 1:
        return texts[0]
    return "".join(f"<g{i}>{text}</g{i}>" for i, text in enumerate(texts))


def parse_payload(payload: str, group_count: int) -> list[str] | None:
    """Recover per-group texts from a model response, or None if unusable."""
    if group_count == 1:
        return [_ANY_TAG_RE.sub("", payload)]

    found: dict[int, str] = {}
    for match in _TAG_RE.finditer(payload):
        index = int(match.group(1))
        if index in found or index >= group_count:
            return None
        found[index] = match.group(2)
    if len(found) != group_count:
        return None
    return [found[i] for i in range(group_count)]


class DocxSegmentTranslator:
    """Translate DOCX segments in batched, concurrent LLM calls."""

    def __init__(
        self,
        *,
        translate_engine: BaseTranslator,
        lang_out: str,
        domain: str | None,
        glossaries: list[Glossary] | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.translate_engine = translate_engine
        self.lang_out = lang_out
        self.domain = domain
        self.glossaries = glossaries or []
        self.max_concurrency = max(
            1, max_concurrency or settings.TRANSLATION_POOL_MAX_WORKERS
        )
        self._tokens = _TokenCounter()

    def build_batches(self, segments: list[Segment]) -> list[SegmentBatch]:
        """Group segments into requests bounded by paragraph and token budgets."""
        max_paragraphs = max(1, settings.LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS)
        max_tokens = max(1, settings.LLM_TRANSLATION_BATCH_MAX_TOKENS)

        batches: list[SegmentBatch] = []
        current: list[Segment] = []
        payloads: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, payloads, current_tokens
            if current:
                batches.append(
                    SegmentBatch(
                        batch_index=len(batches),
                        segments=current,
                        payloads=payloads,
                        token_count=current_tokens,
                    )
                )
            current, payloads, current_tokens = [], [], 0

        for segment in segments:
            payload = build_payload(segment)
            tokens = self._tokens.count(payload)
            over_budget = current and (
                len(current) >= max_paragraphs or current_tokens + tokens > max_tokens
            )
            if over_budget:
                flush()
            current.append(segment)
            payloads.append(payload)
            current_tokens += tokens
        flush()
        return batches

    async def translate(self, segments: list[Segment]) -> TranslationOutcome:
        """Translate every translatable segment, batching and running in parallel."""
        outcome = TranslationOutcome()
        translatable = [segment for segment in segments if segment.is_translatable()]
        if not translatable:
            logger.info("No translatable paragraphs found in DOCX")
            return outcome

        outcome.batches = self.build_batches(translatable)
        logger.info(
            f"Translating {len(translatable)} DOCX paragraphs in "
            f"{len(outcome.batches)} batches (concurrency={self.max_concurrency})"
        )
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run(batch: SegmentBatch) -> None:
            async with semaphore:
                await self._translate_batch(batch, outcome)

        await asyncio.gather(*(run(batch) for batch in outcome.batches))
        return outcome

    async def _translate_batch(
        self, batch: SegmentBatch, outcome: TranslationOutcome
    ) -> None:
        try:
            results = await self._request_batch(batch)
        except Exception:
            logger.exception(
                f"DOCX batch {batch.batch_index} failed; retrying paragraph by paragraph"
            )
            outcome.failed_batches += 1
            await self._retry_individually(batch, outcome)
            return
        self._store_results(batch, results, outcome)

    async def _request_batch(self, batch: SegmentBatch) -> dict[int, str]:
        items = [
            {"id": index, "input": payload}
            for index, payload in enumerate(batch.payloads)
        ]
        prompt = self._build_prompt(items)
        raw = await self.translate_engine.llm_translate_async(
            prompt,
            rate_limit_params={"paragraph_token_count": batch.token_count},
            response_schema=BatchTranslationResponse,
        )
        return self._parse_response(raw, len(items))

    async def _retry_individually(
        self, batch: SegmentBatch, outcome: TranslationOutcome
    ) -> None:
        for segment, payload in zip(batch.segments, batch.payloads, strict=True):
            single = SegmentBatch(
                batch_index=batch.batch_index,
                segments=[segment],
                payloads=[payload],
                token_count=self._tokens.count(payload),
            )
            try:
                results = await self._request_batch(single)
            except Exception:
                logger.exception(
                    f"DOCX paragraph {segment.index} left untranslated after retry"
                )
                outcome.untranslated_segments += 1
                continue
            self._store_results(single, results, outcome)

    def _store_results(
        self,
        batch: SegmentBatch,
        results: dict[int, str],
        outcome: TranslationOutcome,
    ) -> None:
        for position, segment in enumerate(batch.segments):
            payload = results.get(position)
            if payload is None or not payload.strip():
                outcome.untranslated_segments += 1
                continue
            group_texts = parse_payload(payload, len(segment.groups))
            if group_texts is None:
                logger.warning(
                    f"Run tags misaligned for DOCX paragraph {segment.index}; "
                    "collapsing translation onto the first run"
                )
                outcome.tag_fallback_segments += 1
                group_texts = [_ANY_TAG_RE.sub("", payload)] + [""] * (
                    len(segment.groups) - 1
                )
            outcome.translations[segment.index] = group_texts

    def _parse_response(self, raw: str, expected: int) -> dict[int, str]:
        parsed = json.loads(self._strip_wrappers(raw))
        if isinstance(parsed, dict):
            parsed = parsed.get("items", parsed.get("outputs", []))
        if not isinstance(parsed, list):
            raise ValueError(f"Unexpected batch response type: {type(parsed).__name__}")

        results: dict[int, str] = {}
        for item in parsed:
            if not isinstance(item, dict) or "id" not in item:
                raise ValueError("Batch response item missing id field")
            results[int(item["id"])] = str(item.get("output", ""))
        if len(results) != expected:
            raise ValueError(
                f"Batch response length mismatch. Expected {expected}, got {len(results)}"
            )
        return results

    @staticmethod
    def _strip_wrappers(raw: str) -> str:
        text = raw.strip()
        for prefix in ("<json>", "```json", "```"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        for suffix in ("</json>", "```"):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                break
        return text.strip()

    def _build_prompt(self, items: list[dict[str, object]]) -> str:
        json_input = json.dumps(items, ensure_ascii=False, indent=2)
        match_text = "\n".join(str(item["input"]) for item in items)
        usage_rules, tables = self._glossary_blocks(match_text)
        return DOCX_PROMPT_TEMPLATE.substitute(
            role_block=build_domain_role_block(self.domain, self.lang_out),
            lang_out=self.lang_out,
            glossary_usage_rules_block=usage_rules,
            glossary_tables_block=tables,
            json_input_str=json_input,
        )

    def _glossary_blocks(self, batch_text: str) -> tuple[str, str]:
        entries_by_glossary: dict[str, list[tuple[str, str]]] = {}
        for glossary in self.glossaries:
            active = glossary.get_active_entries_for_text(batch_text)
            if active:
                entries_by_glossary[glossary.name] = sorted(set(active))
        if not entries_by_glossary:
            return "", ""

        usage_rules = (
            "## Glossary Usage\n"
            "- Terms in the glossary tables below are MANDATORY translations.\n"
            "- Apply glossary items even inside run tags or when split across spans.\n"
            "- If the glossary does NOT include a term, translate it naturally.\n\n"
        )
        lines: list[str] = ["## Glossary Tables", ""]
        for name, entries in entries_by_glossary.items():
            lines.append(f"### Glossary: {name}")
            lines.append("")
            lines.append("| Source | Target |")
            lines.append("| --- | --- |")
            lines.extend(f"| {source} | {target} |" for source, target in entries)
            lines.append("")
        return usage_rules, "\n".join(lines)
