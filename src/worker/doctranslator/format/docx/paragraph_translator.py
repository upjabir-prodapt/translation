"""Batch paragraph translation for DOCX units.

Mirrors the PDF pipeline's ILTranslatorLLMOnly batching logic
(src/worker/doctranslator/format/pdf/document_il/midend/il_translator_llm_only.py)
but operates on plain TranslatableUnit text instead of the PDF IL tree.
Uses the identical JSON batch schema so it flows through the exact same
BaseTranslator.llm_translate() call path -- same Redis translation
cache (Memorystore via PSC), same per-provider rate limiter, same
Gemini/Claude translator classes, zero new LLM-calling code.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any

import Levenshtein
import orjson
import tiktoken

from src.config.constants import settings
from src.worker.doctranslator.batching import compute_batch_plan
from src.worker.doctranslator.batching import log_batch_plan
from src.worker.doctranslator.format.docx.units import TranslatableUnit
from src.worker.doctranslator.format.pdf.translation_config import get_token_multiplier
from src.worker.doctranslator.translator.translation_cache import build_cache_key
from src.worker.doctranslator.translator.translation_cache import get_translation_cache
from src.worker.doctranslator.translator.translator import BaseTranslator
from src.worker.doctranslator.translator.translator import BatchTranslationResponse

logger = logging.getLogger(__name__)

_TRIM_REPEAT_PATTERN = re.compile(r"[. 。…，]{20,}")

# Units matching these are not translatable text and must never be sent to an
# LLM: bare numbers/dates/percentages, and strings consisting only of
# placeholder tokens, markup tags or punctuation. Mirrors the PDF pipeline's
# is_pure_numeric_paragraph() / is_placeholder_only_paragraph() helpers.
_NUMERIC_ONLY_PATTERN = re.compile(r"[-+]?[\d\s.,:%/()\[\]-]+")
_PLACEHOLDER_ONLY_PATTERN = re.compile(
    r"(?:\s|[<{\[(]/?[a-zA-Z0-9_.-]*[>}\])]|__DLP_TOKEN_\d+__|%[sd]|[^\w\s])+"
)

# Matches one complete {"id": <int>, "output": "<escaped string>"} object,
# even when the surrounding JSON array itself is truncated mid-stream (see
# _partial_parse_truncated_batch / docs/plan.md Section 4.3). Deliberately
# does not require array brackets so it can be applied to a raw truncated
# response body.
_BATCH_ITEM_PATTERN = re.compile(
    r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"output"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
)


def _unescape_json_string(value: str) -> str:
    try:
        return orjson.loads(f'"{value}"')
    except Exception:
        return value


def _partial_parse_truncated_batch(raw_text: str) -> list[dict[str, Any]]:
    """Best-effort recovery of complete items from a truncated batch response.

    When LLM_MAX_OUTPUT_TOKENS cuts a batch's JSON array off mid-object, the
    whole array fails to parse via orjson.loads() even though most items in
    it are well-formed. This walks the raw text with a regex and keeps every
    complete {"id": ..., "output": ...} object it can find, so only the
    last (truncated) item or two need the single-unit fallback instead of
    the entire batch.
    """
    items: list[dict[str, Any]] = []
    for match in _BATCH_ITEM_PATTERN.finditer(raw_text):
        try:
            unit_id = int(match.group(1))
        except ValueError:
            continue
        output_text = _unescape_json_string(match.group(2))
        items.append({"id": unit_id, "output": output_text})
    return items


def _build_prompt(batch: list[TranslatableUnit], lang_out: str) -> str:
    json_input = [
        {"id": unit.unit_id, "input": unit.text, "layout_label": unit.label}
        for unit in batch
    ]
    json_input_str = orjson.dumps(json_input, option=orjson.OPT_INDENT_2).decode()
    return (
        f"You are a professional {lang_out} native translator who needs to "
        f"fluently translate text into {lang_out}.\n\n"
        "## Structure Rules\n"
        "1. Keep the same number of items as the input.\n"
        "2. Treat each input item as an independent, fixed unit.\n"
        "3. Translate ALL human-readable content into "
        f"{lang_out}.\n\n"
        "## Do NOT Modify\n"
        "- Placeholders: `{v1}`, `{name}`, `%s`, `%d`, `[[...]]` -- keep exactly unchanged.\n"
        "- Data-masking tokens matching __DLP_TOKEN_NNNN__ -- copy verbatim.\n"
        "- JSON keys or structure.\n\n"
        "## Output Format\n"
        "Return a JSON array of the same length. For each item, keep the same "
        '"id" and add "output" with the translated text only. No extra text, '
        "no ```json blocks.\n\n"
        "## Here is the input:\n\n"
        f"{json_input_str}"
    )


class DocxParagraphTranslator:
    """Translate a DOCX document's units in batches, mirroring PDF's batching."""

    def __init__(self, translate_engine: BaseTranslator, lang_out: str):
        self.translate_engine = translate_engine
        self.lang_out = lang_out
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = None
        self.ok_count = 0
        self.fallback_count = 0
        self.total_count = 0
        self.skipped_count = 0
        self._last_batch_plan = None
        # B1 (docs/plan.md batch-sizing tuning): mirror the PDF pipeline's
        # CJK-aware paragraph-count scaling (translation_config.py's
        # get_token_multiplier / llm_translation_batch_max_paragraphs) so
        # DOCX batches shrink the same way PDF batches do for CJK language
        # pairs -- tiktoken's gpt-4o encoding under-counts CJK tokens
        # relative to what Gemini/Claude actually consume, so a flat
        # paragraph cap risks oversized CJK batches without this scaling.
        lang_in = str(getattr(translate_engine, "lang_in", "") or "")
        self._token_multiplier = max(get_token_multiplier(lang_in, lang_out), 0.1)

    def _calc_token_count(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, len(text) // 4)
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return max(1, len(text) // 4)

    def _batch_units(self, units: list[TranslatableUnit]) -> list[list[TranslatableUnit]]:
        # Adaptive sizing (see doctranslator/batching.py): aim for roughly one
        # full wave across TRANSLATION_POOL_MAX_WORKERS instead of a flat cap
        # that produced 3 batches for a 12-worker pool.
        total_payload = sum(self._calc_token_count(u.text) for u in units)
        plan = compute_batch_plan(
            total_payload,
            settings.TRANSLATION_POOL_MAX_WORKERS,
            base_max_tokens=settings.LLM_TRANSLATION_BATCH_MAX_TOKENS,
            base_max_items=settings.LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS,
            token_multiplier=self._token_multiplier,
        )
        self._last_batch_plan = plan
        max_tokens = plan.max_tokens
        max_paragraphs = plan.max_items
        batches: list[list[TranslatableUnit]] = []
        current: list[TranslatableUnit] = []
        current_tokens = 0
        for unit in units:
            token_count = self._calc_token_count(unit.text)
            current.append(unit)
            current_tokens += token_count
            if current_tokens > max_tokens or len(current) > max_paragraphs:
                batches.append(current)
                current = []
                current_tokens = 0
        if current:
            batches.append(current)
        return batches

    def _should_skip_llm(self, unit: TranslatableUnit) -> bool:
        """Return True for units that must never reach the LLM.

        Mirrors the PDF pipeline's `_is_paragraph_skippable()` checks
        (min_text_length / pure-numeric / placeholder-only). Without this,
        the 2026-08-24 baseline sent single characters and bare numbers to
        Gemini -- a ~3,381-char boilerplate prompt to translate 5 chars --
        and then logged a `length_ratio` validation failure when the model
        unsurprisingly returned something of a different length.
        """
        text = (unit.text or "").strip()
        if not text:
            return True
        if len(text) < int(settings.LLM_TRANSLATION_MIN_TEXT_LENGTH):
            return True
        if _NUMERIC_ONLY_PATTERN.fullmatch(text):
            return True
        return bool(_PLACEHOLDER_ONLY_PATTERN.fullmatch(text))

    def _validate_translation(self, input_text: str, output_text: str) -> bool:
        """Return True if the translation should be treated as a fallback failure."""
        if not output_text.strip():
            logger.warning("DOCX translation validation failed (empty): output is empty or blank")
            return True

        trimmed_input = _TRIM_REPEAT_PATTERN.sub(".", input_text)
        input_tokens = self._calc_token_count(trimmed_input)
        output_tokens = self._calc_token_count(output_text)

        if (
            trimmed_input == output_text
            and input_tokens > 10
            and not settings.LLM_DISABLE_SAME_TEXT_FALLBACK
        ):
            logger.warning(
                f"DOCX translation validation failed (same_text): output matches input (input_tokens={input_tokens})"
            )
            return True

        if output_tokens == 0 or not (0.3 < output_tokens / max(input_tokens, 1) < 3):
            ratio = output_tokens / max(input_tokens, 1)
            logger.warning(
                f"DOCX translation validation failed (length_ratio): ratio={ratio:.2f} "
                f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
            )
            return True

        if not settings.LLM_DISABLE_SAME_TEXT_FALLBACK and input_tokens > 20:
            edit_distance = Levenshtein.distance(input_text, output_text)
            if edit_distance < 5:
                logger.warning(
                    f"DOCX translation validation failed (edit_distance): distance={edit_distance} (<5, input_tokens={input_tokens})"
                )
                return True

        return False

    def _translate_single_fallback(self, unit: TranslatableUnit) -> str:
        """Translate one unit alone via the plain (non-batch) translate() call."""
        self.fallback_count += 1
        try:
            return self.translate_engine.translate(unit.text) or unit.text
        except Exception:
            logger.warning(
                f"Fallback single-unit translation failed for unit {unit.unit_id}",
                exc_info=True,
            )
            return unit.text

    def _translate_units_fallback_batched(
        self, units: list[TranslatableUnit]
    ) -> dict[int, str]:
        """Retry failed units in small batches before true one-at-a-time calls.

        A single-unit fallback re-sends the full ~760-token boilerplate prompt
        to translate as little as 5 characters; the 2026-08-24 baseline spent
        69s on 45 such calls at ~2.4 output-tok/s. Regrouping the failures
        into small batches amortises that boilerplate across ~10 units. Only
        units that fail *again* fall through to genuine singleton calls, so
        the correctness guarantee is unchanged.
        """
        if not units:
            return {}
        group_size = max(1, int(settings.LLM_FALLBACK_BATCH_SIZE))
        if len(units) <= 1 or group_size <= 1:
            return self._translate_units_fallback_parallel(units)

        groups = [
            units[i : i + group_size] for i in range(0, len(units), group_size)
        ]
        logger.info(
            f"DOCX fallback re-batching {len(units)} unit(s) into "
            f"{len(groups)} group(s) of <= {group_size}"
        )
        results: dict[int, str] = {}
        still_failing: list[TranslatableUnit] = []
        max_workers = min(len(groups), max(1, int(settings.TRANSLATION_POOL_MAX_WORKERS)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_group = {
                executor.submit(self._translate_batch_raw, group): group
                for group in groups
            }
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    group_results = future.result()
                except Exception:
                    logger.warning(
                        "DOCX fallback group failed; deferring to single-unit calls",
                        exc_info=True,
                    )
                    still_failing.extend(group)
                    continue
                for unit in group:
                    text = group_results.get(unit.unit_id)
                    if text and not self._validate_translation(unit.text, text):
                        results[unit.unit_id] = text
                        self._write_unit_cache(unit, text)
                    else:
                        still_failing.append(unit)

        if still_failing:
            logger.info(
                f"DOCX fallback: {len(still_failing)} unit(s) still failing after "
                "re-batching; using single-unit translation"
            )
            results.update(self._translate_units_fallback_parallel(still_failing))
        return results

    def _translate_batch_raw(
        self, batch: list[TranslatableUnit]
    ) -> dict[int, str]:
        """Send one batch prompt and return parsed {unit_id: text}, unvalidated.

        Used by the fallback re-batching path, which applies its own
        validation and decides what still needs a singleton retry. Raises on
        transport/parse failure so the caller can defer the whole group.
        """
        prompt = _build_prompt(batch, self.lang_out)
        raw = self.translate_engine.llm_translate(
            prompt,
            response_schema=BatchTranslationResponse,
            batch_items=len(batch),
        )
        parsed = orjson.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("translations") or parsed.get("results") or []
        out: dict[int, str] = {}
        by_id = {u.unit_id for u in batch}
        for item in parsed or []:
            if not isinstance(item, dict) or "id" not in item:
                continue
            try:
                unit_id = int(item["id"])
            except (TypeError, ValueError):
                continue
            if unit_id in by_id:
                out[unit_id] = str(item.get("output", ""))
        return out

    def _translate_units_fallback_parallel(
        self, units: list[TranslatableUnit]
    ) -> dict[int, str]:
        """Run single-unit fallback translation for `units` concurrently.

        Previously this was a plain sequential `for unit in units: ...`
        loop, so a batch with many rejected/failed units (e.g. 47/414 in a
        sampled job) burned one single-unit LLM call after another inside
        one worker thread. Fanning these out to a small thread pool lets
        them run in parallel, bounded by TRANSLATION_POOL_MAX_WORKERS,
        mirroring the PDF pipeline's dedicated fallback executor.
        """
        if not units:
            return {}
        if len(units) == 1:
            unit = units[0]
            return {unit.unit_id: self._translate_single_fallback(unit)}
        max_workers = min(len(units), max(1, int(settings.TRANSLATION_POOL_MAX_WORKERS)))
        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_unit = {
                executor.submit(self._translate_single_fallback, unit): unit
                for unit in units
            }
            for future in as_completed(future_to_unit):
                unit = future_to_unit[future]
                results[unit.unit_id] = future.result()
        return results

    def _translate_batch(
        self, batch: list[TranslatableUnit]
    ) -> dict[int, str]:
        """Translate one batch; returns {unit_id: translated_text}."""
        prompt = _build_prompt(batch, self.lang_out)
        results: dict[int, str] = {}
        try:
            llm_output = self.translate_engine.llm_translate(
                prompt,
                response_schema=BatchTranslationResponse,
                batch_items=len(batch),
            )
            cleaned = llm_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = orjson.loads(cleaned.strip())
            if isinstance(parsed, dict):
                for key in ("items", "translations", "results", "data"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        parsed = value
                        break
            if not isinstance(parsed, list):
                raise ValueError(f"Unexpected LLM output type: {type(parsed).__name__}")
        except orjson.JSONDecodeError:
            # Truncated JSON array (LLM_MAX_OUTPUT_TOKENS cut the response
            # off mid-object) -- recover whatever complete items we can
            # instead of falling all the way back to per-unit translation
            # for the whole batch (docs/plan.md Section 4.3).
            parsed = _partial_parse_truncated_batch(llm_output)
            if parsed:
                logger.warning(
                    f"Batch translation response truncated for {len(batch)} "
                    f"units; recovered {len(parsed)} complete item(s) via "
                    "partial parse, falling back to single-unit translation "
                    "for the remainder",
                )
            else:
                logger.warning(
                    f"Batch translation failed for {len(batch)} units; "
                    "falling back to single-unit translation for the whole "
                    "batch",
                    exc_info=True,
                )
                results.update(self._translate_units_fallback_parallel(batch))
                self.total_count += len(batch)
                return results
        except Exception:
            logger.warning(
                f"Batch translation failed for {len(batch)} units; falling back "
                "to single-unit translation for the whole batch",
                exc_info=True,
            )
            results.update(self._translate_units_fallback_parallel(batch))
            self.total_count += len(batch)
            return results

        try:
            by_id = {unit.unit_id: unit for unit in batch}
            needs_fallback: list[TranslatableUnit] = []
            for item in parsed:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                unit_id = int(item["id"])
                output_text = str(item.get("output", ""))
                unit = by_id.get(unit_id)
                if unit is None:
                    continue
                if self._validate_translation(unit.text, output_text):
                    needs_fallback.append(unit)
                else:
                    results[unit_id] = output_text
                    self.ok_count += 1
                    self._write_unit_cache(unit, output_text)
                self.total_count += 1

            # Any units missing from the response also need the fallback.
            for unit in batch:
                if unit.unit_id not in results and unit not in needs_fallback:
                    needs_fallback.append(unit)
                    self.total_count += 1

            if needs_fallback:
                results.update(self._translate_units_fallback_batched(needs_fallback))
        except Exception:
            logger.warning(
                f"Batch translation failed for {len(batch)} units; falling back "
                "to single-unit translation for the whole batch",
                exc_info=True,
            )
            results.update(self._translate_units_fallback_parallel(batch))
            self.total_count += len(batch)

        return results

    def _unit_cache_key(self, unit: TranslatableUnit) -> str | None:
        engine = self.translate_engine
        if not unit.text:
            return None
        return build_cache_key(
            provider=str(engine.provider),
            model=str(getattr(engine, "model", "")),
            lang_in=str(getattr(engine, "lang_in", "")),
            lang_out=self.lang_out,
            text=unit.text,
        )

    def _split_cache_hits(
        self, units: list[TranslatableUnit]
    ) -> tuple[dict[int, str], list[TranslatableUnit]]:
        """Check Redis per-unit before batching (docs/plan.md Section 4.4).

        Splitting cache hits out *before* batches are assembled means one
        changed paragraph no longer invalidates an entire batch's worth of
        LLM work -- only cache-miss units get sent to the LLM at all.

        Cache membership is checked via a single pipelined MGET instead of
        N serial round trips (one per unit), which matters for documents
        with hundreds of units and a non-trivial Redis network hop.
        """
        cache = get_translation_cache()
        cache_hits: dict[int, str] = {}
        cache_misses: list[TranslatableUnit] = []

        unit_keys: dict[int, str] = {}
        for unit in units:
            cache_key = self._unit_cache_key(unit)
            if cache_key is not None:
                unit_keys[unit.unit_id] = cache_key

        cached_values = cache.get_many(list(unit_keys.values()))
        for unit in units:
            cache_key = unit_keys.get(unit.unit_id)
            cached = cached_values.get(cache_key) if cache_key is not None else None
            if cached is not None:
                cache_hits[unit.unit_id] = cached
            else:
                cache_misses.append(unit)
        return cache_hits, cache_misses

    def _write_unit_cache(self, unit: TranslatableUnit, translated_text: str) -> None:
        cache_key = self._unit_cache_key(unit)
        if cache_key is None or not translated_text.strip():
            return
        engine = self.translate_engine
        get_translation_cache().set(
            cache_key,
            translated_text,
            provider=str(engine.provider),
            model=str(getattr(engine, "model", "")),
            lang_in=str(getattr(engine, "lang_in", "")),
            lang_out=self.lang_out,
        )

    def translate_all(self, units: list[TranslatableUnit]) -> dict[int, str]:
        """Translate every unit; returns {unit_id: translated_text}.

        Batches run concurrently, bounded by TRANSLATION_POOL_MAX_WORKERS --
        same concurrency model as the PDF pipeline's paragraph translation.
        Cache-hit units (per docs/plan.md Section 4.4) are served directly
        from Redis and never enter a batch prompt at all.
        """
        # Drop non-translatable units (numbers, placeholders, sub-minimum
        # length) before doing any Redis or LLM work -- they are passed
        # through verbatim, exactly as the PDF pipeline already does.
        translatable: list[TranslatableUnit] = []
        results: dict[int, str] = {}
        for unit in units:
            if self._should_skip_llm(unit):
                results[unit.unit_id] = unit.text
                self.skipped_count += 1
            else:
                translatable.append(unit)
        if self.skipped_count:
            logger.info(
                f"DOCX pre-filter skipped {self.skipped_count}/{len(units)} "
                "non-translatable unit(s) before batching"
            )

        cache_hits, cache_miss_units = self._split_cache_hits(translatable)
        results.update(cache_hits)
        self.total_count += len(cache_hits)
        self.ok_count += len(cache_hits)

        batches = self._batch_units(cache_miss_units)
        if self._last_batch_plan is not None:
            log_batch_plan("DocxTranslateParagraphs", self._last_batch_plan, len(batches))
        if not batches:
            self._log_completion(len(cache_hits), batch_count=0)
            return results

        max_workers = max(1, int(settings.TRANSLATION_POOL_MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._translate_batch, batch) for batch in batches]
            for future in as_completed(futures):
                batch_results = future.result()
                results.update(batch_results)

        self._log_completion(len(cache_hits), batch_count=len(batches))
        return results

    def _log_completion(self, cache_hits: int, batch_count: int) -> None:
        """E1 instrumentation: attempt/fallback/cache/batch counters.

        Mirrors ILTranslatorLLMOnly's completion log in the PDF pipeline
        (il_translator_llm_only.py) so both pipelines emit comparable
        structured log lines for latency/throughput investigation.
        """
        cache_stats = get_translation_cache().stats_snapshot()
        logger.info(
            f"DOCX translation completed. Total: {self.total_count}, "
            f"Successful: {self.ok_count}, Fallback: {self.fallback_count}, "
            f"Cache hits (this run): {cache_hits}, Batches: {batch_count}, "
            f"cache_hits_cumulative={cache_stats['cache_hits']} "
            f"cache_misses_cumulative={cache_stats['cache_misses']} "
            f"cache_hit_rate_cumulative={cache_stats['cache_hit_rate']}"
        )
