"""Domain glossary retrieval service."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

from google.api_core.exceptions import PreconditionFailed

from src.config.constants import settings
from src.config.glossary_hygiene import HygieneContext
from src.config.glossary_hygiene import Trust
from src.config.glossary_hygiene import sanitize_term_pairs
from src.config.translation_routing import get_language_display_name
from src.config.translation_routing import normalize_language
from src.repository import get_storage_client
from src.worker.doctranslator.glossary import ExtractedGlossaryTerm
from src.worker.doctranslator.glossary import Glossary
from src.worker.doctranslator.glossary import GlossaryEntry
from src.worker.loaders.utils.path_helpers import get_cache_file_path

logger = logging.getLogger(__name__)

# Bounded retry count for the optimistic-concurrency read-modify-write loop
# against the domain glossary JSON in GCS (see merge_new_terms_into_domain_glossary).
_MERGE_MAX_RETRIES = 5


def _normalize_glossary_language_key(raw: str) -> str | None:
    """Normalize a glossary JSON language key to a canonical ISO code.

    Glossary buckets and per-term `translations` keys are not guaranteed to
    use the same shape: terms this service writes itself use canonical
    codes (`ExtractedGlossaryTerm.source_language` is produced by
    `normalize_language()` upstream), but shipped/seeded glossary assets use
    display names (`"English"`, `"French"`). `normalize_language()` accepts
    both -- `language_mapper.json`'s alias table maps display names, ISO
    codes, and common variants all to the same canonical code -- so routing
    every language key through it here makes bucket and translation lookups
    tolerant of either shape instead of only ever matching one. Returns
    `None` (rather than raising) for a key that matches neither, so callers
    can skip it instead of crashing on unrelated/malformed data.
    """
    try:
        return normalize_language(str(raw))
    except ValueError:
        return None


def _lookup_translation(
    translations: dict[str, Any], target_language_name: str
) -> str | None:
    """Find a term's translation for the target language.

    Tries an exact key match first (the common case: both sides already
    canonical codes), then falls back to comparing normalized forms so a
    canonical-code `target_language_name` still finds a translation stored
    under a legacy display-name key, and vice versa.
    """
    if target_language_name in translations:
        return translations[target_language_name]
    target_norm = _normalize_glossary_language_key(target_language_name)
    if target_norm is None:
        return None
    for raw_key, value in translations.items():
        if _normalize_glossary_language_key(raw_key) == target_norm:
            return value
    return None


class GlossaryService:
    """Load domain-specific terminology from the per-domain JSON in GCS."""

    def __init__(self) -> None:
        # Keyed by (domain, source_language, target_language). The raw JSON is
        # already cached on disk; this caches the *assembled* result, which is
        # where the work is -- every load re-runs the hygiene gate over ~1,500
        # pairs. Same TTL as the file cache, so there is one number to reason
        # about.
        self._pair_cache: dict[tuple[str, str, str], tuple[float, list[Glossary]]] = {}
        self._pair_cache_lock = threading.Lock()

    def _cache_get(self, key: tuple[str, str, str]) -> list[Glossary] | None:
        ttl = float(settings.GLOSSARY_CACHE_TTL_SECONDS)
        with self._pair_cache_lock:
            entry = self._pair_cache.get(key)
            if entry is None:
                return None
            cached_at, glossaries = entry
            if (time.monotonic() - cached_at) >= ttl:
                self._pair_cache.pop(key, None)
                return None
            return glossaries

    def _cache_put(self, key: tuple[str, str, str], glossaries: list[Glossary]) -> None:
        with self._pair_cache_lock:
            self._pair_cache[key] = (time.monotonic(), glossaries)

    def invalidate_cache(self) -> None:
        """Drop the in-process glossary cache. For tests and for a forced reload."""
        with self._pair_cache_lock:
            self._pair_cache.clear()

    def _get_local_glossary_path(self, domain: str) -> Path:
        return get_cache_file_path(f"{domain}.json", settings.GLOSSARIES_DIR)

    def _local_cache_fresh(self, local_path: Path) -> bool:
        try:
            mtime = datetime.fromtimestamp(local_path.stat().st_mtime, tz=UTC)
        except OSError:
            return False
        return datetime.now(UTC) - mtime < timedelta(
            seconds=settings.GLOSSARY_CACHE_TTL_SECONDS
        )

    def _load_local_glossary_json(self, domain: str) -> dict[str, Any] | None:
        local_path = self._get_local_glossary_path(domain)
        if not local_path.exists():
            return None
        if not self._local_cache_fresh(local_path):
            return None
        try:
            return json.loads(local_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Invalid local glossary cache '{local_path}': {e}")
            return None

    def _download_glossary_json(
        self, domain: str, refresh: bool = False
    ) -> dict[str, Any]:
        if not refresh:
            local_data = self._load_local_glossary_json(domain)
            if local_data is not None:
                return local_data

        client = get_storage_client()
        bucket = client.bucket(settings.GCS_BUCKET_NAME)
        blob_path = f"{settings.GCS_ASSETS_PREFIX}/{settings.GCS_GLOSSARIES_PREFIX}/{domain}.json"
        blob = bucket.blob(blob_path)
        raw = blob.download_as_text()
        data = json.loads(raw)
        local_path = self._get_local_glossary_path(domain)
        local_path.write_text(raw, encoding="utf-8")
        logger.info(f"Cached glossary for domain '{domain}' at {local_path}")
        return data

    def prefetch_domain_glossary(self, domain: str) -> bool:
        """Best-effort glossary prefetch into assets root."""
        try:
            self._download_glossary_json(domain, refresh=False)
            return True
        except Exception as e:
            logger.warning(f"Glossary prefetch failed for domain '{domain}': {e}")
            return False

    @staticmethod
    def _language_keys(value: str) -> set[str]:
        """Every spelling a glossary file might use for one language.

        The files on disk are inconsistent: hand-authored sections key their
        translations by display name (``"Spanish"``) while auto-extracted ones
        use the canonical code (``"es"``). Callers pass the code, so a lookup
        that only tried the code silently skipped every curated term -- which
        is why Colt's approved terminology never reached a single translation.
        Matching on the whole set makes both spellings resolve.
        """
        raw = str(value or "").strip()
        if not raw:
            return set()
        keys = {raw, raw.lower(), raw.title()}
        try:
            code = normalize_language(raw)
        except ValueError:
            return keys
        keys.add(code)
        keys.add(get_language_display_name(code))
        return keys

    @classmethod
    def _lookup(cls, payload: dict[str, Any], value: str) -> Any:
        """First value under any spelling of `value`. For a term's translations,
        where one language has exactly one form."""
        for key in cls._language_keys(value):
            if key in payload:
                return payload[key]
        return None

    @classmethod
    def _lookup_all(cls, payload: dict[str, Any], value: str) -> list[Any]:
        """*Every* value under any spelling of `value`.

        Sections, unlike translations, can legitimately be split across
        spellings: the glossary files written before this change carry both an
        ``English`` section (hand-authored) and an ``en`` one (auto-extracted).
        Returning only the first would silently discard one of them.
        """
        return [payload[key] for key in cls._language_keys(value) if key in payload]

    def _load_from_gcs(
        self, *, domain: str, target_language_name: str, source_language: str | None
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
        """Shred the GCS JSON into (curated pairs, learned pairs, preserve)."""
        data = self._download_glossary_json(domain, refresh=False)

        source_language_map = data.get("glossary", {}) or {}
        if source_language:
            sections = [
                s
                for s in self._lookup_all(source_language_map, source_language)
                if isinstance(s, dict)
            ]
        else:
            sections = [s for s in source_language_map.values() if isinstance(s, dict)]

        # Split by declared provenance. A file that does not say is treated as
        # machine-written, which is the safe default: the legacy files mix
        # hand-authored and extracted entries in one undifferentiated structure.
        curated_pairs: list[tuple[str, str]] = []
        learned_pairs: list[tuple[str, str]] = []
        preserve: list[str] = []
        for lang_payload in sections:
            for term in lang_payload.get("terms", []) or []:
                source_term = term.get("source_term")
                translations = term.get("translations", {}) or {}
                target_term = self._lookup(translations, target_language_name)
                if not (source_term and target_term):
                    continue
                pair = (str(source_term), str(target_term))
                if Trust.from_origin(term.get("origin")) is Trust.CURATED:
                    curated_pairs.append(pair)
                else:
                    learned_pairs.append(pair)
            preserve.extend(
                str(t) for t in (lang_payload.get("preserve_as_is") or []) if t
            )

        return curated_pairs, learned_pairs, preserve

    def load_domain_glossary(
        self,
        *,
        domain: str,
        target_language_name: str,
        source_language: str | None = None,
    ) -> list[Glossary]:
        """Load the approved terminology for one language *pair*.

        `source_language` is optional only so older callers keep working.
        Falling back to every source section is the cross-language
        contamination in TRANSLATION_FIX_PLAN.md RC-3: a French->Italian job
        inheriting terms a Spanish->Italian job had written.

        Every entry is re-checked against the hygiene gate on the way out.
        Filtering on read is what makes anything that gets into the file by
        another route inert without waiting for a cleanup.
        """
        cache_key = (domain, source_language or "", target_language_name)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            curated_pairs, learned_pairs, preserve = self._load_from_gcs(
                domain=domain,
                target_language_name=target_language_name,
                source_language=source_language,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load glossary for domain '{domain}' "
                f"({source_language or 'all'}->{target_language_name}): {e}",
                exc_info=True,
            )
            return []

        # Business vocabulary for the person-name heuristic, taken from the
        # hand-authored do-not-translate list rather than a list maintained in
        # code. Non-circular: preserve entries are trusted content, and are
        # never among the pairs being judged.
        context = HygieneContext.from_terms(preserve)
        kept_curated, curated_rejected = sanitize_term_pairs(
            curated_pairs, trust=Trust.CURATED, context=context
        )
        kept_learned, rejected = sanitize_term_pairs(
            learned_pairs, trust=Trust.LEARNED, context=context
        )
        kept = kept_curated + kept_learned
        for reason, count in curated_rejected.items():
            rejected[reason] = rejected.get(reason, 0) + count
        # `preserve_as_is` is hand-authored "do not translate this", so an
        # identity pair there is the intent rather than a defect.
        kept_preserve, preserve_rejected = sanitize_term_pairs(
            [(t, t) for t in preserve], trust=Trust.PRESERVE
        )
        if rejected or preserve_rejected:
            logger.warning(
                f"Dropped unusable glossary entries for domain '{domain}' "
                f"({source_language or 'all'}->{target_language_name}): "
                f"terms={rejected} preserve_as_is={preserve_rejected}"
            )

        entries = [
            GlossaryEntry(
                source=term, target=translation, target_language=target_language_name
            )
            for term, translation in kept + kept_preserve
        ]
        # An empty result is cached too: "this domain has nothing for this
        # pair" is a real answer, and re-deriving it costs the same as a
        # populated one.
        glossaries = (
            [Glossary(name=f"{domain}-glossary", entries=entries)] if entries else []
        )
        self._cache_put(cache_key, glossaries)
        if entries:
            logger.info(
                f"Loaded {len(entries)} glossary term(s) for domain '{domain}' "
                f"({source_language or 'all'}->{target_language_name})"
            )
        return glossaries

    def _blob_path_for_domain(self, domain: str) -> str:
        return f"{settings.GCS_ASSETS_PREFIX}/{settings.GCS_GLOSSARIES_PREFIX}/{domain}.json"

    def _merge_terms_into_json(
        self,
        data: dict[str, Any],
        *,
        target_language_name: str,
        new_terms: list[ExtractedGlossaryTerm],
    ) -> tuple[dict[str, Any], int]:
        """Merge extracted terms into the glossary JSON structure.

        Each term is bucketed under its own `source_language`, not a single
        language for the whole call -- a single extraction batch over a
        mixed-language document can (and should) contribute terms to
        several buckets. See `ExtractedGlossaryTerm` and
        `load_domain_glossary`'s `source_languages` filter for why the
        bucket a term lands in matters at read time.

        Returns (updated_data, added_count). Skips any source_term that
        already has a translation for target_language_name in its own
        bucket (first-writer-wins per term, including terms added by a
        concurrent job since this read).
        """
        # Canonicalise the section key so a job routed as "es" and one routed
        # as "Spanish" write to the same section instead of forking the file.
        try:
            section_key = normalize_language(source_language)
        except ValueError:
            section_key = str(source_language or "").strip().lower()
        try:
            target_key = normalize_language(target_language_name)
        except ValueError:
            target_key = str(target_language_name or "").strip().lower()
        if not section_key or not target_key:
            return data, 0

        # The gate that stops this file becoming what the UAT round found:
        # identity pairs, function words, DLP tokens, names, dates, headings.
        # Judge the newly learned terms against the terminology this domain
        # already holds, so onboarding a domain sharpens the heuristics instead
        # of needing a code change.
        context = HygieneContext.from_terms(
            [
                term.get("source_term", "")
                for section in (data.get("glossary") or {}).values()
                if isinstance(section, dict)
                for term in (section.get("terms") or [])
            ]
            + [
                entry
                for section in (data.get("glossary") or {}).values()
                if isinstance(section, dict)
                for entry in (section.get("preserve_as_is") or [])
            ]
        )
        new_terms, rejected = sanitize_term_pairs(new_terms, context=context)
        if rejected:
            logger.info(
                f"Glossary hygiene rejected {sum(rejected.values())} extracted "
                f"term(s) before merge ({section_key}->{target_key}): {rejected}"
            )
        if not new_terms:
            return data, 0

        glossary_map = data.setdefault("glossary", {})
        lang_payload = glossary_map.setdefault(section_key, {"terms": []})
        terms_list = lang_payload.setdefault("terms", [])

        terms_by_bucket: dict[str, list[ExtractedGlossaryTerm]] = {}
        for term in new_terms:
            if not isinstance(term, ExtractedGlossaryTerm):
                # Defensive: the type hint promises `ExtractedGlossaryTerm`
                # (a NamedTuple with `.source_language`), but a caller
                # passing a plain `(source, target)` tuple -- the old shape,
                # before per-term language attribution existed -- would
                # otherwise hit `AttributeError` on the `.source_language`
                # access below instead of a clear, recoverable log line.
                logger.warning(
                    "Dropping extracted term of unexpected type %s (expected "
                    "ExtractedGlossaryTerm): %r",
                    type(term).__name__,
                    term,
                )
                continue
            if not term.source_language:
                # Should not happen -- callers are expected to drop terms
                # whose reported language didn't normalize before they ever
                # reach here (see term_extractor.py / automatic_term_extractor.py)
                # -- but an ungrouped, un-bucketable term is worse than a
                # skipped one, so fail closed rather than guess a bucket.
                logger.warning(
                    "Dropping extracted term with no source_language: %r", term
                )
                continue
            terms_by_bucket.setdefault(term.source_language, []).append(term)

        added = 0
        for source_term, target_term in new_terms:
            key = source_term.strip().lower()
            if not key or not target_term.strip():
                continue
            existing = existing_by_source.get(key)
            if existing is not None:
                translations = existing.setdefault("translations", {})
                if target_key in translations:
                    # Already has a translation for this language (possibly
                    # added by a concurrent job) -- do not overwrite.
                    continue
                translations[target_key] = target_term
                added += 1
            else:
                new_entry = {
                    "source_term": source_term,
                    "translations": {target_key: target_term},
                }
                terms_list.append(new_entry)
                existing_by_source[key] = new_entry
                added += 1

        return data, added

    def merge_new_terms_into_domain_glossary(
        self,
        *,
        domain: str,
        target_language_name: str,
        new_terms: list[ExtractedGlossaryTerm],
    ) -> bool:
        """Persist newly auto-extracted terms into the domain glossary JSON in GCS.

        Only call this after a translation job has completed successfully --
        terms from failed/low-quality jobs must never reach the shared
        glossary. The hygiene gate runs first, inside `_merge_terms_into_json`.

        Uses optimistic concurrency (GCS `if_generation_match`) so concurrent
        jobs writing to the same domain file cannot silently clobber each
        other's additions; on a generation mismatch the read-modify-write is
        retried up to `_MERGE_MAX_RETRIES` times. First-writer-wins per term,
        so a learned term never displaces a curated one.

        Returns True if any new terms were actually written.
        """
        if not new_terms:
            return False

        written = self._merge_into_gcs(
            domain=domain,
            source_language=source_language,
            target_language_name=target_language_name,
            new_terms=new_terms,
        )
        if written:
            # The next job in this domain should see what this one learned
            # rather than wait out the TTL.
            self.invalidate_cache()
        return written

    def _merge_into_gcs(
        self,
        *,
        domain: str,
        source_language: str,
        target_language_name: str,
        new_terms: list[tuple[str, str]],
    ) -> bool:
        """The read-modify-write cycle. See the caller for the guarantees."""
        blob_path = self._blob_path_for_domain(domain)
        client = get_storage_client()
        bucket = client.bucket(settings.GCS_BUCKET_NAME)

        for attempt in range(_MERGE_MAX_RETRIES):
            blob = bucket.blob(blob_path)
            try:
                blob.reload()
                raw = blob.download_as_text()
                data = json.loads(raw)
                current_generation = blob.generation
            except Exception:
                logger.warning(
                    f"Could not read domain glossary '{domain}' for merge "
                    f"(attempt {attempt + 1}/{_MERGE_MAX_RETRIES})",
                    exc_info=True,
                )
                data = {"glossary": {}}
                current_generation = 0

            merged_data, added_count = self._merge_terms_into_json(
                data,
                target_language_name=target_language_name,
                new_terms=new_terms,
            )
            if added_count == 0:
                logger.info(
                    f"No new terms to merge into domain glossary '{domain}' "
                    f"for language '{target_language_name}' (all already present)"
                )
                return False

            serialized = json.dumps(merged_data, ensure_ascii=False, indent=2)
            try:
                write_blob = bucket.blob(blob_path)
                write_blob.upload_from_string(
                    serialized,
                    content_type="application/json",
                    if_generation_match=current_generation,
                )
            except PreconditionFailed:
                logger.info(
                    f"Domain glossary '{domain}' changed concurrently; "
                    f"retrying merge (attempt {attempt + 1}/{_MERGE_MAX_RETRIES})"
                )
                continue
            except Exception:
                logger.exception(
                    f"Failed to write merged domain glossary '{domain}' to GCS"
                )
                return False

            # Refresh the local per-worker cache immediately so subsequent
            # jobs on this instance see the update without waiting for TTL.
            local_path = self._get_local_glossary_path(domain)
            try:
                local_path.write_text(serialized, encoding="utf-8")
            except Exception:
                logger.debug(
                    f"Failed to refresh local glossary cache for '{domain}' after merge",
                    exc_info=True,
                )

            logger.info(
                f"Merged {added_count} new term(s) into domain glossary "
                f"'{domain}' for language '{target_language_name}'"
            )
            return True

        logger.warning(
            f"Giving up merging new terms into domain glossary '{domain}' "
            f"after {_MERGE_MAX_RETRIES} concurrent-write conflicts"
        )
        return False
