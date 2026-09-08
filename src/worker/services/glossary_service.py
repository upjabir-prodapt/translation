"""Domain glossary retrieval service."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

from google.api_core.exceptions import PreconditionFailed

from src.config.constants import settings
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
    """Load domain-specific glossary from GCS assets."""

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

    def load_domain_glossary(
        self,
        *,
        domain: str,
        target_language_name: str,
        source_languages: Iterable[str] | None = None,
    ) -> list[Glossary]:
        """Load the shared domain glossary, optionally scoped to languages
        actually present in the current job.

        The glossary JSON buckets terms by the source language they were
        extracted from (`ExtractedGlossaryTerm.source_language` --
        `merge_new_terms_into_domain_glossary` writes each term into its own
        bucket). By default (`source_languages=None`) every bucket is
        included, matching the historical behaviour and every existing
        caller that doesn't pass this argument.

        Pass `source_languages` (e.g. `PipelineOrchestrator`'s detected,
        noise-filtered language set for the document being translated) to
        include only the buckets relevant to that document. Without this, a
        term extracted from one document in a language this document never
        contains -- e.g. a French word merged from a past mixed-language
        job -- would still be eligible to literal-string-match this
        document's text purely by coincidence (`GlossaryEntry` matching has
        no language awareness at all; it is a plain string lookup). Scoping
        to the document's own detected languages removes that class of
        false-positive match without needing to change how matching itself
        works.
        """
        try:
            data = self._download_glossary_json(domain, refresh=False)
        except Exception as e:
            logger.warning(
                f"Failed to load glossary for domain '{domain}' and target '{target_language_name}': {e}"
            )
            return []

        allowed_languages = (
            {str(lang).strip().lower() for lang in source_languages}
            if source_languages is not None
            else None
        )

        entries: list[GlossaryEntry] = []
        source_language_map = data.get("glossary", {})
        for bucket_language, lang_payload in source_language_map.items():
            # Normalization only matters for the comparison below: with no
            # filter requested (`allowed_languages is None`), every bucket
            # is included exactly as before regardless of what its key
            # looks like -- a glossary is allowed to hold buckets for
            # languages outside language_mapper.json's *supported* set
            # (e.g. a manually curated reference bucket), and the no-filter
            # path has never validated bucket keys. Only when a caller
            # actually wants to scope by language does an unrecognized key
            # become something to act on, since it can then never be
            # confirmed to belong to the requested set.
            if allowed_languages is not None:
                normalized_bucket = _normalize_glossary_language_key(bucket_language)
                if normalized_bucket is None:
                    logger.warning(
                        "Glossary domain '%s': bucket language '%s' does not "
                        "normalize to a known language; excluding it from "
                        "this language-scoped request.",
                        domain,
                        bucket_language,
                    )
                    continue
                if normalized_bucket not in allowed_languages:
                    continue
            for term in lang_payload.get("terms", []):
                source_term = term.get("source_term")
                translations = term.get("translations", {})
                target_term = _lookup_translation(translations, target_language_name)
                if not source_term or not target_term:
                    continue
                entries.append(
                    GlossaryEntry(
                        source=str(source_term),
                        target=str(target_term),
                        target_language=target_language_name,
                    )
                )
        if not entries:
            return []
        return [Glossary(name=f"{domain}-json-glossary", entries=entries)]

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
        glossary_map = data.setdefault("glossary", {})

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
        for bucket_language, bucket_terms in terms_by_bucket.items():
            lang_payload = glossary_map.setdefault(bucket_language, {"terms": []})
            terms_list = lang_payload.setdefault("terms", [])

            existing_by_source = {
                str(existing.get("source_term", "")).strip().lower(): existing
                for existing in terms_list
                if existing.get("source_term")
            }

            for term in bucket_terms:
                key = term.source.strip().lower()
                if not key or not term.target.strip():
                    continue
                existing = existing_by_source.get(key)
                if existing is not None:
                    translations = existing.setdefault("translations", {})
                    if target_language_name in translations:
                        # Already has a translation for this language
                        # (possibly added by a concurrent job) -- do not
                        # overwrite.
                        continue
                    translations[target_language_name] = term.target
                    added += 1
                else:
                    new_entry = {
                        "source_term": term.source,
                        "translations": {target_language_name: term.target},
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
        terms from failed/low-quality jobs should never reach the shared
        glossary. Uses optimistic concurrency (GCS if_generation_match) so
        concurrent jobs writing to the same domain file cannot silently
        clobber each other's additions; on a generation mismatch, the read-
        modify-write is retried up to _MERGE_MAX_RETRIES times.

        Returns True if any new terms were actually written.
        """
        if not new_terms:
            return False

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
