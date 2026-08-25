"""Domain glossary retrieval service."""

from __future__ import annotations

import json
import logging
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

from google.api_core.exceptions import PreconditionFailed

from src.config.constants import settings
from src.repository import get_storage_client
from src.worker.doctranslator.glossary import Glossary
from src.worker.doctranslator.glossary import GlossaryEntry
from src.worker.loaders.utils.path_helpers import get_cache_file_path

logger = logging.getLogger(__name__)

# Bounded retry count for the optimistic-concurrency read-modify-write loop
# against the domain glossary JSON in GCS (see merge_new_terms_into_domain_glossary).
_MERGE_MAX_RETRIES = 5


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
        self, *, domain: str, target_language_name: str
    ) -> list[Glossary]:
        try:
            data = self._download_glossary_json(domain, refresh=False)
        except Exception as e:
            logger.warning(
                f"Failed to load glossary for domain '{domain}' and target '{target_language_name}': {e}"
            )
            return []

        entries: list[GlossaryEntry] = []
        source_language_map = data.get("glossary", {})
        for _, lang_payload in source_language_map.items():
            for term in lang_payload.get("terms", []):
                source_term = term.get("source_term")
                translations = term.get("translations", {})
                target_term = translations.get(target_language_name)
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
        source_language: str,
        target_language_name: str,
        new_terms: list[tuple[str, str]],
    ) -> tuple[dict[str, Any], int]:
        """Merge (source_term, target_term) pairs into the glossary JSON structure.

        Returns (updated_data, added_count). Skips any source_term that
        already has a translation for target_language_name (first-writer-wins
        per term, including terms added by a concurrent job since this read).
        """
        glossary_map = data.setdefault("glossary", {})
        lang_payload = glossary_map.setdefault(source_language, {"terms": []})
        terms_list = lang_payload.setdefault("terms", [])

        existing_by_source = {
            str(term.get("source_term", "")).strip().lower(): term
            for term in terms_list
            if term.get("source_term")
        }

        added = 0
        for source_term, target_term in new_terms:
            key = source_term.strip().lower()
            if not key or not target_term.strip():
                continue
            existing = existing_by_source.get(key)
            if existing is not None:
                translations = existing.setdefault("translations", {})
                if target_language_name in translations:
                    # Already has a translation for this language (possibly
                    # added by a concurrent job) -- do not overwrite.
                    continue
                translations[target_language_name] = target_term
                added += 1
            else:
                new_entry = {
                    "source_term": source_term,
                    "translations": {target_language_name: target_term},
                }
                terms_list.append(new_entry)
                existing_by_source[key] = new_entry
                added += 1

        return data, added

    def merge_new_terms_into_domain_glossary(
        self,
        *,
        domain: str,
        source_language: str,
        target_language_name: str,
        new_terms: list[tuple[str, str]],
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
                source_language=source_language,
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
