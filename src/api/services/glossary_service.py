"""Domain glossary retrieval service."""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

from src.config.constants import settings
from src.config.logging_config import logger
from src.doctranslator.glossary import Glossary
from src.doctranslator.glossary import GlossaryEntry
from src.loaders.utils.path_helpers import get_cache_file_path
from src.repository import get_storage_client


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
