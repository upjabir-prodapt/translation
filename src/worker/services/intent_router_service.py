"""Intent-based model routing service."""

from __future__ import annotations

import json
import logging
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

from src.config.constants import settings
from src.config.translation_routing import select_model_list
from src.worker.loaders.utils.path_helpers import get_cache_file_path
from src.repository import get_storage_client

logger = logging.getLogger(__name__)


class IntentRouterService:
    """Resolve intent and model priority chain from routing config."""

    def __init__(self, cache_ttl_seconds: int = 300):
        ttl_seconds = cache_ttl_seconds or settings.MODEL_SELECTION_CACHE_TTL_SECONDS
        self._cache_ttl = timedelta(seconds=ttl_seconds)
        self._last_loaded: datetime | None = None
        self._config: dict[str, Any] | None = None

    def build_intent(self, domain: str, source_lang: str, target_lang: str) -> str:
        return f"Intent-{domain.title()}-{source_lang.upper()}-{target_lang.upper()}"

    def _get_model_selection_path(self) -> Path:
        return get_cache_file_path(settings.MODEL_SELECTION_FILENAME)

    def _local_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            return False
        return datetime.now(UTC) - mtime < self._cache_ttl

    def _load_from_gcs(self) -> dict[str, Any]:
        client = get_storage_client()
        bucket = client.bucket(settings.GCS_BUCKET_NAME)
        blob_path = f"{settings.GCS_ASSETS_PREFIX}/{settings.MODEL_SELECTION_FILENAME}"
        blob = bucket.blob(blob_path)
        raw = blob.download_as_text()
        local_path = self._get_model_selection_path()
        local_path.write_text(raw, encoding="utf-8")
        logger.info(f"Updated local model selection cache at {local_path}")
        return json.loads(raw)

    def _load_from_local_fallback(self) -> dict[str, Any]:
        fallback_path = self._get_model_selection_path()
        if fallback_path.exists():
            return json.loads(fallback_path.read_text(encoding="utf-8"))
        return {}

    def sync_model_selection_cache(self, force: bool = False) -> dict[str, Any]:
        """Ensure model selection exists in assets root and cache it in memory."""
        local_config = self._load_from_local_fallback()
        local_path = self._get_model_selection_path()
        if local_config and not force and self._local_cache_fresh(local_path):
            self._config = local_config
            self._last_loaded = datetime.now(UTC)
            logger.info(f"Using fresh local model selection cache from {local_path}")
            return local_config

        try:
            fresh = self._load_from_gcs()
            self._config = fresh
            self._last_loaded = datetime.now(UTC)
            logger.info("Fetched model selection cache from GCS")
            return fresh
        except Exception as e:
            logger.warning(f"Model selection sync failed from GCS: {e}")
            if local_config:
                self._config = local_config
                self._last_loaded = datetime.now(UTC)
                logger.warning(
                    f"Using stale local model selection cache from {local_path}"
                )
                return local_config
            raise

    def _get_config(self) -> dict[str, Any] | list[dict[str, Any]]:
        now = datetime.now(UTC)
        if (
            self._config
            and self._last_loaded
            and (now - self._last_loaded) < self._cache_ttl
        ):
            return self._config
        try:
            self._config = self.sync_model_selection_cache()
        except Exception as e:
            logger.warning(f"Failed to refresh model selection cache in memory: {e}")
            self._config = self._load_from_local_fallback()
        self._last_loaded = now
        return self._config or {}

    def get_model_chain(
        self, *, domain: str, source_lang: str, target_lang: str
    ) -> list[str]:
        # model_selection.json is authoritative and uses source/target/domain matching.
        return select_model_list(source_lang, target_lang, domain)
