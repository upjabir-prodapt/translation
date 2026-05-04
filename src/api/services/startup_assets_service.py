"""Startup orchestration for assets root preflight and background warmup."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from src.api.services.glossary_service import GlossaryService
from src.api.services.intent_router_service import IntentRouterService
from src.config.constants import settings
from src.config.logging import logger
from src.loaders.services.download_service import download_async
from src.loaders.services.warmup_service import WarmupService
from src.loaders.utils.path_helpers import get_cache_file_path
from src.loaders.utils.path_helpers import get_cache_root
from src.loaders.utils.path_helpers import get_subdir_path


class StartupAssetsService:
    """Manage startup preflight and optional background asset warmup."""

    def __init__(self) -> None:
        self._intent_router = IntentRouterService()
        self._glossary_service = GlossaryService()

    @dataclass
    class PreflightStatus:
        assets_root: str
        critical_sync_ok: bool
        critical_files_ready: list[str]

    def ensure_assets_layout(self) -> Path:
        """Ensure the canonical assets root and known subdirectories exist."""
        root = get_cache_root()
        root.mkdir(parents=True, exist_ok=True)
        for subdir in (
            settings.MODELS_DIR,
            settings.METADATA_DIR,
            settings.FONTS_DIR,
            settings.CMAP_DIR,
            settings.TIKTOKEN_DIR,
            settings.GLOSSARIES_DIR,
            "tmp",
        ):
            get_subdir_path(subdir)

        probe = root / ".startup_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return root

    async def _ensure_metadata_indexes(self) -> list[str]:
        ready_files: list[str] = []
        missing_files: list[str] = []
        for filename in (settings.FONT_METADATA_FILENAME, settings.CMAP_METADATA_FILENAME):
            local_path = get_cache_file_path(filename, settings.METADATA_DIR)
            if local_path.exists():
                json.loads(local_path.read_text(encoding="utf-8"))
                ready_files.append(f"{settings.METADATA_DIR}/{filename}")
                continue
            try:
                blob_path = f"{settings.METADATA_DIR}/{filename}"
                await download_async(blob_path, local_path)
                json.loads(local_path.read_text(encoding="utf-8"))
                ready_files.append(blob_path)
                logger.info(f"Downloaded required metadata index '{filename}'")
            except Exception as e:
                logger.warning(f"Metadata preflight failed for '{filename}': {e}")
                missing_files.append(filename)
        if missing_files:
            raise RuntimeError(
                f"Required metadata index files unavailable: {', '.join(missing_files)}"
            )
        return ready_files

    async def run_preflight(self) -> PreflightStatus:
        """Run lightweight startup checks and cache critical routing assets."""
        assets_root = self.ensure_assets_layout()
        logger.info(f"Assets root ready: {assets_root}")

        critical_files_ready: list[str] = []
        self._intent_router.sync_model_selection_cache(force=True)
        critical_files_ready.append(settings.MODEL_SELECTION_FILENAME)
        critical_files_ready.extend(await self._ensure_metadata_indexes())
        required_count = 1 + 2  # model_selection + two metadata index files
        status = self.PreflightStatus(
            assets_root=str(assets_root),
            critical_sync_ok=len(critical_files_ready) >= required_count,
            critical_files_ready=critical_files_ready,
        )
        logger.info(
            f"Startup preflight status: assets_root={status.assets_root} critical_files_ready={status.critical_files_ready}",
            status.assets_root,
            status.critical_files_ready,
        )
        return status

    async def run_background_warmup(self) -> None:
        """Run full warmup in background without blocking API readiness."""
        logger.info("Background asset warmup started")
        try:
            result = await WarmupService().warmup_all()
            logger.info(
                f"Background warmup finished (success={result.success} downloaded={result.downloaded_count} verified={result.verified_count} failed={len(result.failed_assets)})",
                result.success,
                result.downloaded_count,
                result.verified_count,
                len(result.failed_assets),
            )
        except Exception:
            logger.exception("Background warmup failed")

        for domain in settings.STARTUP_PREFETCH_GLOSSARIES:
            self._glossary_service.prefetch_domain_glossary(domain)


def create_background_task(coro):
    """Create a background task with a done callback for logging failures."""
    task = asyncio.create_task(coro)

    def _on_done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
        except asyncio.CancelledError:
            logger.info("Background startup task cancelled")
        except Exception:
            logger.exception("Background startup task failed")

    task.add_done_callback(_on_done)
    return task
