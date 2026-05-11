"""Warmup service for orchestrating asset downloads and verification."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from src.config.constants import settings
from src.loaders.constants import DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256
from src.loaders.constants import TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256
from src.loaders.exceptions import WarmupError
from src.loaders.repositories.cache_repository import get_file_size
from src.loaders.repositories.cache_repository import verify_or_delete
from src.loaders.repositories.metadata_repository import clear_metadata_cache
from src.loaders.repositories.metadata_repository import get_cmap_metadata
from src.loaders.repositories.metadata_repository import get_font_metadata
from src.loaders.services.download_service import download_async
from src.loaders.services.download_service import get_or_download_model_async
from src.loaders.utils.path_helpers import get_cache_file_path
from src.loaders.utils.path_helpers import get_subdir_path
from src.repository.translation_storage_repository import TranslationStorageRepository

logger = logging.getLogger(__name__)


@dataclass
class WarmupResult:
    """Result of a warmup operation."""

    success: bool
    downloaded_count: int = 0
    verified_count: int = 0
    failed_assets: list[str] = None  # type: ignore[assignment]
    elapsed_seconds: float = 0.0
    message: str = ""

    def __post_init__(self):
        if self.failed_assets is None:
            self.failed_assets = []


class WarmupService:
    """Service for warming up (downloading and verifying) all assets."""

    def __init__(
        self,
        storage_repo: TranslationStorageRepository | None = None,
    ) -> None:
        self.storage_repo = storage_repo
        self._download_stats = {"downloaded": 0, "verified": 0, "failed": []}
        self._phase_stats: dict[str, int] = {}

    async def warmup_all(self) -> WarmupResult:
        """Run complete asset warmup.

        Returns:
            WarmupResult with statistics

        Raises:
            WarmupError: If critical assets fail to load
        """
        import time

        logger.info("=" * 60)
        logger.info("Starting asset warmup...")
        logger.info("=" * 60)

        start_time = time.time()
        self._download_stats = {"downloaded": 0, "verified": 0, "failed": []}

        try:
            # Step 1: Bulk sync all files in bounded phases
            await self._bulk_sync_phased()

            # Step 2: Clear metadata cache if new files were downloaded
            clear_metadata_cache()

            # Step 3: Download metadata files
            await self._download_metadata_files()

            # Step 4: Download and verify models in parallel
            await self._warmup_models()

            # Step 5: Verify fonts and CMaps in parallel
            await asyncio.gather(
                self._warmup_fonts(),
                self._warmup_cmaps(),
                self._warmup_tiktoken(),
                return_exceptions=True,
            )

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"Asset warmup complete in {elapsed:.2f}s")
            logger.info(f"  Downloaded: {self._download_stats['downloaded']}")
            logger.info(f"  Verified: {self._download_stats['verified']}")
            if self._phase_stats:
                logger.info(f"  Sync phases: {self._phase_stats}")
            logger.info("=" * 60)

            return WarmupResult(
                success=True,
                downloaded_count=self._download_stats["downloaded"],
                verified_count=self._download_stats["verified"],
                failed_assets=self._download_stats["failed"],
                elapsed_seconds=elapsed,
                message=f"Warmup complete: {self._download_stats['downloaded']} downloaded, "
                f"{self._download_stats['verified']} verified",
            )

        except Exception as e:
            logger.error(f"Asset warmup failed: {e}", exc_info=True)
            raise WarmupError(
                f"Asset warmup failed: {e}",
                failed_assets=self._download_stats["failed"],
            ) from e

    async def _download_blob_if_needed(
        self,
        *,
        rel_path: str,
        blob: Any,
        repo: TranslationStorageRepository,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        """Download one blob if local file is missing or size-mismatched."""
        local_path = get_cache_file_path(rel_path)
        if local_path.exists() and get_file_size(local_path) == blob.size:
            return False

        async with semaphore:
            try:
                await repo.download_asset(rel_path, local_path)
                return True
            except Exception as e:
                logger.warning(f"Failed to sync asset '{rel_path}': {e}")
                self._download_stats["failed"].append(rel_path)
                return False

    async def _bulk_sync_group(
        self,
        *,
        group_name: str,
        rel_paths: list[str],
        blob_map: dict[str, Any],
        repo: TranslationStorageRepository,
        semaphore: asyncio.Semaphore,
    ) -> int:
        """Sync a specific logical group of assets."""
        if not rel_paths:
            self._phase_stats[group_name] = 0
            return 0

        downloaded = 0
        batch_size = max(1, settings.WARMUP_SYNC_CONCURRENCY * 4)
        pending: list[asyncio.Task[bool]] = []

        async def flush_pending() -> None:
            nonlocal downloaded
            if not pending:
                return
            results = await asyncio.gather(*pending, return_exceptions=False)
            downloaded += sum(1 for item in results if item)
            pending.clear()

        for rel_path in rel_paths:
            blob = blob_map[rel_path]
            pending.append(
                asyncio.create_task(
                    self._download_blob_if_needed(
                        rel_path=rel_path,
                        blob=blob,
                        repo=repo,
                        semaphore=semaphore,
                    )
                )
            )
            if len(pending) >= batch_size:
                await flush_pending()

        await flush_pending()
        self._phase_stats[group_name] = downloaded
        logger.info(f"Bulk sync phase '{group_name}' downloaded {downloaded} file(s)")
        return downloaded

    async def _bulk_sync_phased(self) -> int:
        """Bulk sync all assets from GCS using bounded concurrency phases.

        Returns:
            Number of files downloaded
        """
        repo = self.storage_repo or TranslationStorageRepository()

        try:
            logger.info("Bulk sync: Listing GCS objects...")
            blobs = await repo.list_assets()
            self._phase_stats = {}
            prefix = settings.GCS_ASSETS_PREFIX.strip("/")

            blob_map: dict[str, Any] = {}
            for blob in blobs:
                if blob.name is None or blob.name.endswith("/"):
                    continue
                name = blob.name.lstrip("/")
                if not name.startswith(f"{prefix}/"):
                    continue
                rel_path = name[len(prefix) :].lstrip("/")
                if rel_path:
                    blob_map[rel_path] = blob

            if not blob_map:
                logger.info("All assets already cached locally")
                return 0

            semaphore = asyncio.Semaphore(max(1, settings.WARMUP_SYNC_CONCURRENCY))
            remaining = set(blob_map.keys())
            total_downloaded = 0

            for phase_prefix in settings.WARMUP_SYNC_PHASE_PREFIXES:
                phase_prefix = phase_prefix.strip().strip("/")
                if not phase_prefix:
                    continue
                phase_paths = sorted(
                    path for path in remaining if path.startswith(f"{phase_prefix}/")
                )
                total_downloaded += await self._bulk_sync_group(
                    group_name=phase_prefix,
                    rel_paths=phase_paths,
                    blob_map=blob_map,
                    repo=repo,
                    semaphore=semaphore,
                )
                remaining.difference_update(phase_paths)

            if remaining:
                self._phase_stats["unmatched_skipped"] = len(remaining)
                logger.info(
                    f"Bulk sync skipped {len(remaining)} unmatched file(s) outside phase prefixes"
                )

            self._download_stats["downloaded"] += total_downloaded
            logger.info(f"Bulk sync complete ({total_downloaded} files downloaded)")
            return total_downloaded

        except Exception as e:
            logger.error(f"Bulk sync failed: {e}")
            return 0

    async def _download_metadata_files(self) -> None:
        """Download font and cmap metadata JSON files."""
        metadata_files = [
            (settings.FONT_METADATA_FILENAME, settings.METADATA_DIR, "font"),
            (settings.CMAP_METADATA_FILENAME, settings.METADATA_DIR, "cmap"),
        ]

        for filename, subdir, file_type in metadata_files:
            try:
                local_path = get_cache_file_path(filename, subdir)
                blob_path = f"{subdir}/{filename}"
                await download_async(blob_path, local_path, self.storage_repo)
                logger.debug(f"{file_type.capitalize()} metadata downloaded")
            except Exception as e:
                logger.warning(
                    f"Failed to download {file_type} metadata, "
                    f"using local cache if available: {e}"
                )

    async def _warmup_models(self) -> None:
        """Download and verify ML models."""
        logger.info("Warming up models...")

        tasks = [
            get_or_download_model_async(
                settings.DOCLAYOUT_MODEL_FILENAME,
                DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256,
                "DocLayout",
                settings.MODELS_DIR,
            ),
            get_or_download_model_async(
                settings.TABLE_DETECTION_MODEL_FILENAME,
                TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256,
                "Table Detection",
                settings.MODELS_DIR,
            ),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Model warmup failed: {result}")
                self._download_stats["failed"].append(str(result))
            else:
                self._download_stats["verified"] += 1

    async def _warmup_fonts(self) -> None:
        """Verify and download all fonts."""
        logger.info("Warming up fonts...")

        font_metadata = get_font_metadata()

        async def verify_font(font_name: str, meta: Any) -> None:
            try:
                font_path = get_cache_file_path(font_name, settings.FONTS_DIR)
                if verify_or_delete(font_path, meta.sha3_256):
                    self._download_stats["verified"] += 1
                    return

                await download_async(
                    f"{settings.FONTS_DIR}/{font_name}",
                    font_path,
                    self.storage_repo,
                )
                self._download_stats["downloaded"] += 1
                if verify_or_delete(font_path, meta.sha3_256):
                    self._download_stats["verified"] += 1
            except Exception as e:
                logger.warning(f"Failed to verify/download font {font_name}: {e}")
                self._download_stats["failed"].append(font_name)

        tasks = [verify_font(name, meta) for name, meta in font_metadata.items()]

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Verified {len(tasks)} fonts")

    async def _warmup_cmaps(self) -> None:
        """Verify and download all CMap files."""
        logger.info("Warming up CMap files...")

        cmap_metadata = get_cmap_metadata()

        async def verify_cmap(cmap_name: str, meta: Any) -> None:
            try:
                cmap_path = get_cache_file_path(cmap_name, settings.CMAP_DIR)
                if verify_or_delete(cmap_path, meta.sha3_256):
                    self._download_stats["verified"] += 1
                    return

                await download_async(
                    f"{settings.CMAP_DIR}/{cmap_name}",
                    cmap_path,
                    self.storage_repo,
                )
                self._download_stats["downloaded"] += 1
                if verify_or_delete(cmap_path, meta.sha3_256):
                    self._download_stats["verified"] += 1
            except Exception as e:
                logger.warning(f"Failed to verify/download CMap {cmap_name}: {e}")
                self._download_stats["failed"].append(cmap_name)

        tasks = [verify_cmap(name, meta) for name, meta in cmap_metadata.items()]

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Verified {len(tasks)} CMap files")

    async def _warmup_tiktoken(self) -> None:
        """Initialize tiktoken cache."""
        logger.info("Warming up tiktoken cache...")

        try:
            # Ensure directory exists
            tiktoken_dir = get_subdir_path(settings.TIKTOKEN_DIR)

            # Pre-load common encodings
            await asyncio.to_thread(self._init_tiktoken)

            logger.info("Tiktoken cache initialized")
        except Exception as e:
            logger.warning(f"Tiktoken warmup failed (non-critical): {e}")

    def _init_tiktoken(self) -> None:
        """Synchronous tiktoken initialization."""
        from tiktoken import encoding_for_model

        encoding_for_model("gpt-4o")


# Convenience functions for backward compatibility


async def async_warmup(
    storage_repo: TranslationStorageRepository | None = None,
) -> WarmupResult:
    """Run complete asset warmup asynchronously.

    Args:
        storage_repo: Optional storage repository instance

    Returns:
        WarmupResult with statistics
    """
    service = WarmupService(storage_repo)
    return await service.warmup_all()


def warmup(storage_repo: TranslationStorageRepository | None = None) -> WarmupResult:
    """Run complete asset warmup synchronously.

    Args:
        storage_repo: Optional storage repository instance

    Returns:
        WarmupResult with statistics
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # Already in async context - schedule as task
        # This returns a Task, not the result
        return loop.create_task(async_warmup(storage_repo))  # type: ignore[return-value]
    except RuntimeError:
        # No event loop - safe to create one
        return asyncio.run(async_warmup(storage_repo))
