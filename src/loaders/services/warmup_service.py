"""Warmup service for orchestrating asset downloads and verification."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from config.constants import settings
from loaders.constants import CMAP_DIR
from loaders.constants import CMAP_METADATA_FILENAME
from loaders.constants import DOCLAYOUT_MODEL_FILENAME
from loaders.constants import DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256
from loaders.constants import FONT_METADATA_FILENAME
from loaders.constants import FONTS_DIR
from loaders.constants import METADATA_DIR
from loaders.constants import MODELS_DIR
from loaders.constants import TABLE_DETECTION_MODEL_FILENAME
from loaders.constants import TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256
from loaders.constants import TIKTOKEN_DIR
from loaders.exceptions import WarmupError
from loaders.repositories.cache_repository import get_file_size
from loaders.repositories.cache_repository import verify_or_delete
from loaders.repositories.metadata_repository import clear_metadata_cache
from loaders.repositories.metadata_repository import get_cmap_metadata
from loaders.repositories.metadata_repository import get_font_metadata
from loaders.services.download_service import download_async
from loaders.services.download_service import get_or_download_model_async
from loaders.utils.path_helpers import get_cache_file_path
from loaders.utils.path_helpers import get_subdir_path
from worker.repository.worker_storage_repository import WorkerStorageRepository

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
        storage_repo: WorkerStorageRepository | None = None,
    ) -> None:
        self.storage_repo = storage_repo
        self._download_stats = {"downloaded": 0, "verified": 0, "failed": []}

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
            # Step 1: Bulk sync all files
            await self._bulk_sync()

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

    async def _bulk_sync(self) -> int:
        """Bulk sync all assets from GCS to local cache.

        Returns:
            Number of files downloaded
        """
        repo = self.storage_repo or WorkerStorageRepository()

        try:
            logger.info("Bulk sync: Listing GCS objects...")
            blobs = await repo.list_assets()

            download_tasks = []
            download_count = 0

            for blob in blobs:
                # Skip if blob name is missing
                if blob.name is None:
                    continue

                # Skip directories
                if blob.name.endswith("/"):
                    continue

                # Extract relative path
                prefix = settings.GCS_ASSETS_PREFIX
                rel_path = blob.name[len(prefix) :].lstrip("/")

                local_path = get_cache_file_path(rel_path)

                # Check if download needed (missing or size mismatch)
                if not local_path.exists() or get_file_size(local_path) != blob.size:
                    logger.debug(f"Scheduling download: {rel_path}")
                    download_tasks.append(repo.download_asset(rel_path, local_path))
                    download_count += 1

            if download_tasks:
                logger.info(f"Bulk sync: Downloading {len(download_tasks)} files...")
                await asyncio.gather(*download_tasks, return_exceptions=True)
                self._download_stats["downloaded"] += download_count
                logger.info(f"Bulk sync complete ({download_count} files)")
            else:
                logger.info("All assets already cached locally")

            return download_count

        except Exception as e:
            logger.error(f"Bulk sync failed: {e}")
            return 0

    async def _download_metadata_files(self) -> None:
        """Download font and cmap metadata JSON files."""
        metadata_files = [
            (FONT_METADATA_FILENAME, METADATA_DIR, "font"),
            (CMAP_METADATA_FILENAME, METADATA_DIR, "cmap"),
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
                DOCLAYOUT_MODEL_FILENAME,
                DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256,
                "DocLayout",
                MODELS_DIR,
            ),
            get_or_download_model_async(
                TABLE_DETECTION_MODEL_FILENAME,
                TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256,
                "Table Detection",
                MODELS_DIR,
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
                font_path = get_cache_file_path(font_name, FONTS_DIR)
                if verify_or_delete(font_path, meta.sha3_256):
                    self._download_stats["verified"] += 1
                    return

                await download_async(
                    f"{FONTS_DIR}/{font_name}",
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
                cmap_path = get_cache_file_path(cmap_name, CMAP_DIR)
                if verify_or_delete(cmap_path, meta.sha3_256):
                    self._download_stats["verified"] += 1
                    return

                await download_async(
                    f"{CMAP_DIR}/{cmap_name}",
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
            tiktoken_dir = get_subdir_path(TIKTOKEN_DIR)

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
    storage_repo: WorkerStorageRepository | None = None,
) -> WarmupResult:
    """Run complete asset warmup asynchronously.

    Args:
        storage_repo: Optional storage repository instance

    Returns:
        WarmupResult with statistics
    """
    service = WarmupService(storage_repo)
    return await service.warmup_all()


def warmup(storage_repo: WorkerStorageRepository | None = None) -> WarmupResult:
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
