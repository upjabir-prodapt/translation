"""Worker lifecycle management for startup and shutdown hooks."""

import asyncio

from config.logging_config import logger
from loaders import async_warmup


class WorkerLifecycle:
    """Manages worker startup and shutdown lifecycle."""

    _initialized = False
    _shutdown_event: asyncio.Event | None = None

    @classmethod
    async def startup(cls) -> None:
        """Execute startup tasks."""
        if cls._initialized:
            logger.warning("Worker already initialized, skipping startup")
            return

        logger.info("🚀 Starting worker service...")

        try:
            # Warmup assets (downloads fonts, models, cmaps from GCS)
            await async_warmup()

            # Initialize shutdown event
            cls._shutdown_event = asyncio.Event()

            cls._initialized = True
            logger.info("✅ Worker service started successfully")

        except Exception as e:
            logger.exception("❌ Worker startup failed")
            raise RuntimeError(f"Worker startup failed: {e}") from e

    @classmethod
    async def shutdown(cls) -> None:
        """Execute shutdown tasks."""
        if not cls._initialized:
            logger.warning("Worker not initialized, skipping shutdown")
            return

        logger.info("🛑 Shutting down worker service...")

        try:
            # Signal shutdown
            if cls._shutdown_event:
                cls._shutdown_event.set()

            # Cleanup tasks here if needed
            # e.g., close connections, flush buffers, etc.

            cls._initialized = False
            logger.info("✅ Worker service stopped gracefully")

        except Exception as e:
            logger.exception("❌ Error during worker shutdown")
            raise

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if worker is initialized."""
        return cls._initialized

    @classmethod
    async def wait_for_shutdown(cls) -> None:
        """Wait for shutdown signal."""
        if cls._shutdown_event:
            await cls._shutdown_event.wait()
