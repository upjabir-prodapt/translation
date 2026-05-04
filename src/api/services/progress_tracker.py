"""Progress tracking service for translation jobs."""

from datetime import UTC
from datetime import datetime
from typing import Protocol
from typing import Any

from src.config.logging import logger


class JobUpdater(Protocol):
    """Repository contract for progress updates."""

    async def update_job(self, job_id: str, updates: dict[str, Any]) -> None: ...


class ProgressTracker:
    """Tracks and reports job progress with rate limiting."""

    def __init__(
        self,
        updater: JobUpdater,
        job_id: str,
        min_update_interval: float = 2.0,
    ):
        self.updater = updater
        self.job_id = job_id
        self.last_update: datetime | None = None
        self.min_update_interval = min_update_interval

    async def update(
        self,
        progress: float | None,
        current_stage: str | None = None,
        force: bool = False,
    ) -> bool:
        now = datetime.now(UTC)
        if not force and self.last_update:
            time_diff = (now - self.last_update).total_seconds()
            if time_diff < self.min_update_interval:
                logger.debug(f"Skipping progress update for job {self.job_id}")
                return False

        update_data: dict[str, Any] = {"updated_at": now}
        if progress is not None:
            update_data["progress"] = max(0.0, min(1.0, progress))
        if current_stage:
            update_data["current_stage"] = current_stage

        try:
            await self.updater.update_job(self.job_id, update_data)
            self.last_update = now
            return True
        except Exception:
            logger.exception(f"Failed to update progress for job {self.job_id}")
            return False

    async def set_error(self, error_message: str) -> None:
        try:
            await self.updater.update_job(
                self.job_id,
                {
                    "status": "failed",
                    "error_message": error_message,
                    "updated_at": datetime.now(UTC),
                },
            )
        except Exception:
            logger.exception(f"Failed to mark job {self.job_id} as failed")

