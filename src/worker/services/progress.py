"""Progress tracking service for worker jobs."""

from datetime import datetime
from typing import Any

from config.logging import logger
from repository.firestore_repository import FirestoreRepository


class ProgressTracker:
    """Tracks and reports job progress with rate limiting."""

    def __init__(
        self,
        firestore: FirestoreRepository,
        job_id: str,
        min_update_interval: float = 2.0,
    ):
        """
        Initialize progress tracker.

        Args:
            firestore: Firestore repository instance
            job_id: Job identifier
            min_update_interval: Minimum seconds between updates (default: 2.0)
        """
        self.firestore = firestore
        self.job_id = job_id
        self.last_update: datetime | None = None
        self.min_update_interval = min_update_interval

    async def update(
        self,
        progress: float | None,
        current_stage: str | None = None,
        force: bool = False,
    ) -> bool:
        """
        Update job progress with rate limiting.

        Args:
            progress: Progress value (0.0 to 1.0)
            current_stage: Current processing stage description
            force: Force update even if within rate limit interval

        Returns:
            True if update was performed, False if skipped due to rate limiting
        """
        now = datetime.utcnow()

        # Rate limiting check
        if not force and self.last_update:
            time_diff = (now - self.last_update).total_seconds()
            if time_diff < self.min_update_interval:
                logger.debug(
                    f"Skipping progress update for job {self.job_id} (rate limited)"
                )
                return False

        # Build update payload
        update_data: dict[str, Any] = {"updated_at": now}

        if progress is not None:
            update_data["progress"] = max(0.0, min(1.0, progress))  # Clamp to [0, 1]

        if current_stage:
            update_data["current_stage"] = current_stage

        # Persist to Firestore
        try:
            await self.firestore.update_job(self.job_id, update_data)
            self.last_update = now
            logger.debug(
                f"Job {self.job_id} progress: {progress:.2%} - {current_stage or 'N/A'}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update progress for job {self.job_id}: {e}")
            return False

    async def set_error(self, error_message: str) -> None:
        """Mark job as failed with error message."""
        try:
            await self.firestore.update_job(
                self.job_id,
                {
                    "status": "failed",
                    "error_message": error_message,
                    "updated_at": datetime.utcnow(),
                },
            )
            logger.error(f"Job {self.job_id} marked as failed: {error_message}")
        except Exception as e:
            logger.error(f"Failed to mark job {self.job_id} as failed: {e}")
