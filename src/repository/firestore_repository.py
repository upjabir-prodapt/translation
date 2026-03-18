"""Firestore repository for job state management."""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

from google.api_core.exceptions import NotFound
from google.cloud.firestore import AsyncClient

from config.constants import settings
from config.logging import logger
from repository.repository_exception import FirestoreError


class FirestoreRepository:
    """Repository for Firestore operations."""

    def __init__(
        self, client: AsyncClient | None = None, collection: str | None = None
    ):
        """Initialize Firestore repository with async client."""
        self.client = client or AsyncClient(database=settings.FIRESTORE_DATABASE)
        self.collection = collection or settings.FIRESTORE_COLLECTION
        self._collection_ref = self.client.collection(self.collection)

    async def create_job(self, job_id: str, data: dict[str, Any]) -> None:
        """Create a new job document."""
        try:
            # Add timestamps and TTL
            data.update(
                {
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                    "expire_at": datetime.now(UTC)
                    + timedelta(hours=settings.JOB_TTL_HOURS),
                }
            )

            doc_ref = self._collection_ref.document(job_id)
            await doc_ref.set(data)
            logger.info(f"Created job {job_id} in Firestore")

        except Exception as e:
            logger.error(f"Failed to create job {job_id}: {e}")
            raise FirestoreError(
                f"Failed to create job: {e}",
                document_id=job_id,
                collection=self.collection,
            ) from e

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID."""
        try:
            doc_ref = self._collection_ref.document(job_id)
            doc = await doc_ref.get()

            if not doc.exists:
                return None

            data = doc.to_dict()
            return data if data is not None else None

        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            raise FirestoreError(
                f"Failed to get job: {e}",
                document_id=job_id,
                collection=self.collection,
            ) from e

    async def update_job(self, job_id: str, updates: dict[str, Any]) -> bool:
        """Update a job document."""
        try:
            # Always update the timestamp
            updates["updated_at"] = datetime.utcnow()

            doc_ref = self._collection_ref.document(job_id)
            await doc_ref.update(updates)
            logger.debug(f"Updated job {job_id}")
            return True

        except NotFound:
            logger.warning(f"Job {job_id} not found for update")
            return False

        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            raise FirestoreError(
                f"Failed to update job: {e}",
                document_id=job_id,
                collection=self.collection,
            ) from e

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job document."""
        try:
            doc_ref = self._collection_ref.document(job_id)
            await doc_ref.delete()
            logger.info(f"Deleted job {job_id}")
            return True

        except NotFound:
            logger.warning(f"Job {job_id} not found for deletion")
            return False

        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            raise FirestoreError(
                f"Failed to delete job: {e}",
                document_id=job_id,
                collection=self.collection,
            ) from e

    async def list_jobs(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List jobs with optional filtering."""
        try:
            query = self._collection_ref.order_by("created_at", direction="DESCENDING")

            if status:
                query = query.where("status", "==", status)

            # Apply pagination
            query = query.limit(limit).offset(offset)

            docs = query.stream()
            return [data async for doc in docs if (data := doc.to_dict()) is not None]

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise FirestoreError(
                f"Failed to list jobs: {e}", collection=self.collection
            ) from e

    async def claim_job(self, job_id: str) -> bool:
        """Atomically claim a job for processing."""
        try:
            doc_ref = self._collection_ref.document(job_id)

            # Use transaction to atomically claim job
            transaction = self.client.transaction()

            async with transaction:
                snapshot = await doc_ref.get(transaction=transaction)
                if not snapshot.exists:
                    return False

                if snapshot.get("status") != "queued":
                    return False

                # Update status to processing
                transaction.update(
                    doc_ref, {"status": "processing", "updated_at": datetime.utcnow()}
                )
                result = True
            if result:
                logger.info(f"Claimed job {job_id}")
            else:
                logger.warning(f"Could not claim job {job_id}")

            return result

        except Exception as e:
            logger.error(f"Failed to claim job {job_id}: {e}")
            raise FirestoreError(
                f"Failed to claim job: {e}",
                document_id=job_id,
                collection=self.collection,
            ) from e

    async def get_glossary(self, glossary_id: str) -> dict[str, Any] | None:
        """Get a glossary document by ID from the glossary collection."""
        try:
            glossary_ref = self.client.collection(
                settings.FIRESTORE_GLOSSARY_COLLECTION
            ).document(glossary_id)
            doc = await glossary_ref.get()

            if not doc.exists:
                return None

            data = doc.to_dict()
            return data if data is not None else None

        except Exception as e:
            logger.error(f"Failed to get glossary {glossary_id}: {e}")
            raise FirestoreError(
                f"Failed to get glossary: {e}",
                document_id=glossary_id,
                collection=settings.FIRESTORE_GLOSSARY_COLLECTION,
            ) from e

    async def get_jobs_by_status(
        self, status: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get jobs by status (for worker polling)."""
        try:
            query = (
                self._collection_ref.where("status", "==", status)
                .order_by("created_at")
                .limit(limit)
            )

            docs = query.stream()
            return [data async for doc in docs if (data := doc.to_dict()) is not None]

        except Exception as e:
            logger.error(f"Failed to get jobs by status {status}: {e}")
            raise FirestoreError(
                f"Failed to get jobs by status: {e}", collection=self.collection
            ) from e
