"""Repository module with client factories and dependency injection."""

from functools import lru_cache
from typing import Optional

from google.cloud import bigquery
from google.cloud import storage
from google.cloud import tasks_v2
from google.cloud.firestore import AsyncClient

from repository.bigquery_repository import BigQueryRepository
from repository.firestore_repository import FirestoreRepository
from repository.repository_exception import StorageError
from repository.storage_repository import StorageRepository


# LRU cached GCP client factory (expensive to create, reuse across app)
@lru_cache(maxsize=1)
def get_firestore_client() -> AsyncClient:
    """Get cached async Firestore client."""
    from config.constants import settings

    return AsyncClient(database=settings.FIRESTORE_DATABASE)


@lru_cache(maxsize=1)
def get_storage_client() -> storage.Client:
    """Get cached Storage client."""
    return storage.Client()


@lru_cache(maxsize=1)
def get_bigquery_client() -> bigquery.Client:
    """Get cached BigQuery client."""
    from config.constants import settings

    return bigquery.Client(project=settings.GOOGLE_CLOUD_PROJECT_ID)


@lru_cache(maxsize=1)
def get_tasks_client() -> tasks_v2.CloudTasksClient:
    """Get cached Cloud Tasks client."""
    import os

    emulator_host = os.getenv("CLOUD_TASKS_EMULATOR_HOST")

    if emulator_host:
        # When emulator host is set, create client with insecure credentials
        import grpc
        from google.cloud.tasks_v2.services.cloud_tasks import transports

        # Create insecure channel for emulator
        channel = grpc.insecure_channel(emulator_host)
        transport = transports.CloudTasksGrpcTransport(channel=channel)
        return tasks_v2.CloudTasksClient(transport=transport)

    return tasks_v2.CloudTasksClient()


# Lazy singleton instances for repositories
_firestore_repo: FirestoreRepository | None = None
_bigquery_repo: BigQueryRepository | None = None


# Repository dependencies with lazy initialization
def get_firestore_repository() -> FirestoreRepository:
    """Get Firestore repository instance (lazy initialization)."""
    global _firestore_repo
    if _firestore_repo is None:
        _firestore_repo = FirestoreRepository(client=get_firestore_client())
    return _firestore_repo


def get_bigquery_repository() -> BigQueryRepository:
    """Get BigQuery repository instance (lazy initialization)."""
    global _bigquery_repo
    if _bigquery_repo is None:
        _bigquery_repo = BigQueryRepository(client=get_bigquery_client())
    return _bigquery_repo
