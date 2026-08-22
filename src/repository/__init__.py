"""Repository module with client factories and dependency injection."""

from functools import lru_cache

from google.cloud import bigquery
from google.cloud import storage

from src.repository.bigquery_repository import BigQueryRepository


@lru_cache(maxsize=1)
def get_storage_client() -> storage.Client:
    """Get cached Storage client."""
    return storage.Client()


@lru_cache(maxsize=1)
def get_bigquery_client() -> bigquery.Client:
    """Get cached BigQuery client."""
    from src.config.constants import settings

    return bigquery.Client(project=settings.GOOGLE_CLOUD_PROJECT)


# Lazy singleton instances for repositories
_bigquery_repo: BigQueryRepository | None = None


# Repository dependencies with lazy initialization
def get_bigquery_repository() -> BigQueryRepository:
    """Get BigQuery repository instance (lazy initialization)."""
    global _bigquery_repo
    if _bigquery_repo is None:
        _bigquery_repo = BigQueryRepository(client=get_bigquery_client())
    return _bigquery_repo
