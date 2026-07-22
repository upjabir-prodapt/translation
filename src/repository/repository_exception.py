"""Repository exception classes."""

from typing import Any


class StorageError(Exception):
    """Exception raised for GCS storage operations."""

    def __init__(
        self, message: str, operation: str | None = None, path: str | None = None
    ):
        self.message = message
        self.operation = operation
        self.path = path
        details = {"operation": operation, "path": path}
        self.details = {k: v for k, v in details.items() if v is not None}
        super().__init__(self.message)


class RepositoryError(Exception):
    """Base repository exception."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class BigQueryError(RepositoryError):
    """Exception raised for BigQuery operations."""

    def __init__(
        self,
        message: str,
        dataset: str | None = None,
        table: str | None = None,
        query: str | None = None,
    ):
        details = {"dataset": dataset, "table": table, "query": query}
        super().__init__(message, {k: v for k, v in details.items() if v is not None})
