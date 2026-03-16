"""Worker-specific exceptions."""

from typing import Any


class WorkerError(Exception):
    """Base exception for BabelDOC Worker."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(WorkerError):
    """Raised when worker input validation fails."""

    def __init__(self, message: str, field: str | None = None):
        details = {"field": field} if field else {}
        super().__init__(message, details)


class FileProcessingError(WorkerError):
    """Raised when file processing fails during a worker task."""

    def __init__(self, message: str, file_id: str | None = None):
        details = {"file_id": file_id} if file_id else {}
        super().__init__(message, details)


class JobNotFoundError(WorkerError):
    """Raised when a task refers to a job that is not found."""

    def __init__(self, job_id: str):
        message = f"Job {job_id} not found"
        super().__init__(message, {"job_id": job_id})


class JobAlreadyCompletedError(WorkerError):
    """Raised when trying to process a completed job."""

    def __init__(self, job_id: str):
        message = f"Job {job_id} is already completed"
        super().__init__(message, {"job_id": job_id})


class TranslationError(WorkerError):
    """Raised when the translation service fails."""

    def __init__(
        self, message: str, job_id: str | None = None, stage: str | None = None
    ):
        details = {"job_id": job_id, "stage": stage}
        details = {k: v for k, v in details.items() if v is not None}
        super().__init__(message, details)


class StorageError(WorkerError):
    """Raised when storage operations fail in the worker."""

    def __init__(
        self, message: str, operation: str | None = None, path: str | None = None
    ):
        details = {"operation": operation, "path": path}
        details = {k: v for k, v in details.items() if v is not None}
        super().__init__(message, details)
