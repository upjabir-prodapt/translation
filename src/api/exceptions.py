"""Custom API exceptions."""

from typing import Any

from fastapi import HTTPException
from fastapi import status


class BabelDocError(Exception):
    """Base exception for BabelDOC API."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(BabelDocError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None):
        details = {"field": field} if field else {}
        super().__init__(message, details)


class FileProcessingError(BabelDocError):
    """Raised when file processing fails."""

    def __init__(self, message: str, file_id: str | None = None):
        details = {"file_id": file_id} if file_id else {}
        super().__init__(message, details)


class JobNotFoundError(BabelDocError):
    """Raised when a job is not found."""

    def __init__(self, job_id: str):
        message = f"Job {job_id} not found"
        super().__init__(message, {"job_id": job_id})


class JobAlreadyCompletedError(BabelDocError):
    """Raised when trying to modify a completed job."""

    def __init__(self, job_id: str):
        message = f"Job {job_id} is already completed"
        super().__init__(message, {"job_id": job_id})


class TranslationError(BabelDocError):
    """Raised when translation fails."""

    def __init__(
        self, message: str, job_id: str | None = None, stage: str | None = None
    ):
        details = {"job_id": job_id, "stage": stage}
        details = {k: v for k, v in details.items() if v is not None}
        super().__init__(message, details)


class StorageError(BabelDocError):
    """Raised when storage operations fail."""

    def __init__(
        self, message: str, operation: str | None = None, path: str | None = None
    ):
        details = {"operation": operation, "path": path}
        details = {k: v for k, v in details.items() if v is not None}
        super().__init__(message, details)


# HTTP Exception helpers
def create_http_exception(
    status_code: int,
    message: str,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> HTTPException:
    """Create a standardized HTTP exception."""
    content = {
        "error": {
            "message": message,
            "code": error_code or "UNKNOWN_ERROR",
            "details": details or {},
        }
    }
    return HTTPException(status_code=status_code, detail=content)


# Common HTTP exceptions
def not_found_error(resource: str, identifier: str | None = None) -> HTTPException:
    """Create a 404 Not Found error."""
    return create_http_exception(
        status_code=status.HTTP_404_NOT_FOUND,
        message=f"{resource} not found",
        error_code="NOT_FOUND",
        details={"identifier": identifier},
    )


def validation_error(message: str, field: str | None = None) -> HTTPException:
    """Create a 400 Validation Error."""
    return create_http_exception(
        status_code=status.HTTP_400_BAD_REQUEST,
        message=message,
        error_code="VALIDATION_ERROR",
        details={"field": field} if field else {},
    )


def internal_error(message: str) -> HTTPException:
    """Create a 500 Internal Server Error."""
    return create_http_exception(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message=message,
        error_code="INTERNAL_ERROR",
    )
