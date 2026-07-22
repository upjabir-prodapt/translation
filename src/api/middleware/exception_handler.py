"""Global exception handling middleware."""

import asyncio
import logging

from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.exceptions import BabelDocError
from src.api.exceptions import ConfigurationError
from src.api.exceptions import FileProcessingError
from src.api.exceptions import JobAlreadyCompletedError
from src.api.exceptions import JobNotFoundError
from src.api.exceptions import ReviewNotFoundError
from src.api.exceptions import StorageError
from src.api.exceptions import TranslationError
from src.api.exceptions import ValidationError

logger = logging.getLogger(__name__)


async def exception_handler_middleware(request: Request, call_next):
    """Global exception handling middleware."""
    try:
        return await call_next(request)
    except Exception as exc:
        # Handle coroutines that were raised instead of exceptions
        if asyncio.iscoroutine(exc):
            try:
                await exc
            except Exception as nested_exc:
                return handle_exception(nested_exc)
            return handle_exception(RuntimeError("Coroutine was raised as exception"))
        return handle_exception(exc)


def handle_exception(exc: Exception) -> JSONResponse:
    """Handle exceptions and return appropriate JSON responses."""

    # Safely convert exception to string
    try:
        exc_str = str(exc)
    except Exception:
        exc_str = repr(exc)

    # Pass through FastAPI/Starlette HTTP exceptions unchanged
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
            if isinstance(exc.detail, dict)
            else {"error": {"message": str(exc.detail), "code": "HTTP_ERROR"}},
        )

    # Pydantic request validation errors
    if isinstance(exc, RequestValidationError):
        errors = [
            {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        logger.warning("Request validation error: %s", errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": {
                    "message": "Request validation failed",
                    "code": "VALIDATION_ERROR",
                    "details": errors,
                }
            },
        )

    # Configuration errors (missing files, bad settings) — must check before BabelDocError
    if isinstance(exc, ConfigurationError):
        logger.error(f"Configuration error: {exc.message}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "CONFIGURATION_ERROR"}},
        )

    # Input validation errors
    if isinstance(exc, ValidationError):
        logger.warning("Validation error: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": {"message": exc.message, "code": "VALIDATION_ERROR"}},
        )

    if isinstance(exc, FileProcessingError):
        logger.error(f"File processing error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "FILE_ERROR"}},
        )

    if isinstance(exc, JobNotFoundError):
        logger.warning(f"Job not found: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": {"message": exc.message, "code": "JOB_NOT_FOUND"}},
        )

    if isinstance(exc, ReviewNotFoundError):
        logger.warning(f"Review not found: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": {"message": exc.message, "code": "REVIEW_NOT_FOUND"}},
        )

    if isinstance(exc, JobAlreadyCompletedError):
        logger.warning(f"Job already completed: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": {"message": exc.message, "code": "JOB_ALREADY_COMPLETED"}
            },
        )

    if isinstance(exc, StorageError):
        logger.error(f"Storage error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "STORAGE_ERROR"}},
        )

    if isinstance(exc, TranslationError):
        logger.error(f"Translation error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "TRANSLATION_ERROR"}},
        )

    if isinstance(exc, BabelDocError):
        logger.error(f"BabelDoc error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "BABELDOC_ERROR"}},
        )

    # ValueError from language/domain normalization surfaced outside service layer
    if isinstance(exc, ValueError):
        logger.warning("Value error: %s", exc_str)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": {"message": exc_str, "code": "VALIDATION_ERROR"}},
        )

    # RuntimeError from missing config (e.g. language_mapper.json not found)
    if isinstance(exc, RuntimeError):
        logger.error(f"Runtime error: {exc_str}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc_str, "code": "CONFIGURATION_ERROR"}},
        )

    # Unknown exceptions
    logger.error("Unhandled exception: %s", exc_str, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {"message": "Internal server error", "code": "INTERNAL_ERROR"}
        },
    )
