"""Worker exception handling (no dependency on src.api)."""

from __future__ import annotations

import logging

from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def exception_handler_middleware(request: Request, call_next):
    """Map unexpected errors to JSON 500; pass through HTTPException."""
    try:
        return await call_next(request)
    except HTTPException as exc:
        if exc.status_code >= 500:
            logger.exception(
                "Worker HTTPException %s on %s %s: %s",
                exc.status_code,
                request.method,
                request.url.path,
                exc.detail,
            )
        raise
    except Exception as exc:
        logger.exception(
            "Unhandled worker error on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": str(exc) or "Internal server error",
            },
        )
