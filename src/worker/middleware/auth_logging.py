"""Authentication and logging middleware for worker."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from config.logging import logger


class WorkerAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for authenticating Cloud Tasks requests."""

    async def dispatch(self, request: Request, call_next):
        """Verify request is from Cloud Tasks."""
        # Skip auth for health/ready endpoints
        if request.url.path in ["/health", "/ready"]:
            return await call_next(request)

        # For production, verify Cloud Tasks headers
        # X-CloudTasks-TaskName, X-CloudTasks-QueueName, etc.
        # For now, just log and proceed
        logger.debug(f"Processing request: {request.method} {request.url.path}")

        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    async def dispatch(self, request: Request, call_next):
        """Log request and response details."""
        # Log incoming request
        logger.info(f"→ {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            logger.info(
                f"← {request.method} {request.url.path} - {response.status_code}"
            )
            return response
        except Exception as e:
            logger.exception(f"✗ {request.method} {request.url.path} - Error: {e}")
            raise
