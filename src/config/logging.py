"""Logging configuration for the API service."""

import logging
import sys

from loguru import logger

from src.config.constants import settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru."""

    def emit(self, record):
        """Emit log record through loguru."""
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        message = record.getMessage().replace("\n", " ").replace("\r", " ")
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, message
        )


def setup_logging():
    """Configure logging for the application."""
    # Remove default handlers
    logger.remove()

    # Add console handler (terminal only)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level=settings.LOG_LEVEL,
    )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Set specific loggers
    logging.getLogger("uvicorn").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]

    logger.info("Logging configured")


# Auto-setup logging on import
setup_logging()
