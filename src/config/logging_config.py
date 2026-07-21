"""Logging configuration — stdlib logging with GcpJsonFormatter writing to stdout."""

import json
import logging
import sys
from datetime import UTC
from datetime import datetime

from src.config.constants import settings


class GcpJsonFormatter(logging.Formatter):
    """Formats log records as structured JSON for Cloud Logging.

    Reads otelTraceID, otelSpanID, otelTraceSampled injected by
    LoggingInstrumentor and maps them to the Cloud Logging correlation fields:
      logging.googleapis.com/trace
      logging.googleapis.com/spanId
      logging.googleapis.com/trace_sampled
    """

    _SEVERITY = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def __init__(self, project_id: str) -> None:
        super().__init__()
        self._project_id = project_id

    def format(self, record: logging.LogRecord) -> str:
        # Sanitize message — prevent log injection (CWE-117)
        message = record.getMessage().replace("\n", " ").replace("\r", " ")

        payload: dict = {
            "severity": self._SEVERITY.get(record.levelname, record.levelname),
            "message": message,
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "logger": record.name,
            "logging.googleapis.com/sourceLocation": {
                "file": record.filename,
                "line": str(record.lineno),
                "function": record.funcName,
            },
        }

        trace_id: str = getattr(record, "otelTraceID", "")
        span_id: str = getattr(record, "otelSpanID", "")
        sampled: bool = getattr(record, "otelTraceSampled", False)

        if trace_id:
            payload["logging.googleapis.com/trace"] = (
                f"projects/{self._project_id}/traces/{trace_id}"
            )
        if span_id:
            payload["logging.googleapis.com/spanId"] = span_id
        if trace_id or span_id:
            payload["logging.googleapis.com/trace_sampled"] = sampled

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exception"] = record.exc_text

        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure root logger with GcpJsonFormatter writing to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(GcpJsonFormatter(project_id=settings.GOOGLE_CLOUD_PROJECT))

    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)
    root.handlers.clear()
    root.addHandler(handler)

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        log = logging.getLogger(name)
        log.handlers.clear()
        log.propagate = True

    logging.getLogger(__name__).info("Logging configured")


# Auto-setup on import — same behaviour as previous Loguru-based config
setup_logging()
