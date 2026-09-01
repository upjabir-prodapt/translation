"""Run the Translation Worker (uvicorn).

Usage:
    python main_worker.py
"""

import os

import uvicorn

# Cloud Run sends SIGTERM and SIGKILLs the container ~10s later. Without an
# explicit graceful-shutdown timeout uvicorn waits indefinitely for the
# executor thread running the (uncancellable, synchronous) translation
# pipeline, so the process is always SIGKILLed on deploy/scale-down. Exiting
# just inside the grace window turns those into clean shutdowns.
DEFAULT_GRACEFUL_SHUTDOWN_SECONDS = 8


def main() -> None:
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")  # nosec B104  # noqa: S104
    graceful_shutdown = int(
        os.environ.get(
            "UVICORN_GRACEFUL_SHUTDOWN_SECONDS", DEFAULT_GRACEFUL_SHUTDOWN_SECONDS
        )
    )
    print(f"Starting worker server on {host}:{port}...", flush=True)
    uvicorn.run(
        "src.worker.main:app",
        host=host,
        port=port,
        timeout_graceful_shutdown=graceful_shutdown,
    )


if __name__ == "__main__":
    main()
