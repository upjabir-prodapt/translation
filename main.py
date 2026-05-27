"""Run the Translation API (uvicorn) from the repository root.

Usage:
    python main.py
"""

import os

import uvicorn
from src.api.main import app


def main() -> None:
    port = int(os.environ.get("PORT", 8000))
    # HOST defaults to 0.0.0.0 for containerised deployments; override via env var
    # to restrict binding in non-container environments (e.g. HOST=127.0.0.1 locally).
    host = os.environ.get("HOST", "0.0.0.0")  # nosec B104  # noqa: S104
    print(f"Starting server on {host}:{port}...", flush=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
