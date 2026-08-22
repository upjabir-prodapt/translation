"""Run the Translation Worker (uvicorn).

Usage:
    python main_worker.py
"""

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")  # nosec B104  # noqa: S104
    print(f"Starting worker server on {host}:{port}...", flush=True)
    uvicorn.run("src.worker.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
