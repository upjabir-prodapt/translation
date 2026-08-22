"""Run the public Translation API (uvicorn).

Usage:
    python main_api.py
"""

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")  # nosec B104  # noqa: S104
    print(f"Starting API server on {host}:{port}...", flush=True)
    uvicorn.run("src.api.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
