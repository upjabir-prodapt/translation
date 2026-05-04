"""Run the Translation API (uvicorn) from the repository root.

Usage:
    python main.py
"""
import os
import uvicorn
from src.api.main import app

def main() -> None:
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
