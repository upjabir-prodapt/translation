"""Run the Translation API (uvicorn) from the repository root.

Usage:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import os

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
