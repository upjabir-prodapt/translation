from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from typing import Any

try:
    __version__ = version("pdfminer.six")
except PackageNotFoundError:
    # package is not installed, return default
    __version__ = "0.0"


def __getattr__(name: str) -> Any:
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(
        {*globals().keys(), "utils", "settings", "pdfexceptions", "psexceptions"}
    )


if __name__ == "__main__":
    print(__version__)
