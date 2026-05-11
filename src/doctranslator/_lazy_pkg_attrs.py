"""PEP 562 helpers for sparse ``__init__.py`` packages under ``src.doctranslator``.

Imports like ``from src.doctranslator.utils import memory`` expect the parent package to expose the
``memory`` submodule attribute. This module loads submodules on demand via
``importlib.import_module``.
"""

from __future__ import annotations

import importlib
from typing import Any


def import_submodule_or_raise(parent_name: str, name: str) -> Any:
    try:
        return importlib.import_module(f"{parent_name}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"module {parent_name!r} has no attribute {name!r}"
        ) from exc
