"""Document IL frontend; lazy submodule exports."""

from __future__ import annotations

from typing import Any

from ....._lazy_pkg_attrs import import_submodule_or_raise


def __getattr__(name: str) -> Any:
    return import_submodule_or_raise(__name__, name)


def __dir__() -> list[str]:
    return sorted(globals().keys())
