from __future__ import annotations

import os
import sys
from pathlib import Path

import babeldoc.main as main_module


def _has_config_arg(argv: list[str]) -> bool:
    for arg in argv[1:]:
        if arg in {"-c", "--config"}:
            return True
        if arg.startswith("--config="):
            return True
    return False


def _candidate_config_paths() -> list[Path]:
    paths: list[Path] = []

    env_config = os.environ.get("BABELDOC_CONFIG")
    if env_config:
        paths.append(Path(env_config).expanduser())

    paths.append(Path.cwd() / "config" / "config.toml")
    paths.append(Path(__file__).resolve().parent.parent / "config" / "config.toml")
    return paths


def _inject_default_config(argv: list[str]) -> list[str]:
    if _has_config_arg(argv):
        return argv

    for config_path in _candidate_config_paths():
        if config_path.is_file():
            return [argv[0], "-c", str(config_path), *argv[1:]]

    return argv


def cli() -> None:
    sys.argv = _inject_default_config(sys.argv)
    main_module.cli()


if __name__ == "__main__":
    cli()
