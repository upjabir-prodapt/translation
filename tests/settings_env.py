"""
Centralized test environment values.

Usage in a test:
    from tests.settings_env import apply_test_settings_env

    def test_something(monkeypatch):
        apply_test_settings_env(monkeypatch)
        # now import settings-dependent code

IMPORTANT: call apply_test_settings_env(monkeypatch) BEFORE importing any
module that triggers `from src.config.constants import settings`, because
_DOTENV_FILE = load_dotenv_file() runs at import time.

By default, tests/conftest.py sets DOTENV_PATH to tests/test.env so the full
Settings schema is satisfied without hardcoded defaults.
"""

from pathlib import Path

from dotenv import dotenv_values
from pytest import MonkeyPatch

_TEST_ENV_PATH = Path(__file__).resolve().parent / "test.env"


def apply_test_settings_env(monkeypatch: MonkeyPatch) -> None:
    """Set all required env vars from tests/test.env via monkeypatch."""
    monkeypatch.setenv("DOTENV_DISABLE", "true")
    for key, value in dotenv_values(_TEST_ENV_PATH).items():
        if value is not None:
            monkeypatch.setenv(key, value)
