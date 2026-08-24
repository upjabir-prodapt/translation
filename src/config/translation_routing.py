"""Language normalization and model-chain routing utilities."""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config.constants import settings

logger = logging.getLogger(__name__)

SUPPORTED_DOMAINS = {"commercial", "legal", "finance", "hr", "operations"}


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """One model-chain entry: which model to use and (optionally) which
    Vertex AI region to pin it to.

    `region=None` means "use the process-wide default region"
    (`settings.GOOGLE_CLOUD_LOCATION`), preserving existing behavior for
    every model that doesn't declare a `"region"` in model_selection.json.
    """

    model_id: str
    region: str | None = None


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {path}: {e}") from e


@lru_cache(maxsize=1)
def get_language_mapper() -> dict[str, str]:
    """Load language aliases to canonical language codes."""
    mapper_path = settings.PROJECT_ROOT / "src" / "config" / "language_mapper.json"
    try:
        raw_mapper = _load_json(mapper_path)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"language_mapper.json is missing at {mapper_path}. "
            "Ensure the configuration file exists before starting the service."
        ) from e
    return {
        str(k).strip().lower(): str(v).strip().lower() for k, v in raw_mapper.items()
    }


def normalize_language(value: str) -> str:
    """Normalize language names/codes into canonical code."""
    normalized = str(value).strip().lower()
    mapper = get_language_mapper()
    if normalized in mapper:
        return mapper[normalized]
    raise ValueError(f"Unsupported language '{value}'")


# Plain-language display names for the canonical language codes produced by
# normalize_language()/language_mapper.json. Used anywhere a human-readable
# language name is shown to end users (e.g. document cover pages) instead of
# the internal short code ("en", "fr", ...).
LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
}


def get_language_display_name(value: str | None) -> str:
    """Return a plain-language display name for a language code or name.

    Accepts either a canonical code ("fr"), an alias/full name known to
    language_mapper.json ("french"), or an already-plain value ("auto").
    Falls back to a title-cased version of the input rather than exposing a
    raw internal code to end users.
    """
    if not value or not str(value).strip():
        return "N/A"
    try:
        code = normalize_language(value)
    except ValueError:
        code = str(value).strip().lower()
    return LANGUAGE_DISPLAY_NAMES.get(code) or str(value).strip().title()


def normalize_domain(value: str) -> str:
    """Normalize domain name into lowercase domain key."""
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported domain '{value}'")
    return normalized


@lru_cache(maxsize=1)
def get_model_selection_entries() -> list[dict[str, Any]]:
    """Load model routing entries from assets/model_selection.json."""
    model_selection_path = settings.assets_root_path / settings.MODEL_SELECTION_FILENAME
    raw_data = _load_json(model_selection_path)
    raw_entries: list[dict[str, Any]]
    if isinstance(raw_data, dict):
        raw_entries = [raw_data]
    elif isinstance(raw_data, list):
        raw_entries = [item for item in raw_data if isinstance(item, dict)]
    else:
        raise ValueError(
            "model_selection.json must contain an object or a list of objects"
        )

    normalized_entries: list[dict[str, Any]] = []
    for entry in raw_entries:
        try:
            normalized_in = normalize_language(str(entry.get("source_language", "")))
            normalized_out = normalize_language(str(entry.get("target_language", "")))
            normalized_domain = normalize_domain(str(entry.get("domain", "")))
        except ValueError:
            continue

        normalized_entry = dict(entry)
        normalized_entry["source_language"] = normalized_in
        normalized_entry["target_language"] = normalized_out
        normalized_entry["domain"] = normalized_domain
        normalized_entries.append(normalized_entry)

    return normalized_entries


def _extract_model_list(entry: dict[str, Any]) -> list[ModelRoute]:
    """Extract ordered model routes (model_id + optional region) from route entry."""
    chain = entry.get("model_chain")
    if not isinstance(chain, list) or not chain:
        return []

    sorted_chain = sorted(chain, key=lambda item: item.get("priority", 999))
    model_list: list[ModelRoute] = []
    for item in sorted_chain:
        model_id = item.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            region = item.get("region")
            model_list.append(
                ModelRoute(
                    model_id=model_id.strip(),
                    region=str(region).strip() if region else None,
                )
            )
    return model_list


def select_model_list(lang_in: str, lang_out: str, domain: str) -> list[ModelRoute]:
    """Resolve model list by direct route match in model_selection.json.

    This lookup is intentionally lightweight: it compares source/target/domain
    case-insensitively against entries in model_selection.json and returns the
    chain sorted by priority.
    """
    normalized_in = str(lang_in).strip().lower()
    normalized_out = str(lang_out).strip().lower()
    normalized_domain = str(domain).strip().lower()

    entries = get_model_selection_entries()
    for entry in entries:
        entry_in = str(entry.get("source_language", "")).strip().lower()
        entry_out = str(entry.get("target_language", "")).strip().lower()
        entry_domain = str(entry.get("domain", "")).strip().lower()

        if (
            entry_in == normalized_in
            and entry_out == normalized_out
            and entry_domain == normalized_domain
        ):
            models = _extract_model_list(entry)
            if models:
                return models[: max(1, settings.MAX_MODEL_ATTEMPTS)]

    logger.warning(
        "No model route found for source='%s', target='%s', domain='%s'. "
        "Falling back to default Gemini model '%s' (region: %s).",
        normalized_in,
        normalized_out,
        normalized_domain,
        settings.GEMINI_MODEL,
        settings.GEMINI_MODEL_REGION or "default",
    )
    return [
        ModelRoute(
            model_id=settings.GEMINI_MODEL,
            region=settings.GEMINI_MODEL_REGION or None,
        )
    ]
