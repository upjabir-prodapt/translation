"""Language normalization and model-chain routing utilities."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.constants import settings

SUPPORTED_DOMAINS = {"commercial", "legal", "finance", "hr", "operations"}


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


def normalize_domain(value: str) -> str:
    """Normalize domain name into lowercase domain key."""
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported domain '{value}'")
    return normalized


@lru_cache(maxsize=1)
def get_model_selection_entries() -> list[dict[str, Any]]:
    """Load model routing entries from assets/model_selection.json."""
    model_selection_path = Path(str(settings.CACHE_FOLDER)) / "model_selection.json"
    raw_data = _load_json(model_selection_path)
    if isinstance(raw_data, dict):
        return [raw_data]
    if isinstance(raw_data, list):
        return [item for item in raw_data if isinstance(item, dict)]
    raise ValueError("model_selection.json must contain an object or a list of objects")


def _extract_model_list(entry: dict[str, Any]) -> list[str]:
    """Extract ordered model IDs from route entry."""
    chain = entry.get("model_chain")
    if not isinstance(chain, list) or not chain:
        return []

    sorted_chain = sorted(chain, key=lambda item: item.get("priority", 999))
    model_list: list[str] = []
    for item in sorted_chain:
        model_id = item.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            model_list.append(model_id.strip())
    return model_list


def select_model_list(lang_in: str, lang_out: str, domain: str) -> list[str]:
    """Resolve a model list by source, target and domain."""
    normalized_in = normalize_language(lang_in)
    normalized_out = normalize_language(lang_out)
    normalized_domain = normalize_domain(domain)

    entries = get_model_selection_entries()
    for entry in entries:
        try:
            entry_in = normalize_language(str(entry.get("source_language", "")))
            entry_out = normalize_language(str(entry.get("target_language", "")))
            entry_domain = normalize_domain(str(entry.get("domain", "")))
        except ValueError:
            continue

        if (
            entry_in == normalized_in
            and entry_out == normalized_out
            and entry_domain == normalized_domain
        ):
            models = _extract_model_list(entry)
            if models:
                return models[:2]

    raise ValueError(
        "No model route found for "
        f"source='{normalized_in}', target='{normalized_out}', domain='{normalized_domain}'"
    )
