"""Vertex LLM rate catalog, sourced exclusively from pricing_catalog.json.

pricing_catalog.json (mounted into every runtime environment via the GCS
asset cache) is the single source of truth for LLM pricing. There is no
env-var fallback and no env-var override mechanism -- the asset cache mount
is guaranteed present wherever this service runs, so a missing or malformed
catalog file is a hard configuration error, not a degrade-silently case.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from src.config.constants import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RateTier:
    """Pricing tier keyed by maximum input tokens (None = long/default tier)."""

    max_input_tokens: int | None
    input_cost_per_1k: float
    output_cost_per_1k: float
    cache_hit_cost_per_1k: float = 0.0
    cache_write_5m_cost_per_1k: float = 0.0
    cache_write_1h_cost_per_1k: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelRateEntry:
    provider: str
    model_id: str
    region: str | None
    tiers: tuple[RateTier, ...]
    context_window_tokens: int | None = None


def _entry_key(
    provider: str, model_id: str, region: str | None
) -> tuple[str, str, str | None]:
    return provider, model_id.strip().lower(), region


def _model_rate_entry_from_dict(item: dict[str, Any]) -> ModelRateEntry:
    tiers_raw = item.get("tiers") or []
    tiers = tuple(
        RateTier(
            max_input_tokens=tier.get("max_input_tokens"),
            input_cost_per_1k=float(tier["input_cost_per_1k"]),
            output_cost_per_1k=float(tier["output_cost_per_1k"]),
            cache_hit_cost_per_1k=float(tier.get("cache_hit_cost_per_1k", 0.0)),
            cache_write_5m_cost_per_1k=float(
                tier.get("cache_write_5m_cost_per_1k", 0.0)
            ),
            cache_write_1h_cost_per_1k=float(
                tier.get("cache_write_1h_cost_per_1k", 0.0)
            ),
        )
        for tier in tiers_raw
    )
    region = item.get("region")
    context_window_tokens = item.get("context_window_tokens")
    return ModelRateEntry(
        provider=str(item["provider"]),
        model_id=str(item["model_id"]).strip().lower(),
        region=str(region) if region not in (None, "") else None,
        tiers=tiers,
        context_window_tokens=(
            int(context_window_tokens) if context_window_tokens is not None else None
        ),
    )


@lru_cache(maxsize=1)
def load_pricing_catalog_entries(catalog_path: Path) -> tuple[ModelRateEntry, ...]:
    """Load rate entries from the mounted pricing_catalog.json asset file.

    pricing_catalog.json is the sole source of LLM pricing. It is mounted
    into every runtime environment (local, CI, and Cloud Run via the GCS
    FUSE asset cache), so a missing or malformed file is treated as a hard
    configuration error rather than silently falling back to an empty
    catalog.
    """
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"pricing_catalog.json not found at {catalog_path}. This file is "
            "the sole source of LLM pricing and must be present in the "
            "mounted asset cache."
        )
    try:
        with catalog_path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in pricing catalog {catalog_path}: {exc}"
        ) from exc

    if not isinstance(raw_data, list):
        raise ValueError(
            "pricing_catalog.json must contain a JSON array of rate entries"
        )

    return tuple(_model_rate_entry_from_dict(item) for item in raw_data)


def build_rate_catalog_from_settings(settings: Settings) -> tuple[ModelRateEntry, ...]:
    """Build the active rate catalog exclusively from pricing_catalog.json."""
    catalog_path = settings.assets_root_path / settings.PRICING_CATALOG_FILENAME
    return load_pricing_catalog_entries(catalog_path)
