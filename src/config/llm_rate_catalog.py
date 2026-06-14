"""Vertex LLM rate catalog with env-based overrides."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from src.config.constants import Settings

logger = logging.getLogger(__name__)

PRO_SHORT_CONTEXT_MAX_INPUT_TOKENS = 200_000


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


def _entry_key(
    provider: str, model_id: str, region: str | None
) -> tuple[str, str, str | None]:
    return provider, model_id.strip().lower(), region


def _parse_override_json(raw: str) -> tuple[ModelRateEntry, ...]:
    text = (raw or "").strip()
    if not text or text == "[]":
        return ()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM_RATE_CATALOG_OVERRIDE_JSON must be a JSON array of rate entries"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError("LLM_RATE_CATALOG_OVERRIDE_JSON must be a JSON array")

    return tuple(_model_rate_entry_from_dict(item) for item in payload)


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
    return ModelRateEntry(
        provider=str(item["provider"]),
        model_id=str(item["model_id"]).strip().lower(),
        region=str(region) if region not in (None, "") else None,
        tiers=tiers,
    )


def build_rate_catalog_from_settings(settings: Settings) -> tuple[ModelRateEntry, ...]:
    """Build the active catalog from per-model env vars plus optional JSON overrides."""
    entries: dict[tuple[str, str, str | None], ModelRateEntry] = {}

    def put(entry: ModelRateEntry) -> None:
        entries[_entry_key(entry.provider, entry.model_id, entry.region)] = entry

    put(
        ModelRateEntry(
            provider="gemini_vertexai",
            model_id="gemini-2.5-pro",
            region=None,
            tiers=(
                RateTier(
                    max_input_tokens=PRO_SHORT_CONTEXT_MAX_INPUT_TOKENS,
                    input_cost_per_1k=float(
                        settings.GEMINI_2_5_PRO_SHORT_INPUT_COST_PER_1K
                    ),
                    output_cost_per_1k=float(
                        settings.GEMINI_2_5_PRO_SHORT_OUTPUT_COST_PER_1K
                    ),
                    cache_hit_cost_per_1k=float(
                        settings.GEMINI_2_5_PRO_SHORT_CACHE_HIT_COST_PER_1K
                    ),
                ),
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=float(
                        settings.GEMINI_2_5_PRO_LONG_INPUT_COST_PER_1K
                    ),
                    output_cost_per_1k=float(
                        settings.GEMINI_2_5_PRO_LONG_OUTPUT_COST_PER_1K
                    ),
                    cache_hit_cost_per_1k=float(
                        settings.GEMINI_2_5_PRO_LONG_CACHE_HIT_COST_PER_1K
                    ),
                ),
            ),
        )
    )
    put(
        ModelRateEntry(
            provider="gemini_vertexai",
            model_id="gemini-2.5-flash",
            region=None,
            tiers=(
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=float(
                        settings.GEMINI_2_5_FLASH_INPUT_COST_PER_1K
                    ),
                    output_cost_per_1k=float(
                        settings.GEMINI_2_5_FLASH_OUTPUT_COST_PER_1K
                    ),
                    cache_hit_cost_per_1k=float(
                        settings.GEMINI_2_5_FLASH_CACHE_HIT_COST_PER_1K
                    ),
                ),
            ),
        )
    )
    put(
        ModelRateEntry(
            provider="gemini_vertexai",
            model_id="gemini-2.5-flash-lite",
            region=None,
            tiers=(
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=float(
                        settings.GEMINI_2_5_FLASH_LITE_INPUT_COST_PER_1K
                    ),
                    output_cost_per_1k=float(
                        settings.GEMINI_2_5_FLASH_LITE_OUTPUT_COST_PER_1K
                    ),
                    cache_hit_cost_per_1k=float(
                        settings.GEMINI_2_5_FLASH_LITE_CACHE_HIT_COST_PER_1K
                    ),
                ),
            ),
        )
    )
    claude_region = settings.CLAUDE_VERTEX_REGION or None
    put(
        ModelRateEntry(
            provider="claude",
            model_id=settings.CLAUDE_MODEL.strip().lower(),
            region=claude_region,
            tiers=(
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=float(settings.CLAUDE_INPUT_COST_PER_1K),
                    output_cost_per_1k=float(settings.CLAUDE_OUTPUT_COST_PER_1K),
                    cache_hit_cost_per_1k=float(settings.CLAUDE_CACHE_HIT_COST_PER_1K),
                    cache_write_5m_cost_per_1k=float(
                        settings.CLAUDE_CACHE_WRITE_5M_COST_PER_1K
                    ),
                    cache_write_1h_cost_per_1k=float(
                        settings.CLAUDE_CACHE_WRITE_1H_COST_PER_1K
                    ),
                ),
            ),
        )
    )

    for entry in _parse_override_json(settings.LLM_RATE_CATALOG_OVERRIDE_JSON):
        put(entry)

    return tuple(entries.values())
