"""Rate catalog env override tests."""

import pytest
from src.config.constants import settings
from src.config.llm_rate_catalog import ModelRateEntry
from src.config.llm_rate_catalog import RateTier
from src.config.llm_rate_catalog import build_rate_catalog_from_settings


def _sample_catalog() -> tuple[ModelRateEntry, ...]:
    return (
        ModelRateEntry(
            provider="gemini_vertexai",
            model_id="gemini-2.5-flash",
            region=None,
            tiers=(
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=0.0003,
                    output_cost_per_1k=0.0025,
                    cache_hit_cost_per_1k=0.00003,
                ),
            ),
        ),
        ModelRateEntry(
            provider="gemini_vertexai",
            model_id="gemini-2.5-pro",
            region=None,
            tiers=(
                RateTier(
                    max_input_tokens=200_000,
                    input_cost_per_1k=0.00125,
                    output_cost_per_1k=0.01,
                    cache_hit_cost_per_1k=0.00013,
                ),
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=0.0025,
                    output_cost_per_1k=0.015,
                    cache_hit_cost_per_1k=0.00025,
                ),
            ),
        ),
    )


class TestBuildRateCatalogFromSettings:
    def test_builds_gemini_flash_and_pro_from_env(self):
        catalog = build_rate_catalog_from_settings(settings)
        by_model = {entry.model_id: entry for entry in catalog}
        assert "gemini-2.5-flash" in by_model
        assert "gemini-2.5-pro" in by_model
        assert by_model["gemini-2.5-flash"].tiers[0].input_cost_per_1k == pytest.approx(
            float(settings.GEMINI_2_5_FLASH_INPUT_COST_PER_1K)
        )
        assert len(by_model["gemini-2.5-pro"].tiers) == 2

    def test_json_override_replaces_matching_entry(self, monkeypatch):
        override = (
            '[{"provider":"gemini_vertexai","model_id":"gemini-2.5-flash","region":null,'
            '"tiers":[{"max_input_tokens":null,"input_cost_per_1k":0.99,"output_cost_per_1k":0.88}]}]'
        )
        monkeypatch.setattr(settings, "LLM_RATE_CATALOG_OVERRIDE_JSON", override)
        catalog = build_rate_catalog_from_settings(settings)
        flash = next(e for e in catalog if e.model_id == "gemini-2.5-flash")
        assert flash.tiers[0].input_cost_per_1k == pytest.approx(0.99)
        assert flash.tiers[0].output_cost_per_1k == pytest.approx(0.88)
