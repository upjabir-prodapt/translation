"""Rate catalog file-loading tests. pricing_catalog.json is the single
source of LLM pricing -- there is no env-var fallback or override.
"""

import json

import pytest
from src.config.constants import settings
from src.config.llm_rate_catalog import build_rate_catalog_from_settings
from src.config.llm_rate_catalog import load_pricing_catalog_entries


def _write_catalog(tmp_path, entries: list[dict]) -> None:
    catalog_path = tmp_path / "pricing_catalog.json"
    catalog_path.write_text(json.dumps(entries), encoding="utf-8")


class TestBuildRateCatalogFromSettings:
    def test_builds_catalog_from_pricing_catalog_json(self, monkeypatch, tmp_path):
        _write_catalog(
            tmp_path,
            [
                {
                    "provider": "gemini_vertexai",
                    "model_id": "gemini-2.5-flash",
                    "region": "europe-west1",
                    "context_window_tokens": 1048576,
                    "tiers": [
                        {
                            "max_input_tokens": None,
                            "input_cost_per_1k": 0.0003,
                            "output_cost_per_1k": 0.0025,
                        }
                    ],
                },
                {
                    "provider": "gemini_vertexai",
                    "model_id": "gemini-2.5-pro",
                    "region": "europe-west1",
                    "tiers": [
                        {
                            "max_input_tokens": 200_000,
                            "input_cost_per_1k": 0.00125,
                            "output_cost_per_1k": 0.01,
                        },
                        {
                            "max_input_tokens": None,
                            "input_cost_per_1k": 0.0025,
                            "output_cost_per_1k": 0.015,
                        },
                    ],
                },
            ],
        )
        load_pricing_catalog_entries.cache_clear()
        monkeypatch.setattr(settings, "ASSETS_ROOT", str(tmp_path))
        try:
            catalog = build_rate_catalog_from_settings(settings)
            by_model = {entry.model_id: entry for entry in catalog}
            assert "gemini-2.5-flash" in by_model
            assert "gemini-2.5-pro" in by_model
            assert by_model[
                "gemini-2.5-flash"
            ].tiers[0].input_cost_per_1k == pytest.approx(0.0003)
            assert by_model["gemini-2.5-flash"].context_window_tokens == 1048576
            assert len(by_model["gemini-2.5-pro"].tiers) == 2
        finally:
            load_pricing_catalog_entries.cache_clear()

    def test_missing_pricing_catalog_json_raises(self, monkeypatch, tmp_path):
        load_pricing_catalog_entries.cache_clear()
        monkeypatch.setattr(settings, "ASSETS_ROOT", str(tmp_path))
        try:
            with pytest.raises(FileNotFoundError, match="pricing_catalog.json"):
                build_rate_catalog_from_settings(settings)
        finally:
            load_pricing_catalog_entries.cache_clear()


class TestLoadPricingCatalogEntries:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        load_pricing_catalog_entries.cache_clear()
        try:
            with pytest.raises(FileNotFoundError, match="pricing_catalog.json"):
                load_pricing_catalog_entries(tmp_path / "does_not_exist.json")
        finally:
            load_pricing_catalog_entries.cache_clear()

    def test_invalid_json_raises(self, tmp_path):
        bad_path = tmp_path / "pricing_catalog.json"
        bad_path.write_text("not json", encoding="utf-8")
        load_pricing_catalog_entries.cache_clear()
        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_pricing_catalog_entries(bad_path)
        finally:
            load_pricing_catalog_entries.cache_clear()

    def test_non_list_json_raises(self, tmp_path):
        bad_path = tmp_path / "pricing_catalog.json"
        bad_path.write_text('{"not": "a list"}', encoding="utf-8")
        load_pricing_catalog_entries.cache_clear()
        try:
            with pytest.raises(ValueError, match="must contain a JSON array"):
                load_pricing_catalog_entries(bad_path)
        finally:
            load_pricing_catalog_entries.cache_clear()

    def test_context_window_tokens_round_trips(self, tmp_path):
        catalog_path = tmp_path / "pricing_catalog.json"
        catalog_path.write_text(
            json.dumps(
                [
                    {
                        "provider": "claude",
                        "model_id": "claude-sonnet-4-6",
                        "region": "europe-west1",
                        "context_window_tokens": 1000000,
                        "tiers": [
                            {
                                "max_input_tokens": None,
                                "input_cost_per_1k": 0.0033,
                                "output_cost_per_1k": 0.0165,
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        load_pricing_catalog_entries.cache_clear()
        try:
            entries = load_pricing_catalog_entries(catalog_path)
            assert entries[0].context_window_tokens == 1000000
        finally:
            load_pricing_catalog_entries.cache_clear()

    def test_context_window_tokens_defaults_to_none(self, tmp_path):
        catalog_path = tmp_path / "pricing_catalog.json"
        catalog_path.write_text(
            json.dumps(
                [
                    {
                        "provider": "gemini_vertexai",
                        "model_id": "gemini-2.5-flash",
                        "region": None,
                        "tiers": [
                            {
                                "max_input_tokens": None,
                                "input_cost_per_1k": 0.0003,
                                "output_cost_per_1k": 0.0025,
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        load_pricing_catalog_entries.cache_clear()
        try:
            entries = load_pricing_catalog_entries(catalog_path)
            assert entries[0].context_window_tokens is None
        finally:
            load_pricing_catalog_entries.cache_clear()

            load_pricing_catalog_entries.cache_clear()
