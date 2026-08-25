"""Vertex LLM cost service tests."""

import json
from dataclasses import dataclass

import pytest
from src.config.constants import settings
from src.config.llm_rate_catalog import ModelRateEntry
from src.config.llm_rate_catalog import RateTier
from src.worker.doctranslator.translator.usage import TokenUsage
from src.worker.services.llm_cost_service import VertexLLMCostService


def _regional_test_catalog() -> tuple[ModelRateEntry, ...]:
    return (
        ModelRateEntry(
            provider="gemini_vertexai",
            model_id="gemini-3.5-flash",
            region="europe-west3",
            tiers=(
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=0.00165,
                    output_cost_per_1k=0.0099,
                    cache_hit_cost_per_1k=0.000165,
                ),
            ),
        ),
    )


def _default_test_catalog() -> tuple[ModelRateEntry, ...]:
    return (
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
            provider="claude",
            model_id="claude-sonnet-4-6",
            region="europe-west1",
            tiers=(
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=0.0033,
                    output_cost_per_1k=0.0165,
                    cache_hit_cost_per_1k=0.00033,
                ),
            ),
        ),
    )


@dataclass
class _Chunk:
    chunk_index: int
    token_count: int


class TestVertexLLMCostService:
    @pytest.fixture
    def service(self):
        return VertexLLMCostService(catalog=_default_test_catalog())

    def test_gemini_flash_rates(self, service):
        breakdown = service.calculate_attempt_cost(
            model_id="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        assert breakdown.provider == "gemini_vertexai"
        assert breakdown.input_rate_per_1k == pytest.approx(0.0003)
        assert breakdown.output_rate_per_1k == pytest.approx(0.0025)
        assert breakdown.total_cost_usd == pytest.approx(0.0003 + 0.00125)

    def test_gemini_pro_long_context_tier(self, service):
        short = service.calculate_attempt_cost(
            model_id="gemini-2.5-pro",
            prompt_tokens=100_000,
            completion_tokens=1000,
        )
        long_ctx = service.calculate_attempt_cost(
            model_id="gemini-2.5-pro",
            prompt_tokens=250_000,
            completion_tokens=1000,
        )
        assert short.input_rate_per_1k == pytest.approx(0.00125)
        assert long_ctx.input_rate_per_1k == pytest.approx(0.0025)
        assert long_ctx.output_rate_per_1k == pytest.approx(0.015)

    def test_claude_sonnet_rates(self, service):
        breakdown = service.calculate_attempt_cost(
            model_id="claude-sonnet-4-6",
            prompt_tokens=2000,
            completion_tokens=1000,
        )
        assert breakdown.provider == "claude"
        assert breakdown.input_rate_per_1k == pytest.approx(0.0033)
        assert breakdown.output_rate_per_1k == pytest.approx(0.0165)
        expected = (2000 / 1000) * 0.0033 + (1000 / 1000) * 0.0165
        assert breakdown.total_cost_usd == pytest.approx(expected)

    def test_cache_hit_tokens_billed_at_cache_rate(self, service):
        breakdown = service.calculate_cost(
            model_id="claude-sonnet-4-6",
            usage=TokenUsage(
                input_tokens=1000,
                output_tokens=500,
                cache_hit_input_tokens=400,
            ),
        )
        billable_input = 600
        expected = (
            (billable_input / 1000) * 0.0033
            + (500 / 1000) * 0.0165
            + (400 / 1000) * 0.00033
        )
        assert breakdown.total_cost_usd == pytest.approx(expected)

    def test_per_chunk_allocation_sums_to_total(self, service):
        chunks = [_Chunk(0, 300), _Chunk(1, 700)]
        records = service.compute_per_chunk_costs(
            chunks=chunks,
            model_id="gemini-2.5-flash",
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert len(records) == 2
        total = service.calculate_attempt_cost(
            model_id="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
        ).total_cost_usd
        assert sum(r["cost_usd"] for r in records) == pytest.approx(total)

    def test_per_chunk_allocation_distributes_tokens_proportionally(self, service):
        chunks = [_Chunk(0, 1500), _Chunk(1, 2000), _Chunk(2, 2500)]
        records = service.compute_per_chunk_costs(
            chunks=chunks,
            model_id="gemini-2.5-flash",
            total_input_tokens=6000,
            total_output_tokens=2400,
        )
        assert len(records) == 3
        assert sum(r["tokens_input"] for r in records) == pytest.approx(6000, abs=3)
        assert sum(r["tokens_output"] for r in records) == pytest.approx(2400, abs=3)
        assert records[2]["tokens_input"] > records[0]["tokens_input"]

    def test_per_chunk_records_have_required_fields(self, service):
        records = service.compute_per_chunk_costs(
            chunks=[_Chunk(0, 1000)],
            model_id="gemini-2.5-flash",
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        rec = records[0]
        assert rec["chunk_index"] == 0
        assert rec["tokens_input"] == 1000
        assert rec["tokens_output"] == 500
        assert rec["cost_usd"] >= 0.0

    def test_zero_tokens_produces_zero_chunk_costs(self, service):
        chunks = [_Chunk(0, 1000), _Chunk(1, 2000)]
        records = service.compute_per_chunk_costs(
            chunks=chunks,
            model_id="gemini-2.5-flash",
            total_input_tokens=0,
            total_output_tokens=0,
        )
        for rec in records:
            assert rec["tokens_input"] == 0
            assert rec["tokens_output"] == 0
            assert rec["cost_usd"] == 0.0

    def test_unknown_model_raises_value_error(self, service):
        with pytest.raises(ValueError, match="No pricing catalog entry found"):
            service.calculate_attempt_cost(
                model_id="gemini-9.9-nonexistent",
                prompt_tokens=100,
                completion_tokens=50,
            )

    def test_explicit_region_override_resolves_regional_entry(self):
        service = VertexLLMCostService(catalog=_regional_test_catalog())
        breakdown = service.calculate_attempt_cost(
            model_id="gemini-3.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
            region="europe-west3",
        )
        assert breakdown.region == "europe-west3"
        assert breakdown.input_rate_per_1k == pytest.approx(0.00165)
        assert breakdown.output_rate_per_1k == pytest.approx(0.0099)
        assert breakdown.total_cost_usd == pytest.approx(0.00165 + 0.00495)

    def test_region_none_falls_back_to_process_default(self, monkeypatch):
        service = VertexLLMCostService(catalog=_regional_test_catalog())
        monkeypatch.setattr(
            "src.worker.services.llm_cost_service.settings.GOOGLE_CLOUD_LOCATION",
            "europe-west3",
        )
        entry = service.resolve_rate_entry("gemini-3.5-flash")
        assert entry.region == "europe-west3"


class TestRegionalEntryOrdering:
    """Guards a silent-mispricing trap in resolve_rate_entry().

    The lookup returns `matches[0]` and its filter accepts
    `entry.region is None or entry.region == region`. If a global
    (`region: null`) entry is listed *before* the region-specific one in
    pricing_catalog.json, the cheaper global rate silently wins and every
    cost figure is quietly wrong -- with no error anywhere.
    """

    @staticmethod
    def _entry(region, input_rate, output_rate):
        return ModelRateEntry(
            provider="gemini_vertexai",
            model_id="gemini-3.5-flash",
            region=region,
            tiers=(
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=input_rate,
                    output_cost_per_1k=output_rate,
                ),
            ),
        )

    def test_regional_entry_first_wins(self):
        catalog = (
            self._entry("europe-west3", 0.00165, 0.0099),
            self._entry(None, 0.0015, 0.009),
        )
        entry = VertexLLMCostService(catalog=catalog).resolve_rate_entry(
            "gemini-3.5-flash", "europe-west3"
        )
        assert entry.region == "europe-west3"
        assert entry.tiers[0].input_cost_per_1k == pytest.approx(0.00165)

    def test_global_entry_first_silently_shadows_the_regional_rate(self):
        """Documents the failure mode this ordering requirement prevents."""
        catalog = (
            self._entry(None, 0.0015, 0.009),
            self._entry("europe-west3", 0.00165, 0.0099),
        )
        entry = VertexLLMCostService(catalog=catalog).resolve_rate_entry(
            "gemini-3.5-flash", "europe-west3"
        )
        # Wrong rate, no exception -- hence the ordering rule and the
        # WARNING emitted by _log_resolved_entry().
        assert entry.region is None
        assert entry.tiers[0].input_cost_per_1k == pytest.approx(0.0015)

    def test_shipped_catalog_orders_regional_before_global(self):
        """The real mounted catalog must satisfy the ordering rule."""
        catalog_path = settings.assets_root_path / settings.PRICING_CATALOG_FILENAME
        if not catalog_path.is_file():
            pytest.skip("pricing_catalog.json not present in this environment")
        entries = json.loads(catalog_path.read_text(encoding="utf-8"))
        seen_global = {}
        for entry in entries:
            key = (entry.get("provider"), entry.get("model_id"))
            if entry.get("region") is None:
                seen_global[key] = True
            elif seen_global.get(key):
                pytest.fail(
                    f"pricing_catalog.json lists a global entry for {key} before "
                    f"its region={entry['region']} entry; the regional rate will "
                    "be silently ignored by resolve_rate_entry()."
                )
