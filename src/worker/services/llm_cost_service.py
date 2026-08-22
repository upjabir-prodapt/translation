"""Centralized Vertex LLM cost calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.constants import settings
from src.config.llm_rate_catalog import ModelRateEntry
from src.config.llm_rate_catalog import RateTier
from src.config.llm_rate_catalog import build_rate_catalog_from_settings
from src.worker.doctranslator.translator.resolver import infer_provider
from src.worker.doctranslator.translator.usage import TokenUsage

if TYPE_CHECKING:
    from src.worker.doctranslator.format.pdf.split_manager import SplitPoint


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    provider: str
    model_id: str
    region: str | None
    input_tokens: int
    output_tokens: int
    cache_hit_input_tokens: int
    cache_write_input_tokens: int
    input_rate_per_1k: float
    output_rate_per_1k: float
    cache_hit_rate_per_1k: float
    input_cost_usd: float
    output_cost_usd: float
    cache_hit_cost_usd: float
    total_cost_usd: float

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "region": self.region,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_input_tokens": self.cache_hit_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "input_rate_per_1k": self.input_rate_per_1k,
            "output_rate_per_1k": self.output_rate_per_1k,
            "cache_hit_rate_per_1k": self.cache_hit_rate_per_1k,
            "input_cost_usd": self.input_cost_usd,
            "output_cost_usd": self.output_cost_usd,
            "cache_hit_cost_usd": self.cache_hit_cost_usd,
            "total_cost_usd": self.total_cost_usd,
        }


class VertexLLMCostService:
    """Resolve model rates and compute attempt/chunk costs."""

    def __init__(self, catalog: tuple[ModelRateEntry, ...] | None = None):
        self._catalog = catalog or build_rate_catalog_from_settings(settings)

    def resolve_rate_entry(self, model_id: str) -> ModelRateEntry:
        provider = infer_provider(model_id).value
        normalized = model_id.strip().lower()
        region = (
            settings.CLAUDE_VERTEX_REGION
            if provider == "claude"
            else settings.GOOGLE_CLOUD_LOCATION
        )

        matches = [
            entry
            for entry in self._catalog
            if entry.provider == provider
            and entry.model_id == normalized
            and (entry.region is None or entry.region == region)
        ]
        if matches:
            return matches[0]

        prefix_matches = [
            entry
            for entry in self._catalog
            if entry.provider == provider
            and normalized.startswith(entry.model_id)
            and (entry.region is None or entry.region == region)
        ]
        if prefix_matches:
            return max(prefix_matches, key=lambda e: len(e.model_id))

        return self._env_fallback_entry(provider, normalized, region)

    def _env_fallback_entry(
        self, provider: str, model_id: str, region: str | None
    ) -> ModelRateEntry:
        if provider == "gemini_vertexai":
            return ModelRateEntry(
                provider=provider,
                model_id=model_id,
                region=region,
                tiers=(
                    RateTier(
                        max_input_tokens=None,
                        input_cost_per_1k=float(settings.GEMINI_INPUT_COST_PER_1K),
                        output_cost_per_1k=float(settings.GEMINI_OUTPUT_COST_PER_1K),
                    ),
                ),
            )
        return ModelRateEntry(
            provider=provider,
            model_id=model_id,
            region=region,
            tiers=(
                RateTier(
                    max_input_tokens=None,
                    input_cost_per_1k=float(settings.CLAUDE_INPUT_COST_PER_1K),
                    output_cost_per_1k=float(settings.CLAUDE_OUTPUT_COST_PER_1K),
                    cache_hit_cost_per_1k=float(settings.CLAUDE_CACHE_HIT_COST_PER_1K),
                ),
            ),
        )

    def select_tier(self, entry: ModelRateEntry, input_tokens: int) -> RateTier:
        tiered = [t for t in entry.tiers if t.max_input_tokens is not None]
        if tiered:
            tiered.sort(key=lambda t: t.max_input_tokens or 0)
            for tier in tiered:
                if input_tokens <= tier.max_input_tokens:
                    return tier
        for tier in entry.tiers:
            if tier.max_input_tokens is None:
                return tier
        return entry.tiers[-1]

    def calculate_cost(
        self,
        *,
        model_id: str,
        usage: TokenUsage,
    ) -> CostBreakdown:
        entry = self.resolve_rate_entry(model_id)
        tier = self.select_tier(entry, usage.input_tokens)

        billable_input = max(usage.input_tokens - usage.cache_hit_input_tokens, 0)
        input_cost = (billable_input / 1000) * tier.input_cost_per_1k
        output_cost = (usage.output_tokens / 1000) * tier.output_cost_per_1k
        cache_hit_cost = (
            usage.cache_hit_input_tokens / 1000
        ) * tier.cache_hit_cost_per_1k

        total = input_cost + output_cost + cache_hit_cost
        return CostBreakdown(
            provider=entry.provider,
            model_id=model_id,
            region=entry.region,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_hit_input_tokens=usage.cache_hit_input_tokens,
            cache_write_input_tokens=usage.cache_write_input_tokens,
            input_rate_per_1k=tier.input_cost_per_1k,
            output_rate_per_1k=tier.output_cost_per_1k,
            cache_hit_rate_per_1k=tier.cache_hit_cost_per_1k,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            cache_hit_cost_usd=cache_hit_cost,
            total_cost_usd=total,
        )

    def calculate_attempt_cost(
        self,
        *,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_tokens: int = 0,
    ) -> CostBreakdown:
        usage = TokenUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cache_hit_input_tokens=cache_hit_tokens,
        )
        return self.calculate_cost(model_id=model_id, usage=usage)

    def compute_per_chunk_costs(
        self,
        *,
        chunks: list[SplitPoint],
        model_id: str,
        total_input_tokens: int,
        total_output_tokens: int,
        cache_hit_tokens: int = 0,
    ) -> list[dict]:
        """Distribute total token usage proportionally across chunks."""
        total_breakdown = self.calculate_attempt_cost(
            model_id=model_id,
            prompt_tokens=total_input_tokens,
            completion_tokens=total_output_tokens,
            cache_hit_tokens=cache_hit_tokens,
        )
        total_chunk_tokens = sum(c.token_count for c in chunks)
        records: list[dict] = []
        allocated_input = 0
        allocated_output = 0
        allocated_cache = 0

        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            weight = (
                chunk.token_count / total_chunk_tokens if total_chunk_tokens > 0 else 0
            )

            if is_last:
                chunk_input = total_input_tokens - allocated_input
                chunk_output = total_output_tokens - allocated_output
                chunk_cache = cache_hit_tokens - allocated_cache
            else:
                chunk_input = round(total_input_tokens * weight)
                chunk_output = round(total_output_tokens * weight)
                chunk_cache = round(cache_hit_tokens * weight)
                allocated_input += chunk_input
                allocated_output += chunk_output
                allocated_cache += chunk_cache

            chunk_breakdown = self.calculate_attempt_cost(
                model_id=model_id,
                prompt_tokens=chunk_input,
                completion_tokens=chunk_output,
                cache_hit_tokens=chunk_cache,
            )
            records.append(
                {
                    "chunk_index": chunk.chunk_index,
                    "tokens_input": chunk_input,
                    "tokens_output": chunk_output,
                    "cache_hit_tokens": chunk_cache,
                    "cost_usd": chunk_breakdown.total_cost_usd,
                    **chunk_breakdown.to_dict(),
                }
            )

        if records and total_breakdown.total_cost_usd:
            drift = total_breakdown.total_cost_usd - sum(r["cost_usd"] for r in records)
            records[-1]["cost_usd"] = round(records[-1]["cost_usd"] + drift, 8)

        return records


_default_cost_service: VertexLLMCostService | None = None


def get_vertex_llm_cost_service() -> VertexLLMCostService:
    global _default_cost_service
    if _default_cost_service is None:
        _default_cost_service = VertexLLMCostService()
    return _default_cost_service
