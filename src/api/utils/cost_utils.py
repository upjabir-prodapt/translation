"""Token usage and cost utilities for per-chunk cost attribution."""

from __future__ import annotations

from src.doctranslator.format.pdf.split_manager import SplitPoint

MAX_JOB_COST_USD = 10.0


def compute_chunk_cost(
    tokens_input: int,
    tokens_output: int,
    input_rate_per_1k: float,
    output_rate_per_1k: float,
) -> float:
    """Return cost in USD for a single chunk given its token counts and model rates."""
    return (tokens_input / 1000.0) * input_rate_per_1k + (
        tokens_output / 1000.0
    ) * output_rate_per_1k


def validate_job_cost(cost_usd: float) -> None:
    """Raise ValueError when total job cost exceeds the $10.00 guardrail."""
    if cost_usd > MAX_JOB_COST_USD:
        raise ValueError(
            f"Job cost ${cost_usd:.4f} exceeds the ${MAX_JOB_COST_USD:.2f} guardrail"
        )


def compute_per_chunk_costs(
    chunks: list[SplitPoint],
    total_input_tokens: int,
    total_output_tokens: int,
    input_rate_per_1k: float,
    output_rate_per_1k: float,
) -> list[dict]:
    """Distribute job-level token counts across chunks proportionally by estimated token_count.

    Returns one record per chunk with keys:
        chunk_index, tokens_input, tokens_output, cost_usd
    """
    total_estimated = sum(c.token_count for c in chunks) or 1
    records = []
    for chunk in chunks:
        weight = chunk.token_count / total_estimated
        chunk_input = round(total_input_tokens * weight)
        chunk_output = round(total_output_tokens * weight)
        chunk_cost = compute_chunk_cost(
            chunk_input, chunk_output, input_rate_per_1k, output_rate_per_1k
        )
        records.append(
            {
                "chunk_index": chunk.chunk_index,
                "tokens_input": chunk_input,
                "tokens_output": chunk_output,
                "cost_usd": round(chunk_cost, 6),
            }
        )
    return records
