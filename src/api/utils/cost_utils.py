"""Token usage and cost utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.doctranslator.format.pdf.split_manager import SplitPoint

MAX_JOB_COST_USD = 10.0


def compute_chunk_cost(
    input_tokens: int,
    output_tokens: int,
    input_rate_per_1k: float,
    output_rate_per_1k: float,
) -> float:
    """Return the USD cost for a single chunk given token counts and rates per 1K tokens."""
    return (input_tokens / 1000) * input_rate_per_1k + (
        output_tokens / 1000
    ) * output_rate_per_1k


def compute_per_chunk_costs(
    chunks: list[SplitPoint],
    total_input_tokens: int,
    total_output_tokens: int,
    input_rate_per_1k: float,
    output_rate_per_1k: float,
) -> list[dict]:
    """Distribute total token usage proportionally across chunks by token_count.

    Returns one record per chunk with chunk_index, tokens_input, tokens_output, cost_usd.
    """
    total_chunk_tokens = sum(c.token_count for c in chunks)
    records: list[dict] = []
    allocated_input = 0
    allocated_output = 0

    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        weight = chunk.token_count / total_chunk_tokens if total_chunk_tokens > 0 else 0

        if is_last:
            chunk_input = total_input_tokens - allocated_input
            chunk_output = total_output_tokens - allocated_output
        else:
            chunk_input = round(total_input_tokens * weight)
            chunk_output = round(total_output_tokens * weight)
            allocated_input += chunk_input
            allocated_output += chunk_output

        records.append(
            {
                "chunk_index": chunk.chunk_index,
                "tokens_input": chunk_input,
                "tokens_output": chunk_output,
                "cost_usd": compute_chunk_cost(
                    chunk_input, chunk_output, input_rate_per_1k, output_rate_per_1k
                ),
            }
        )

    return records


def validate_job_cost(cost_usd: float) -> None:
    """Raise ValueError when total job cost exceeds the $10.00 guardrail."""
    if cost_usd > MAX_JOB_COST_USD:
        raise ValueError(
            f"Job cost ${cost_usd:.4f} exceeds the ${MAX_JOB_COST_USD:.2f} guardrail"
        )
