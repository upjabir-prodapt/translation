"""Token usage and cost utilities."""

from __future__ import annotations

MAX_JOB_COST_USD = 10.0


def aggregate_chunk_cost_records(records: list[dict]) -> dict[str, float | int]:
    """Sum per-chunk token and cost records into job-level totals."""
    return {
        "input_tokens": sum(int(r.get("tokens_input", 0)) for r in records),
        "output_tokens": sum(int(r.get("tokens_output", 0)) for r in records),
        "cost_usd": round(sum(float(r.get("cost_usd", 0.0)) for r in records), 8),
    }


def validate_job_cost(cost_usd: float) -> None:
    """Raise ValueError when total job cost exceeds the $10.00 guardrail."""
    if cost_usd > MAX_JOB_COST_USD:
        raise ValueError(
            f"Job cost ${cost_usd:.4f} exceeds the ${MAX_JOB_COST_USD:.2f} guardrail"
        )
