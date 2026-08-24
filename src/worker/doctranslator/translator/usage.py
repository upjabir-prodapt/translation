"""Normalized token usage for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-agnostic token accounting for one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    # Reasoning/"thinking" tokens. Vertex bills these as output tokens but
    # reports them in a separate `thoughts_token_count` field that is NOT
    # included in `candidates_token_count`. Tracking them separately keeps
    # cost reporting honest and makes thinking-budget tuning measurable.
    thinking_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def billable_output_tokens(self) -> int:
        """Output tokens as billed by Vertex (response + reasoning)."""
        return self.output_tokens + self.thinking_tokens

    @classmethod
    def from_gemini_usage(cls, usage) -> TokenUsage:
        if not usage:
            return cls()
        return cls(
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            cache_hit_input_tokens=int(
                getattr(usage, "cached_content_token_count", 0) or 0
            ),
            thinking_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
        )

    @classmethod
    def from_claude_usage(cls, usage) -> TokenUsage:
        if not usage:
            return cls()
        return cls(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_hit_input_tokens=int(
                getattr(usage, "cache_read_input_tokens", 0) or 0
            ),
            cache_write_input_tokens=int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
        )

    def merge(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_hit_input_tokens=self.cache_hit_input_tokens
            + other.cache_hit_input_tokens,
            cache_write_input_tokens=self.cache_write_input_tokens
            + other.cache_write_input_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
        )
