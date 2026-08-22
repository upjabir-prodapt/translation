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

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

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
        )
