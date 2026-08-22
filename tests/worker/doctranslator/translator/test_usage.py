"""Token usage parsing tests."""

from src.worker.doctranslator.translator.usage import TokenUsage


class _GeminiUsage:
    prompt_token_count = 100
    candidates_token_count = 50
    total_token_count = 150
    cached_content_token_count = 10


class _ClaudeUsage:
    input_tokens = 200
    output_tokens = 80
    cache_read_input_tokens = 15
    cache_creation_input_tokens = 5


class TestTokenUsage:
    def test_from_gemini_usage(self):
        usage = TokenUsage.from_gemini_usage(_GeminiUsage())
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_hit_input_tokens == 10
        assert usage.total_tokens == 150

    def test_from_claude_usage(self):
        usage = TokenUsage.from_claude_usage(_ClaudeUsage())
        assert usage.input_tokens == 200
        assert usage.output_tokens == 80
        assert usage.cache_hit_input_tokens == 15
        assert usage.cache_write_input_tokens == 5

    def test_merge(self):
        left = TokenUsage(input_tokens=10, output_tokens=5)
        right = TokenUsage(input_tokens=3, output_tokens=2, cache_hit_input_tokens=1)
        merged = left.merge(right)
        assert merged.input_tokens == 13
        assert merged.output_tokens == 7
        assert merged.cache_hit_input_tokens == 1
