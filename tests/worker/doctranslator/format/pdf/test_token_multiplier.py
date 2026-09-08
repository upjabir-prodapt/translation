"""`get_token_multiplier` must follow the document's real language mix.

tiktoken's gpt-4o encoding under-counts CJK relative to what Gemini/Claude
actually consume, so CJK batches have to hold fewer paragraphs or the
request overruns the model's context and the output comes back truncated.

Keying that decision off `lang_in`/`lang_out` alone was safe only while
mixed-language documents were rejected outright. Now that they are
translated, a document declared `en` targeting `de` can legitimately be a
third Japanese by character count, and the declared pair says nothing about
it.

tests/test.env fixes LLM_TOKEN_MULTIPLIER_DEFAULT=1.0 and
LLM_TOKEN_MULTIPLIER_CJK=0.5.
"""

from __future__ import annotations

import pytest
from src.worker.doctranslator.format.pdf.translation_config import get_token_multiplier

CJK = 0.5
DEFAULT = 1.0


class TestDeclaredLanguagePair:
    @pytest.mark.parametrize(
        ("lang_in", "lang_out", "expected"),
        [
            ("en", "de", DEFAULT),
            ("en", "ja", CJK),
            ("zh", "en", CJK),
            ("ko", "fr", CJK),
            ("en", "zh-cn", CJK),
        ],
    )
    def test_declared_pair_still_decides_on_its_own(self, lang_in, lang_out, expected):
        assert get_token_multiplier(lang_in, lang_out) == pytest.approx(expected)


class TestDetectedContent:
    def test_detected_cjk_widens_a_non_cjk_pair(self):
        assert get_token_multiplier("en", "de", ["en", "ja"]) == pytest.approx(CJK)

    def test_detected_non_cjk_leaves_the_default(self):
        assert get_token_multiplier("en", "de", ["en", "fr", "es"]) == pytest.approx(
            DEFAULT
        )

    def test_empty_detection_falls_back_to_the_declared_pair(self):
        assert get_token_multiplier("en", "de", []) == pytest.approx(DEFAULT)
        assert get_token_multiplier("en", "ja", []) == pytest.approx(CJK)

    def test_none_is_accepted(self):
        """Direct callers and tests pass nothing; behaviour must not change."""
        assert get_token_multiplier("en", "de", None) == pytest.approx(DEFAULT)

    def test_detected_cjk_cannot_narrow_an_already_cjk_pair(self):
        assert get_token_multiplier("ja", "en", ["en"]) == pytest.approx(CJK)

    @pytest.mark.parametrize("code", ["ja", "zh", "ko", "zh-cn", "zh-tw"])
    def test_every_cjk_code_is_recognised_in_the_distribution(self, code):
        assert get_token_multiplier("en", "de", ["en", code]) == pytest.approx(CJK)
