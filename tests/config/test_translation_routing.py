"""
Unit tests for config/translation_routing.py.

Tests cover language normalization, domain validation, model list selection,
and path/cache behaviour. The lru_cache is cleared between relevant tests to
ensure isolation.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.translation_routing import (
    SUPPORTED_DOMAINS,
    _extract_model_list,
    get_language_mapper,
    normalize_domain,
    normalize_language,
    select_model_list,
)
from fixtures.sample_data import MODEL_SELECTION_LIST


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_caches():
    """Clear lru_cache state so tests are isolated."""
    get_language_mapper.cache_clear()
    # select_model_list uses get_model_selection_entries which also has lru_cache
    from src.config.translation_routing import get_model_selection_entries

    get_model_selection_entries.cache_clear()


# ---------------------------------------------------------------------------
# normalize_language
# ---------------------------------------------------------------------------


class TestNormalizeLanguage:
    """Tests that language names and codes map to canonical codes."""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("en", "en"),
            ("english", "en"),
            ("English", "en"),
            ("eng", "en"),
            ("en-us", "en"),
            ("es", "es"),
            ("Spanish", "es"),
            ("spa", "es"),
            ("fr", "fr"),
            ("French", "fr"),
            ("de", "de"),
            ("German", "de"),
            ("zh", "zh"),
            ("Chinese", "zh"),
            ("zh-cn", "zh"),
            ("Mandarin", "zh"),
            ("ja", "ja"),
            ("Japanese", "ja"),
            ("ar", "ar"),
            ("Arabic", "ar"),
            ("auto", "auto"),
        ],
    )
    def test_known_languages(self, input_val, expected):
        assert normalize_language(input_val) == expected

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("  english  ", "en"),  # whitespace stripped
            ("ENGLISH", "en"),      # case insensitive
            ("EN", "en"),
            ("EN-US", "en"),
        ],
    )
    def test_case_and_whitespace_handling(self, input_val, expected):
        assert normalize_language(input_val) == expected

    @pytest.mark.parametrize(
        "unsupported",
        ["klingon", "elvish", "xx", "latin", "esperanto"],
    )
    def test_unsupported_language_raises(self, unsupported):
        with pytest.raises(ValueError, match="Unsupported language"):
            normalize_language(unsupported)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            normalize_language("")


# ---------------------------------------------------------------------------
# normalize_domain
# ---------------------------------------------------------------------------


class TestNormalizeDomain:
    @pytest.mark.parametrize("domain", list(SUPPORTED_DOMAINS))
    def test_valid_domains(self, domain):
        assert normalize_domain(domain) == domain

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("Commercial", "commercial"),
            ("LEGAL", "legal"),
            ("  finance  ", "finance"),
            ("HR", "hr"),
        ],
    )
    def test_case_and_whitespace_handling(self, input_val, expected):
        assert normalize_domain(input_val) == expected

    @pytest.mark.parametrize(
        "bad",
        ["marketing", "healthcare", "science", "engineering", ""],
    )
    def test_invalid_domain_raises(self, bad):
        with pytest.raises(ValueError, match="Unsupported domain"):
            normalize_domain(bad)


# ---------------------------------------------------------------------------
# SUPPORTED_DOMAINS set
# ---------------------------------------------------------------------------


class TestSupportedDomains:
    def test_contains_all_expected(self):
        assert SUPPORTED_DOMAINS == {"commercial", "legal", "finance", "hr", "operations"}


# ---------------------------------------------------------------------------
# _extract_model_list (private helper)
# ---------------------------------------------------------------------------


class TestExtractModelList:
    def test_sorted_by_priority(self):
        entry = {
            "model_chain": [
                {"model_id": "gpt-4o", "priority": 2},
                {"model_id": "gpt-4o-mini", "priority": 1},
            ]
        }
        result = _extract_model_list(entry)
        assert result == ["gpt-4o-mini", "gpt-4o"]

    def test_empty_chain_returns_empty(self):
        assert _extract_model_list({"model_chain": []}) == []

    def test_no_chain_key_returns_empty(self):
        assert _extract_model_list({}) == []

    def test_invalid_chain_type_returns_empty(self):
        assert _extract_model_list({"model_chain": "invalid"}) == []

    def test_skips_blank_model_ids(self):
        entry = {
            "model_chain": [
                {"model_id": "gpt-4o-mini", "priority": 1},
                {"model_id": "  ", "priority": 2},
            ]
        }
        result = _extract_model_list(entry)
        assert result == ["gpt-4o-mini"]

    def test_strips_model_id_whitespace(self):
        entry = {
            "model_chain": [{"model_id": "  gpt-4o-mini  ", "priority": 1}]
        }
        result = _extract_model_list(entry)
        assert result == ["gpt-4o-mini"]


# ---------------------------------------------------------------------------
# select_model_list — mocked get_model_selection_entries
# ---------------------------------------------------------------------------


class TestSelectModelList:
    def setup_method(self):
        _clear_caches()

    def teardown_method(self):
        _clear_caches()

    def test_matching_route_returns_model_list(self):
        with patch(
            "config.translation_routing.get_model_selection_entries",
            return_value=MODEL_SELECTION_LIST,
        ):
            models = select_model_list("en", "es", "commercial")
        assert "gpt-4o-mini" in models

    def test_returns_at_most_two_models(self):
        """select_model_list returns at most 2 models (iterative fallback)."""
        with patch(
            "config.translation_routing.get_model_selection_entries",
            return_value=MODEL_SELECTION_LIST,
        ):
            models = select_model_list("english", "spanish", "commercial")
        assert len(models) <= 2

    def test_no_matching_route_raises(self):
        with patch(
            "config.translation_routing.get_model_selection_entries",
            return_value=MODEL_SELECTION_LIST,
        ):
            with pytest.raises(ValueError, match="No model route found"):
                select_model_list("en", "ja", "legal")

    def test_invalid_source_lang_raises(self):
        with patch(
            "config.translation_routing.get_model_selection_entries",
            return_value=MODEL_SELECTION_LIST,
        ):
            with pytest.raises(ValueError):
                select_model_list("klingon", "es", "commercial")

    def test_invalid_target_lang_raises(self):
        with patch(
            "config.translation_routing.get_model_selection_entries",
            return_value=MODEL_SELECTION_LIST,
        ):
            with pytest.raises(ValueError):
                select_model_list("en", "klingon", "commercial")

    def test_invalid_domain_raises(self):
        with patch(
            "config.translation_routing.get_model_selection_entries",
            return_value=MODEL_SELECTION_LIST,
        ):
            with pytest.raises(ValueError):
                select_model_list("en", "es", "science")

    def test_accepts_full_language_names(self):
        with patch(
            "config.translation_routing.get_model_selection_entries",
            return_value=MODEL_SELECTION_LIST,
        ):
            models = select_model_list("English", "Spanish", "commercial")
        assert len(models) >= 1

    def test_entries_with_invalid_langs_are_skipped(self):
        """Entries with unrecognised lang codes in model_selection.json are skipped."""
        bad_entry = {
            "source_language": "klingon",
            "target_language": "elvish",
            "domain": "commercial",
            "model_chain": [{"model_id": "model-x", "priority": 1}],
        }
        entries = [bad_entry, MODEL_SELECTION_LIST[0]]
        with patch(
            "config.translation_routing.get_model_selection_entries",
            return_value=entries,
        ):
            models = select_model_list("en", "es", "commercial")
        assert "gpt-4o-mini" in models


# ---------------------------------------------------------------------------
# get_language_mapper — cache behaviour
# ---------------------------------------------------------------------------


class TestGetLanguageMapper:
    def setup_method(self):
        get_language_mapper.cache_clear()

    def teardown_method(self):
        get_language_mapper.cache_clear()

    def test_returns_dict(self):
        mapper = get_language_mapper()
        assert isinstance(mapper, dict)

    def test_contains_english(self):
        mapper = get_language_mapper()
        assert "english" in mapper
        assert mapper["english"] == "en"

    def test_cache_returns_same_object(self):
        m1 = get_language_mapper()
        m2 = get_language_mapper()
        assert m1 is m2  # lru_cache returns same instance

    def test_missing_mapper_file_raises(self, tmp_path, monkeypatch):
        """If the language_mapper.json is missing, a RuntimeError is raised."""
        get_language_mapper.cache_clear()
        monkeypatch.setattr(
            "config.translation_routing.settings",
            MagicMock(PROJECT_ROOT=tmp_path),
        )
        with pytest.raises(RuntimeError, match="language_mapper.json is missing"):
            get_language_mapper()
