from unittest.mock import patch

import pytest
from src.config.translation_routing import ModelRoute
from src.config.translation_routing import get_language_display_name
from src.config.translation_routing import get_language_mapper
from src.config.translation_routing import get_model_selection_entries
from src.config.translation_routing import normalize_domain
from src.config.translation_routing import normalize_language
from src.config.translation_routing import select_model_list


class TestTranslationRouting:
    def test_normalize_domain_success(self):
        assert normalize_domain(" LEGAL ") == "legal"
        assert normalize_domain("finance") == "finance"

    def test_normalize_domain_failure(self):
        with pytest.raises(ValueError, match="Unsupported domain"):
            normalize_domain("unknown")

    @patch("src.config.translation_routing._load_json")
    def test_get_language_mapper_success(self, mock_load):
        get_language_mapper.cache_clear()
        try:
            mock_load.return_value = {"English": "en", "French": "fr"}
            mapper = get_language_mapper()
            assert mapper["english"] == "en"
        finally:
            get_language_mapper.cache_clear()

    def test_normalize_language_success(self):
        with patch(
            "src.config.translation_routing.get_language_mapper",
            return_value={"en": "en", "english": "en"},
        ):
            assert normalize_language("English") == "en"

    @patch("src.config.translation_routing.get_model_selection_entries")
    def test_select_model_list_success(self, mock_get_entries):
        mock_get_entries.return_value = [
            {
                "source_language": "en",
                "target_language": "fr",
                "domain": "legal",
                "model_chain": [{"model_id": "m1", "priority": 1}],
            }
        ]
        models = select_model_list("en", "fr", "legal")
        assert models == [ModelRoute(model_id="m1", region=None)]

    @patch("src.config.translation_routing.get_model_selection_entries")
    def test_select_model_list_with_region(self, mock_get_entries):
        mock_get_entries.return_value = [
            {
                "source_language": "en",
                "target_language": "de",
                "domain": "commercial",
                "model_chain": [
                    {
                        "model_id": "gemini-3.5-flash",
                        "priority": 1,
                        "region": "europe-west3",
                    },
                    {"model_id": "gemini-2.5-pro", "priority": 2},
                ],
            }
        ]
        models = select_model_list("en", "de", "commercial")
        assert models == [
            ModelRoute(model_id="gemini-3.5-flash", region="europe-west3"),
            ModelRoute(model_id="gemini-2.5-pro", region=None),
        ]

    @patch("src.config.translation_routing.get_model_selection_entries")
    def test_select_model_list_failure(self, mock_get_entries):
        mock_get_entries.return_value = []
        result = select_model_list("en", "de", "legal")
        assert len(result) == 1
        assert isinstance(result[0], ModelRoute)
        assert result[0].model_id == "gemini-3.5-flash"
        assert result[0].region == "europe-west3"

    @patch("src.config.translation_routing._load_json")
    def test_get_model_selection_entries_list(self, mock_load):
        get_model_selection_entries.cache_clear()
        try:
            mock_load.return_value = [
                {
                    "source_language": "English",
                    "target_language": "French",
                    "domain": "legal",
                    "model_chain": [{"model_id": "m1"}],
                }
            ]
            with patch(
                "src.config.translation_routing.normalize_language",
                side_effect=["en", "fr"],
            ):
                res = get_model_selection_entries()
                assert len(res) == 1
                assert res[0]["source_language"] == "en"
        finally:
            get_model_selection_entries.cache_clear()

    @patch("src.config.translation_routing._load_json")
    def test_get_model_selection_entries_invalid(self, mock_load):
        get_model_selection_entries.cache_clear()
        try:
            mock_load.return_value = "not a list or dict"
            with pytest.raises(ValueError, match="must contain an object"):
                get_model_selection_entries()
        finally:
            get_model_selection_entries.cache_clear()

    def test_get_language_display_name_from_code(self):
        assert get_language_display_name("en") == "English"
        assert get_language_display_name("fr") == "French"

    def test_get_language_display_name_from_alias(self):
        assert get_language_display_name("French") == "French"
        assert get_language_display_name("ENGLISH") == "English"

    def test_get_language_display_name_unknown_falls_back_to_title_case(self):
        assert get_language_display_name("auto") == "Auto"

    def test_get_language_display_name_empty(self):
        assert get_language_display_name(None) == "N/A"
        assert get_language_display_name("") == "N/A"
