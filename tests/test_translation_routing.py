from config.translation_routing import normalize_domain
from config.translation_routing import normalize_language
from config.translation_routing import select_model_list


def test_normalize_language_aliases():
    assert normalize_language("english") == "en"
    assert normalize_language("JP") == "ja"
    assert normalize_language("spa") == "es"


def test_select_model_list_exact_match(monkeypatch):
    monkeypatch.setattr(
        "config.translation_routing.get_model_selection_entries",
        lambda: [
            {
                "source_language": "English",
                "target_language": "Spanish",
                "domain": "HR",
                "model_chain": [
                    {"priority": 2, "model_id": "gemini-2.5-flash"},
                    {"priority": 1, "model_id": "gemini-2.5-pro"},
                ],
            }
        ],
    )

    assert select_model_list("en", "es", "hr") == ["gemini-2.5-pro", "gemini-2.5-flash"]


def test_select_model_list_domain_specific(monkeypatch):
    monkeypatch.setattr(
        "config.translation_routing.get_model_selection_entries",
        lambda: [
            {
                "source_language": "en",
                "target_language": "es",
                "domain": "finance",
                "model_chain": [
                    {"priority": 1, "model_id": "gpt-4o-mini"},
                    {"priority": 2, "model_id": "gpt-4.1-mini"},
                ],
            }
        ],
    )

    assert select_model_list("english", "spanish", "finance") == [
        "gpt-4o-mini",
        "gpt-4.1-mini",
    ]


def test_normalize_domain():
    assert normalize_domain("HR") == "hr"
