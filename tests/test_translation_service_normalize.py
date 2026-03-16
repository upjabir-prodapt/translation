from api.schemas.requests import TranslateRequest
from api.services.translation_service import TranslationService


def test_normalize_config_includes_model_list(monkeypatch):
    monkeypatch.setattr(
        "api.services.translation_service.select_model_list",
        lambda lang_in, lang_out, domain: ["gemini-2.5-pro", "gemini-2.5-flash"],
    )

    service = object.__new__(TranslationService)
    request = TranslateRequest(
        domain="hr",
        lang_in="eng",
        lang_out="spa",
        user="user@example.com",
        department="PeopleOps",
    )

    config = service._normalize_config(request)

    assert config["lang_in"] == "en"
    assert config["lang_out"] == "es"
    assert config["domain"] == "hr"
    assert config["model_list"] == ["gemini-2.5-pro", "gemini-2.5-flash"]
