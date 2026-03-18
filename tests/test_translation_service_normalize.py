from api.schemas.requests import TranslateRequest
from api.services.translation_service import TranslationService


def test_normalize_config_uses_auto_source_language():
    service = object.__new__(TranslationService)
    request = TranslateRequest(
        domain="hr",
        lang_in="auto",
        lang_out="spa",
        user="user@example.com",
        department="PeopleOps",
    )

    config = service._normalize_config(request)

    assert config["lang_in"] == "auto"
    assert config["lang_out"] == "es"
    assert config["domain"] == "hr"
