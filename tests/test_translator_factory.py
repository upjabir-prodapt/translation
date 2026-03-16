from babeldoc.translator import factory


class DummyTranslator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_create_translator_dispatches_openai(monkeypatch):
    monkeypatch.setattr(factory, "OpenAITranslator", DummyTranslator)
    translator = factory.create_translator(
        "gpt-4o-mini",
        lang_in="en",
        lang_out="es",
        qps=4,
    )
    assert isinstance(translator, DummyTranslator)
    assert translator.kwargs["model"] == "gpt-4o-mini"


def test_create_translator_dispatches_gemini(monkeypatch):
    monkeypatch.setattr(factory, "GeminiVertexAITranslator", DummyTranslator)
    translator = factory.create_translator(
        "gemini-2.5-pro",
        lang_in="en",
        lang_out="es",
        qps=4,
    )
    assert isinstance(translator, DummyTranslator)
    assert translator.kwargs["model"] == "gemini-2.5-pro"


def test_create_translator_from_model_list_uses_first_model(monkeypatch):
    models = []

    def fake_create_translator(model_name, **kwargs):
        models.append(model_name)
        return DummyTranslator(model=model_name, **kwargs)

    monkeypatch.setattr(factory, "create_translator", fake_create_translator)
    monkeypatch.setattr(factory, "set_translate_rate_limiter", lambda max_qps: None)

    primary = factory.create_translator_from_model_list(
        ["gemini-2.5-pro", "gemini-2.5-flash", "gpt-4o-mini"],
        lang_in="en",
        lang_out="es",
        qps=8,
    )

    assert models == ["gemini-2.5-pro"]
    assert isinstance(primary, DummyTranslator)
