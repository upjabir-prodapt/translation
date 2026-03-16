from pathlib import Path

from worker.services.processor import JobProcessor


class DummyProgressTracker:
    async def update(self, *_args, **_kwargs):
        return None


class DummyTranslationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_translation_config_filters_unrelated_keys(monkeypatch, tmp_path):
    processor = JobProcessor(DummyProgressTracker())

    monkeypatch.setattr(
        "worker.services.processor.create_translator_from_model_list",
        lambda model_list, **kwargs: "translator",
    )
    monkeypatch.setattr(
        "worker.services.processor.TranslationConfig",
        DummyTranslationConfig,
    )
    monkeypatch.setattr(processor, "_get_doc_layout_model", lambda: object())
    monkeypatch.setattr(processor, "_get_table_model", lambda: object())

    config = {
        "input_file": str(tmp_path / "input.pdf"),
        "output_dir": tmp_path / "output",
        "lang_in": "en",
        "lang_out": "es",
        "model_list": ["gemini-2.5-pro", "gemini-2.5-flash"],
        "custom_system_prompt": "Keep legal nuance",
        "domain": "hr",
        "user": "someone",
        "department": "peopleops",
    }

    (tmp_path / "input.pdf").write_bytes(b"%PDF-1.4")
    built = processor._build_translation_config(config, Path(config["output_dir"]))
    kwargs = built.kwargs

    assert kwargs["translator"] == "translator"
    assert kwargs["term_extraction_translator"] == "translator"
    assert kwargs["custom_system_prompt"] == "Keep legal nuance"
    assert "domain" not in kwargs
    assert "user" not in kwargs
    assert "department" not in kwargs
