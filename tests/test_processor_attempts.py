import asyncio
from pathlib import Path

from worker.services.processor import JobProcessor
from worker.services.quality_judge import QualityJudgeResult


class DummyProgressTracker:
    async def update(self, *_args, **_kwargs):
        return None


class _Counter:
    def __init__(self, value: int):
        self.value = value


class DummyTranslator:
    def __init__(self):
        self.prompt_token_count = _Counter(100)
        self.completion_token_count = _Counter(50)
        self.token_count = _Counter(150)
        self.cache_hit_prompt_token_count = _Counter(10)


class DummyTranslationConfig:
    def __init__(self, working_dir: Path):
        self.translator = DummyTranslator()
        self.term_extraction_token_usage = {
            "total_tokens": 20,
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "cache_hit_prompt_tokens": 0,
        }
        self.working_dir = working_dir


class _FakeTextBox:
    def __init__(self, text: str):
        self._text = text

    def get_text(self) -> str:
        return self._text


def test_detect_source_language_fails_when_page_has_more_than_two_languages(
    monkeypatch,
):
    processor = JobProcessor(DummyProgressTracker())

    monkeypatch.setattr(
        "worker.services.processor.extract_pages",
        lambda _path: [
            [_FakeTextBox("English"), _FakeTextBox("French"), _FakeTextBox("German")]
        ],
    )
    detected_languages = iter(["en", "fr", "de"])
    monkeypatch.setattr(
        processor,
        "_detect_language_for_text",
        lambda _text: next(detected_languages),
    )
    monkeypatch.setattr(
        "worker.services.processor.LTTextContainer",
        _FakeTextBox,
    )

    try:
        processor.detect_source_language("input.pdf")
    except ValueError as exc:
        assert str(exc) == "Detected more than 2 languages on page 1: de, en, fr"
    else:
        raise AssertionError("Expected detect_source_language to fail")


def test_translate_retries_until_quality_pass(monkeypatch, tmp_path):
    processor = JobProcessor(DummyProgressTracker())

    build_calls = []
    cover_calls = []

    def fake_build(config, _output_dir):
        attempt_index = config["attempt_index"]
        working_dir = tmp_path / f"iter_{attempt_index}"
        working_dir.mkdir(parents=True, exist_ok=True)
        (working_dir / "translate_tracking.json").write_text(
            '{"page":[{"paragraph":[{"input":"a","output":"b"}]}]}',
            encoding="utf-8",
        )
        build_calls.append(attempt_index)
        return DummyTranslationConfig(working_dir)

    async def fake_run(_translation_config, config):
        out = Path(config["output_dir"]) / "mono.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"%PDF-1.4")
        return {"mono_pdf_path": out, "page_count": 1, "translations": []}

    quality_results = [
        QualityJudgeResult(0.4, 0.4, 0.4, 0.4, False, ["retry"], "judge"),
        QualityJudgeResult(0.9, 0.9, 0.9, 0.9, True, ["good"], "judge"),
    ]

    def fake_eval(**_kwargs):
        return quality_results.pop(0)

    def fake_build_cover_metadata(_translation_config, attempt_config, quality_result):
        return {
            "selected_model": attempt_config["selected_model"],
            "score": quality_result.final_score,
        }

    def fake_apply_cover_pages(_translation_config, attempt_result, metadata):
        cover_calls.append((attempt_result["model_id"], metadata))

    monkeypatch.setattr(processor, "_build_translation_config", fake_build)
    monkeypatch.setattr(processor, "_run_single_attempt", fake_run)
    monkeypatch.setattr(processor, "_evaluate_attempt_quality", fake_eval)
    monkeypatch.setattr(
        processor, "_build_cover_page_metadata", fake_build_cover_metadata
    )
    monkeypatch.setattr(processor, "_apply_cover_pages", fake_apply_cover_pages)

    result = asyncio.run(
        processor.translate(
            {
                "job_id": "job-1",
                "input_file": str(tmp_path / "input.pdf"),
                "output_dir": tmp_path / "output",
                "lang_in": "en",
                "lang_out": "es",
                "model_list": ["model-A", "model-B"],
            }
        )
    )

    assert build_calls == [1, 2]
    assert result["attempt_index"] == 2
    assert result["model_id"] == "model-B"
    assert len(result["attempts"]) == 2
    assert cover_calls == [("model-B", {"selected_model": "model-B", "score": 0.9})]
