import json

from worker.services.quality_judge import GoogleADKJudgeAgent
from worker.services.quality_judge import extract_attempt_text


def test_extract_attempt_text_reads_tracking_file(tmp_path):
    tracking = {
        "page": [
            {
                "paragraph": [
                    {"input": "Hello world.", "output": "Hola mundo."},
                    {"input": "How are you?", "output": "Como estas?"},
                ]
            }
        ]
    }
    path = tmp_path / "translate_tracking.json"
    path.write_text(json.dumps(tracking), encoding="utf-8")

    source, translated = extract_attempt_text(tmp_path)

    assert "Hello world." in source
    assert "Hola mundo." in translated


def test_google_adk_judge_fallback_without_client():
    judge = GoogleADKJudgeAgent(model="gemini-2.5-flash")
    judge._client = None

    result = judge.evaluate(
        source_text="This is one sentence. This is second sentence.",
        translated_text="Esta es una frase. Esta es segunda frase.",
    )

    assert 0.0 <= result.final_score <= 1.0
    assert result.model == "gemini-2.5-flash"
