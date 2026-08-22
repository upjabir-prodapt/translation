import json
from unittest.mock import patch

import pytest
from fixtures.docx_builder import build_docx
from fixtures.docx_builder import paragraph
from fixtures.docx_builder import run
from src.worker.docxtranslator.package import DocxPackage
from src.worker.docxtranslator.segment_translator import DocxSegmentTranslator
from src.worker.docxtranslator.segment_translator import build_payload
from src.worker.docxtranslator.segment_translator import parse_payload
from src.worker.docxtranslator.segments import collect_segments


def segments_for(body: str, tmp_path):
    path = tmp_path / "doc.docx"
    path.write_bytes(build_docx(body))
    return collect_segments(DocxPackage.open(path).text_parts())


class FakeEngine:
    """Stand-in translator that echoes a scripted response per call."""

    def __init__(self, handler=None, error: Exception | None = None):
        self.handler = handler
        self.error = error
        self.calls: list[str] = []

    async def llm_translate_async(
        self, prompt, rate_limit_params=None, response_schema=None
    ):
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.handler(prompt)


def echo_upper(prompt: str) -> str:
    """Return each input uppercased, preserving ids and run tags."""
    payload = prompt.split("## Here is the input:", 1)[1].strip()
    items = json.loads(payload)
    return json.dumps(
        {
            "items": [
                {"id": item["id"], "output": item["input"].upper()} for item in items
            ]
        }
    )


@pytest.fixture
def translator():
    return DocxSegmentTranslator(
        translate_engine=FakeEngine(handler=echo_upper),
        lang_out="Spanish",
        domain="legal",
    )


class TestBuildPayload:
    def test_single_group_is_plain_text(self, tmp_path):
        segments = segments_for(paragraph(run("Hello world")), tmp_path)
        assert build_payload(segments[0]) == "Hello world"

    def test_multiple_groups_are_tagged(self, tmp_path):
        segments = segments_for(paragraph(run("Due "), run("now", bold=True)), tmp_path)
        assert build_payload(segments[0]) == "<g0>Due </g0><g1>now</g1>"


class TestParsePayload:
    def test_single_group_strips_stray_tags(self):
        assert parse_payload("<g0>Hola</g0>", 1) == ["Hola"]

    def test_tagged_payload_parsed_in_order(self):
        assert parse_payload("<g0>Uno </g0><g1>dos</g1>", 2) == ["Uno ", "dos"]

    def test_empty_span_allowed(self):
        assert parse_payload("<g0>todo</g0><g1></g1>", 2) == ["todo", ""]

    def test_missing_tag_rejected(self):
        assert parse_payload("<g0>solo</g0>", 2) is None

    def test_duplicate_tag_rejected(self):
        assert parse_payload("<g0>a</g0><g0>b</g0>", 2) is None

    def test_out_of_range_tag_rejected(self):
        assert parse_payload("<g0>a</g0><g5>b</g5>", 2) is None

    def test_multiline_span_parsed(self):
        assert parse_payload("<g0>line one\nline two</g0><g1>b</g1>", 2) == [
            "line one\nline two",
            "b",
        ]


class TestBuildBatches:
    def test_respects_paragraph_budget(self, translator, tmp_path):
        body = "".join(paragraph(run(f"Paragraph {i}")) for i in range(7))
        segments = segments_for(body, tmp_path)
        with patch(
            "src.worker.docxtranslator.segment_translator.settings"
        ) as mock_settings:
            mock_settings.LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS = 3
            mock_settings.LLM_TRANSLATION_BATCH_MAX_TOKENS = 100_000
            batches = translator.build_batches(segments)
        assert [len(batch.segments) for batch in batches] == [3, 3, 1]
        assert [batch.batch_index for batch in batches] == [0, 1, 2]

    def test_respects_token_budget(self, translator, tmp_path):
        body = "".join(paragraph(run("word " * 50)) for _ in range(4))
        segments = segments_for(body, tmp_path)
        with patch(
            "src.worker.docxtranslator.segment_translator.settings"
        ) as mock_settings:
            mock_settings.LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS = 100
            mock_settings.LLM_TRANSLATION_BATCH_MAX_TOKENS = 60
            batches = translator.build_batches(segments)
        assert len(batches) > 1

    def test_oversized_single_segment_still_batched(self, translator, tmp_path):
        segments = segments_for(paragraph(run("word " * 500)), tmp_path)
        with patch(
            "src.worker.docxtranslator.segment_translator.settings"
        ) as mock_settings:
            mock_settings.LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS = 10
            mock_settings.LLM_TRANSLATION_BATCH_MAX_TOKENS = 5
            batches = translator.build_batches(segments)
        assert len(batches) == 1


class TestTranslate:
    async def test_translates_every_segment(self, translator, tmp_path):
        segments = segments_for(
            paragraph(run("alpha")) + paragraph(run("beta")), tmp_path
        )
        outcome = await translator.translate(segments)
        assert outcome.translations == {0: ["ALPHA"], 1: ["BETA"]}
        assert outcome.failed_batches == 0

    async def test_skips_untranslatable_segments(self, translator, tmp_path):
        segments = segments_for(
            paragraph(run("alpha")) + paragraph(run("1000")), tmp_path
        )
        outcome = await translator.translate(segments)
        assert set(outcome.translations) == {0}

    async def test_no_llm_call_when_nothing_translatable(self, tmp_path):
        engine = FakeEngine(handler=echo_upper)
        translator = DocxSegmentTranslator(
            translate_engine=engine, lang_out="Spanish", domain=None
        )
        segments = segments_for(paragraph(run("1000")), tmp_path)
        outcome = await translator.translate(segments)
        assert engine.calls == []
        assert outcome.translations == {}

    async def test_tagged_segment_round_trips(self, translator, tmp_path):
        segments = segments_for(paragraph(run("Due "), run("now", bold=True)), tmp_path)
        outcome = await translator.translate(segments)
        assert outcome.translations == {0: ["DUE ", "NOW"]}
        assert outcome.tag_fallback_segments == 0

    async def test_misaligned_tags_fall_back_to_first_run(self, tmp_path):
        def drop_second_tag(prompt):
            payload = prompt.split("## Here is the input:", 1)[1].strip()
            items = json.loads(payload)
            return json.dumps(
                {
                    "items": [
                        {"id": item["id"], "output": "Vencimiento ahora"}
                        for item in items
                    ]
                }
            )

        translator = DocxSegmentTranslator(
            translate_engine=FakeEngine(handler=drop_second_tag),
            lang_out="Spanish",
            domain=None,
        )
        segments = segments_for(paragraph(run("Due "), run("now", bold=True)), tmp_path)
        outcome = await translator.translate(segments)
        assert outcome.tag_fallback_segments == 1
        assert outcome.translations == {0: ["Vencimiento ahora", ""]}

    async def test_batch_failure_retries_each_paragraph(self, tmp_path):
        state = {"batch_seen": False}

        def fail_batch_then_succeed(prompt):
            payload = prompt.split("## Here is the input:", 1)[1].strip()
            items = json.loads(payload)
            if len(items) > 1:
                state["batch_seen"] = True
                raise RuntimeError("batch too big")
            return echo_upper(prompt)

        translator = DocxSegmentTranslator(
            translate_engine=FakeEngine(handler=fail_batch_then_succeed),
            lang_out="Spanish",
            domain=None,
        )
        segments = segments_for(
            paragraph(run("alpha")) + paragraph(run("beta")), tmp_path
        )
        outcome = await translator.translate(segments)
        assert state["batch_seen"] is True
        assert outcome.failed_batches == 1
        assert outcome.translations == {0: ["ALPHA"], 1: ["BETA"]}

    async def test_total_failure_counts_untranslated(self, tmp_path):
        translator = DocxSegmentTranslator(
            translate_engine=FakeEngine(error=RuntimeError("model down")),
            lang_out="Spanish",
            domain=None,
        )
        segments = segments_for(paragraph(run("alpha")), tmp_path)
        outcome = await translator.translate(segments)
        assert outcome.translations == {}
        assert outcome.untranslated_segments == 1

    async def test_empty_output_counted_as_untranslated(self, tmp_path):
        def blank(prompt):
            payload = prompt.split("## Here is the input:", 1)[1].strip()
            items = json.loads(payload)
            return json.dumps(
                {"items": [{"id": item["id"], "output": "  "} for item in items]}
            )

        translator = DocxSegmentTranslator(
            translate_engine=FakeEngine(handler=blank),
            lang_out="Spanish",
            domain=None,
        )
        segments = segments_for(paragraph(run("alpha")), tmp_path)
        outcome = await translator.translate(segments)
        assert outcome.translations == {}
        assert outcome.untranslated_segments == 1


class TestResponseParsing:
    def test_accepts_fenced_json(self, translator):
        raw = '```json\n{"items": [{"id": 0, "output": "hola"}]}\n```'
        assert translator._parse_response(raw, 1) == {0: "hola"}

    def test_accepts_bare_array(self, translator):
        assert translator._parse_response('[{"id": 0, "output": "hola"}]', 1) == {
            0: "hola"
        }

    def test_rejects_length_mismatch(self, translator):
        with pytest.raises(ValueError, match="length mismatch"):
            translator._parse_response('{"items": [{"id": 0, "output": "a"}]}', 2)

    def test_rejects_missing_id(self, translator):
        with pytest.raises(ValueError, match="missing id"):
            translator._parse_response('{"items": [{"output": "a"}]}', 1)


class TestGlossaryBlocks:
    def test_no_glossary_yields_empty_blocks(self, translator):
        assert translator._glossary_blocks("some text") == ("", "")

    def test_matched_terms_render_a_table(self, tmp_path):
        from src.worker.doctranslator.glossary import Glossary
        from src.worker.doctranslator.glossary import GlossaryEntry

        glossary = Glossary(
            name="legal",
            entries=[GlossaryEntry("indemnity", "indemnización", "Spanish")],
        )
        translator = DocxSegmentTranslator(
            translate_engine=FakeEngine(handler=echo_upper),
            lang_out="Spanish",
            domain="legal",
            glossaries=[glossary],
        )
        usage, tables = translator._glossary_blocks("Limitation of indemnity applies")
        assert "MANDATORY" in usage
        assert "| indemnity | indemnización |" in tables
