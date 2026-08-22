import json
import zipfile

import pytest
from fixtures.docx_builder import W_NS
from fixtures.docx_builder import build_docx
from fixtures.docx_builder import paragraph
from fixtures.docx_builder import run
from fixtures.docx_builder import simple_docx
from fixtures.docx_builder import table
from src.worker.docxtranslator.package import DocxPackage
from src.worker.services.dlp_service import DlpProvider
from src.worker.services.dlp_service import DlpResult
from src.worker.services.docx_translation_service import DocxTranslationService


class FakeEngine:
    """Echoes inputs with a prefix, preserving ids and run tags."""

    def __init__(self, transform=None):
        self.transform = transform or (lambda text: f"ES:{text}")
        self.seen_inputs: list[str] = []

    async def llm_translate_async(
        self, prompt, rate_limit_params=None, response_schema=None
    ):
        payload = prompt.split("## Here is the input:", 1)[1].strip()
        items = json.loads(payload)
        outputs = []
        for item in items:
            self.seen_inputs.append(item["input"])
            outputs.append({"id": item["id"], "output": self.transform(item["input"])})
        return json.dumps({"items": outputs})


class StubDlp:
    """Masks a fixed name and records what it was asked to mask."""

    def __init__(self, pii_value="Jane Doe", placeholder="__DLP_TOKEN_0001__"):
        self.secret = pii_value
        self.token = placeholder
        self.chunks_seen: list[str] = []

    def select_provider(self):
        return DlpProvider.REGEX_FALLBACK

    def mask_chunks(self, *, job_id, chunks, source_language, token_counter_start=0):
        self.chunks_seen = list(chunks)
        masked = [chunk.replace(self.secret, self.token) for chunk in chunks]
        rows = (
            [{"token": self.token, "original_value": self.secret}]
            if any(self.secret in chunk for chunk in chunks)
            else []
        )
        return DlpResult(
            masked_chunks=masked,
            token_rows=rows,
            dlp_provider=DlpProvider.REGEX_FALLBACK,
        )


@pytest.fixture
def service():
    return DocxTranslationService()


def docx_at(tmp_path, name="in.docx", content=None):
    path = tmp_path / name
    path.write_bytes(content if content is not None else simple_docx())
    return path


def output_text(path):
    root = DocxPackage.open(path).part_root("word/document.xml")
    return [node.text or "" for node in root.iter(f"{{{W_NS}}}t")]


class TestTranslateDocument:
    async def test_produces_a_docx_not_a_pdf(self, service, tmp_path):
        out = tmp_path / "out.docx"
        result = await service.translate_document(
            input_path=docx_at(tmp_path),
            output_path=out,
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        assert result.output_path == out
        assert out.exists()
        assert zipfile.is_zipfile(out)

    async def test_translates_body_and_table_text(self, service, tmp_path):
        out = tmp_path / "out.docx"
        await service.translate_document(
            input_path=docx_at(tmp_path),
            output_path=out,
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        texts = "".join(output_text(out))
        assert "ES:Payment Terms" in texts
        assert "ES:Item" in texts
        assert "ES:Licence fee" in texts

    async def test_numeric_cell_left_untouched(self, service, tmp_path):
        out = tmp_path / "out.docx"
        await service.translate_document(
            input_path=docx_at(tmp_path),
            output_path=out,
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        assert "1000" in output_text(out)

    async def test_tables_remain_editable_tables(self, service, tmp_path):
        source = docx_at(
            tmp_path, content=build_docx(table(["Item", "Amount"], ["Licence", "1000"]))
        )
        out = tmp_path / "out.docx"
        await service.translate_document(
            input_path=source,
            output_path=out,
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        root = DocxPackage.open(out).part_root("word/document.xml")
        assert len(root.findall(f".//{{{W_NS}}}tbl")) == 1
        assert len(root.findall(f".//{{{W_NS}}}tr")) == 2
        assert len(root.findall(f".//{{{W_NS}}}tc")) == 4
        assert root.find(f".//{{{W_NS}}}tblGrid") is not None

    async def test_non_text_parts_preserved_byte_for_byte(self, service, tmp_path):
        source = docx_at(tmp_path)
        out = tmp_path / "out.docx"
        await service.translate_document(
            input_path=source,
            output_path=out,
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        with zipfile.ZipFile(source) as before, zipfile.ZipFile(out) as after:
            assert before.namelist() == after.namelist()
            assert before.read("[Content_Types].xml") == after.read(
                "[Content_Types].xml"
            )

    async def test_headers_and_footers_translated(self, service, tmp_path):
        header = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:hdr xmlns:w="{W_NS}">{paragraph(run("Confidential"))}</w:hdr>'
        )
        source = docx_at(
            tmp_path,
            content=build_docx(
                paragraph(run("Body")), extra_parts={"word/header1.xml": header}
            ),
        )
        out = tmp_path / "out.docx"
        await service.translate_document(
            input_path=source,
            output_path=out,
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        header_root = DocxPackage.open(out).part_root("word/header1.xml")
        assert header_root.find(f".//{{{W_NS}}}t").text == "ES:Confidential"

    async def test_result_counts_reported(self, service, tmp_path):
        result = await service.translate_document(
            input_path=docx_at(tmp_path),
            output_path=tmp_path / "out.docx",
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        # 6 paragraphs total; "1000" is not translatable.
        assert result.segment_count == 6
        assert result.translated_segment_count == 5
        assert result.batch_count >= 1
        assert result.failed_batches == 0
        assert result.untranslated_segments == 0

    async def test_source_and_translated_text_captured(self, service, tmp_path):
        result = await service.translate_document(
            input_path=docx_at(tmp_path),
            output_path=tmp_path / "out.docx",
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        assert "Payment Terms" in result.source_text
        assert "ES:Payment Terms" in result.translated_text

    async def test_bold_span_preserved_through_translation(self, service, tmp_path):
        source = docx_at(
            tmp_path,
            content=build_docx(
                paragraph(run("Due "), run("within 30 days", bold=True))
            ),
        )
        out = tmp_path / "out.docx"
        await service.translate_document(
            input_path=source,
            output_path=out,
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        root = DocxPackage.open(out).part_root("word/document.xml")
        bold_runs = [
            r for r in root.iter(f"{{{W_NS}}}r") if r.find(f"{{{W_NS}}}rPr") is not None
        ]
        assert len(bold_runs) == 1
        assert "within 30 days" in (bold_runs[0].find(f"{{{W_NS}}}t").text or "")

    async def test_empty_document_succeeds(self, service, tmp_path):
        source = docx_at(tmp_path, content=build_docx(paragraph()))
        out = tmp_path / "out.docx"
        result = await service.translate_document(
            input_path=source,
            output_path=out,
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
        )
        assert out.exists()
        assert result.segment_count == 0


class TestDlpIntegration:
    async def test_pii_masked_before_model_and_restored_after(self, tmp_path):
        dlp = StubDlp()
        service = DocxTranslationService(dlp_service=dlp)
        source = docx_at(
            tmp_path, content=build_docx(paragraph(run("Signed by Jane Doe today")))
        )
        out = tmp_path / "out.docx"
        engine = FakeEngine()

        result = await service.translate_document(
            input_path=source,
            output_path=out,
            translate_engine=engine,
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
            enable_dlp=True,
        )

        # The model never saw the real name...
        assert all("Jane Doe" not in seen for seen in engine.seen_inputs)
        assert any("__DLP_TOKEN_0001__" in seen for seen in engine.seen_inputs)
        # ...but the output document has it back, and no token leaked.
        texts = "".join(output_text(out))
        assert "Jane Doe" in texts
        assert "__DLP_TOKEN_" not in texts
        assert result.dlp_provider == DlpProvider.REGEX_FALLBACK
        assert result.dlp_token_rows

    async def test_masking_skipped_when_disabled(self, tmp_path):
        dlp = StubDlp()
        service = DocxTranslationService(dlp_service=dlp)
        await service.translate_document(
            input_path=docx_at(tmp_path),
            output_path=tmp_path / "out.docx",
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
            enable_dlp=False,
        )
        assert dlp.chunks_seen == []

    async def test_untranslated_paragraph_is_unmasked(self, tmp_path):
        dlp = StubDlp()
        service = DocxTranslationService(dlp_service=dlp)
        # "Jane Doe 2000" has letters so it is masked; the model fails, so the
        # paragraph is never rewritten — the token must still be restored.
        source = docx_at(
            tmp_path, content=build_docx(paragraph(run("Jane Doe signature")))
        )
        out = tmp_path / "out.docx"

        class BrokenEngine:
            async def llm_translate_async(self, *args, **kwargs):
                raise RuntimeError("model down")

        result = await service.translate_document(
            input_path=source,
            output_path=out,
            translate_engine=BrokenEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
            enable_dlp=True,
        )
        texts = "".join(output_text(out))
        assert "__DLP_TOKEN_" not in texts
        assert "Jane Doe signature" in texts
        assert result.untranslated_segments == 1

    async def test_masking_is_per_run_group(self, tmp_path):
        dlp = StubDlp()
        service = DocxTranslationService(dlp_service=dlp)
        source = docx_at(
            tmp_path,
            content=build_docx(paragraph(run("Signed "), run("by", bold=True))),
        )
        await service.translate_document(
            input_path=source,
            output_path=tmp_path / "out.docx",
            translate_engine=FakeEngine(),
            lang_in="English",
            lang_out="Spanish",
            domain="legal",
            job_id="job1",
            enable_dlp=True,
        )
        assert dlp.chunks_seen == ["Signed ", "by"]
