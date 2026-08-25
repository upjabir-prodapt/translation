from docx import Document
from src.worker.doctranslator.format.docx.cover_page import add_cover_page
from src.worker.doctranslator.format.pdf.translation_config import TranslationCoverPageMetadata


def test_add_cover_page_with_judge(tmp_path):
    doc = Document()
    doc.add_paragraph("Original body paragraph 1")

    meta = TranslationCoverPageMetadata(
        original_language="en",
        target_language="fr",
        model_used="gemini-3.5-flash",
        domain="commercial",
        translation_date="2026-08-23 00:00:00 UTC",
        confidence_score=0.899,
        translated_sections="All paragraphs",
        judge_model="gemini-3.5-flash",
    )
    add_cover_page(doc, meta)
    out = tmp_path / "cover_test.docx"
    doc.save(str(out))

    reopened = Document(str(out))
    texts = [p.text for p in reopened.paragraphs]
    assert texts[0] == "AI Translated Document"
    assert any("Confidence score: 0.90 (89.9%) via gemini-3.5-flash" in t for t in texts)


def test_add_cover_page_without_judge(tmp_path):
    doc = Document()
    doc.add_paragraph("Original body paragraph 1")

    meta = TranslationCoverPageMetadata(
        original_language="en",
        target_language="fr",
        model_used="gemini-3.5-flash",
        domain="commercial",
        translation_date="2026-08-23 00:00:00 UTC",
        confidence_score=None,
        translated_sections="All paragraphs",
        judge_model=None,
    )
    add_cover_page(doc, meta)
    out = tmp_path / "cover_no_judge_test.docx"
    doc.save(str(out))

    reopened = Document(str(out))
    texts = [p.text for p in reopened.paragraphs]
    assert texts[0] == "AI Translated Document"
    assert any("Original language: English" in t for t in texts)
    assert any("Target language: French" in t for t in texts)
    assert not any("Confidence score" in t for t in texts)

