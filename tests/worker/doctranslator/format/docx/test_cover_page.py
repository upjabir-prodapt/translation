from docx import Document
from src.worker.doctranslator.format.docx.cover_page import add_cover_page
from src.worker.doctranslator.format.pdf.translation_config import TranslationCoverPageMetadata


def test_add_cover_page(tmp_path):
    doc = Document()
    doc.add_paragraph("Original body paragraph 1")
    doc.add_paragraph("Original body paragraph 2")

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
    assert any("cover page summarizes" in t for t in texts)
    assert any("Original language: English" in t for t in texts)
    assert any("Target language: French" in t for t in texts)
    assert any("AI generated translation" in t for t in texts)
    assert "Original body paragraph 1" in texts
    assert "Original body paragraph 2" in texts
    # original body content must still come after the cover page content
    assert texts.index("Original body paragraph 1") > texts.index("AI Translated Document")
