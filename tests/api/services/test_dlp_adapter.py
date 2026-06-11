"""Tests for IL-aware DLP masking adapter."""

from src.api.services.dlp_service import DlpProvider
from src.api.services.dlp_service import DlpService
from src.doctranslator.format.pdf.dlp_adapter import apply_dlp_to_document
from src.doctranslator.format.pdf.dlp_adapter import unmask_document_with_tokens
from src.doctranslator.format.pdf.document_il import il_version_1


def test_apply_dlp_to_document_masks_paragraph_unicode():
    docs = il_version_1.Document(
        page=[
            il_version_1.Page(
                pdf_paragraph=[
                    il_version_1.PdfParagraph(unicode="Reach me at alice@example.com")
                ]
            )
        ]
    )

    result = apply_dlp_to_document(
        docs=docs,
        dlp_service=DlpService(),
        job_id="job-1",
        source_language="en",
    )

    assert result.applied is True
    assert result.chunk_count == 1
    assert result.dlp_provider in DlpProvider
    assert docs.page[0].pdf_paragraph[0].unicode == "Reach me at __DLP_TOKEN_0001__"
    assert result.token_rows[0]["chunk_index"] == 0


def test_apply_dlp_to_document_uses_stable_chunk_index_order():
    docs = il_version_1.Document(
        page=[
            il_version_1.Page(
                pdf_paragraph=[
                    il_version_1.PdfParagraph(unicode=""),
                    il_version_1.PdfParagraph(unicode="Email bob@example.com"),
                ]
            ),
            il_version_1.Page(
                pdf_paragraph=[
                    il_version_1.PdfParagraph(unicode="Call +1 (212) 555-1212")
                ]
            ),
        ]
    )

    result = apply_dlp_to_document(
        docs=docs,
        dlp_service=DlpService(),
        job_id="job-2",
        source_language="ja",
    )

    assert result.applied is True
    assert result.chunk_count == 2
    assert result.dlp_provider in DlpProvider
    assert result.token_rows[0]["chunk_index"] == 0
    assert result.token_rows[1]["chunk_index"] == 1
    assert docs.page[0].pdf_paragraph[1].unicode == "Email __DLP_TOKEN_0001__"
    # International phone pattern matches "+1 (212) 555-1212" including the "+" prefix.
    assert docs.page[1].pdf_paragraph[0].unicode == "Call __DLP_TOKEN_0002__"


def test_unmask_document_with_tokens_restores_original_values():
    docs = il_version_1.Document(
        page=[
            il_version_1.Page(
                pdf_paragraph=[
                    il_version_1.PdfParagraph(
                        unicode="Please contact __DLP_TOKEN_0001__ immediately."
                    )
                ]
            )
        ]
    )
    restored = unmask_document_with_tokens(
        docs=docs,
        token_rows=[
            {
                "token": "__DLP_TOKEN_0001__",
                "original_value": "alice@example.com",
            }
        ],
    )
    assert restored == 1
    assert (
        docs.page[0].pdf_paragraph[0].unicode
        == "Please contact alice@example.com immediately."
    )
