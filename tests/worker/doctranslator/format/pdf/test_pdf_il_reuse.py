"""Tests for C1: PDF IL-tree reuse across model attempts."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.doctranslator.format.pdf.document_il import il_version_1
from src.worker.doctranslator.format.pdf.high_level import _do_translate_single
from src.worker.doctranslator.format.pdf.translation_config import (
    SharedContextCrossSplitPart,
)
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig


def test_shared_context_il_cache():
    """Verify SharedContextCrossSplitPart stores and retrieves parsed IL documents."""
    sc = SharedContextCrossSplitPart()
    assert not sc.has_cached_il_docs()
    assert sc.get_cached_il_doc(0) is None

    mock_doc = il_version_1.Document()
    sc.set_cached_il_doc(0, mock_doc)

    assert sc.has_cached_il_docs()
    assert sc.get_cached_il_doc(0) is mock_doc
    assert sc.get_cached_il_doc(1) is None


@patch("src.worker.doctranslator.format.pdf.high_level.PDFCreater")
@patch("src.worker.doctranslator.format.pdf.high_level.Typesetting")
@patch("src.worker.doctranslator.format.pdf.high_level._run_translation_phase")
@patch(
    "src.worker.doctranslator.format.pdf.high_level._run_layout_and_structure_phases"
)
@patch("src.worker.doctranslator.format.pdf.high_level._build_il_document")
@patch("src.worker.doctranslator.format.pdf.high_level._prepare_working_pdf")
def test_do_translate_single_caches_il_on_first_attempt_and_reuses_on_retry(
    mock_prepare,
    mock_build_il,
    _mock_run_layout,
    mock_run_translation,
    mock_typesetting,
    _mock_pdf_creater,
    tmp_path: Path,
):
    """C1: _do_translate_single caches IL on first attempt and reuses on retry."""
    input_pdf = tmp_path / "sample.pdf"
    input_pdf.write_bytes(b"%PDF-1.4 dummy")

    mock_doc_mupdf = MagicMock()
    temp_pdf = tmp_path / "input.pdf"
    mediabox = {0: [0, 0, 100, 100]}
    mock_prepare.return_value = (mock_doc_mupdf, str(temp_pdf), mediabox)

    il_doc_1 = il_version_1.Document()
    il_doc_1.page = [MagicMock()]
    mock_build_il.return_value = il_doc_1
    _mock_run_layout.return_value = il_doc_1

    mock_creater_instance = MagicMock()
    mock_creater_instance.write.return_value = MagicMock()
    _mock_pdf_creater.return_value = mock_creater_instance

    shared_context = SharedContextCrossSplitPart()

    config_1 = MagicMock(spec=TranslationConfig)
    config_1.input_file = str(input_pdf)
    config_1.debug = False
    config_1.enable_dlp = False
    config_1.only_include_translated_page = False
    config_1.only_parse_generate_pdf = False
    config_1.watermark_output_mode = 0
    config_1.split_part_index = 0
    config_1.shared_context_cross_split_part = shared_context

    pm = MagicMock()

    # --- Attempt 1: Not cached -> builds IL and runs layout phases ---
    _do_translate_single(pm, config_1)

    assert mock_build_il.call_count == 1
    assert _mock_run_layout.call_count == 1
    assert mock_run_translation.call_count == 1
    assert mock_typesetting.call_count == 1
    assert shared_context.has_cached_il_docs()
    assert shared_context.get_cached_il_doc(0) is not None

    # Reset mocks for Attempt 2
    mock_build_il.reset_mock()
    _mock_run_layout.reset_mock()
    mock_run_translation.reset_mock()

    # --- Attempt 2: Cached -> skips build_il and layout phases! ---
    config_2 = MagicMock(spec=TranslationConfig)
    config_2.input_file = str(input_pdf)
    config_2.debug = False
    config_2.enable_dlp = False
    config_2.only_include_translated_page = False
    config_2.only_parse_generate_pdf = False
    config_2.watermark_output_mode = 0
    config_2.split_part_index = 0
    config_2.shared_context_cross_split_part = shared_context

    _do_translate_single(pm, config_2)

    # _build_il_document and _run_layout_and_structure_phases must NOT be called on retry
    mock_build_il.assert_not_called()
    _mock_run_layout.assert_not_called()
    # But translation phase runs for the new attempt
    assert mock_run_translation.call_count == 1
