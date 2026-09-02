"""Typesetting must report a paragraph it could not lay out.

`_find_optimal_scale_and_layout()` walks the scale down to min_scale and,
if nothing ever fits, returns having written no composition -- and it
swallowed every layout exception on the way. The paragraph then renders
with no characters and its text is silently missing from the delivered PDF;
the first (and only) signal was pdf_creater's "Unable to export paragraphs
that have not yet been formatted", raised a stage later with no idea why.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.doctranslator.format.pdf.document_il import Box
from src.worker.doctranslator.format.pdf.document_il.midend.typesetting import (
    Typesetting,
)


def _typesetting():
    instance = Typesetting.__new__(Typesetting)
    instance.is_cjk = False
    # Space expansion is unavailable, so the scale walk runs to min_scale.
    instance.get_max_bottom_space = MagicMock(side_effect=ValueError("no space"))
    instance.get_max_right_space = MagicMock(side_effect=ValueError("no space"))
    return instance


def _paragraph(text: str = "Verwandtschaftsverhältnis"):
    paragraph = MagicMock()
    paragraph.box = Box(x=228.2, y=704.4, x2=303.4, y2=713.0)
    paragraph.debug_id = "9f1Bh"
    paragraph.unicode = text
    paragraph.layout_label = "fallback_line"
    return paragraph


def _run(instance, paragraph, *, apply_layout, layout_result=None, layout_error=None):
    kwargs = (
        {"side_effect": layout_error}
        if layout_error is not None
        else {"return_value": layout_result or ([], False)}
    )
    with patch.object(Typesetting, "_layout_typesetting_units", **kwargs):
        return instance._find_optimal_scale_and_layout(
            paragraph, MagicMock(), [MagicMock()], 1.0, True, apply_layout
        )


class TestGiveUpIsReported:
    def test_nothing_fits_is_logged_as_an_error(self, caplog):
        paragraph = _paragraph()
        with caplog.at_level(logging.ERROR):
            _run(_typesetting(), paragraph, apply_layout=True)
        assert "could not fit paragraph 9f1Bh" in caplog.text

    def test_the_report_carries_the_diagnostic_context(self, caplog):
        paragraph = _paragraph("x" * 3000)
        with caplog.at_level(logging.ERROR):
            _run(_typesetting(), paragraph, apply_layout=True)
        # Box size, text length and layout label are what explain the failure,
        # and none of them are recoverable once the renderer reports it.
        assert "75.2x8.6pt" in caplog.text
        assert "text_length=3000" in caplog.text
        assert "fallback_line" in caplog.text

    def test_long_text_is_truncated_in_the_log(self, caplog):
        with caplog.at_level(logging.ERROR):
            _run(_typesetting(), _paragraph("y" * 5000), apply_layout=True)
        assert len(caplog.text) < 2000

    def test_measuring_pass_does_not_report(self, caplog):
        """apply_layout=False only computes a scale; nothing is lost yet."""
        with caplog.at_level(logging.ERROR):
            _run(_typesetting(), _paragraph(), apply_layout=False)
        assert caplog.text == ""

    def test_successful_layout_does_not_report(self, caplog):
        unit = MagicMock()
        unit.render.return_value = ([], [], [])
        with caplog.at_level(logging.WARNING):
            scale, units = _run(
                _typesetting(),
                _paragraph(),
                apply_layout=True,
                layout_result=([unit], True),
            )
        assert scale == 1.0
        assert units == [unit]
        assert caplog.text == ""


class TestSwallowedLayoutErrorsAreSurfaced:
    def test_layout_exception_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING):
            _run(
                _typesetting(),
                _paragraph(),
                apply_layout=True,
                layout_error=ValueError("bad glyph"),
            )
        assert "bad glyph" in caplog.text
        assert "9f1Bh" in caplog.text

    def test_layout_exception_is_logged_once_not_per_scale(self, caplog):
        """The loop retries ~20 scales; one report is enough."""
        with caplog.at_level(logging.WARNING):
            _run(
                _typesetting(),
                _paragraph(),
                apply_layout=True,
                layout_error=ValueError("bad glyph"),
            )
        assert caplog.text.count("bad glyph") == 1
