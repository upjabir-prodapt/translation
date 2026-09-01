"""Typesetting must never emit an empty composition for non-empty text.

Production defect: when a paragraph's text could not be made to fit its
box at any scale down to `min_scale = 0.1`, `_find_optimal_scale_and_layout`
returned without ever committing a layout. `render_paragraph` had already
cleared `pdf_paragraph_composition = []`, so the paragraph reached
`pdf_creater.render_paragraph_to_char` with non-empty `unicode` and an
empty composition -- it was silently dropped from the output PDF and only
an ERROR was logged.

Chosen behaviour: force-render at minimum scale and allow overflow.
Overflowing text is visible and fixable; silently lost text is not.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.worker.doctranslator.format.pdf.document_il import Box
from src.worker.doctranslator.format.pdf.document_il.midend.typesetting import (
    Typesetting,
)


def _make_typesetting() -> Typesetting:
    typesetting = Typesetting.__new__(Typesetting)
    typesetting.is_cjk = False
    typesetting.translation_config = MagicMock()
    typesetting.font_mapper = MagicMock()
    # Space expansion must not rescue the layout in these tests.
    typesetting.get_max_bottom_space = MagicMock(return_value=1000.0)
    typesetting.get_max_right_space = MagicMock(return_value=-1000.0)
    typesetting._update_paragraph_render_order = MagicMock()
    return typesetting


def _make_paragraph(unicode_text: str = "Verwandtschaftsverhältnis" * 40):
    paragraph = MagicMock()
    # A ~75 x 9 pt box, as in the incident.
    paragraph.box = Box(x=100.0, y=500.0, x2=175.0, y2=509.0)
    paragraph.unicode = unicode_text
    paragraph.debug_id = "p-42"
    paragraph.layout_label = "plain text"
    paragraph.optimal_scale = 0.1
    paragraph.pdf_paragraph_composition = []
    return paragraph


def _make_unit(can_passthrough: bool = False):
    unit = MagicMock()
    unit.can_passthrough = can_passthrough
    unit.render.return_value = ([MagicMock()], [], [])
    return unit


def _never_fits(units, *_args, **_kwargs):
    """Stand-in for `_layout_typesetting_units` that always overflows."""
    return list(units), False


def _yields_nothing(*_args, **_kwargs):
    """Stand-in for the pathological `return [], False` early exit."""
    return [], False


class TestForcedLayoutInFindOptimalScale:
    def test_force_commits_a_layout_when_nothing_fits(self):
        typesetting = _make_typesetting()
        typesetting._layout_typesetting_units = MagicMock(side_effect=_never_fits)
        paragraph = _make_paragraph()
        page = MagicMock()
        page.pdf_curve = []
        page.pdf_form = []
        units = [_make_unit() for _ in range(3)]

        scale, typeset_units = typesetting._find_optimal_scale_and_layout(
            paragraph, page, units, 1.0, apply_layout=True, force=True
        )

        assert scale == pytest.approx(0.1)
        assert typeset_units is not None
        assert paragraph.pdf_paragraph_composition != []
        assert len(paragraph.pdf_paragraph_composition) == 3

    def test_without_force_the_composition_stays_empty(self):
        """Pins the pre-existing default so the forced path is opt-in."""
        typesetting = _make_typesetting()
        typesetting._layout_typesetting_units = MagicMock(side_effect=_never_fits)
        paragraph = _make_paragraph()
        page = MagicMock()

        _scale, typeset_units = typesetting._find_optimal_scale_and_layout(
            paragraph, page, [_make_unit()], 1.0, apply_layout=True, force=False
        )

        assert typeset_units is None
        assert paragraph.pdf_paragraph_composition == []

    def test_force_without_apply_layout_does_not_mutate(self):
        typesetting = _make_typesetting()
        typesetting._layout_typesetting_units = MagicMock(side_effect=_never_fits)
        paragraph = _make_paragraph()

        typesetting._find_optimal_scale_and_layout(
            paragraph, MagicMock(), [_make_unit()], 1.0, apply_layout=False, force=True
        )

        assert paragraph.pdf_paragraph_composition == []

    def test_forced_pass_yielding_no_units_logs_error_and_stays_empty(self, caplog):
        typesetting = _make_typesetting()
        typesetting._layout_typesetting_units = MagicMock(side_effect=_yields_nothing)
        paragraph = _make_paragraph()

        with caplog.at_level("ERROR"):
            _scale, typeset_units = typesetting._find_optimal_scale_and_layout(
                paragraph,
                MagicMock(),
                [_make_unit()],
                1.0,
                apply_layout=True,
                force=True,
            )

        assert typeset_units is None
        assert paragraph.pdf_paragraph_composition == []
        assert "Forced typesetting pass produced no units" in caplog.text
        assert "p-42" in caplog.text


class TestRenderParagraphFallback:
    def test_render_paragraph_forces_a_layout_rather_than_dropping_text(self, caplog):
        typesetting = _make_typesetting()
        typesetting._layout_typesetting_units = MagicMock(side_effect=_never_fits)
        units = [_make_unit() for _ in range(2)]
        typesetting.create_typesetting_units = MagicMock(return_value=units)
        paragraph = _make_paragraph()
        page = MagicMock()
        page.pdf_curve = []
        page.pdf_form = []

        with caplog.at_level("ERROR"):
            typesetting.render_paragraph(paragraph, page, {})

        assert paragraph.pdf_paragraph_composition != []
        assert len(paragraph.pdf_paragraph_composition) == 2
        assert "forcing a min-scale layout" in caplog.text
        assert "p-42" in caplog.text

    def test_passthrough_paragraphs_are_untouched(self):
        typesetting = _make_typesetting()
        typesetting._layout_typesetting_units = MagicMock(side_effect=_never_fits)
        typesetting.create_passthrough_composition = MagicMock(
            return_value=["passthrough"]
        )
        typesetting.create_typesetting_units = MagicMock(
            return_value=[_make_unit(can_passthrough=True)]
        )
        paragraph = _make_paragraph()

        typesetting.render_paragraph(paragraph, MagicMock(), {})

        assert paragraph.pdf_paragraph_composition == ["passthrough"]
        assert paragraph.scale == 1.0
