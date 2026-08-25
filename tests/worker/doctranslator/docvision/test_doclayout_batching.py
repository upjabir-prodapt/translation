"""ONNX layout inference must actually batch pages.

`OnnxModel.predict()` has always supported batched inference via
`ONNX_LAYOUT_BATCH_SIZE`, but `handle_document()` used to call
`self.predict(image)[0]` once per page, making the setting dead config and
leaving layout parsing fully sequential (58.1s for 20 pages in the
2026-08-24 baseline).
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from src.worker.doctranslator.docvision.doclayout import OnnxModel


def _page(number):
    page = MagicMock()
    page.page_number = number
    return page


def _model():
    """Build an OnnxModel without loading real ONNX weights."""
    model = OnnxModel.__new__(OnnxModel)
    model.lock = MagicMock()
    model.lock.__enter__ = MagicMock(return_value=None)
    model.lock.__exit__ = MagicMock(return_value=False)
    return model


class _FakePixmap:
    height = 4
    width = 4
    samples = bytes(4 * 4 * 3)


@pytest.fixture
def patched_render():
    with patch(
        "src.worker.doctranslator.docvision.doclayout.get_no_rotation_img",
        return_value=_FakePixmap(),
    ) as render:
        yield render


class TestHandleDocumentBatching:
    @pytest.mark.parametrize(
        ("pages", "batch_size", "expected_predict_calls"),
        [
            (20, 4, 5),  # the baseline PDF: 20 pages / 4 = 5 inference calls
            (20, 1, 20),  # batch_size=1 preserves the old one-call-per-page path
            (3, 4, 1),  # fewer pages than the batch size -> a single call
            (9, 4, 3),  # ragged final batch (4 + 4 + 1)
        ],
    )
    def test_pages_are_grouped_into_predict_calls(
        self, patched_render, pages, batch_size, expected_predict_calls
    ):
        model = _model()
        page_list = [_page(i) for i in range(pages)]
        seen_batch_sizes = []

        def fake_predict(images):
            seen_batch_sizes.append(len(images))
            return [MagicMock() for _ in images]

        model.predict = fake_predict
        with patch(
            "src.worker.doctranslator.docvision.doclayout.settings.ONNX_LAYOUT_BATCH_SIZE",
            batch_size,
        ):
            results = list(
                model.handle_document(page_list, MagicMock(), MagicMock(), MagicMock())
            )

        assert len(seen_batch_sizes) == expected_predict_calls
        assert sum(seen_batch_sizes) == pages
        assert max(seen_batch_sizes) <= batch_size
        assert len(results) == pages

    def test_yield_order_matches_input_order(self, patched_render):
        """Downstream assigns layouts by position, so order must be preserved."""
        model = _model()
        page_list = [_page(i) for i in range(10)]
        model.predict = lambda images: [MagicMock() for _ in images]

        with patch(
            "src.worker.doctranslator.docvision.doclayout.settings.ONNX_LAYOUT_BATCH_SIZE",
            4,
        ):
            yielded = [
                page.page_number
                for page, _ in model.handle_document(
                    page_list, MagicMock(), MagicMock(), MagicMock()
                )
            ]

        assert yielded == list(range(10))

    def test_each_page_gets_its_own_result_and_image(self, patched_render):
        """A batch must not hand the same YoloResult to every page in it."""
        model = _model()
        page_list = [_page(i) for i in range(4)]
        model.predict = lambda images: [
            MagicMock(name=f"r{i}") for i in range(len(images))
        ]
        saved = []

        with patch(
            "src.worker.doctranslator.docvision.doclayout.settings.ONNX_LAYOUT_BATCH_SIZE",
            4,
        ):
            pairs = list(
                model.handle_document(
                    page_list,
                    MagicMock(),
                    MagicMock(),
                    lambda _img, res, num: saved.append((id(res), num)),
                )
            )

        assert len({id(result) for _, result in pairs}) == 4
        assert [num for _, num in saved] == [1, 2, 3, 4]  # 1-based page numbers

    def test_cancellation_is_checked_between_batches(self, patched_render):
        model = _model()
        page_list = [_page(i) for i in range(8)]
        model.predict = lambda images: [MagicMock() for _ in images]
        config = MagicMock()

        with patch(
            "src.worker.doctranslator.docvision.doclayout.settings.ONNX_LAYOUT_BATCH_SIZE",
            4,
        ):
            list(model.handle_document(page_list, MagicMock(), config, MagicMock()))

        assert config.raise_if_cancelled.call_count == 2

    def test_images_are_decoded_correctly(self, patched_render):
        """Guards the frombuffer/reshape/BGR-flip that feeds the model."""
        model = _model()
        captured = {}

        def fake_predict(images):
            captured["images"] = images
            return [MagicMock() for _ in images]

        model.predict = fake_predict
        with patch(
            "src.worker.doctranslator.docvision.doclayout.settings.ONNX_LAYOUT_BATCH_SIZE",
            2,
        ):
            list(
                model.handle_document([_page(0)], MagicMock(), MagicMock(), MagicMock())
            )

        image = captured["images"][0]
        assert isinstance(image, np.ndarray)
        assert image.shape == (4, 4, 3)
