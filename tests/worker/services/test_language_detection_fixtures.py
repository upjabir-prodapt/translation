"""Regression tests against the real PDFs in `docs/test_docs/`.

`Colt-profile-EN_2026_JMamends.pdf` is the document that motivated this
whole change: an entirely English company brochure whose short, list-like
blocks (bare city-name lists, ALL-CAPS slogans, executive-name lists) were
confidently mis-tagged Catalan/Tagalog/French/German, tripping the old
mixed-language guard and failing the job. The other two are ordinary
long-form English documents that must keep passing.

These run the real lingua detector through the real production code path
(`JobProcessor.detect_source_language_with_distribution`), so they are the
only tests here that would catch a detector or threshold regression that
the mocked unit tests cannot see.

`docs/test_docs/` is not committed, so each test skips when its fixture is
absent rather than failing CI. The character-level behaviour these assert is
additionally covered by the mocked distributions in
`test_language_detection_core.py` and `test_pipeline_input_consistency.py`,
which do run everywhere.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.worker.services.language_detection_core import supported_language_share
from src.worker.services.processor_service import JobProcessor

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "docs" / "test_docs"

FIXTURES = [
    "Colt-profile-EN_2026_JMamends.pdf",
    "HR Policy Manual 2023.pdf",
    "22-promptengg.pdf",
]


@pytest.fixture
def processor() -> JobProcessor:
    return JobProcessor(progress_tracker=MagicMock())


@pytest.mark.parametrize("filename", FIXTURES)
def test_english_fixtures_detect_as_english_with_full_coverage(processor, filename):
    path = FIXTURE_DIR / filename
    if not path.is_file():
        pytest.skip(f"fixture not present: {path}")

    winner, distribution = processor.detect_source_language_with_distribution(path)

    assert winner == "en"
    # Not just "en wins": every language that survives the noise floor must
    # be translatable, or the pipeline's coverage gate would fail the job.
    assert supported_language_share(distribution) == pytest.approx(1.0)


def test_colt_brochure_has_no_spurious_secondary_languages(processor):
    """The specific regression: at the previous confidence floor this
    document reported Catalan, Tagalog, French, German and Spanish
    alongside English. Those blocks must now come back undetermined and be
    excluded from the distribution entirely, not merely outvoted."""
    path = FIXTURE_DIR / "Colt-profile-EN_2026_JMamends.pdf"
    if not path.is_file():
        pytest.skip(f"fixture not present: {path}")

    _winner, distribution = processor.detect_source_language_with_distribution(path)

    assert set(distribution) == {"en"}
