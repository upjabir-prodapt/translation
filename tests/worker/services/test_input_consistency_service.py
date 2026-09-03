"""Unit tests for `input_consistency_service`: sampling, fail-closed error
conversion, and the per-source_hash classification cache.

The orchestrator-level policy these feed is covered in
tests/worker/services/test_pipeline_input_consistency.py.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from src.config.constants import settings
from src.worker.services import input_consistency_service as ics
from src.worker.services.input_consistency_service import DomainCheckUnavailableError
from src.worker.services.input_consistency_service import DomainClassification
from src.worker.services.input_consistency_service import DomainClassificationError


@pytest.fixture(autouse=True)
def _clear_cache():
    ics._classification_cache.clear()
    yield
    ics._classification_cache.clear()


def _classification(domain: str = "legal") -> DomainClassification:
    return DomainClassification(
        domain=domain, confidence=0.9, reason="A supply agreement between parties."
    )


class TestSampling:
    def test_short_document_returned_whole(self):
        assert ics._sample_blocks(["hello world", "second block"], 4000) == (
            "hello world\nsecond block"
        )

    def test_whitespace_is_collapsed_and_empties_dropped(self):
        assert ics._sample_blocks(["  a \n\n b  ", "   ", "c"], 4000) == "a b\nc"

    def test_no_blocks_yields_empty_string(self):
        assert ics._sample_blocks([], 4000) == ""
        assert ics._sample_blocks(["", "   "], 4000) == ""

    def test_budget_is_respected(self):
        blocks = [f"block number {i} with some filler text" for i in range(500)]
        sample = ics._sample_blocks(blocks, 200)
        assert len(sample.replace("\n", "")) <= 200

    def test_sampling_spans_the_document_rather_than_truncating(self):
        """A cover page is the weakest domain evidence a document has.

        The excerpt must therefore reach material from the end of the
        document, not just the first N characters. Uses the real default
        budget: an earlier implementation strided by a fixed block count and
        silently collapsed to "the first N blocks" here.
        """
        blocks = [f"UNIQUE{i} " + "padding text here" for i in range(400)]
        sample = ics._sample_blocks(blocks, 4000)
        assert "UNIQUE0" in sample
        late_hits = [i for i in range(300, 400) if f"UNIQUE{i}" in sample]
        assert late_hits, "sample never reached the tail of the document"

    def test_long_blocks_are_spread_and_stay_readable(self):
        """With page-sized blocks the excerpts must span and stay meaningful."""
        blocks = [f"PAGE{i} " + ("lorem ipsum dolor " * 200) for i in range(40)]
        sample = ics._sample_blocks(blocks, 4000)
        assert len(sample) <= 4000 + 64  # plus newline separators
        seen = [i for i in range(40) if f"PAGE{i} " in sample]
        assert seen[0] == 0
        assert seen[-1] >= 30, "sample never reached the tail of the document"
        # Every excerpt must be long enough to carry a domain signal.
        assert all(
            len(line) >= ics._MIN_SAMPLE_EXCERPT_CHARS for line in sample.splitlines()
        )


class TestParseClassification:
    class _Response:
        def __init__(self, parsed=None, text=""):
            self.parsed = parsed
            self.text = text

    def test_uses_sdk_parsed_object(self):
        expected = _classification()
        assert ics._parse_classification(self._Response(parsed=expected)) is expected

    def test_falls_back_to_raw_json_text(self):
        raw = '{"domain": "hr", "confidence": 0.8, "reason": "Leave policy."}'
        parsed = ics._parse_classification(self._Response(text=raw))
        assert parsed.domain == "hr"

    def test_strips_markdown_code_fences(self):
        raw = '```json\n{"domain": "hr", "confidence": 0.8, "reason": "Leave."}\n```'
        assert ics._parse_classification(self._Response(text=raw)).domain == "hr"

    def test_empty_response_raises_retryable_error(self):
        with pytest.raises(DomainClassificationError):
            ics._parse_classification(self._Response(text="   "))

    def test_unparseable_response_raises_retryable_error(self):
        with pytest.raises(DomainClassificationError):
            ics._parse_classification(self._Response(text="not json at all"))


class TestFailClosedConversion:
    """Every non-verdict outcome becomes DomainCheckUnavailableError."""

    async def test_sampling_failure(self, tmp_path: Path):
        with patch.object(
            ics, "sample_document_text", side_effect=OSError("cannot open")
        ):
            with pytest.raises(DomainCheckUnavailableError, match="could not sample"):
                await ics._do_classify(
                    job_id="j",
                    path=tmp_path / "x.pdf",
                    is_docx=False,
                    source_language="en",
                    enable_dlp=False,
                )

    async def test_sample_too_short(self, tmp_path: Path):
        with patch.object(ics, "sample_document_text", return_value="tiny"):
            with pytest.raises(DomainCheckUnavailableError, match="too short"):
                await ics._do_classify(
                    job_id="j",
                    path=tmp_path / "x.pdf",
                    is_docx=False,
                    source_language="en",
                    enable_dlp=False,
                )

    async def test_dlp_masking_failure(self, tmp_path: Path):
        """Rather than send unmasked text, the guard reports itself down."""
        with (
            patch.object(ics, "sample_document_text", return_value="x" * 500),
            patch.object(ics, "_mask_sample", side_effect=RuntimeError("dlp down")),
        ):
            with pytest.raises(DomainCheckUnavailableError, match="DLP masking"):
                await ics._do_classify(
                    job_id="j",
                    path=tmp_path / "x.pdf",
                    is_docx=False,
                    source_language="en",
                    enable_dlp=True,
                )

    async def test_llm_call_failure(self, tmp_path: Path):
        with (
            patch.object(ics, "sample_document_text", return_value="x" * 500),
            patch.object(
                ics, "_classify_blocking", side_effect=RuntimeError("vertex 503")
            ),
        ):
            with pytest.raises(DomainCheckUnavailableError, match="call failed"):
                await ics._do_classify(
                    job_id="j",
                    path=tmp_path / "x.pdf",
                    is_docx=False,
                    source_language="en",
                    enable_dlp=False,
                )

    async def test_dlp_is_skipped_when_disabled(self, tmp_path: Path):
        with (
            patch.object(ics, "sample_document_text", return_value="x" * 500),
            patch.object(ics, "_mask_sample") as mask,
            patch.object(
                ics, "_classify_blocking", return_value=(_classification(), 0.01)
            ),
        ):
            classification, cost = await ics._do_classify(
                job_id="j",
                path=tmp_path / "x.pdf",
                is_docx=False,
                source_language="en",
                enable_dlp=False,
            )

        mask.assert_not_called()
        assert classification.domain == "legal"
        assert cost == 0.01


class TestClassificationCache:
    """Siblings of a multi-target batch share one call and one verdict."""

    async def test_siblings_share_one_call(self, tmp_path: Path):
        calls: list[str] = []

        def _fake(sample: str):
            calls.append(sample)
            return _classification(), 0.03

        with (
            patch.object(ics, "sample_document_text", return_value="x" * 500),
            patch.object(ics, "_classify_blocking", side_effect=_fake),
        ):
            verdicts = await asyncio.gather(
                *[
                    ics.classify_document_domain(
                        job_id=f"job-{i}",
                        path=tmp_path / "x.pdf",
                        is_docx=False,
                        source_language="en",
                        enable_dlp=False,
                        source_hash="hash-abc",
                    )
                    for i in range(3)
                ]
            )

        assert len(calls) == 1
        assert {v.classification.domain for v in verdicts} == {"legal"}
        # Only the sibling that made the call is billed for it.
        assert sorted(v.cost_usd for v in verdicts) == [0.0, 0.0, 0.03]

    async def test_failure_is_shared_then_evicted(self, tmp_path: Path):
        """Concurrent siblings fail identically, but a later job retries."""
        with (
            patch.object(ics, "sample_document_text", return_value="x" * 500),
            patch.object(
                ics, "_classify_blocking", side_effect=RuntimeError("vertex 503")
            ),
        ):
            results = await asyncio.gather(
                *[
                    ics.classify_document_domain(
                        job_id=f"job-{i}",
                        path=tmp_path / "x.pdf",
                        is_docx=False,
                        source_language="en",
                        enable_dlp=False,
                        source_hash="hash-fail",
                    )
                    for i in range(3)
                ],
                return_exceptions=True,
            )

        assert all(isinstance(r, DomainCheckUnavailableError) for r in results)
        # A transient outage must not be memoised for the rest of the process.
        assert "hash-fail" not in ics._classification_cache

        with (
            patch.object(ics, "sample_document_text", return_value="x" * 500),
            patch.object(
                ics, "_classify_blocking", return_value=(_classification(), 0.01)
            ),
        ):
            verdict = await ics.classify_document_domain(
                job_id="job-retry",
                path=tmp_path / "x.pdf",
                is_docx=False,
                source_language="en",
                enable_dlp=False,
                source_hash="hash-fail",
            )
        assert verdict.classification.domain == "legal"

    async def test_no_source_hash_bypasses_the_cache(self, tmp_path: Path):
        calls: list[str] = []

        def _fake(sample: str):
            calls.append(sample)
            return _classification(), 0.01

        with (
            patch.object(ics, "sample_document_text", return_value="x" * 500),
            patch.object(ics, "_classify_blocking", side_effect=_fake),
        ):
            for _ in range(2):
                await ics.classify_document_domain(
                    job_id="job-solo",
                    path=tmp_path / "x.pdf",
                    is_docx=False,
                    source_language="en",
                    enable_dlp=False,
                    source_hash="",
                )

        assert len(calls) == 2
        assert not ics._classification_cache

    async def test_cache_is_bounded(self, tmp_path: Path):
        with (
            patch.object(ics, "sample_document_text", return_value="x" * 500),
            patch.object(
                ics, "_classify_blocking", return_value=(_classification(), 0.0)
            ),
            patch.object(ics, "_MAX_CACHED_CLASSIFICATIONS", 4),
        ):
            for i in range(10):
                await ics.classify_document_domain(
                    job_id=f"job-{i}",
                    path=tmp_path / "x.pdf",
                    is_docx=False,
                    source_language="en",
                    enable_dlp=False,
                    source_hash=f"hash-{i}",
                )

        assert len(ics._classification_cache) <= 4


class TestSampleBudgetSetting:
    def test_sample_chars_setting_is_honoured(self, tmp_path: Path):
        blocks = ["word " * 500]
        with (
            patch.object(settings, "DOMAIN_CLASSIFIER_SAMPLE_CHARS", 100),
            patch.object(ics, "_extract_pdf_blocks", return_value=blocks),
        ):
            sample = ics.sample_document_text(tmp_path / "x.pdf", is_docx=False)
        assert len(sample) <= 100
