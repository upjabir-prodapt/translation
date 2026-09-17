"""Tests for the two pre-translation input-consistency guards in
`PipelineOrchestrator._execute_pipeline`:

  * language: (1) the document's dominant detected language must match the
    declared `source_language`, and (2) enough of the document overall must
    be in a language this service can translate at all;
  * the declared domain vs. the domain an LLM reads the document as.

Both guards are fail-closed: a document that cannot be verified is failed
rather than translated unchecked.

The language guard's history: it originally rejected both "mixed-language"
documents and "declared language doesn't match detection" outright. Both
rejections were removed (mixed documents translated as-is, source_language
treated as routing-only) in favour of a share-based coverage-only check.
That coverage-only design left a gap -- a document confidently mismatched
against its declared language, or one whose untranslatable content was
merely *fragmented* across several individually-small unsupported
languages, could both slip through -- so the declared-vs-dominant
comparison was reinstated, and coverage is now computed on the raw
(non-noise-filtered) distribution instead of the noise-filtered one. Mixed
documents where every language present is *supported* still translate as
-is; only a genuinely wrong declared language or an untranslatable majority
fails the job.

Structured to match tests/worker/services/test_pipeline_language_distribution.py,
which covers the neighbouring Phase C.5 guards.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.constants import settings
from src.worker.services.input_consistency_service import DomainCheckUnavailableError
from src.worker.services.input_consistency_service import DomainClassification
from src.worker.services.input_consistency_service import DomainVerdict
from src.worker.services.pipeline_orchestrator import _UNDETECTED_DOMAIN
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator
from src.worker.services.temp_workspace_service import TempWorkspaceService


def _job_data(
    job_id: str,
    *,
    source_language: str = "en",
    target_language: str = "es",
    domain: str = "legal",
) -> dict:
    return {
        "source_document": {
            "gcs_uri": f"gs://bucket/input/{job_id}/file.pdf",
            "original_filename": "input.pdf",
        },
        "translation_config": {
            "source_language": source_language,
            "target_language": target_language,
            "domain": domain,
        },
        "cost_attribution": {"user_id": "user-1"},
        "processing_options": {"enable_dlp": False},
    }


def _attempt_result(tmp_path: Path) -> dict:
    out_pdf = tmp_path / "mono.pdf"
    out_pdf.write_bytes(b"%PDF-1.4\n")
    return {
        "attempt_index": 1,
        "model_id": "gemini-2.5-flash",
        "mono_pdf_path": str(out_pdf),
        "token_usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 400,
            "estimated_cost_usd": 0.05,
        },
        "quality_report": {"final_score": 0.9, "pass_fail": True},
        "chunks_processed": 3,
        "dlp_token_rows": [],
    }


class FakeProcessor:
    def __init__(self, progress_tracker, result: dict):
        del progress_tracker
        self._result = result

    async def translate(self, config):
        del config
        return self._result


def _verdict(domain: str, confidence: float, cost_usd: float = 0.0) -> DomainVerdict:
    return DomainVerdict(
        classification=DomainClassification(
            domain=domain,
            confidence=confidence,
            reason="It sets out employee leave entitlement and payroll terms.",
        ),
        cost_usd=cost_usd,
    )


@contextmanager
def _patched_orchestrator(bigquery, storage, tmp_path: Path, attempt_result: dict):
    orchestrator = PipelineOrchestrator(
        bigquery=bigquery,
        storage=storage,
        temp_workspace_service=TempWorkspaceService(base_dir=tmp_path / "work"),
    )
    orchestrator.intent_router = MagicMock()
    orchestrator.intent_router.build_intent.return_value = "legal_en_es"
    orchestrator.intent_router.get_model_chain.return_value = ["gemini-2.5-flash"]
    orchestrator.glossary_service = MagicMock()
    orchestrator.glossary_service.load_domain_glossary.return_value = []
    orchestrator.assembly_service = MagicMock()
    orchestrator.assembly_service.upload_output = AsyncMock(
        return_value="gs://bucket/output.pdf"
    )
    with (
        patch(
            "src.worker.services.pipeline_orchestrator.JobProcessor",
            lambda progress_tracker: FakeProcessor(progress_tracker, attempt_result),
        ),
        patch.object(
            orchestrator,
            "_compute_accumulated_chunk_costs",
            new=AsyncMock(
                return_value={
                    "input_tokens": 1000,
                    "output_tokens": 400,
                    "cost_usd": 0.05,
                    "chunk_count": 3,
                }
            ),
        ),
    ):
        yield orchestrator


@pytest.fixture
def pipeline_mocks(tmp_path: Path):
    bigquery = AsyncMock()
    bigquery.patch_translation_job = AsyncMock()
    storage = AsyncMock()
    storage.download_file = AsyncMock(
        side_effect=lambda _blob, path: path.write_bytes(b"%PDF-1.4\n")
    )
    return bigquery, storage, tmp_path


def _status_calls(bigquery, status: str) -> list:
    return [
        call
        for call in bigquery.patch_translation_job.call_args_list
        if call[0][1].get("status") == status
    ]


def _error_message(bigquery) -> str:
    failed = _status_calls(bigquery, "failed")
    assert len(failed) == 1
    return failed[0][0][1]["error_message"]


async def _run(orchestrator, job_id: str, job_data: dict) -> None:
    await orchestrator._execute_pipeline(job_id, job_data, MagicMock())


@contextmanager
def _detector(orchestrator, result, *, raises=None):
    """Stub language detection with a fixed verdict or failure."""
    kwargs = {"side_effect": raises} if raises is not None else {"return_value": result}
    with patch.object(
        orchestrator.language_detector, "detect_with_distribution", **kwargs
    ) as stub:
        yield stub


@contextmanager
def _classifier(verdict=None, *, raises=None):
    """Stub the domain classifier the orchestrator calls."""
    kwargs = (
        {"side_effect": raises} if raises is not None else {"return_value": verdict}
    )
    with patch(
        "src.worker.services.pipeline_orchestrator.classify_document_domain",
        new=AsyncMock(**kwargs),
    ) as stub:
        yield stub


# ---------------------------------------------------------------------------
# Guard 1 -- supported-language coverage
# ---------------------------------------------------------------------------


class TestDeclaredLanguageIsAGate:
    """The document's dominant detected language must match the declared
    `source_language` -- users select exactly one source language per job,
    and a document confidently dominated by a different one was very
    likely submitted under the wrong declaration."""

    async def test_wrong_declared_language_now_fails_job(self, pipeline_mocks):
        """A German document declared `en` fails, naming both languages.

        `de` is itself a supported language (coverage is 1.0, so the
        coverage check alone would pass it), but the dominant detected
        language doesn't match what was declared, which is exactly the
        scenario this check exists to catch.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("de", Counter({"de": 900}))),
                _classifier(_verdict("legal", 0.95)),
            ):
                await _run(
                    orchestrator,
                    "job-lang-mismatch",
                    _job_data("job-lang-mismatch", source_language="en"),
                )

        message = _error_message(bigquery)
        assert "English" in message
        assert "German" in message
        assert _status_calls(bigquery, "completed") == []

    async def test_unknown_declared_language_still_fails(self, pipeline_mocks):
        """Not a detection comparison -- an unroutable value fails on its own.

        The API validates `source_language` against language_mapper.json, so
        this is only reachable by a direct-to-worker submission. Left
        unchecked it would fall silently through `select_model_list` to the
        default model.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 900}))),
                _classifier(_verdict("legal", 0.95)),
            ):
                await _run(
                    orchestrator,
                    "job-lang-unknown",
                    _job_data("job-lang-unknown", source_language="klingon"),
                )

        assert "Unsupported source language 'klingon'" in _error_message(bigquery)

    async def test_detection_runs_even_when_language_is_declared(self, pipeline_mocks):
        """The old gate skipped detection entirely for explicit languages."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))) as detector,
                _classifier(_verdict("legal", 0.95)),
            ):
                await _run(
                    orchestrator,
                    "job-detect-always",
                    _job_data("job-detect-always", source_language="en"),
                )

        assert detector.call_count == 1
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_matching_monolingual_document_completes(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 1200}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-lang-match",
                    _job_data("job-lang-match", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_declared_language_alias_is_normalized(self, pipeline_mocks):
        """A job row carrying "English" rather than "en" still matches."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-lang-alias",
                    _job_data("job-lang-alias", source_language="English"),
                )

        assert _status_calls(bigquery, "failed") == []

    async def test_guard_can_be_disabled(self, pipeline_mocks, caplog):
        """The kill switch lets an unsupported job through, but still logs it.

        Logging while disabled is what makes a shadow rollout possible: the
        real rejection rate can be measured before anyone's job is failed.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "LANGUAGE_MISMATCH_CHECK_ENABLED", False),
                _detector(orchestrator, ("ru", Counter({"ru": 900}))),
                _classifier(_verdict("legal", 0.99)),
                caplog.at_level("WARNING"),
            ):
                await _run(
                    orchestrator,
                    "job-lang-disabled",
                    _job_data("job-lang-disabled", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1
        assert any(
            "below threshold" in record.message and "(guard disabled)" in record.message
            for record in caplog.records
        )

    async def test_coverage_is_logged_on_success(self, pipeline_mocks, caplog):
        """Pass or fail, the distribution and coverage are logged.

        This is the data the rollout tunes `LANGUAGE_SUPPORTED_MIN_COVERAGE`
        from, so it must not be emitted only on rejection.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "LANGUAGE_MISMATCH_CHECK_ENABLED", False),
                _detector(orchestrator, ("en", Counter({"en": 550, "de": 450}))),
                _classifier(_verdict("legal", 0.99)),
                caplog.at_level("INFO"),
            ):
                await _run(
                    orchestrator,
                    "job-coverage-log",
                    _job_data("job-coverage-log", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert any(
            "supported-language coverage 1.000" in record.getMessage()
            for record in caplog.records
        )

    async def test_disabled_guard_allows_undetectable_document(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "LANGUAGE_MISMATCH_CHECK_ENABLED", False),
                _detector(orchestrator, ("en", Counter())),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-empty-disabled",
                    _job_data("job-empty-disabled", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1


class TestMixedLanguageGuard:
    """The mixed-language decision is one global dominance threshold.

    A document fails only when no language reaches
    `LANGUAGE_DETECTION_MIN_DOMINANT_SHARE` (60%) of the detected
    characters. There is no per-block veto: the guard this replaced
    rejected a document as soon as any second language appeared, which
    made one unlucky text block fatal.
    """

    async def test_dominant_language_above_threshold_translates(self, pipeline_mocks):
        """A 90/10 split is not "mixed" -- it translates.

        Under the old purity rule this exact distribution failed the job.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500, "fr": 500}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-mixed",
                    _job_data("job-mixed", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_skylight_contact_block_no_longer_fails_the_job(self, pipeline_mocks):
        """Regression: the real distribution from an all-English PDF.

        A Colt support document whose escalation matrix is a list of names
        and email addresses ("1st Level: <name> (<name>@colt.net)") measured
        en 75% / sq 25% -- the contact block is ~70% personal names, which
        lingua scores as Albanian at 0.95 confidence. The document is
        entirely English and must translate.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 806, "sq": 269}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-skylight",
                    _job_data("job-skylight", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_dominant_language_below_threshold_fails(self, pipeline_mocks):
        """A 55/45 split has no single source language to translate from."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 550, "de": 450}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-half",
                    _job_data("job-half", source_language="en"),
                )

        message = _error_message(bigquery)
        assert "mixed-language" in message
        assert "60%" in message
        assert "English" in message
        assert "German" in message

    async def test_threshold_is_measured_after_incidental_languages_drop(
        self, pipeline_mocks
    ):
        """A long tail of detector noise must not fail a monolingual document.

        English holds only 58% of the raw characters here, but every other
        language is individually incidental (<5%), so once they are excluded
        English holds all of the text that counts. Measuring the 60% on the
        raw distribution instead would reject this document.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        distribution = Counter({"en": 5800})
        distribution.update({f"x{i}": 280 for i in range(15)})

        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", distribution)),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-noisy-tail",
                    _job_data("job-noisy-tail", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_short_document_is_not_rejected_as_mixed(self, pipeline_mocks):
        """Below the evidence floor, prefer translating over failing.

        Mirrors the allowance `resolve_dominant_language` makes, so the
        detection core and this guard cannot disagree on the same document.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 120, "de": 100}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-short-mixed",
                    _job_data("job-short-mixed", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_mixed_message_wins_over_mismatch_message(self, pipeline_mocks):
        """A document that is both mixed and mis-declared reports 'mixed'.

        Reporting the mismatch first would tell the user to resubmit as
        German, only for that resubmission to fail again as mixed.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("ru", Counter({"ru": 1000}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-unsupported",
                    _job_data("job-unsupported", source_language="en"),
                )

        message = _error_message(bigquery)
        assert "mixed-language" in message
        assert "Please correct the source language" not in message

    async def test_threshold_is_configurable(self, pipeline_mocks):
        """One setting moves the decision, and it moves both layers.

        `min_dominant_language_share()` reads `settings` on every call
        precisely so raising the bar here cannot leave the detection core
        judging the same document by a different number.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "LANGUAGE_DETECTION_MIN_DOMINANT_SHARE", 0.85),
                _detector(orchestrator, ("en", Counter({"en": 800, "fr": 200}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-strict",
                    _job_data("job-strict", source_language="en"),
                )

        message = _error_message(bigquery)
        assert "mixed-language" in message
        assert "85%" in message


class TestUnverifiableLanguage:
    """Fail-closed: a document whose language cannot be read is rejected."""

    async def test_detection_failure_fails_job(self, pipeline_mocks):
        """Detection no longer falls back to trusting the declared value."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(
                    orchestrator,
                    None,
                    raises=ValueError("no extractable text layer"),
                ),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-undetectable",
                    _job_data("job-undetectable", source_language="en"),
                )

        assert "no extractable text layer" in _error_message(bigquery)

    async def test_empty_distribution_fails_job(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter())),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-empty-dist",
                    _job_data("job-empty-dist", source_language="en"),
                )

        assert "long or distinctive enough to identify its language" in _error_message(
            bigquery
        )


# ---------------------------------------------------------------------------
# Guard 2 -- domain
# ---------------------------------------------------------------------------


class TestDomainMismatchGuard:
    """A confidently contradicted domain fails the job; a hedged one does not."""

    async def test_confident_domain_mismatch_fails_job(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("hr", 0.95)),
            ):
                await _run(
                    orchestrator,
                    "job-domain-mismatch",
                    _job_data("job-domain-mismatch", domain="legal"),
                )

        message = _error_message(bigquery)
        assert "legal" in message
        assert "hr" in message
        assert "payroll" in message

    async def test_low_confidence_mismatch_is_allowed_through(self, pipeline_mocks):
        """Domains overlap; only a confident contradiction is worth failing."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("hr", 0.55)),
            ):
                await _run(
                    orchestrator,
                    "job-domain-hedged",
                    _job_data("job-domain-hedged", domain="legal"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_confidence_exactly_at_floor_fails(self, pipeline_mocks):
        """The floor is inclusive: 0.70 is confident enough to fail."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "DOMAIN_CLASSIFIER_MIN_CONFIDENCE", 0.70),
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("hr", 0.70)),
            ):
                await _run(
                    orchestrator,
                    "job-domain-floor",
                    _job_data("job-domain-floor", domain="legal"),
                )

        assert "hr" in _error_message(bigquery)

    async def test_matching_domain_is_not_flagged(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-domain-match",
                    _job_data("job-domain-match", domain="legal"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_unavailable_classifier_fails_job(self, pipeline_mocks):
        """Fail-closed: an unverifiable domain is not translated unchecked."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(raises=DomainCheckUnavailableError("vertex timeout")),
            ):
                await _run(
                    orchestrator,
                    "job-domain-unavailable",
                    _job_data("job-domain-unavailable", domain="legal"),
                )

        message = _error_message(bigquery)
        assert "could not be verified" in message
        # The wording must be distinguishable from a real mismatch so support
        # can tell an outage apart from a wrong declaration.
        assert "appears to be" not in message

    async def test_guard_can_be_disabled(self, pipeline_mocks):
        """The kill switch keeps jobs flowing during a classifier outage."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "DOMAIN_MISMATCH_CHECK_ENABLED", False),
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(raises=DomainCheckUnavailableError("vertex down")) as stub,
            ):
                await _run(
                    orchestrator,
                    "job-domain-disabled",
                    _job_data("job-domain-disabled", domain="legal"),
                )

        stub.assert_not_awaited()
        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_unsupported_declared_domain_fails_job(self, pipeline_mocks):
        """A direct-to-worker row with a domain the API would have rejected."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("legal", 0.99)) as stub,
            ):
                await _run(
                    orchestrator,
                    "job-domain-bogus",
                    _job_data("job-domain-bogus", domain="general"),
                )

        stub.assert_not_awaited()
        assert "Unsupported document domain" in _error_message(bigquery)


class TestDomainCheckCostAccounting:
    """The classifier's spend must land on the job, unlike the judge's."""

    async def test_classification_cost_is_added_to_job_cost(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("legal", 0.99, cost_usd=0.02)),
            ):
                await _run(
                    orchestrator,
                    "job-domain-cost",
                    _job_data("job-domain-cost", domain="legal"),
                )

        # 0.05 from the stubbed chunk costs + 0.02 for the classification.
        bigquery.write_cost_attribution.assert_awaited_once()
        recorded = bigquery.write_cost_attribution.await_args[0][0]
        assert recorded["cost_usd"] == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# Auto-detection -- no domain declared
# ---------------------------------------------------------------------------


def _persisted_config(bigquery) -> dict:
    """The translation_config the orchestrator persisted before translating."""
    calls = [
        call[0][1]["translation_config"]
        for call in bigquery.patch_translation_job.call_args_list
        if "translation_config" in call[0][1]
    ]
    assert calls, "routing metadata was never persisted"
    return calls[-1]


class TestDomainAutoDetection:
    """With no declared domain, the classifier's verdict *is* the domain.

    The guard semantics invert here. There is no claim to contradict, so
    nothing in this path may fail the job -- it either adopts what was
    detected or falls back.
    """

    async def test_detected_domain_is_adopted_and_routed_on(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("hr", 0.93)) as stub,
            ):
                await _run(
                    orchestrator,
                    "job-domain-auto",
                    _job_data("job-domain-auto", domain=""),
                )

        assert _status_calls(bigquery, "failed") == []
        # Everything downstream must route on the detected domain rather than
        # on the empty declaration: prompt profile, glossary and model chain
        # all key on this one value.
        stub.assert_awaited_once()
        assert _persisted_config(bigquery)["domain"] == "hr"
        glossary = orchestrator.glossary_service.load_domain_glossary
        assert glossary.call_args.kwargs["domain"] == "hr"
        chain = orchestrator.intent_router.get_model_chain
        assert chain.call_args.kwargs["domain"] == "hr"

    async def test_missing_domain_key_is_treated_as_undeclared(self, pipeline_mocks):
        """A direct-to-worker row may omit the field entirely."""
        bigquery, storage, tmp_path = pipeline_mocks
        job_data = _job_data("job-domain-absent")
        del job_data["translation_config"]["domain"]
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("finance", 0.91)),
            ):
                await _run(orchestrator, "job-domain-absent", job_data)

        assert _status_calls(bigquery, "failed") == []
        assert _persisted_config(bigquery)["domain"] == "finance"

    async def test_low_confidence_detection_is_still_adopted(self, pipeline_mocks):
        """The confidence floor decides whether to override a human.

        With nobody to override, a hedged verdict is still the best evidence
        available and beats the fallback.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(_verdict("operations", 0.41)),
            ):
                await _run(
                    orchestrator,
                    "job-domain-auto-hedged",
                    _job_data("job-domain-auto-hedged", domain=""),
                )

        assert _status_calls(bigquery, "failed") == []
        assert _persisted_config(bigquery)["domain"] == "operations"

    async def test_unavailable_detection_falls_back_instead_of_failing(
        self, pipeline_mocks
    ):
        """The inverted fail-closed rule.

        A declared job fails here, because its claim could not be verified.
        An undeclared job must not: the submitter made no claim, and has no
        corrective action to take on a Vertex or DLP outage.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 500}))),
                _classifier(raises=DomainCheckUnavailableError("vertex down")),
            ):
                await _run(
                    orchestrator,
                    "job-domain-auto-outage",
                    _job_data("job-domain-auto-outage", domain=""),
                )

        assert _status_calls(bigquery, "failed") == []
        assert _persisted_config(bigquery)["domain"] == _UNDETECTED_DOMAIN
