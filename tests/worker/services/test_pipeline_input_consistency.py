"""Tests for the two pre-translation input-consistency guards in
`PipelineOrchestrator._execute_pipeline`:

  * the declared source language vs. the language actually detected, plus the
    rejection of mixed-language documents;
  * the declared domain vs. the domain an LLM reads the document as.

Both guards are fail-closed: a document that cannot be verified is failed
rather than translated unchecked.

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
# Guard 1 -- source language
# ---------------------------------------------------------------------------


class TestSourceLanguageMismatchGuard:
    """The document must be written in the declared language."""

    async def test_wrong_language_fails_job(self, pipeline_mocks):
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
        """The kill switch lets a mismatched job through, but still logs it.

        Logging while disabled is what makes a shadow rollout possible: the
        real rejection rate can be measured before anyone's job is failed.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "LANGUAGE_MISMATCH_CHECK_ENABLED", False),
                _detector(orchestrator, ("de", Counter({"de": 900}))),
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
            "source language mismatch (guard disabled)" in record.message
            for record in caplog.records
        )

    async def test_disabled_guard_still_logs_mixed_language(
        self, pipeline_mocks, caplog
    ):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "LANGUAGE_MISMATCH_CHECK_ENABLED", False),
                _detector(orchestrator, ("en", Counter({"en": 900, "de": 100}))),
                _classifier(_verdict("legal", 0.99)),
                caplog.at_level("WARNING"),
            ):
                await _run(
                    orchestrator,
                    "job-mixed-disabled",
                    _job_data("job-mixed-disabled", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert any(
            "mixed-language document detected (guard disabled)" in record.message
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
    """Mixed-language translation is out of scope, so mixed documents fail."""

    async def test_mixed_document_fails_even_when_declared_dominates(
        self, pipeline_mocks
    ):
        """The declared language winning the vote is not enough.

        The document is 90% English and English was declared, so there is no
        mismatch -- but it still contains a second language and mixed-language
        translation is not supported.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                _detector(orchestrator, ("en", Counter({"en": 900, "de": 100}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-mixed",
                    _job_data("job-mixed", source_language="en"),
                )

        message = _error_message(bigquery)
        assert "more than one language" in message
        assert "English" in message
        assert "German" in message

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
                _detector(orchestrator, ("de", Counter({"de": 550, "en": 450}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-mixed-and-wrong",
                    _job_data("job-mixed-and-wrong", source_language="en"),
                )

        message = _error_message(bigquery)
        assert "more than one language" in message
        assert "Please correct the source language" not in message

    async def test_threshold_tolerates_configured_noise(self, pipeline_mocks):
        """Raising the threshold absorbs stray detector noise.

        Detection is per text block, so a monolingual document can still emit
        a few foreign-looking blocks. The setting exists so that can be
        tolerated without a code change.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "LANGUAGE_MIXED_MAX_SECONDARY_SHARE", 0.10),
                _detector(orchestrator, ("en", Counter({"en": 960, "fr": 40}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-noise",
                    _job_data("job-noise", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_noise_above_threshold_still_fails(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "LANGUAGE_MIXED_MAX_SECONDARY_SHARE", 0.10),
                _detector(orchestrator, ("en", Counter({"en": 800, "fr": 200}))),
                _classifier(_verdict("legal", 0.99)),
            ):
                await _run(
                    orchestrator,
                    "job-noise-over",
                    _job_data("job-noise-over", source_language="en"),
                )

        assert "more than one language" in _error_message(bigquery)


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

        assert "Unable to detect a source language" in _error_message(bigquery)


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
