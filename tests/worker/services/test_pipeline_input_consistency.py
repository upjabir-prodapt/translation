"""Tests for the two pre-translation input-consistency guards in
`PipelineOrchestrator._execute_pipeline`:

  * the declared source language vs. the language actually detected;
  * the declared domain vs. the domain an LLM reads the document as.

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
from src.worker.services.input_consistency_service import DomainClassification
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator
from src.worker.services.temp_workspace_service import TempWorkspaceService


def _job_data(
    job_id: str,
    *,
    source_language: str | None = "auto",
    target_language: str = "es",
    domain: str = "general",
) -> dict:
    config: dict = {"target_language": target_language, "domain": domain}
    if source_language is not None:
        config["source_language"] = source_language
    return {
        "source_document": {
            "gcs_uri": f"gs://bucket/input/{job_id}/file.pdf",
            "original_filename": "input.pdf",
        },
        "translation_config": config,
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


@contextmanager
def _patched_orchestrator(bigquery, storage, tmp_path: Path, attempt_result: dict):
    orchestrator = PipelineOrchestrator(
        bigquery=bigquery,
        storage=storage,
        temp_workspace_service=TempWorkspaceService(base_dir=tmp_path / "work"),
    )
    orchestrator.intent_router = MagicMock()
    orchestrator.intent_router.build_intent.return_value = "general_xx_es"
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


async def _run(orchestrator, job_id: str, job_data: dict) -> None:
    await orchestrator._execute_pipeline(job_id, job_data, MagicMock())


class TestSourceLanguageMismatchGuard:
    """A declared source language absent from the document fails the job."""

    async def test_declared_language_absent_from_document_fails_job(
        self, pipeline_mocks
    ):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with patch.object(
                orchestrator.language_detector,
                "detect_with_distribution",
                return_value=("de", Counter({"de": 900, "en": 20})),
            ):
                await _run(
                    orchestrator,
                    "job-lang-mismatch",
                    _job_data("job-lang-mismatch", source_language="en"),
                )

        failed = _status_calls(bigquery, "failed")
        assert len(failed) == 1
        message = failed[0][0][1]["error_message"]
        assert "English" in message
        assert "German" in message

    async def test_detection_runs_even_when_language_is_declared(self, pipeline_mocks):
        """The old gate skipped detection entirely for explicit languages."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with patch.object(
                orchestrator.language_detector,
                "detect_with_distribution",
                return_value=("en", Counter({"en": 500})),
            ) as detector:
                await _run(
                    orchestrator,
                    "job-detect-always",
                    _job_data("job-detect-always", source_language="en"),
                )

        assert detector.call_count == 1
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_bilingual_document_does_not_fail(self, pipeline_mocks):
        """The declared language losing a plurality vote is not a mismatch.

        A near-even split is a mixed-language document, not a wrong
        language: the declared language really is present, so failing here
        would reject documents the user described correctly.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with patch.object(
                orchestrator.language_detector,
                "detect_with_distribution",
                return_value=("de", Counter({"de": 550, "en": 450})),
            ):
                await _run(
                    orchestrator,
                    "job-bilingual",
                    _job_data("job-bilingual", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_undetectable_document_keeps_declared_language(self, pipeline_mocks):
        """Absence of evidence is not evidence of mismatch.

        Both detectors raise when no confidently detectable text is found
        (the PDF path raising the *scanned PDF* wording). A short but valid
        document that translates fine must not start failing now that
        detection runs on every job.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with patch.object(
                orchestrator.language_detector,
                "detect_with_distribution",
                side_effect=ValueError("no extractable text layer"),
            ):
                await _run(
                    orchestrator,
                    "job-undetectable",
                    _job_data("job-undetectable", source_language="en"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_undetectable_document_still_fails_on_auto(self, pipeline_mocks):
        """With nothing declared there is no language to fall back to."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with patch.object(
                orchestrator.language_detector,
                "detect_with_distribution",
                side_effect=ValueError("no extractable text layer"),
            ):
                await _run(
                    orchestrator,
                    "job-undetectable-auto",
                    _job_data("job-undetectable-auto", source_language="auto"),
                )

        failed = _status_calls(bigquery, "failed")
        assert len(failed) == 1
        assert "no extractable text layer" in failed[0][0][1]["error_message"]


class TestDomainMismatchGuard:
    """A confidently contradicted domain fails the job; a hedged one does not."""

    @staticmethod
    def _classifier(domain: str, confidence: float) -> AsyncMock:
        return AsyncMock(
            return_value=DomainClassification(
                domain=domain,
                confidence=confidence,
                reason="It sets out employee leave entitlement and payroll terms.",
            )
        )

    async def test_confident_domain_mismatch_fails_job(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(
                    orchestrator.language_detector,
                    "detect_with_distribution",
                    return_value=("en", Counter({"en": 500})),
                ),
                patch(
                    "src.worker.services.pipeline_orchestrator."
                    "classify_document_domain",
                    new=self._classifier("hr", 0.95),
                ),
            ):
                await _run(
                    orchestrator,
                    "job-domain-mismatch",
                    _job_data(
                        "job-domain-mismatch", source_language="en", domain="legal"
                    ),
                )

        failed = _status_calls(bigquery, "failed")
        assert len(failed) == 1
        message = failed[0][0][1]["error_message"]
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
                patch.object(
                    orchestrator.language_detector,
                    "detect_with_distribution",
                    return_value=("en", Counter({"en": 500})),
                ),
                patch(
                    "src.worker.services.pipeline_orchestrator."
                    "classify_document_domain",
                    new=self._classifier("hr", 0.55),
                ),
            ):
                await _run(
                    orchestrator,
                    "job-domain-hedged",
                    _job_data(
                        "job-domain-hedged", source_language="en", domain="legal"
                    ),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_unavailable_classifier_does_not_fail_job(self, pipeline_mocks):
        """No opinion is not a mismatch -- that is a measurement failure."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(
                    orchestrator.language_detector,
                    "detect_with_distribution",
                    return_value=("en", Counter({"en": 500})),
                ),
                patch(
                    "src.worker.services.pipeline_orchestrator."
                    "classify_document_domain",
                    new=AsyncMock(return_value=None),
                ),
            ):
                await _run(
                    orchestrator,
                    "job-domain-unavailable",
                    _job_data(
                        "job-domain-unavailable", source_language="en", domain="legal"
                    ),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1

    async def test_matching_domain_is_not_flagged(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(
                    orchestrator.language_detector,
                    "detect_with_distribution",
                    return_value=("en", Counter({"en": 500})),
                ),
                patch(
                    "src.worker.services.pipeline_orchestrator."
                    "classify_document_domain",
                    new=self._classifier("legal", 0.99),
                ),
            ):
                await _run(
                    orchestrator,
                    "job-domain-match",
                    _job_data("job-domain-match", source_language="en", domain="legal"),
                )

        assert _status_calls(bigquery, "failed") == []
        assert len(_status_calls(bigquery, "completed")) == 1
