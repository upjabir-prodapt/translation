"""Phase C.5 tests: language-distribution persistence, unsupported-language
warning, and the source-equals-target guard in the pipeline orchestrator's
`_execute_pipeline`.

The declared-vs-detected language and domain guards themselves live in
`test_pipeline_input_consistency.py`; the domain guard is switched off here so
these tests exercise only the C.5 behaviour they were written for.
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
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator
from src.worker.services.temp_workspace_service import TempWorkspaceService


def _job_data(
    job_id: str, *, source_language: str = "de", target_language: str = "es"
) -> dict:
    return {
        "source_document": {
            "gcs_uri": f"gs://bucket/input/{job_id}/file.pdf",
            "original_filename": "input.pdf",
        },
        "translation_config": {
            "source_language": source_language,
            "target_language": target_language,
            "domain": "legal",
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
        patch.object(settings, "DOMAIN_MISMATCH_CHECK_ENABLED", False),
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


class TestLanguageDistributionPersistence:
    """C.5.1: the full per-language Counter reaches the BigQuery payload."""

    async def test_detected_languages_persisted_to_bigquery(self, pipeline_mocks):
        """The whole Counter is persisted, not just the winning language.

        A mixed de/en document needs no threshold tweak any more: both
        languages are supported, so coverage is 1.0 and the job runs to the
        point where it persists the distribution.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)

        with _patched_orchestrator(
            bigquery, storage, tmp_path, attempt_result
        ) as orchestrator:
            with (
                patch.object(
                    orchestrator.language_detector,
                    "detect_with_distribution",
                    return_value=("de", Counter({"de": 700, "en": 300})),
                ),
            ):
                pipeline_span = MagicMock()
                await orchestrator._execute_pipeline(
                    "job-mixed-lang", _job_data("job-mixed-lang"), pipeline_span
                )

        routing_calls = [
            c
            for c in bigquery.patch_translation_job.call_args_list
            if "translation_config" in c[0][1]
            and "detected_languages" in c[0][1]["translation_config"]
        ]
        assert len(routing_calls) >= 1
        persisted = routing_calls[0][0][1]["translation_config"]["detected_languages"]
        assert persisted == {"de": 700, "en": 300}


class TestGlossaryScopedToDetectedLanguages:
    """The domain glossary load is scoped to the document's own detected,
    noise-filtered languages -- see GlossaryService.load_domain_glossary's
    `source_languages` docstring for why this matters: glossary matching
    is a plain literal-string lookup with no language awareness, so an
    unscoped load could apply a term extracted from one document's foreign
    passage to an unrelated document that merely happens to contain the
    same literal string.
    """

    async def test_detected_languages_are_passed_as_the_glossary_filter(
        self, pipeline_mocks
    ):
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with patch.object(
                orchestrator.language_detector,
                "detect_with_distribution",
                return_value=("de", Counter({"de": 700, "en": 300})),
            ):
                await orchestrator._execute_pipeline(
                    "job-glossary-scope", _job_data("job-glossary-scope"), MagicMock()
                )

        orchestrator.glossary_service.load_domain_glossary.assert_called_once()
        call_kwargs = (
            orchestrator.glossary_service.load_domain_glossary.call_args.kwargs
        )
        assert call_kwargs["source_languages"] == ["de", "en"]

    async def test_empty_distribution_falls_back_to_no_filter(self, pipeline_mocks):
        """An empty distribution (only reachable with the language guard
        disabled) passes None, not an empty list -- an empty list would
        exclude every bucket and zero out glossary consistency for a
        document we simply couldn't read, which is not a more correct
        outcome than the historical unfiltered behaviour."""
        bigquery, storage, tmp_path = pipeline_mocks
        with _patched_orchestrator(
            bigquery, storage, tmp_path, _attempt_result(tmp_path)
        ) as orchestrator:
            with (
                patch.object(settings, "LANGUAGE_MISMATCH_CHECK_ENABLED", False),
                patch.object(
                    orchestrator.language_detector,
                    "detect_with_distribution",
                    return_value=("de", Counter()),
                ),
            ):
                await orchestrator._execute_pipeline(
                    "job-glossary-empty", _job_data("job-glossary-empty"), MagicMock()
                )

        call_kwargs = (
            orchestrator.glossary_service.load_domain_glossary.call_args.kwargs
        )
        assert call_kwargs["source_languages"] is None


class TestUnsupportedLanguageWarning:
    """C.5.3: WARN when the dominant detected language is outside the
    configured language_mapper.json set."""

    async def test_unsupported_dominant_language_logs_warning(
        self, pipeline_mocks, caplog
    ):
        """The routing-fallback warning fires, and the job separately fails.

        This WARN (`select_model_list` falling back to the default model)
        is independent of `_assert_language_supported`'s own checks. The
        job fails here because a wholly-`nl` document has 0% coverage in a
        translatable language (Guard 1's coverage check, which runs before
        its declared-vs-dominant comparison) -- not because the declared
        value ("de") mismatches the dominant detected one ("nl"), even
        though that would also fail it; coverage is checked first
        specifically so an untranslatable document is told that, rather
        than told it doesn't match its declared source.
        """
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)

        with _patched_orchestrator(
            bigquery, storage, tmp_path, attempt_result
        ) as orchestrator:
            with patch.object(
                orchestrator.language_detector,
                "detect_with_distribution",
                return_value=("nl", Counter({"nl": 500})),
            ):
                pipeline_span = MagicMock()
                with caplog.at_level("WARNING"):
                    await orchestrator._execute_pipeline(
                        "job-unsupported-lang",
                        _job_data("job-unsupported-lang"),
                        pipeline_span,
                    )

        assert any(
            "outside the configured language set" in record.message
            for record in caplog.records
        )
        failed_calls = [
            c
            for c in bigquery.patch_translation_job.call_args_list
            if c[0][1].get("status") == "failed"
        ]
        assert len(failed_calls) == 1


class TestSourceEqualsTargetGuard:
    """C.5.4: reject when the source language equals the requested target.

    The API blocks this at the schema layer, so the worker-side check is
    defensive: it covers direct-to-worker submissions and job rows written
    before the validator existed.
    """

    async def test_source_equals_target_fails_job(self, pipeline_mocks):
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)

        with _patched_orchestrator(
            bigquery, storage, tmp_path, attempt_result
        ) as orchestrator:
            with patch.object(
                orchestrator.language_detector,
                "detect_with_distribution",
                return_value=("es", Counter({"es": 500})),
            ):
                pipeline_span = MagicMock()
                await orchestrator._execute_pipeline(
                    "job-same-lang",
                    _job_data(
                        "job-same-lang", source_language="es", target_language="es"
                    ),
                    pipeline_span,
                )

        failed_calls = [
            c
            for c in bigquery.patch_translation_job.call_args_list
            if c[0][1].get("status") == "failed"
        ]
        assert len(failed_calls) == 1
        assert "same as" in failed_calls[0][0][1]["error_message"]

    async def test_missing_source_language_fails_job(self, pipeline_mocks):
        """A legacy row still carrying the retired "auto" sentinel fails."""
        bigquery, storage, tmp_path = pipeline_mocks
        attempt_result = _attempt_result(tmp_path)

        with _patched_orchestrator(
            bigquery, storage, tmp_path, attempt_result
        ) as orchestrator:
            with patch.object(
                orchestrator.language_detector,
                "detect_with_distribution",
                return_value=("de", Counter({"de": 500})),
            ):
                pipeline_span = MagicMock()
                await orchestrator._execute_pipeline(
                    "job-auto-legacy",
                    _job_data("job-auto-legacy", source_language="auto"),
                    pipeline_span,
                )

        failed_calls = [
            c
            for c in bigquery.patch_translation_job.call_args_list
            if c[0][1].get("status") == "failed"
        ]
        assert len(failed_calls) == 1
        assert "No source language" in failed_calls[0][0][1]["error_message"]
