"""Translation pipeline orchestration.

Always executed by the Cloud Tasks worker (TranslateTaskHandler -> here).
The API_USE_BACKGROUND_PIPELINE dev/local flag additionally allows the API
process to invoke this same orchestrator in-process (bypassing Cloud Tasks)
purely for local testing convenience -- it is not a separate "API-only"
runtime mode.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections import Counter
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode

from src.config.constants import settings
from src.config.tracing import set_root_span
from src.config.tracing import set_root_span_attributes
from src.config.tracing import tracer_pipeline
from src.config.translation_routing import get_language_display_name
from src.config.translation_routing import normalize_domain
from src.config.translation_routing import normalize_language
from src.repository.api_storage_repository import APIStorageRepository
from src.repository.bigquery_repository import BigQueryRepository
from src.repository.repository_exception import BigQueryError
from src.repository.repository_exception import StorageError
from src.worker.doctranslator.doctranslator_exception.DocTranslatorException import (
    ScannedPDFError,
)
from src.worker.doctranslator.format.txt.txt_docx_bridge import docx_path_to_txt_bytes
from src.worker.doctranslator.format.txt.txt_docx_bridge import txt_bytes_to_docx_bytes
from src.worker.services.assembly_service import AssemblyService
from src.worker.services.docx_job_processor import DocxJobProcessor
from src.worker.services.glossary_service import GlossaryService
from src.worker.services.input_consistency_service import DomainCheckUnavailableError
from src.worker.services.input_consistency_service import classify_document_domain
from src.worker.services.intent_router_service import IntentRouterService
from src.worker.services.language_detection_core import get_supported_languages
from src.worker.services.language_detection_service import LanguageDetectionService
from src.worker.services.llm_cost_service import get_vertex_llm_cost_service
from src.worker.services.processor_service import JobProcessor
from src.worker.services.shared_document_prep import PreparedDocument
from src.worker.services.shared_document_prep import get_shared_document_prep_cache
from src.worker.services.temp_workspace_service import TempWorkspaceService
from src.worker.services.translation_job_session import TranslationJobSessionManager
from src.worker.utils.cost_utils import validate_job_cost

logger = logging.getLogger(__name__)

_OMIT = object()

# Process-wide bound on concurrent in-process pipeline executions. This was
# previously defined in Settings (MAX_CONCURRENT_JOBS) but never actually
# enforced anywhere, so in dev mode (API_USE_BACKGROUND_PIPELINE=true) an
# unbounded number of concurrent asyncio tasks could each spin up their own
# thread pools / ONNX inference calls, risking memory blowup under load.
_pipeline_semaphore = asyncio.Semaphore(max(1, int(settings.MAX_CONCURRENT_JOBS)))

# User-facing wording for document-shape worker exceptions, matching
# PDFValidator._assert_has_text_layer's API-side rejection message
# (implementation_plan.md Phase B.3.2) so a job that slips past the fast
# API-side check (or is submitted via the Cloud Tasks worker endpoint
# directly, bypassing PDFValidator) still fails with the same clear
# wording instead of leaking "Translation failed: Scanned PDF detected."
_NO_TEXT_LAYER_MESSAGE = (
    "This PDF has no extractable text layer (scanned or image-only). "
    "OCR is not supported — please supply a text-based PDF."
)


def _user_facing_error_message(exc: Exception) -> str:
    """Map internal worker exceptions to the wording shown to end users."""
    if isinstance(exc, ScannedPDFError):
        return _NO_TEXT_LAYER_MESSAGE
    return str(exc)


def _declared_language(value: str | None) -> str | None:
    """Return the source language the user actually declared, else None.

    The API requires a concrete source language, so in practice this always
    returns one. `None` means the job record predates that requirement (or
    was submitted straight to the worker): either the field is missing, or it
    holds the retired "auto" sentinel that used to mean auto-detect.
    """
    cleaned = str(value or "").strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return cleaned


def _language_for_user(code: str) -> str:
    """Human-readable language name for an end-user error message.

    `get_language_display_name` title-cases codes it has no name for
    ("nl" -> "Nl"), which reads like a typo mid-sentence, so codes it cannot
    name are described instead of displayed.
    """
    display = get_language_display_name(code)
    if display.strip().lower() == str(code).strip().lower():
        return f"a different language (detected as '{code}')"
    return display


def _extract_model_version(model_id: str) -> str | None:
    """Extract a human-readable version string from a model ID.

    Examples:
        "gemini-2.5-flash" -> "2.5 flash"
        "claude-opus-4-7"  -> "opus 4-7"
    """
    if not model_id:
        return None
    for prefix in ("gemini-", "claude-"):
        if model_id.lower().startswith(prefix):
            return model_id[len(prefix) :].replace("-", " ", 1)
    return model_id


def _attempt_index_to_variant(attempt_index: int) -> str:
    """Convert a 1-based attempt index to an A/B/C variant label."""
    idx = max(1, attempt_index) - 1
    return chr(ord("A") + idx) if idx < 26 else str(attempt_index)


class _PipelineProgressTracker:
    """Logs translation progress to the process logger only (not BigQuery)."""

    def __init__(self, *, job_id: str) -> None:
        self.job_id = job_id

    async def update(
        self,
        progress: float | None,
        current_stage: str | None = None,
        force: bool = False,
    ) -> bool:
        del force
        if progress is not None or current_stage:
            logger.info(
                f"job_id={self.job_id} progress={progress} stage={current_stage}"
            )
        import asyncio

        await asyncio.sleep(0)
        return True


class PipelineOrchestrator:
    """Execute the translation pipeline for one job (always worker-driven)."""

    def __init__(
        self,
        *,
        bigquery: BigQueryRepository,
        storage: APIStorageRepository,
        temp_workspace_service: TempWorkspaceService | None = None,
        session_manager: TranslationJobSessionManager | None = None,
    ):
        self.bigquery = bigquery
        self.storage = storage
        self.temp_workspace_service = temp_workspace_service or TempWorkspaceService()
        self.session_manager = session_manager or TranslationJobSessionManager()
        self.language_detector = LanguageDetectionService()
        self.intent_router = IntentRouterService()
        self.glossary_service = GlossaryService()
        self.assembly_service = AssemblyService(storage=storage)

    def _extract_blob_path(self, uri_or_blob: str) -> str:
        if uri_or_blob.startswith("gs://"):
            without_scheme = uri_or_blob[5:]
            if without_scheme.startswith(f"{settings.GCS_BUCKET_NAME}/"):
                return without_scheme[len(settings.GCS_BUCKET_NAME) + 1 :]
            return without_scheme
        return uri_or_blob

    async def _update_status(
        self,
        job_id: str,
        *,
        status: str,
        result: Any = _OMIT,
        completed_at: Any = _OMIT,
        error_message: Any = _OMIT,
    ) -> None:
        fields: dict[str, Any] = {"status": status}
        if result is not _OMIT:
            fields["result"] = result
        if completed_at is not _OMIT:
            fields["completed_at"] = completed_at
        if error_message is not _OMIT:
            fields["error_message"] = error_message
        await self.bigquery.patch_translation_job(job_id, fields)

    async def _mark_job_failed(
        self,
        job_id: str,
        exc: Exception,
        pipeline_span,
        *,
        stage: str,
    ) -> None:
        """Record failure in session, OTel, logs, and BigQuery.

        The BigQuery `error_message` field (surfaced verbatim to end
        users via JobService/_job_error_message) uses
        `_user_facing_error_message()` so document-shape failures like
        `ScannedPDFError` read as the same clear wording as the API-side
        rejection instead of the internal "Translation failed: Scanned
        PDF detected." string (implementation_plan.md Phase B.3.2). The
        session/span/log trail keeps the raw `str(exc)` for debugging.
        """
        await self.session_manager.mark_failure(
            job_id,
            stage=stage,
            error_message=str(exc),
        )
        pipeline_span.set_status(Status(StatusCode.ERROR, str(exc)))
        pipeline_span.record_exception(exc)

        if isinstance(exc, BigQueryError):
            logger.critical(
                "BigQuery persistence failed for job %s at stage %s: %s",
                job_id,
                stage,
                exc,
            )
        elif isinstance(exc, StorageError):
            logger.critical(
                "GCS write failed for job %s after all retry attempts. Error: %s",
                job_id,
                exc,
            )
        else:
            logger.error(
                "Pipeline failed for job %s at stage %s. Exception: %s",
                job_id,
                stage,
                exc,
            )

        try:
            await self._update_status(
                job_id,
                status="failed",
                completed_at=datetime.now(UTC),
                error_message=_user_facing_error_message(exc),
            )
            await self.session_manager.mark_persistence_step(
                job_id, "failed_status_patch", succeeded=True
            )
        except BigQueryError as patch_exc:
            logger.critical(
                "Failed to mark job %s as failed in BigQuery after stage %s error: %s",
                job_id,
                stage,
                patch_exc,
            )
            raise patch_exc from exc

    async def run(self, job_id: str, job_data: dict[str, Any], parent_ctx=None) -> None:
        token = otel_context.attach(parent_ctx) if parent_ctx is not None else None
        try:
            # Bound the number of pipelines executing concurrently in this
            # process (MAX_CONCURRENT_JOBS). Jobs beyond the limit simply
            # wait here rather than each spinning up their own thread
            # pools / ONNX inference sessions unbounded.
            async with _pipeline_semaphore:
                await self._run_pipeline(job_id, job_data)
        finally:
            if token is not None:
                otel_context.detach(token)

    async def _run_pipeline(self, job_id: str, job_data: dict[str, Any]) -> None:
        translation_config = job_data["translation_config"]

        with tracer_pipeline.start_as_current_span(
            "pipeline.run",
            kind=SpanKind.INTERNAL,
            attributes={
                "translation.job_id": job_id,
                "translation.source_lang": str(
                    translation_config.get("source_language", "auto")
                ),
                "translation.target_lang": str(
                    translation_config.get("target_language", "")
                ),
                "translation.domain": str(translation_config.get("domain", "")),
            },
        ) as pipeline_span:
            set_root_span(pipeline_span)
            await self._execute_pipeline(job_id, job_data, pipeline_span)

    async def _compute_accumulated_chunk_costs(
        self,
        *,
        job_id: str,
        local_input_path: Path,
        token_usage: dict[str, Any],
        model_id: str,
        split_page_ranges: list | None = None,
    ) -> dict[str, float | int]:
        """Compute proportional per-chunk costs and return job totals.

        `split_page_ranges` are the parts the translation actually ran, as
        recorded by `_dispatch_translation`. Re-deriving chunks here with a
        default-constructed `StructureAwareSplitStrategy` was a second,
        independent split decision that could -- and did -- disagree with the
        first: one run attributed cost across 52 re-derived chunks for a
        document that was translated as 40 parts. Re-deriving is kept only as
        a fallback for jobs that never went down the split path.
        """
        from src.worker.doctranslator.format.pdf.split_manager import SplitPoint
        from src.worker.doctranslator.format.pdf.split_manager import (
            StructureAwareSplitStrategy,
        )
        from src.worker.utils.cost_utils import aggregate_chunk_cost_records

        if split_page_ranges:
            chunks = [
                SplitPoint(
                    start_page=int(start),
                    end_page=int(end),
                    chunk_index=index,
                    token_count=int(tokens),
                )
                for index, (start, end, tokens) in enumerate(split_page_ranges)
            ]
        else:

            class _Cfg:
                input_file = local_input_path

            chunks = StructureAwareSplitStrategy().determine_split_points(_Cfg())
        cost_service = get_vertex_llm_cost_service()
        records = cost_service.compute_per_chunk_costs(
            chunks=chunks,
            model_id=model_id,
            total_input_tokens=int(token_usage.get("prompt_tokens", 0)),
            total_output_tokens=int(token_usage.get("completion_tokens", 0)),
            cache_hit_tokens=int(token_usage.get("cache_hit_prompt_tokens", 0)),
        )
        if not records:
            raise RuntimeError(
                f"No chunk cost records produced for job {job_id} — cannot attribute cost"
            )
        totals = aggregate_chunk_cost_records(records)
        totals["chunk_count"] = len(records)
        return totals

    def _convert_txt_to_docx(self, txt_path: Path) -> Path:
        """Wrap a downloaded `.txt` file's content in a `.docx` and return its path.

        Plain-text documents are translated by reusing the native DOCX
        pipeline (see `src/worker/doctranslator/format/txt/txt_docx_bridge.py`)
        rather than a bespoke format-specific translator.
        """
        docx_path = txt_path.with_suffix(".docx")
        docx_bytes = txt_bytes_to_docx_bytes(txt_path.read_bytes())
        docx_path.write_bytes(docx_bytes)
        return docx_path

    async def _prepare_input_solo(
        self,
        *,
        job_id: str,
        blob_path: str,
        source_doc: dict[str, Any],
        workspace_input_dir: Path,
        local_input_path: Path,
    ) -> tuple[Path, str, Counter[str]]:
        """Download/detect for a job with no sibling batch jobs.

        DOCX files are translated natively (no PDF conversion) -- see
        docs/architecture/pdf-vs-docx-translation-architecture.md. Source
        language detection for DOCX uses a lightweight text-extraction
        variant since LanguageDetectionService's default path is pymupdf/PDF
        specific.

        Returns `(local_path, detected_source_lang, language_distribution)`.
        The language returned is the one *detected in the document*, never
        the one the user declared -- `_execute_pipeline` routes on the
        declared language and uses this only to check it.
        """
        del job_id
        await self.storage.download_file(blob_path, local_input_path)

        if source_doc.get("format") == "txt":
            local_input_path = self._convert_txt_to_docx(local_input_path)

        detected_lang, language_distribution = self._detect_source_language(
            local_input_path,
            is_docx=source_doc.get("format") in ("docx", "txt"),
        )
        return local_input_path, detected_lang, language_distribution

    def _detect_source_language(
        self, input_path: Path, *, is_docx: bool
    ) -> tuple[str, Counter[str]]:
        """Detect the source language -- always, for every job.

        Detection used to run only when the source language was omitted or
        "auto". The API now requires a concrete source language, so under the
        old gate detection would never have run at all and a document in the
        wrong language would be translated blindly. It now runs
        unconditionally so `_execute_pipeline` can compare what the user
        declared against what the document actually contains.

        Failures deliberately propagate. A document whose language cannot be
        determined cannot be checked, and an unverifiable document fails
        rather than being translated unchecked. Both detectors raise when no
        confidently detectable text is found, and the PDF path raises the
        scanned-PDF wording specifically, so the message the user sees
        already explains the real problem.
        """
        return self.language_detector.detect_with_distribution(
            input_path, is_docx=is_docx
        )

    async def _prepare_input_shared(
        self,
        *,
        job_id: str,
        source_hash: str,
        source_doc: dict[str, Any],
        workspace_input_dir: Path,
        local_input_path: Path,
    ) -> tuple[Path, str, Counter[str]]:
        """Download/convert/detect once per `source_hash`, shared across sibling jobs.

        Sibling jobs from the same multi-target-language batch (same
        source_hash, different target language) that happen to run
        concurrently in this process reuse one another's download/convert/
        detect work instead of repeating it. The shared cache stores its
        prepared file under a shared scratch dir; each job copies it into
        its own workspace so subsequent per-job mutation (DLP masking,
        typesetting, etc) never touches the shared file.

        Returns `(local_path, detected_source_lang, language_distribution)` --
        see `_prepare_input_solo`. Detection is part of the shared work, and
        no longer depends on what any one sibling declared, so the whole
        prepared result is genuinely language-independent and every sibling
        of a batch is checked against one identical distribution.
        """
        cache = get_shared_document_prep_cache()
        blob_path = self._extract_blob_path(source_doc["gcs_uri"])
        shared_scratch_root = (
            settings.temp_root_path / settings.TEMP_JOBS_ROOT / "_shared_prep"
        )

        async def _do_prepare() -> PreparedDocument:
            scratch_dir = shared_scratch_root / source_hash
            scratch_dir.mkdir(parents=True, exist_ok=True)
            filename = source_doc.get("original_filename", "input.pdf")
            shared_path = scratch_dir / filename
            await self.storage.download_file(blob_path, shared_path)

            if source_doc.get("format") == "txt":
                shared_path = self._convert_txt_to_docx(shared_path)

            detected_lang, distribution = self._detect_source_language(
                shared_path,
                is_docx=source_doc.get("format") in ("docx", "txt"),
            )

            return PreparedDocument(
                local_path=shared_path,
                detected_source_language=detected_lang,
                detected_language_distribution=distribution,
            )

        try:
            prepared = await cache.get_or_prepare(
                source_hash=source_hash,
                job_id=job_id,
                job_local_dir=workspace_input_dir,
                prepare_fn=_do_prepare,
            )
        except Exception:
            # If the shared prep failed (e.g. the first job's download
            # failed), fall back to doing it solo for this job rather than
            # letting every sibling job fail from one shared error.
            logger.warning(
                "[shared_prep] shared prep failed for source_hash=%s; "
                "falling back to solo prep for job=%s",
                source_hash,
                job_id,
                exc_info=True,
            )
            return await self._prepare_input_solo(
                job_id=job_id,
                blob_path=blob_path,
                source_doc=source_doc,
                workspace_input_dir=workspace_input_dir,
                local_input_path=local_input_path,
            )

        # Copy (not move) the shared file into this job's own workspace so
        # later per-job mutation never affects sibling jobs still reading it.
        workspace_input_dir.mkdir(parents=True, exist_ok=True)
        job_local_path = workspace_input_dir / prepared.local_path.name
        await asyncio.to_thread(shutil.copy2, prepared.local_path, job_local_path)

        return (
            job_local_path,
            prepared.detected_source_language,
            prepared.detected_language_distribution,
        )

    def _assert_language_matches(
        self,
        *,
        job_id: str,
        declared_source_lang: str,
        language_distribution: Counter[str],
    ) -> None:
        """Fail the job unless the document is monolingual in the declared language.

        Two separate rejections, checked in this order:

        1. **Mixed language.** Mixed-language translation is out of scope, so
           a document containing more than one language is rejected outright
           regardless of what was declared.
        2. **Wrong language.** The single language present must be the one
           the user declared.

        The mixed check runs first deliberately. Reporting the mismatch first
        on a document that is *both* mixed and mis-declared would tell the
        user to resubmit as German, only for that resubmission to fail again
        as mixed -- a two-step dead end.

        `LANGUAGE_MIXED_MAX_SECONDARY_SHARE` defaults to 0.0, i.e. any second
        detected language fails. It exists because detection is per text
        block and stray blocks do occur in real documents; raise it to
        tolerate that noise without a code change.

        The distribution is logged even when the guard is switched off, which
        is what makes a shadow rollout possible: run with
        `LANGUAGE_MISMATCH_CHECK_ENABLED=false`, read how often real documents
        would have been rejected, then tune the threshold and switch it on.
        """
        enabled = bool(settings.LANGUAGE_MISMATCH_CHECK_ENABLED)

        total_chars = sum(language_distribution.values())
        if not language_distribution or total_chars <= 0:
            logger.warning(
                "Job %s: language detection produced no evidence; the declared "
                "language '%s' cannot be verified.",
                job_id,
                declared_source_lang,
            )
            if not enabled:
                return
            # An unverifiable document is failed rather than translated
            # unchecked.
            raise ValueError(
                "Unable to detect a source language: no sufficiently long, "
                "confidently detectable text was found. This document cannot "
                "be checked against the source language you selected."
            )

        shares = {
            code: count / total_chars for code, count in language_distribution.items()
        }
        dominant_code, dominant_chars = language_distribution.most_common(1)[0]
        dominant_share = dominant_chars / total_chars

        try:
            declared_code = normalize_language(declared_source_lang)
        except ValueError:
            # Not a language_mapper.json alias. The API validator rejects
            # these, so this is a direct-to-worker submission whose declared
            # language cannot be compared against anything.
            logger.warning(
                "Job %s: declared source language '%s' is not a known language;"
                " detected distribution was %s.",
                job_id,
                declared_source_lang,
                dict(language_distribution),
            )
            if not enabled:
                return
            raise ValueError(
                f"Unsupported source language '{declared_source_lang}'."
            ) from None

        logger.info(
            "Job %s: detected language distribution %s (dominant '%s' at "
            "%.1f%%), declared '%s'",
            job_id,
            {code: round(share, 4) for code, share in shares.items()},
            dominant_code,
            dominant_share * 100,
            declared_code,
        )

        max_secondary = float(settings.LANGUAGE_MIXED_MAX_SECONDARY_SHARE)
        secondary = {
            code: share
            for code, share in shares.items()
            if code != dominant_code and share > max_secondary
        }
        if secondary:
            detected_summary = ", ".join(
                f"{_language_for_user(code)} {share * 100:.0f}%"
                for code, share in sorted(
                    shares.items(), key=lambda item: item[1], reverse=True
                )
            )
            logger.warning(
                "Job %s: mixed-language document%s -- distribution %s",
                job_id,
                " rejected" if enabled else " detected (guard disabled)",
                dict(language_distribution),
            )
            if not enabled:
                return
            raise ValueError(
                "This document contains more than one language "
                f"({detected_summary}). Translating mixed-language documents "
                "is not supported. Please submit a document written in a "
                "single language."
            )

        if dominant_code != declared_code:
            logger.warning(
                "Job %s: source language mismatch%s -- declared '%s', document "
                "is '%s' (%.1f%% of detected text)",
                job_id,
                "" if enabled else " (guard disabled)",
                declared_code,
                dominant_code,
                dominant_share * 100,
            )
            if not enabled:
                return
            raise ValueError(
                f"You selected {_language_for_user(declared_code)} as the "
                "source language, but this document is written in "
                f"{_language_for_user(dominant_code)}. Please correct the "
                "source language and submit again."
            )

    async def _assert_domain_matches(
        self,
        *,
        job_id: str,
        domain: str,
        local_input_path: Path,
        is_docx: bool,
        source_lang: str,
        enable_dlp: bool,
        source_hash: str,
    ) -> float:
        """Fail the job when the declared domain contradicts the document.

        Returns the USD cost of the classification call so it can be added to
        the job's total. That is zero when the guard is disabled or when a
        sibling job of the same batch already paid for the call.

        Only a contradiction the classifier is confident about fails the job.
        Business domains genuinely overlap -- an HR policy is full of
        contractual language, a finance document is full of regulatory
        language -- so a disagreement below
        `DOMAIN_CLASSIFIER_MIN_CONFIDENCE` is logged and allowed through.
        That floor, not the on/off flag, is the real false-positive control.

        A classifier that cannot answer at all raises `DomainCheckUnavailableError`
        and fails the job: an unverifiable document is not translated
        unchecked. Note that this makes a Vertex or DLP outage fail jobs
        permanently, since a failed job is terminal and Cloud Tasks does not
        redeliver it -- `DOMAIN_MISMATCH_CHECK_ENABLED` is the kill switch.
        """
        if not settings.DOMAIN_MISMATCH_CHECK_ENABLED:
            return 0.0

        try:
            declared_domain = normalize_domain(domain)
        except ValueError:
            # The API restricts domain to SUPPORTED_DOMAINS, so this is a
            # direct-to-worker submission. There is nothing to classify
            # against, and the guard is fail-closed.
            raise ValueError(f"Unsupported document domain '{domain}'.") from None

        try:
            verdict = await classify_document_domain(
                job_id=job_id,
                path=local_input_path,
                is_docx=is_docx,
                source_language=source_lang,
                enable_dlp=enable_dlp,
                source_hash=source_hash,
            )
        except DomainCheckUnavailableError as exc:
            logger.warning(
                "Job %s: domain check could not run (%s); failing the job "
                "because the declared domain could not be verified.",
                job_id,
                exc,
            )
            # Deliberately distinct wording from a real mismatch so support
            # can tell an outage apart from a wrong declaration.
            raise ValueError(
                "The document domain could not be verified for this "
                "document, so the translation was not started. Please try "
                "again."
            ) from exc

        classification = verdict.classification
        if classification.domain == declared_domain:
            return verdict.cost_usd

        min_confidence = float(settings.DOMAIN_CLASSIFIER_MIN_CONFIDENCE)
        if classification.confidence < min_confidence:
            logger.info(
                "Job %s: document classified as '%s' but declared '%s'; "
                "confidence %.2f is below the %.2f floor, allowing the job "
                "through. Reason: %s",
                job_id,
                classification.domain,
                declared_domain,
                classification.confidence,
                min_confidence,
                classification.reason,
            )
            return verdict.cost_usd

        logger.warning(
            "Job %s: domain mismatch -- declared '%s', classified '%s' "
            "(confidence %.2f). Reason: %s",
            job_id,
            declared_domain,
            classification.domain,
            classification.confidence,
            classification.reason,
        )
        # Model-authored text lands in BigQuery's error_message and is shown
        # verbatim to the user, so it is whitespace-collapsed and bounded.
        reason = " ".join(str(classification.reason).split())[:300]
        raise ValueError(
            f"You selected '{declared_domain}' as the document domain, but "
            f"this document appears to be a '{classification.domain}' "
            f"document. {reason} Please select the correct domain and submit "
            "again."
        )

    async def _execute_pipeline(
        self, job_id: str, job_data: dict[str, Any], pipeline_span
    ) -> None:
        workspace = self.temp_workspace_service.create(job_id)
        await self.session_manager.start(job_id, workspace)
        current_stage = "initialize"
        source_hash_for_release: str | None = None
        try:
            source_doc = job_data["source_document"]
            translation_config = job_data["translation_config"]
            cost_attribution = job_data["cost_attribution"]

            current_stage = "processing_status_patch"
            await self._update_status(job_id, status="processing")
            await self.session_manager.mark_persistence_step(
                job_id, "processing_status_patch", succeeded=True
            )

            input_filename = source_doc.get("original_filename", "input.pdf")
            local_input_path = workspace.input_dir / input_filename
            blob_path = self._extract_blob_path(source_doc["gcs_uri"])
            source_hash = str(job_data.get("source_hash") or "").strip()
            requested_source_lang = translation_config.get("source_language")

            current_stage = "download_input"
            if source_hash:
                # Multi-target-language batches submit N sibling jobs that
                # share one source document. Download + DOCX->PDF conversion
                # + source-language auto-detect are 100% language-independent,
                # so when several sibling jobs for the same source_hash run
                # concurrently in this process, only the first one actually
                # does this work -- the rest await and reuse its result
                # instead of re-downloading/re-converting/re-detecting.
                (
                    local_input_path,
                    detected_source_lang,
                    language_distribution,
                ) = await self._prepare_input_shared(
                    job_id=job_id,
                    source_hash=source_hash,
                    source_doc=source_doc,
                    workspace_input_dir=workspace.input_dir,
                    local_input_path=local_input_path,
                )
                source_hash_for_release = source_hash

            else:
                (
                    local_input_path,
                    detected_source_lang,
                    language_distribution,
                ) = await self._prepare_input_solo(
                    job_id=job_id,
                    blob_path=blob_path,
                    source_doc=source_doc,
                    workspace_input_dir=workspace.input_dir,
                    local_input_path=local_input_path,
                )

            await self.session_manager.set_input_path(job_id, local_input_path)

            target_lang = translation_config["target_language"]
            domain = translation_config["domain"]

            # Everything downstream routes on the language the user
            # *declared*; the detected one exists only to check that claim.
            source_lang = _declared_language(requested_source_lang)

            # Defensive: the API requires a concrete source language and
            # rejects source == target, so neither branch is reachable
            # through it. They stay for direct-to-worker submissions and for
            # legacy BigQuery rows written before the field became mandatory
            # (those still carry the "auto" sentinel), which would otherwise
            # flow into routing and the job record as a missing language.
            if not source_lang:
                raise ValueError(
                    "No source language was specified for this job. Please "
                    "resubmit with the document's source language."
                )
            if source_lang == target_lang:
                raise ValueError(
                    f"Source language '{source_lang}' is the same as the "
                    f"requested target language '{target_lang}'. No "
                    "translation is needed."
                )

            # C.5.3: warn (do not fail) when the detected dominant
            # language falls outside the configured support set --
            # select_model_list() silently falls back to the default
            # Gemini model in this case, which is worth surfacing in logs.
            # Keyed on the detected language rather than the routing one:
            # the declared value is validated against language_mapper.json
            # at the API, so it can never be the thing that is unsupported.
            if (
                detected_source_lang
                and detected_source_lang not in get_supported_languages()
            ):
                logger.warning(
                    "Job %s: detected dominant source language '%s' is outside "
                    "the configured language set; model routing will fall back "
                    "to the default model.",
                    job_id,
                    detected_source_lang,
                )

            # Guard 1: the document must actually be written in the declared
            # language, and must be monolingual. Detection runs on every job
            # now, so this is the first point at which the two can be
            # compared.
            current_stage = "language_consistency_check"
            self._assert_language_matches(
                job_id=job_id,
                declared_source_lang=source_lang,
                language_distribution=language_distribution,
            )

            processing_options = job_data.get("processing_options") or {}
            enable_dlp = bool(
                processing_options.get(
                    "enable_dlp",
                    translation_config.get(
                        "enable_dlp",
                        getattr(settings, "GOOGLE_DLP_ENABLED", True),
                    ),
                )
            )

            # Guard 2: the declared domain must match what the document
            # actually is (an HR policy submitted as `legal`). Runs after
            # `enable_dlp` is resolved because the text sampled for
            # classification is sent to Vertex and must respect the same
            # masking the translation path applies. Cached per source_hash
            # inside the service, so a multi-target batch classifies once.
            current_stage = "domain_consistency_check"
            domain_check_cost_usd = await self._assert_domain_matches(
                job_id=job_id,
                domain=domain,
                local_input_path=local_input_path,
                is_docx=source_doc.get("format") in ("docx", "txt"),
                source_lang=source_lang,
                enable_dlp=enable_dlp,
                source_hash=source_hash,
            )

            enable_judge = bool(settings.QUALITY_JUDGE_ENABLED)
            intent = self.intent_router.build_intent(domain, source_lang, target_lang)
            model_chain = self.intent_router.get_model_chain(
                domain=domain,
                source_lang=source_lang,
                target_lang=target_lang,
            )

            if not model_chain:
                raise ValueError(f"No models configured for intent {intent}")

            await self.session_manager.set_routing(
                job_id,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain,
                intent=intent,
                model_chain=model_chain,
            )

            # Persist detected language and routing decision immediately so failed jobs remain analysable.
            # C.5.1: also persist the full per-language distribution (not
            # just the winner) so a mixed-language document's actual
            # composition is queryable later instead of being discarded.
            try:
                await self.bigquery.patch_translation_job(
                    job_id,
                    {
                        "translation_config": {
                            **translation_config,
                            "source_language": source_lang,
                            "target_language": target_lang,
                            "domain": domain,
                            "enable_dlp": enable_dlp,
                            "detected_languages": dict(language_distribution),
                        },
                        "result": {"intent": intent},
                    },
                )
            except Exception:
                logger.warning(
                    f"Failed to persist initial routing metadata to BigQuery for job {job_id}",
                    exc_info=True,
                )

            glossaries = self.glossary_service.load_domain_glossary(
                domain=domain,
                target_language_name=target_lang,
            )

            is_txt = source_doc.get("format") == "txt"
            is_docx = source_doc.get("format") == "docx" or is_txt

            current_stage = "translate"
            if is_docx:
                # Native DOCX translation: direct OOXML manipulation, no PDF
                # conversion, no LibreOffice subprocess -- see
                # docs/architecture/pdf-vs-docx-translation-architecture.md.
                docx_processor = DocxJobProcessor(
                    glossary_service=self.glossary_service
                )
                docx_config = {
                    "job_id": job_id,
                    "input_file": str(local_input_path),
                    "output_dir": str(workspace.attempts_dir),
                    "lang_in": source_lang,
                    "lang_out": target_lang,
                    "domain": domain,
                    "model_list": model_chain,
                    "max_model_attempts": max(1, settings.MAX_MODEL_ATTEMPTS),
                    "enable_dlp": enable_dlp,
                    "enable_judge": enable_judge,
                    "auto_extract_glossary": True,
                    # UAT EC-01/EC-08: the domain glossary used to be
                    # write-only on the DOCX path -- terms were merged into
                    # it after a job and never read back for the next one,
                    # so an approved rendering (or a do-not-translate brand
                    # name) had no effect on DOCX or TXT output.
                    "glossaries": glossaries,
                    # Plain-text jobs are unwrapped back to .txt after
                    # translation (see docx_path_to_txt_bytes below); a DOCX
                    # cover page would leak formatted disclaimer paragraphs
                    # into what the user expects to be clean translated text,
                    # so it is only added for genuine .docx deliverables.
                    "add_cover_page": not is_txt,
                }
                attempt_result = await docx_processor.translate(docx_config)
                if not attempt_result:
                    raise RuntimeError(
                        "No attempt report produced by DOCX translation pipeline"
                    )
                mono_pdf_path = attempt_result.get("output_path")
            else:
                tracker = _PipelineProgressTracker(job_id=job_id)
                processor = JobProcessor(progress_tracker=tracker)
                processor_config = {
                    "job_id": job_id,
                    "input_file": str(local_input_path),
                    "output_dir": str(workspace.attempts_dir),
                    "lang_in": source_lang,
                    "lang_out": target_lang,
                    "domain": domain,
                    "intent": intent,
                    "model_list": model_chain,
                    "max_model_attempts": max(1, settings.MAX_MODEL_ATTEMPTS),
                    "glossaries": glossaries,
                    "add_cover_page": True,
                    "no_dual": True,
                    "enable_dlp": enable_dlp,
                    "enable_judge": enable_judge,
                }
                attempt_result = await processor.translate(processor_config)
                if not attempt_result:
                    raise RuntimeError(
                        "No attempt report produced by translation pipeline"
                    )
                mono_pdf_path = attempt_result.get("mono_pdf_path")

            quality_rpt = attempt_result.get("quality_report") or {}
            attempt_idx = int(attempt_result.get("attempt_index") or 1)
            token_usage = attempt_result.get("token_usage") or {}

            await self.session_manager.record_attempt(
                job_id,
                {
                    "attempt_index": attempt_idx,
                    "model_id": attempt_result.get("model_id"),
                    "attempt_dir": str(workspace.attempts_dir),
                    "status": "completed",
                    "quality_score": quality_rpt.get("final_score"),
                    "token_usage": token_usage,
                    "cost_usd": token_usage.get("estimated_cost_usd"),
                    "output_path": mono_pdf_path,
                },
            )

            current_stage = "write_dlp_tokens"
            await self.bigquery.write_dlp_tokens(
                list(attempt_result.get("dlp_token_rows") or [])
            )
            await self.session_manager.mark_persistence_step(
                job_id, "write_dlp_tokens", succeeded=True
            )

            raw_output_name = str(
                source_doc.get("output_filename")
                or source_doc.get("original_filename")
                or ("output.docx" if is_docx else "output.pdf")
            )
            # Native DOCX translation produces a .docx output (no format
            # conversion); only the legacy PDF pipeline needs the
            # .docx -> .pdf filename rewrite.
            if not is_docx and raw_output_name.lower().endswith(".docx"):
                raw_output_name = raw_output_name[:-5] + ".pdf"
            preferred_output_name = raw_output_name

            # A .txt input was translated via the DOCX bridge (see
            # _convert_txt_to_docx); unwrap the winning .docx attempt back to
            # plain text before uploading so the output file matches the
            # original .txt format.
            if is_txt and mono_pdf_path:
                txt_output_path = Path(str(mono_pdf_path)).with_suffix(".txt")
                txt_output_path.write_bytes(
                    docx_path_to_txt_bytes(Path(str(mono_pdf_path)))
                )
                mono_pdf_path = txt_output_path

            selected_output_uri = None

            current_stage = "upload_output"
            if mono_pdf_path:
                selected_output_uri = await self.assembly_service.upload_output(
                    job_id=job_id,
                    local_path=Path(str(mono_pdf_path)),
                    preferred_filename=preferred_output_name,
                )
                await self.session_manager.set_output(
                    job_id,
                    output_path=mono_pdf_path,
                    output_gcs_uri=selected_output_uri,
                )

            current_stage = "compute_chunk_costs"
            accumulated_chunk_costs = await self._compute_accumulated_chunk_costs(
                job_id=job_id,
                local_input_path=local_input_path,
                token_usage=token_usage,
                model_id=str(attempt_result.get("model_id") or ""),
                split_page_ranges=attempt_result.get("split_page_ranges"),
            )
            attributed_input_tokens = int(accumulated_chunk_costs["input_tokens"])
            attributed_output_tokens = int(accumulated_chunk_costs["output_tokens"])
            # The domain guard's LLM call is part of what this job spent, so
            # it belongs in the total the cost ceiling is checked against.
            total_cost_usd = float(accumulated_chunk_costs["cost_usd"]) + float(
                domain_check_cost_usd
            )
            cache_hit_tokens = int(token_usage.get("cache_hit_prompt_tokens", 0))
            chunk_count = int(accumulated_chunk_costs.get("chunk_count", 0))

            await self.session_manager.set_token_totals(
                job_id,
                input_tokens=attributed_input_tokens,
                output_tokens=attributed_output_tokens,
                cache_hit_tokens=cache_hit_tokens,
                total_cost_usd=total_cost_usd,
                chunk_count=chunk_count,
            )

            current_stage = "validate_job_cost"
            validate_job_cost(total_cost_usd)

            completed_at = datetime.now(UTC)

            result_payload = {
                "output_gcs_uri": selected_output_uri,
                "token_count": attributed_input_tokens + attributed_output_tokens,
                "cost_usd": total_cost_usd,
                "intent": intent,
                "model_used": attempt_result.get("model_id"),
                "model_version": _extract_model_version(
                    attempt_result.get("model_id") or ""
                ),
                "confidence_score": float(quality_rpt.get("final_score", 0.0))
                if quality_rpt.get("final_score") is not None
                else None,
                "ab_variant": _attempt_index_to_variant(attempt_idx),
                "chunks": int(attempt_result.get("chunks_processed") or 0) or None,
                "retry_count": max(0, attempt_idx - 1),
                "quality_report": attempt_result.get("quality_report"),
                "dlp_provider": attempt_result.get("dlp_provider"),
                "dlp_chunk_mode": attempt_result.get("dlp_chunk_mode"),
                "attempts": attempt_result.get("attempts") or [],
            }

            current_stage = "write_cost_attribution"
            await self.bigquery.write_cost_attribution(
                {
                    "job_id": job_id,
                    "user_id": cost_attribution.get("user_id"),
                    "business_unit": cost_attribution.get("business_unit"),
                    "organization": cost_attribution.get("organization"),
                    "model_id": attempt_result.get("model_id"),
                    "intent": intent,
                    "input_tokens": attributed_input_tokens,
                    "output_tokens": attributed_output_tokens,
                    "cost_usd": total_cost_usd,
                    "timestamp": completed_at.isoformat(),
                }
            )
            await self.session_manager.mark_persistence_step(
                job_id, "write_cost_attribution", succeeded=True
            )

            current_stage = "completed_status_patch"
            await self._update_status(
                job_id,
                status="completed",
                result=result_payload,
                completed_at=completed_at,
                error_message=None,
            )
            await self.session_manager.mark_persistence_step(
                job_id, "completed_status_patch", succeeded=True
            )
            await self.session_manager.finish(job_id, "completed")
            logger.info(f"Translation pipeline finished for job {job_id}")

            # Auto-extracted glossary terms are only ever persisted to the
            # shared domain glossary AFTER the job has been marked completed
            # successfully -- terms from failed/low-quality attempts must
            # never reach the shared glossary that every future job reads
            # from (see docs/architecture/pdf-vs-docx-translation-architecture.md).
            if is_docx and attempt_result.get("extracted_terms"):
                try:
                    docx_processor.persist_extracted_terms(attempt_result)
                except Exception:
                    logger.warning(
                        f"Failed to persist auto-extracted glossary terms for job {job_id}",
                        exc_info=True,
                    )

            quality_report = attempt_result.get("quality_report") or {}
            set_root_span_attributes(
                {
                    "pipeline.model_used": str(attempt_result.get("model_id") or ""),
                    "pipeline.intent": intent,
                    "pipeline.source_lang": source_lang,
                    "pipeline.target_lang": target_lang,
                    "pipeline.domain": domain,
                    "pipeline.attempt_index": attempt_idx,
                    "pipeline.total_tokens": attributed_input_tokens
                    + attributed_output_tokens,
                    "pipeline.prompt_tokens": attributed_input_tokens,
                    "pipeline.completion_tokens": attributed_output_tokens,
                    "pipeline.estimated_cost_usd": total_cost_usd,
                    "pipeline.quality_final_score": float(
                        quality_report.get("final_score", 0.0)
                    ),
                    "pipeline.quality_passed": bool(
                        quality_report.get("pass_fail", False)
                    ),
                    "pipeline.quality_alignment": float(
                        quality_report.get("alignment_score", 0.0)
                    ),
                    "pipeline.quality_omission": float(
                        quality_report.get("omission_score", 0.0)
                    ),
                    "pipeline.quality_hallucination": float(
                        quality_report.get("hallucination_score", 0.0)
                    ),
                    "pipeline.judge_model": str(quality_report.get("model") or ""),
                }
            )
        except Exception as exc:
            await self._mark_job_failed(job_id, exc, pipeline_span, stage=current_stage)
        finally:
            try:
                snapshot = await self.session_manager.snapshot(job_id)
                logger.info(
                    "Translation job session snapshot for %s: %s",
                    job_id,
                    snapshot,
                )
            except KeyError:
                pass
            await self.session_manager.discard(job_id)
            self.temp_workspace_service.cleanup(job_id)
            if source_hash_for_release is not None:
                await get_shared_document_prep_cache().release(
                    source_hash_for_release, job_id
                )
