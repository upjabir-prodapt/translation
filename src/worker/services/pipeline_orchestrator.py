"""API-only background translation orchestration."""

from __future__ import annotations

import logging
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode

from src.worker.services.assembly_service import AssemblyService
from src.worker.services.glossary_service import GlossaryService
from src.worker.services.intent_router_service import IntentRouterService
from src.worker.services.language_detection_service import LanguageDetectionService
from src.worker.services.llm_cost_service import get_vertex_llm_cost_service
from src.worker.services.processor_service import JobProcessor
from src.worker.services.temp_workspace_service import TempWorkspaceService
from src.worker.services.translation_job_session import TranslationJobSessionManager
from src.worker.utils.cost_utils import validate_job_cost
from src.worker.utils.docx_converter import convert_docx_to_pdf
from src.config.constants import settings
from src.config.tracing import set_root_span
from src.config.tracing import set_root_span_attributes
from src.config.tracing import tracer_pipeline
from src.repository.api_storage_repository import APIStorageRepository
from src.repository.bigquery_repository import BigQueryRepository
from src.repository.repository_exception import BigQueryError
from src.repository.repository_exception import StorageError

logger = logging.getLogger(__name__)

_OMIT = object()


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
    """Execute translation pipeline in API background process."""

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
        """Record failure in session, OTel, logs, and BigQuery."""
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
                result=None,
                completed_at=datetime.now(UTC),
                error_message=str(exc),
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
    ) -> dict[str, float | int]:
        """Split the PDF into chunks, compute proportional costs, and return job totals."""
        from src.worker.utils.cost_utils import aggregate_chunk_cost_records
        from src.worker.doctranslator.format.pdf.split_manager import (
            StructureAwareSplitStrategy,
        )

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

    async def _execute_pipeline(
        self, job_id: str, job_data: dict[str, Any], pipeline_span
    ) -> None:
        workspace = self.temp_workspace_service.create(job_id)
        await self.session_manager.start(job_id, workspace)
        current_stage = "initialize"
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

            current_stage = "download_input"
            await self.storage.download_file(blob_path, local_input_path)

            if source_doc.get("format") == "docx":
                logger.info(f"Converting DOCX to PDF for job {job_id}")
                local_input_path = convert_docx_to_pdf(
                    local_input_path, workspace.input_dir
                )

            await self.session_manager.set_input_path(job_id, local_input_path)

            source_lang = translation_config.get("source_language")
            if not source_lang or source_lang == "auto":
                source_lang = self.language_detector.detect(local_input_path)

            target_lang = translation_config["target_language"]
            domain = translation_config["domain"]
            processing_options = job_data.get("processing_options") or {}
            enable_dlp = bool(processing_options.get("enable_dlp", False))
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

            glossaries = self.glossary_service.load_domain_glossary(
                domain=domain,
                target_language_name=target_lang,
            )

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
            }

            current_stage = "translate"
            attempt_result = await processor.translate(processor_config)
            if not attempt_result:
                raise RuntimeError("No attempt report produced by translation pipeline")

            quality_rpt = attempt_result.get("quality_report") or {}
            attempt_idx = int(attempt_result.get("attempt_index") or 1)
            token_usage = attempt_result.get("token_usage") or {}
            mono_pdf_path = attempt_result.get("mono_pdf_path")

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
                or "output.pdf"
            )
            if raw_output_name.lower().endswith(".docx"):
                raw_output_name = raw_output_name[:-5] + ".pdf"
            preferred_output_name = raw_output_name
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
            )
            attributed_input_tokens = int(accumulated_chunk_costs["input_tokens"])
            attributed_output_tokens = int(accumulated_chunk_costs["output_tokens"])
            total_cost_usd = float(accumulated_chunk_costs["cost_usd"])
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
