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

from src.api.services.assembly_service import AssemblyService
from src.api.services.cover_page_service import CoverPageService
from src.api.services.glossary_service import GlossaryService
from src.api.services.intent_router_service import IntentRouterService
from src.api.services.language_detection_service import LanguageDetectionService
from src.api.services.processor_service import JobProcessor
from src.api.services.temp_workspace_service import TempWorkspaceService
from src.api.utils.docx_converter import convert_docx_to_pdf
from src.config.constants import settings
from src.config.tracing import set_root_span
from src.config.tracing import set_root_span_attributes
from src.config.tracing import tracer_pipeline
from src.repository.api_storage_repository import APIStorageRepository
from src.repository.bigquery_repository import BigQueryRepository

logger = logging.getLogger(__name__)

_OMIT = object()


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
    ):
        self.bigquery = bigquery
        self.storage = storage
        self.temp_workspace_service = temp_workspace_service or TempWorkspaceService()
        self.language_detector = LanguageDetectionService()
        self.intent_router = IntentRouterService()
        self.glossary_service = GlossaryService()
        self.cover_page_service = CoverPageService()
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

    async def _execute_pipeline(
        self, job_id: str, job_data: dict[str, Any], pipeline_span
    ) -> None:
        workspace = self.temp_workspace_service.create(job_id)
        try:
            source_doc = job_data["source_document"]
            translation_config = job_data["translation_config"]
            cost_attribution = job_data["cost_attribution"]

            await self._update_status(job_id, status="processing")

            input_filename = source_doc.get("original_filename", "input.pdf")
            local_input_path = workspace.input_dir / input_filename
            blob_path = self._extract_blob_path(source_doc["gcs_uri"])
            await self.storage.download_file(blob_path, local_input_path)

            if source_doc.get("format") == "docx":
                logger.info(f"Converting DOCX to PDF for job {job_id}")
                local_input_path = convert_docx_to_pdf(
                    local_input_path, workspace.input_dir
                )

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
            attempt_result = await processor.translate(processor_config)
            await self.bigquery.write_dlp_tokens(
                list(attempt_result.get("dlp_token_rows") or [])
            )
            if not attempt_result:
                raise RuntimeError("No attempt report produced by translation pipeline")
            mono_pdf_path = attempt_result.get("mono_pdf_path")
            raw_output_name = str(
                source_doc.get("output_filename")
                or source_doc.get("original_filename")
                or "output.pdf"
            )
            if raw_output_name.lower().endswith(".docx"):
                raw_output_name = raw_output_name[:-5] + ".pdf"
            preferred_output_name = raw_output_name
            selected_output_uri = None
            if mono_pdf_path:
                selected_output_uri = await self.assembly_service.upload_output(
                    job_id=job_id,
                    local_path=Path(str(mono_pdf_path)),
                    preferred_filename=preferred_output_name,
                )

            result_payload = {
                "output_gcs_uri": selected_output_uri,
                "token_count": int(
                    (attempt_result.get("token_usage") or {}).get("total_tokens", 0)
                ),
                "cost_usd": float(
                    (attempt_result.get("token_usage") or {}).get(
                        "estimated_cost_usd", 0.0
                    )
                ),
                "intent": intent,
                "model_used": attempt_result.get("model_id"),
                "retry_count": max(0, attempt_result.get("attempt_index")),
                "quality_report": attempt_result.get("quality_report"),
                "dlp_provider": attempt_result.get("dlp_provider"),
                "dlp_chunk_mode": attempt_result.get("dlp_chunk_mode"),
            }

            completed_at = datetime.now(UTC)
            await self._update_status(
                job_id,
                status="completed",
                result=result_payload,
                completed_at=completed_at,
                error_message=None,
            )

            await self.bigquery.write_cost_attribution(
                {
                    "job_id": job_id,
                    "user_id": cost_attribution.get("user_id"),
                    "business_unit": cost_attribution.get("business_unit"),
                    "organization": cost_attribution.get("organization"),
                    "model_id": attempt_result.get("model_id"),
                    "intent": intent,
                    "input_tokens": int(
                        (attempt_result.get("token_usage") or {}).get(
                            "prompt_tokens", 0
                        )
                    ),
                    "output_tokens": int(
                        (attempt_result.get("token_usage") or {}).get(
                            "completion_tokens", 0
                        )
                    ),
                    "cost_usd": float(
                        (attempt_result.get("token_usage") or {}).get(
                            "estimated_cost_usd", 0.0
                        )
                    ),
                    "timestamp": completed_at.isoformat(),
                }
            )
            logger.info(f"Translation pipeline finished for job {job_id}")

            # ── Propagate final summary to the root pipeline span ──────────
            quality_report = attempt_result.get("quality_report") or {}
            token_usage = attempt_result.get("token_usage") or {}
            set_root_span_attributes(
                {
                    "pipeline.model_used": str(attempt_result.get("model_id") or ""),
                    "pipeline.intent": intent,
                    "pipeline.source_lang": source_lang,
                    "pipeline.target_lang": target_lang,
                    "pipeline.domain": domain,
                    "pipeline.attempt_index": int(
                        attempt_result.get("attempt_index", 1)
                    ),
                    "pipeline.total_tokens": int(token_usage.get("total_tokens", 0)),
                    "pipeline.prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
                    "pipeline.completion_tokens": int(
                        token_usage.get("completion_tokens", 0)
                    ),
                    "pipeline.estimated_cost_usd": float(
                        token_usage.get("estimated_cost_usd", 0.0)
                    ),
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
            logger.error(f"Pipeline failed for job {job_id}. Exception: {exc}")
            pipeline_span.set_status(Status(StatusCode.ERROR, str(exc)))
            pipeline_span.record_exception(exc)
            await self._update_status(
                job_id,
                status="failed",
                result=None,
                completed_at=datetime.now(UTC),
                error_message=str(exc),
            )
        finally:
            self.temp_workspace_service.cleanup(job_id)
