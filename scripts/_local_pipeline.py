"""Local stand-ins for BigQuery/GCS plus the runner behind `local_translate.py`.

Kept out of the CLI module so every import here happens *after*
`local_translate.apply_env_overrides()` has set its env vars -- `Settings`
is constructed at `src.config.constants` import time and never re-read.

Nothing in this file re-implements a pipeline stage. The real
`PipelineOrchestrator` runs; only the two I/O boundaries it talks to are
swapped:

  `LocalObjectStore`  stands in for `APIStorageRepository` (GCS)
  `LocalJobStore`     stands in for `BigQueryRepository`
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import uuid
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

# Imported for its import-time side effect: configures the root logger with
# the same GcpJsonFormatter the API and worker processes use.
import src.config.logging_config  # noqa: F401
from src.api.exceptions import ValidationError
from src.api.schemas.requests import CostAttributionInput
from src.api.schemas.requests import DocumentInput
from src.api.schemas.requests import ProcessingOptions
from src.api.schemas.requests import TranslateRequest
from src.api.schemas.requests import TranslationConfigInput
from src.api.services.translation_service import _validate_and_describe
from src.config.constants import settings
from src.repository.storage_repository import StoragePath
from src.shared import job_status
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger("local_translate")

_FORMAT_BY_SUFFIX = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def force_utf8_console() -> None:
    """Make stdout/stderr UTF-8 so log records survive a cp1252 console.

    `GcpJsonFormatter` serialises with `ensure_ascii=False`, and several
    user-facing rejection messages contain an em dash. On Cloud Run the
    stream is UTF-8 already; in a Windows terminal it is cp1252, where the
    first such record would raise `UnicodeEncodeError` inside the logging
    handler and lose the message.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def use_text_logging() -> None:
    """Swap the Cloud Logging JSON formatter for a readable one-line format."""
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)


# ---------------------------------------------------------------------------
# Local stand-in for GCS
# ---------------------------------------------------------------------------


class LocalObjectStore:
    """A filesystem tree shaped like the GCS bucket, addressed by blob path.

    Implements exactly the surface `PipelineOrchestrator` and
    `AssemblyService` use, with the same signatures and the same
    `gs://bucket/blob` return values, so the orchestrator's own
    `_extract_blob_path()` keeps working unchanged.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket_name = settings.GCS_BUCKET_NAME

    # -- path helpers (same semantics as StorageRepository) ----------------

    def build_job_path(
        self,
        job_id: str,
        folder: str | None = None,
        filename: str | None = None,
        prefix: str | None = None,
    ) -> str:
        return StoragePath(
            prefix=prefix or settings.GCS_TRANSLATION_PREFIX,
            job_id=job_id,
            folder=folder,
            filename=filename,
        ).build()

    def _uri(self, blob_path: str) -> str:
        return f"gs://{self.bucket_name}/{blob_path}"

    def _local_path(self, blob_path: str) -> Path:
        return self.root / blob_path

    def local_path_for_uri(self, uri: str) -> Path:
        blob_path = uri
        if blob_path.startswith("gs://"):
            blob_path = blob_path[5:]
            if blob_path.startswith(f"{self.bucket_name}/"):
                blob_path = blob_path[len(self.bucket_name) + 1 :]
        return self._local_path(blob_path)

    # -- the APIStorageRepository surface the pipeline actually calls -------

    async def upload_input_pdf(
        self, file_content: bytes, filename: str, job_id: str
    ) -> str:
        blob_path = self.build_job_path(
            job_id=job_id, folder=settings.GCS_INPUT_FOLDER, filename=filename
        )
        return await self.upload_file(source=file_content, blob_path=blob_path)

    async def upload_file(
        self,
        source: Path | bytes,
        blob_path: str,
        file_type: Any = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        del file_type, metadata
        destination = self._local_path(blob_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, bytes | bytearray):
            await asyncio.to_thread(destination.write_bytes, bytes(source))
        else:
            await asyncio.to_thread(shutil.copy2, Path(source), destination)
        logger.info(
            "[local-gcs] upload %s (%d bytes)", blob_path, destination.stat().st_size
        )
        return self._uri(blob_path)

    async def download_file(
        self, blob_path: str, local_path: Path, create_dirs: bool = True
    ) -> Path:
        source = self._local_path(blob_path)
        if not source.exists():
            raise FileNotFoundError(f"[local-gcs] no such object: {blob_path}")
        if create_dirs:
            local_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source, local_path)
        logger.info("[local-gcs] download %s -> %s", blob_path, local_path)
        return local_path

    async def delete_job_files(self, job_id: str) -> int:
        prefix = self._local_path(f"{settings.GCS_TRANSLATION_PREFIX}/{job_id}")
        if not prefix.exists():
            return 0
        count = sum(1 for path in prefix.rglob("*") if path.is_file())
        await asyncio.to_thread(shutil.rmtree, prefix, True)
        logger.info("[local-gcs] deleted %d object(s) for job %s", count, job_id)
        return count


# ---------------------------------------------------------------------------
# Local stand-in for BigQuery
# ---------------------------------------------------------------------------


class LocalJobStore:
    """Holds job rows in memory and logs every write BigQuery would have taken.

    No dataset, no table, no network. `find_recent_duplicate_job` always
    reports "no duplicate": the idempotency window is a property of the
    shared jobs table, which does not exist here.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.dlp_token_rows: list[dict[str, Any]] = []
        self.cost_rows: list[dict[str, Any]] = []

    async def upsert_translation_job(self, job_data: dict[str, Any]) -> None:
        job_id = str(job_data["job_id"])
        self.jobs[job_id] = dict(job_data)
        logger.info(
            "[local-bq] upsert_translation_job job_id=%s status=%s "
            "(no BigQuery row written)",
            job_id,
            job_data.get("status"),
        )

    async def patch_translation_job(self, job_id: str, updates: dict[str, Any]) -> None:
        record = self.jobs.setdefault(str(job_id), {"job_id": job_id})
        record.update(updates)
        logger.info(
            "[local-bq] patch_translation_job job_id=%s fields=%s status=%s",
            job_id,
            sorted(updates),
            updates.get("status", record.get("status")),
        )

    async def get_translation_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(str(job_id))

    async def find_recent_duplicate_job(self, **kwargs: Any) -> None:
        del kwargs
        return None

    async def write_dlp_tokens(self, rows: list[dict[str, Any]]) -> None:
        self.dlp_token_rows.extend(rows or [])
        logger.info("[local-bq] write_dlp_tokens rows=%d", len(rows or []))

    async def write_cost_attribution(self, data: dict[str, Any]) -> None:
        self.cost_rows.append(dict(data))
        logger.info(
            "[local-bq] write_cost_attribution job_id=%s cost_usd=%s "
            "tokens_in=%s tokens_out=%s",
            data.get("job_id"),
            data.get("cost_usd"),
            data.get("input_tokens"),
            data.get("output_tokens"),
        )


# ---------------------------------------------------------------------------
# One job
# ---------------------------------------------------------------------------


def _output_filename(validated_filename: str, target_language: str) -> str:
    """Turn `report.pdf` plus `fr` into `report.fr.pdf`.

    The service names the output after the input, which is safe when the two
    live in different GCS folders. Here they can land in the same directory,
    so the target language is folded into the name rather than the source
    document being overwritten.
    """
    name = Path(validated_filename)
    return f"{name.stem}.{target_language}{name.suffix}"


def _build_job_data(
    *,
    job_id: str,
    request: TranslateRequest,
    metadata: dict[str, Any],
    input_uri: str,
    source_hash: str,
    output_filename: str,
) -> dict[str, Any]:
    """The job record `TranslationService._do_submit` would have written.

    Kept field-for-field identical to the real one: `PipelineOrchestrator`
    reads `source_document`, `translation_config`, `processing_options`,
    `cost_attribution` and `source_hash` out of it, so a key missing here
    would be a divergence from the deployed path rather than a local
    convenience.
    """
    now = datetime.now(UTC)
    config = request.translation_config
    return {
        "job_id": job_id,
        "status": job_status.QUEUED,
        "progress": 0.0,
        "source_document": {
            "gcs_uri": input_uri,
            "format": request.document.format,
            "page_count": metadata.get("page_count"),
            "source_language": config.source_language,
            "original_filename": metadata["filename"],
            "output_filename": output_filename,
            "file_size_bytes": metadata["size_bytes"],
            "checksum": metadata["checksum"],
        },
        "translation_config": {
            "source_language": config.source_language,
            "target_language": config.target_language,
            "domain": config.domain,
            "enable_dlp": request.processing_options.enable_dlp,
        },
        "cost_attribution": request.cost_attribution.model_dump(),
        "processing_options": {
            "enable_dlp": request.processing_options.enable_dlp,
            "enable_chunking": request.processing_options.enable_chunking,
            "priority": request.processing_options.priority,
        },
        "error_message": None,
        "result": None,
        "timestamps": {"submitted_at": now, "completed_at": None},
        "config": {
            "job_id": job_id,
            "lang_in": config.source_language,
            "lang_out": config.target_language,
            "domain": config.domain,
        },
        "source_hash": source_hash,
        "submitted_at": now,
        "completed_at": None,
    }


async def translate_one(
    input_path: Path,
    *,
    args: argparse.Namespace,
    store: LocalObjectStore,
    jobs: LocalJobStore,
    orchestrator: PipelineOrchestrator,
) -> dict[str, Any]:
    """Validate, run the pipeline, and write the output beside the input.

    Returns a summary dict and never raises for an expected rejection (a bad
    file, a mixed-language document, a wrong domain): those come back as a
    non-completed status carrying the message the API would have returned.
    """
    doc_format = _FORMAT_BY_SUFFIX[input_path.suffix.lower()]
    content = input_path.read_bytes()
    logger.info(
        "Submitting %s (%s, %d bytes) %s -> %s [domain=%s]",
        input_path,
        doc_format,
        len(content),
        args.source_language,
        args.target_language,
        args.domain,
    )

    # 1. Request-schema validation: the same models FastAPI validates the
    #    POST body against (extension, base64, domain, language codes,
    #    source != target).
    try:
        request = TranslateRequest(
            document=DocumentInput(
                content=base64.b64encode(content).decode("ascii"),
                format=doc_format,
                filename=input_path.name,
            ),
            translation_config=TranslationConfigInput(
                source_language=args.source_language,
                target_language=args.target_language,
                domain=args.domain,
            ),
            cost_attribution=CostAttributionInput(
                user_id=args.user_id,
                business_unit=args.business_unit,
                organization=args.organization,
            ),
            processing_options=ProcessingOptions(enable_dlp=not args.no_dlp),
        )
    except PydanticValidationError as exc:
        messages = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        logger.error("Request validation failed for %s: %s", input_path, messages)
        return {"input": input_path, "status": "rejected", "error": messages}

    # 2. File validation: PDF magic bytes / size / encryption / copy
    #    permission / text layer, DOCX OOXML structure, TXT UTF-8.
    try:
        metadata = _validate_and_describe(content, request.document)
    except ValidationError as exc:
        field = exc.details.get("field")
        logger.error(
            "File validation failed for %s%s: %s",
            input_path,
            f" (field={field})" if field else "",
            exc.message,
        )
        return {"input": input_path, "status": "rejected", "error": exc.message}

    logger.info(
        "File validation passed for %s: pages=%s size=%s checksum=%s",
        metadata["filename"],
        metadata.get("page_count"),
        metadata["size_bytes"],
        metadata["checksum"][:12],
    )

    # 3. Stage the input where the pipeline expects to download it from, and
    #    build the job record instead of writing one to BigQuery.
    job_id = str(uuid.uuid4())
    source_hash = hashlib.sha256(content).hexdigest()
    input_uri = await store.upload_input_pdf(
        file_content=content, filename=metadata["filename"], job_id=job_id
    )
    output_filename = _output_filename(
        metadata["filename"], request.translation_config.target_language
    )
    job_data = _build_job_data(
        job_id=job_id,
        request=request,
        metadata=metadata,
        input_uri=input_uri,
        source_hash=source_hash,
        output_filename=output_filename,
    )
    await jobs.upsert_translation_job(job_data)

    # 4. The real pipeline: language detection plus the mixed/wrong-language
    #    guard, the domain guard, DLP, model routing, translation, the
    #    quality judge and costing. It records failure on the job record
    #    rather than raising.
    logger.info("Running translation pipeline in-process for job %s", job_id)
    await orchestrator.run(job_id=job_id, job_data=job_data)

    record = await jobs.get_translation_job(job_id) or {}
    status = str(record.get("status") or "unknown")
    if status != "completed":
        error = str(record.get("error_message") or "Pipeline did not complete")
        logger.error("Job %s %s: %s", job_id, status, error)
        return {
            "input": input_path,
            "job_id": job_id,
            "status": status,
            "error": error,
        }

    result = record.get("result") or {}
    output_uri = result.get("output_gcs_uri")
    if not output_uri:
        logger.error("Job %s completed but produced no output file", job_id)
        return {
            "input": input_path,
            "job_id": job_id,
            "status": "failed",
            "error": "Pipeline completed without producing an output file",
        }

    # 5. Deliver the result next to the input (or into --output-dir).
    output_dir = args.output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / output_filename
    if destination.resolve() == input_path.resolve():
        destination = output_dir / f"{destination.stem}.translated{destination.suffix}"
    await asyncio.to_thread(
        shutil.copy2, store.local_path_for_uri(str(output_uri)), destination
    )
    logger.info("Wrote translated output to %s", destination)

    return {
        "input": input_path,
        "job_id": job_id,
        "status": "completed",
        "output": destination,
        "model": result.get("model_used"),
        "cost_usd": result.get("cost_usd"),
        "tokens": result.get("token_count"),
        "quality": result.get("confidence_score"),
        "detected_languages": (record.get("translation_config") or {}).get(
            "detected_languages"
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _collect_inputs(target: Path) -> list[Path]:
    if target.is_dir():
        files = sorted(
            path
            for path in target.iterdir()
            if path.is_file() and path.suffix.lower() in _FORMAT_BY_SUFFIX
        )
        if not files:
            raise SystemExit(f"No .pdf/.docx/.txt files found in {target}")
        return files
    if not target.exists():
        raise SystemExit(f"No such file: {target}")
    if target.suffix.lower() not in _FORMAT_BY_SUFFIX:
        raise SystemExit(
            f"Unsupported file type '{target.suffix}'. "
            f"Supported: {', '.join(sorted(_FORMAT_BY_SUFFIX))}"
        )
    return [target]


async def _run_all(
    inputs: list[Path], args: argparse.Namespace, scratch_root: Path
) -> list[dict[str, Any]]:
    store = LocalObjectStore(scratch_root / "gcs")
    jobs = LocalJobStore()
    orchestrator = PipelineOrchestrator(bigquery=jobs, storage=store)

    results: list[dict[str, Any]] = []
    for input_path in inputs:
        results.append(
            await translate_one(
                input_path,
                args=args,
                store=store,
                jobs=jobs,
                orchestrator=orchestrator,
            )
        )
    return results


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72, file=sys.stderr)
    print("Local translation summary", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    for result in results:
        marker = "OK  " if result["status"] == "completed" else "FAIL"
        print(f"{marker} {result['input']}", file=sys.stderr)
        if result["status"] == "completed":
            print(f"     output  : {result['output']}", file=sys.stderr)
            print(
                f"     model   : {result.get('model')}  "
                f"tokens={result.get('tokens')}  "
                f"cost=${result.get('cost_usd')}  "
                f"quality={result.get('quality')}",
                file=sys.stderr,
            )
            if result.get("detected_languages"):
                print(
                    f"     detected: {json.dumps(result['detected_languages'])}",
                    file=sys.stderr,
                )
        else:
            print(f"     {result['status']}: {result['error']}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)


def run_cli(args: argparse.Namespace) -> int:
    force_utf8_console()
    if args.log_format == "text":
        use_text_logging()

    inputs = _collect_inputs(args.input)
    logger.info(
        "local_translate: %d document(s); dlp=%s domain_check=%s "
        "language_check=%s judge=%s",
        len(inputs),
        settings.GOOGLE_DLP_ENABLED and not args.no_dlp,
        settings.DOMAIN_MISMATCH_CHECK_ENABLED,
        settings.LANGUAGE_MISMATCH_CHECK_ENABLED,
        settings.QUALITY_JUDGE_ENABLED,
    )

    scratch_root = Path(tempfile.mkdtemp(prefix="local-translate-"))
    logger.info("Local stand-in object store: %s", scratch_root)
    try:
        results = asyncio.run(_run_all(inputs, args, scratch_root))
    finally:
        if args.keep_scratch:
            logger.info("Keeping local object store at %s", scratch_root)
        else:
            shutil.rmtree(scratch_root, ignore_errors=True)

    _print_summary(results)
    return 0 if all(result["status"] == "completed" for result in results) else 1
