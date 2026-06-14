"""In-process runtime session state for concurrent translation jobs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from src.api.services.temp_workspace_service import JobWorkspace


@dataclass
class TranslationAttemptSession:
    """Runtime summary for one model attempt within a job."""

    attempt_index: int
    model_id: str
    attempt_dir: str | None = None
    status: str = "pending"
    quality_score: float | None = None
    token_usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    output_path: str | None = None
    error_message: str | None = None


@dataclass
class TranslationJobSession:
    """Runtime state for one active translation job."""

    job_id: str
    started_at: datetime
    workspace_root: str
    input_dir: str
    attempts_dir: str
    final_dir: str
    logs_dir: str
    status: str = "processing"
    input_path: str | None = None
    output_gcs_uri: str | None = None
    output_path: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    domain: str | None = None
    intent: str | None = None
    model_chain: list[str] = field(default_factory=list)
    selected_model: str | None = None
    attempts: list[TranslationAttemptSession] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    chunk_count: int = 0
    persistence_steps: dict[str, bool] = field(default_factory=dict)
    failure_stage: str | None = None
    error_message: str | None = None
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "paths": {
                "workspace_root": self.workspace_root,
                "input_dir": self.input_dir,
                "input_path": self.input_path,
                "attempts_dir": self.attempts_dir,
                "final_dir": self.final_dir,
                "logs_dir": self.logs_dir,
                "output_path": self.output_path,
                "output_gcs_uri": self.output_gcs_uri,
            },
            "routing": {
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "domain": self.domain,
                "intent": self.intent,
                "model_chain": list(self.model_chain),
                "selected_model": self.selected_model,
            },
            "attempts": [
                {
                    "attempt_index": a.attempt_index,
                    "model_id": a.model_id,
                    "attempt_dir": a.attempt_dir,
                    "status": a.status,
                    "quality_score": a.quality_score,
                    "token_usage": dict(a.token_usage),
                    "cost_usd": a.cost_usd,
                    "output_path": a.output_path,
                    "error_message": a.error_message,
                }
                for a in self.attempts
            ],
            "totals": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_hit_tokens": self.cache_hit_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": self.cost_usd,
                "chunk_count": self.chunk_count,
            },
            "persistence_steps": dict(self.persistence_steps),
            "failure_stage": self.failure_stage,
            "error_message": self.error_message,
        }


class TranslationJobSessionManager:
    """Registry of active translation job sessions (one per job_id)."""

    def __init__(self) -> None:
        self._sessions: dict[str, TranslationJobSession] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        job_id: str,
        workspace: JobWorkspace,
    ) -> TranslationJobSession:
        async with self._lock:
            if job_id in self._sessions:
                raise ValueError(f"Session already active for job {job_id}")
            session = TranslationJobSession(
                job_id=job_id,
                started_at=datetime.now(UTC),
                workspace_root=str(workspace.root),
                input_dir=str(workspace.input_dir),
                attempts_dir=str(workspace.attempts_dir),
                final_dir=str(workspace.final_dir),
                logs_dir=str(workspace.logs_dir),
            )
            self._sessions[job_id] = session
            return session

    def _require(self, job_id: str) -> TranslationJobSession:
        session = self._sessions.get(job_id)
        if session is None:
            raise KeyError(f"No active session for job {job_id}")
        return session

    async def set_input_path(self, job_id: str, path: Path | str) -> None:
        async with self._lock:
            self._require(job_id).input_path = str(path)

    async def set_routing(
        self,
        job_id: str,
        *,
        source_lang: str,
        target_lang: str,
        domain: str,
        intent: str,
        model_chain: list[str],
    ) -> None:
        async with self._lock:
            session = self._require(job_id)
            session.source_lang = source_lang
            session.target_lang = target_lang
            session.domain = domain
            session.intent = intent
            session.model_chain = list(model_chain)

    async def record_attempt(
        self, job_id: str, attempt_summary: dict[str, Any]
    ) -> None:
        async with self._lock:
            session = self._require(job_id)
            attempt = TranslationAttemptSession(
                attempt_index=int(attempt_summary.get("attempt_index", 0)),
                model_id=str(attempt_summary.get("model_id", "")),
                attempt_dir=attempt_summary.get("attempt_dir"),
                status=str(attempt_summary.get("status", "completed")),
                quality_score=attempt_summary.get("quality_score"),
                token_usage=dict(attempt_summary.get("token_usage") or {}),
                cost_usd=attempt_summary.get("cost_usd"),
                output_path=attempt_summary.get("output_path"),
                error_message=attempt_summary.get("error_message"),
            )
            session.attempts.append(attempt)
            session.selected_model = attempt.model_id

    async def set_output(
        self,
        job_id: str,
        *,
        output_path: Path | str | None = None,
        output_gcs_uri: str | None = None,
    ) -> None:
        async with self._lock:
            session = self._require(job_id)
            if output_path is not None:
                session.output_path = str(output_path)
            if output_gcs_uri is not None:
                session.output_gcs_uri = output_gcs_uri

    async def set_token_totals(
        self,
        job_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int = 0,
        total_cost_usd: float,
        chunk_count: int = 0,
    ) -> None:
        async with self._lock:
            session = self._require(job_id)
            session.input_tokens = int(input_tokens)
            session.output_tokens = int(output_tokens)
            session.cache_hit_tokens = int(cache_hit_tokens)
            session.total_tokens = session.input_tokens + session.output_tokens
            session.cost_usd = float(total_cost_usd)
            session.chunk_count = int(chunk_count)

    async def mark_persistence_step(
        self, job_id: str, step: str, *, succeeded: bool = True
    ) -> None:
        async with self._lock:
            self._require(job_id).persistence_steps[step] = succeeded

    async def mark_failure(
        self, job_id: str, *, stage: str, error_message: str
    ) -> None:
        async with self._lock:
            session = self._require(job_id)
            session.status = "failed"
            session.failure_stage = stage
            session.error_message = error_message

    async def snapshot(self, job_id: str) -> dict[str, Any]:
        async with self._lock:
            return self._require(job_id).to_dict()

    async def finish(
        self,
        job_id: str,
        status: str,
        *,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            session = self._require(job_id)
            session.status = status
            session.error_message = error_message
            session.finished_at = datetime.now(UTC)
            return session.to_dict()

    async def discard(self, job_id: str) -> None:
        async with self._lock:
            self._sessions.pop(job_id, None)

    async def active_count(self) -> int:
        async with self._lock:
            return len(self._sessions)
