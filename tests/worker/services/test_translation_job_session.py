"""Tests for per-job translation session manager."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from src.worker.services.temp_workspace_service import TempWorkspaceService
from src.worker.services.translation_job_session import TranslationJobSessionManager


@pytest.fixture
def workspace(tmp_path: Path):
    service = TempWorkspaceService(base_dir=tmp_path / "jobs")
    return service.create("job-a")


@pytest.fixture
def manager() -> TranslationJobSessionManager:
    return TranslationJobSessionManager()


class TestTranslationJobSessionManager:
    @pytest.mark.asyncio
    async def test_start_and_snapshot(self, manager, workspace):
        session = await manager.start("job-a", workspace)
        assert session.job_id == "job-a"
        assert session.workspace_root == str(workspace.root)

        snapshot = await manager.snapshot("job-a")
        assert snapshot["job_id"] == "job-a"
        assert snapshot["paths"]["input_dir"] == str(workspace.input_dir)
        assert snapshot["status"] == "processing"

    @pytest.mark.asyncio
    async def test_tracks_paths_tokens_and_attempts(self, manager, workspace):
        await manager.start("job-a", workspace)
        await manager.set_input_path("job-a", workspace.input_dir / "input.pdf")
        await manager.set_routing(
            "job-a",
            source_lang="ja",
            target_lang="en",
            domain="commercial",
            intent="Intent-Commercial-JA-EN",
            model_chain=["claude-sonnet-4-6", "gemini-2.5-flash"],
        )
        await manager.record_attempt(
            "job-a",
            {
                "attempt_index": 1,
                "model_id": "claude-sonnet-4-6",
                "attempt_dir": str(workspace.attempts_dir),
                "status": "completed",
                "quality_score": 0.91,
                "token_usage": {"prompt_tokens": 1000, "completion_tokens": 400},
                "cost_usd": 0.05,
                "output_path": str(workspace.final_dir / "out.pdf"),
            },
        )
        await manager.set_output(
            "job-a",
            output_path=workspace.final_dir / "out.pdf",
            output_gcs_uri="gs://bucket/out.pdf",
        )
        await manager.set_token_totals(
            "job-a",
            input_tokens=1000,
            output_tokens=400,
            cache_hit_tokens=100,
            total_cost_usd=0.05,
            chunk_count=5,
        )
        await manager.mark_persistence_step("job-a", "write_cost_attribution")

        snapshot = await manager.snapshot("job-a")
        assert snapshot["paths"]["input_path"].endswith("input.pdf")
        assert snapshot["routing"]["selected_model"] == "claude-sonnet-4-6"
        assert snapshot["totals"]["input_tokens"] == 1000
        assert snapshot["totals"]["output_tokens"] == 400
        assert snapshot["totals"]["cost_usd"] == pytest.approx(0.05)
        assert snapshot["totals"]["chunk_count"] == 5
        assert snapshot["persistence_steps"]["write_cost_attribution"] is True
        assert snapshot["paths"]["output_gcs_uri"] == "gs://bucket/out.pdf"

    @pytest.mark.asyncio
    async def test_finish_and_discard(self, manager, workspace):
        await manager.start("job-a", workspace)
        finished = await manager.finish("job-a", "completed")
        assert finished["status"] == "completed"
        assert finished["finished_at"] is not None

        await manager.discard("job-a")
        assert await manager.active_count() == 0
        with pytest.raises(KeyError):
            await manager.snapshot("job-a")

    @pytest.mark.asyncio
    async def test_concurrent_sessions_remain_isolated(self, manager, tmp_path: Path):
        service = TempWorkspaceService(base_dir=tmp_path / "jobs")
        ws_a = service.create("job-a")
        ws_b = service.create("job-b")

        await manager.start("job-a", ws_a)
        await manager.start("job-b", ws_b)

        async def update_job_a() -> None:
            await manager.set_token_totals(
                "job-a",
                input_tokens=100,
                output_tokens=50,
                total_cost_usd=0.01,
            )

        async def update_job_b() -> None:
            await manager.set_token_totals(
                "job-b",
                input_tokens=900,
                output_tokens=300,
                total_cost_usd=0.09,
            )

        await asyncio.gather(update_job_a(), update_job_b())

        snap_a = await manager.snapshot("job-a")
        snap_b = await manager.snapshot("job-b")
        assert snap_a["totals"]["input_tokens"] == 100
        assert snap_b["totals"]["input_tokens"] == 900
        assert snap_a["paths"]["workspace_root"] != snap_b["paths"]["workspace_root"]

    @pytest.mark.asyncio
    async def test_mark_failure_records_stage(self, manager, workspace):
        await manager.start("job-a", workspace)
        await manager.mark_failure(
            "job-a",
            stage="write_cost_attribution",
            error_message="BigQuery insert failed",
        )
        snapshot = await manager.snapshot("job-a")
        assert snapshot["status"] == "failed"
        assert snapshot["failure_stage"] == "write_cost_attribution"
        assert snapshot["error_message"] == "BigQuery insert failed"
