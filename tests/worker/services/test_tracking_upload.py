import json
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.repository.storage_repository import FileType
from src.repository.translation_storage_repository import TranslationStorageRepository
from src.worker.services.model_attempt_orchestrator import (
    _upload_sampled_attempt_artifacts,
)
from src.worker.services.model_attempt_orchestrator import is_job_sampled_for_tracking


class TestTrackingSampling:
    def test_sampling_rate_0_percent(self, monkeypatch):
        from src.config.constants import settings

        monkeypatch.setattr(settings, "TRACKING_SAMPLE_PERCENTAGE", 0)
        assert is_job_sampled_for_tracking("job-123") is False
        assert is_job_sampled_for_tracking("job-456") is False

    def test_sampling_rate_100_percent(self, monkeypatch):
        from src.config.constants import settings

        monkeypatch.setattr(settings, "TRACKING_SAMPLE_PERCENTAGE", 100)
        assert is_job_sampled_for_tracking("job-123") is True
        assert is_job_sampled_for_tracking("job-456") is True

    def test_sampling_deterministic_by_job_id(self, monkeypatch):
        from src.config.constants import settings

        monkeypatch.setattr(settings, "TRACKING_SAMPLE_PERCENTAGE", 50)
        res1 = is_job_sampled_for_tracking("job-fixed-abc")
        res2 = is_job_sampled_for_tracking("job-fixed-abc")
        assert res1 == res2

    def test_empty_job_id_returns_false(self):
        assert is_job_sampled_for_tracking("") is False


class TestUploadAttemptArtifacts:
    @pytest.mark.asyncio
    async def test_upload_attempt_artifacts_success(self, tmp_path: Path):
        repo = TranslationStorageRepository.__new__(TranslationStorageRepository)
        repo.upload_file = AsyncMock(
            side_effect=["gs://b/track.json", "gs://b/qual.json"]
        )
        repo.build_job_path = MagicMock(
            side_effect=[
                "translation-service/job-1/input/iter_1_translate_tracking.json",
                "translation-service/job-1/input/iter_1_quality_report.json",
            ]
        )

        tracking_file = tmp_path / "translate_tracking.json"
        tracking_file.write_text(json.dumps({"page": []}))

        quality_file = tmp_path / "quality_report.json"
        quality_file.write_text(json.dumps({"final_score": 0.9}))

        uris = await repo.upload_attempt_artifacts(
            job_id="job-1",
            attempt_index=1,
            tracking_path=tracking_file,
            quality_report_path=quality_file,
        )

        assert len(uris) == 2
        assert repo.upload_file.call_count == 2
        calls = repo.upload_file.call_args_list
        assert calls[0].kwargs["file_type"] == FileType.JSON
        assert calls[0].kwargs["metadata"]["artifact_type"] == "tracking"
        assert calls[1].kwargs["metadata"]["artifact_type"] == "quality_report"

    @pytest.mark.asyncio
    async def test_upload_attempt_artifacts_missing_files_ignored(self):
        repo = TranslationStorageRepository.__new__(TranslationStorageRepository)
        repo.upload_file = AsyncMock()

        uris = await repo.upload_attempt_artifacts(
            job_id="job-1",
            attempt_index=1,
            tracking_path=Path("/nonexistent/tracking.json"),
            quality_report_path=None,
        )

        assert uris == []
        repo.upload_file.assert_not_called()


class TestOrchestratorSampledUploadHelper:
    @pytest.mark.asyncio
    async def test_upload_skipped_when_dlp_not_applied(self, tmp_path: Path):
        mock_config = MagicMock()
        mock_config.dlp_applied_pre_translation = False
        mock_config.dlp_provider = None
        mock_config.working_dir = tmp_path

        with patch(
            "src.worker.services.model_attempt_orchestrator.get_translation_storage_repository"
        ) as mock_repo_getter:
            await _upload_sampled_attempt_artifacts(
                job_id="job-1",
                attempt_index=1,
                translation_config=mock_config,
            )
            mock_repo_getter.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_skipped_when_not_sampled(self, tmp_path: Path, monkeypatch):
        from src.config.constants import settings

        monkeypatch.setattr(settings, "TRACKING_SAMPLE_PERCENTAGE", 0)

        mock_config = MagicMock()
        mock_config.dlp_applied_pre_translation = True
        mock_config.working_dir = tmp_path

        with patch(
            "src.worker.services.model_attempt_orchestrator.get_translation_storage_repository"
        ) as mock_repo_getter:
            await _upload_sampled_attempt_artifacts(
                job_id="job-1",
                attempt_index=1,
                translation_config=mock_config,
            )
            mock_repo_getter.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_invoked_when_dlp_applied_and_sampled(
        self, tmp_path: Path, monkeypatch
    ):
        from src.config.constants import settings

        monkeypatch.setattr(settings, "TRACKING_SAMPLE_PERCENTAGE", 100)

        tracking_file = tmp_path / "translate_tracking.json"
        tracking_file.write_text(json.dumps({"page": []}))

        quality_file = tmp_path / "quality_report.json"
        quality_file.write_text(json.dumps({"final_score": 0.95}))

        mock_config = MagicMock()
        mock_config.dlp_applied_pre_translation = True
        mock_config.working_dir = tmp_path

        mock_storage = AsyncMock()
        with patch(
            "src.worker.services.model_attempt_orchestrator.get_translation_storage_repository",
            return_value=mock_storage,
        ):
            await _upload_sampled_attempt_artifacts(
                job_id="job-1",
                attempt_index=1,
                translation_config=mock_config,
            )
            mock_storage.upload_attempt_artifacts.assert_awaited_once_with(
                job_id="job-1",
                attempt_index=1,
                tracking_path=tracking_file,
                quality_report_path=quality_file,
            )
