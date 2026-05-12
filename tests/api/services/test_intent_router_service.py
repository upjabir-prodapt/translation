
import pytest
from unittest.mock import MagicMock, patch
from src.api.services.intent_router_service import IntentRouterService
from pathlib import Path
import json
from datetime import datetime, UTC

@pytest.fixture
def service():
    return IntentRouterService()

class TestIntentRouterService:
    def test_build_intent(self, service):
        assert service.build_intent("legal", "en", "fr") == "Intent-Legal-EN-FR"
        assert service.build_intent("COMMERCIAL", "es", "ja") == "Intent-Commercial-ES-JA"

    @patch("src.api.services.intent_router_service.get_storage_client")
    @patch("src.api.services.intent_router_service.get_cache_file_path")
    def test_load_from_gcs(self, mock_path, mock_client, service):
        mock_blob = MagicMock()
        mock_blob.download_as_text.return_value = '{"models": []}'
        mock_client.return_value.bucket.return_value.blob.return_value = mock_blob
        mock_file = MagicMock(spec=Path)
        mock_path.return_value = mock_file
        
        res = service._load_from_gcs()
        assert res == {"models": []}
        mock_file.write_text.assert_called_once()

    def test_get_model_chain(self, service):
        with patch("src.api.services.intent_router_service.select_model_list") as mock_select:
            mock_select.return_value = ["m1"]
            assert service.get_model_chain(domain="d", source_lang="en", target_lang="fr") == ["m1"]

    def test_local_cache_fresh_not_exists(self, service):
        assert service._local_cache_fresh(Path("nonexistent")) is False

    def test_local_cache_fresh_exists_and_valid(self, service):
        mock_file = MagicMock(spec=Path)
        mock_file.exists.return_value = True
        # Set mtime to just now
        mock_file.stat.return_value.st_mtime = datetime.now(UTC).timestamp()
        assert service._local_cache_fresh(mock_file) is True

    def test_local_cache_fresh_exists_and_expired(self, service):
        mock_file = MagicMock(spec=Path)
        mock_file.exists.return_value = True
        # Set mtime to long ago
        mock_file.stat.return_value.st_mtime = (datetime.now(UTC) - service._cache_ttl * 2).timestamp()
        assert service._local_cache_fresh(mock_file) is False

    @patch("src.api.services.intent_router_service.IntentRouterService._load_from_local_fallback")
    @patch("src.api.services.intent_router_service.IntentRouterService._local_cache_fresh", return_value=True)
    def test_sync_model_selection_cache_no_force_fresh(self, mock_fresh, mock_local, service):
        mock_local.return_value = {"cached": True}
        res = service.sync_model_selection_cache(force=False)
        assert res == {"cached": True}
        assert service._config == {"cached": True}

    @patch("src.api.services.intent_router_service.IntentRouterService._load_from_local_fallback")
    def test_sync_model_selection_cache_gcs_failure(self, mock_local, service):
        mock_local.return_value = {"stale": True}
        with patch.object(service, "_load_from_gcs", side_effect=Exception("GCS Fail")):
            res = service.sync_model_selection_cache(force=True)
            assert res == {"stale": True}

    def test_get_config_in_memory_fresh(self, service):
        service._config = {"mem": True}
        service._last_loaded = datetime.now(UTC)
        res = service._get_config()
        assert res == {"mem": True}

    @patch("src.api.services.intent_router_service.IntentRouterService.sync_model_selection_cache")
    def test_get_config_refresh(self, mock_sync, service):
        service._config = {"old": True}
        service._last_loaded = datetime.now(UTC) - service._cache_ttl * 2
        mock_sync.return_value = {"new": True}
        res = service._get_config()
        assert res == {"new": True}
