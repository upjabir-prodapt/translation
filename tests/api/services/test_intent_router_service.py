import pytest
from src.api.services.intent_router_service import IntentRouterService
from unittest.mock import MagicMock, patch
from pathlib import Path
import json

@pytest.fixture
def service():
    return IntentRouterService()

class TestIntentRouterService:
    def test_build_intent(self, service):
        assert service.build_intent("legal", "en", "fr") == "Intent-Legal-EN-FR"

    @patch("src.api.services.intent_router_service.get_storage_client")
    @patch("src.api.services.intent_router_service.get_cache_file_path")
    def test_load_from_gcs(self, mock_path, mock_client, service):
        mock_blob = MagicMock()
        mock_blob.download_as_text.return_value = '{"models": []}'
        mock_client.return_value.bucket.return_value.blob.return_value = mock_blob
        mock_path.return_value = MagicMock(spec=Path)
        
        res = service._load_from_gcs()
        assert res == {"models": []}

    def test_get_model_chain(self, service):
        with patch("src.api.services.intent_router_service.select_model_list") as mock_select:
            mock_select.return_value = ["m1"]
            assert service.get_model_chain(domain="d", source_lang="en", target_lang="fr") == ["m1"]

    def test_local_cache_fresh_not_exists(self, service):
        assert service._local_cache_fresh(Path("nonexistent")) is False
