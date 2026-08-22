from datetime import UTC
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.services.glossary_service import GlossaryService


@pytest.fixture
def service():
    with patch("src.worker.services.glossary_service.get_storage_client"):
        return GlossaryService()


class TestGlossaryService:
    def test_get_local_glossary_path(self, service):
        path = service._get_local_glossary_path("test_domain")
        assert "test_domain" in str(path)

    def test_local_cache_fresh_not_exists(self, service):
        assert service._local_cache_fresh(Path("nonexistent")) is False

    def test_local_cache_fresh_exists(self, service):
        mock_file = MagicMock(spec=Path)
        mock_file.stat.return_value.st_mtime = datetime.now(UTC).timestamp()
        assert service._local_cache_fresh(mock_file) is True

    def test_load_local_glossary_json_exists(self, service):
        with patch.object(
            service, "_get_local_glossary_path", return_value=Path("fake.json")
        ):
            with patch.object(service, "_local_cache_fresh", return_value=True):
                with patch(
                    "src.worker.services.glossary_service.Path.exists",
                    return_value=True,
                ):
                    with patch(
                        "src.worker.services.glossary_service.Path.read_text",
                        return_value='{"terms": []}',
                    ):
                        res = service._load_local_glossary_json("domain")
                        assert res == {"terms": []}

    def test_load_local_glossary_json_invalid(self, service):
        with patch.object(
            service, "_get_local_glossary_path", return_value=Path("fake.json")
        ):
            with patch.object(service, "_local_cache_fresh", return_value=True):
                with patch(
                    "src.worker.services.glossary_service.Path.exists",
                    return_value=True,
                ):
                    with patch(
                        "src.worker.services.glossary_service.Path.read_text",
                        return_value="invalid json",
                    ):
                        assert service._load_local_glossary_json("domain") is None

    def test_load_domain_glossary_success(self, service):
        data = {
            "glossary": {
                "en": {
                    "terms": [{"source_term": "court", "translations": {"fr": "cour"}}]
                }
            }
        }
        with patch.object(service, "_download_glossary_json", return_value=data):
            res = service.load_domain_glossary(
                domain="domain", target_language_name="fr"
            )
            assert len(res) == 1
            assert res[0].entries[0].source == "court"

    def test_load_domain_glossary_failure(self, service):
        with patch.object(
            service, "_download_glossary_json", side_effect=Exception("Fail")
        ):
            assert (
                service.load_domain_glossary(domain="domain", target_language_name="fr")
                == []
            )

    def test_prefetch_domain_glossary(self, service):
        with patch.object(service, "_download_glossary_json") as mock_down:
            assert service.prefetch_domain_glossary("domain") is True
            mock_down.assert_called_once_with("domain", refresh=False)

    def test_download_glossary_json_success(self, service, tmp_path):
        with patch(
            "src.worker.services.glossary_service.get_storage_client"
        ) as mock_client:
            mock_blob = MagicMock()
            mock_blob.download_as_text.return_value = '{"terms": []}'
            mock_client.return_value.bucket.return_value.blob.return_value = mock_blob

            p = tmp_path / "gloss.json"
            with patch.object(service, "_get_local_glossary_path", return_value=p):
                res = service._download_glossary_json("domain", refresh=True)
                assert res == {"terms": []}
                assert p.exists()
