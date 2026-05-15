import pytest
from src.api.services.dlp_service import DlpService


@pytest.fixture
def service():
    return DlpService()


class TestDlpService:
    def test_select_provider(self, service):
        assert service.select_provider("en") == "google_cloud_dlp"
        assert service.select_provider("fr") == "vertex_ai_dlp"

    def test_iter_chunk_windows(self, service):
        chunks = ["abc", "def", "ghi"]
        # Max chars 4 -> window [(0, 1), (1, 2), (2, 3)]
        windows = service._iter_chunk_windows(chunks, "google_cloud_dlp", 4)
        assert windows == [(0, 1), (1, 2), (2, 3)]

        # Max chars 10 -> window [(0, 3)]
        windows = service._iter_chunk_windows(chunks, "google_cloud_dlp", 10)
        assert windows == [(0, 3)]

    def test_mask_chunks(self, service):
        chunks = ["My email is test@example.com", "Call me at +1234567890"]
        res = service.mask_chunks(job_id="job1", chunks=chunks, source_language="en")

        assert "__DLP_TOKEN_0001__" in res.masked_chunks[0]
        assert "__DLP_TOKEN_0002__" in res.masked_chunks[1]
        assert len(res.token_rows) == 2
        assert res.token_rows[0]["info_type"] == "EMAIL_ADDRESS"
        assert res.token_rows[1]["info_type"] == "PHONE_NUMBER"
