"""Tests for DlpService behavior."""

from src.api.services.dlp_service import DlpService


def test_google_dlp_windows_respect_character_limit():
    service = DlpService()
    windows = service._iter_chunk_windows(
        chunks=["a" * 120000, "b" * 120000, "c" * 120000],
        provider="google_cloud_dlp",
        max_chars_per_request=300000,
    )
    assert windows == [(0, 2), (2, 3)]


def test_mask_chunks_accepts_token_counter_start():
    service = DlpService()
    result = service.mask_chunks(
        job_id="job-1",
        chunks=["Email me at alice@example.com"],
        source_language="en",
        token_counter_start=4,
    )
    assert result.masked_chunks == ["Email me at __DLP_TOKEN_0005__"]
    assert result.token_rows[0]["token"] == "__DLP_TOKEN_0005__"
