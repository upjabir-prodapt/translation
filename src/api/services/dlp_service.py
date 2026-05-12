"""PII masking service with language-aware routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config.constants import settings


@dataclass(slots=True)
class DlpResult:
    """Masked output payload."""

    masked_chunks: list[str]
    token_rows: list[dict]
    dlp_provider: str


class DlpService:
    """Apply deterministic placeholder masking for PII-like patterns."""

    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")

    def select_provider(self, source_language: str) -> str:
        normalized = (source_language or "").lower()
        return "google_cloud_dlp" if normalized in {"en", "eng", "english"} else "vertex_ai_dlp"

    def _iter_chunk_windows(
        self, chunks: list[str], provider: str, max_chars_per_request: int
    ) -> list[tuple[int, int]]:
        """Return index windows that respect provider request character budgets."""
        if provider != "google_cloud_dlp":
            return [(0, len(chunks))]
        if max_chars_per_request <= 0:
            return [(0, len(chunks))]
        windows: list[tuple[int, int]] = []
        start = 0
        current_chars = 0
        for idx, chunk in enumerate(chunks):
            chunk_len = len(chunk or "")
            if start == idx:
                current_chars = chunk_len
                continue
            if current_chars + chunk_len > max_chars_per_request:
                windows.append((start, idx))
                start = idx
                current_chars = chunk_len
            else:
                current_chars += chunk_len
        windows.append((start, len(chunks)))
        return windows

    def mask_chunks(
        self,
        *,
        job_id: str,
        chunks: list[str],
        source_language: str,
        token_counter_start: int = 0,
    ) -> DlpResult:
        provider = self.select_provider(source_language)
        token_rows: list[dict] = []
        masked_chunks: list[str] = list(chunks)
        token_counter = max(0, int(token_counter_start))
        max_chars = int(settings.GOOGLE_DLP_MAX_CHARS_PER_REQUEST)

        for start, end in self._iter_chunk_windows(chunks, provider, max_chars):
            for chunk_index in range(start, end):
                masked_text = masked_chunks[chunk_index]
                for pattern, info_type in (
                    (self.EMAIL_PATTERN, "EMAIL_ADDRESS"),
                    (self.PHONE_PATTERN, "PHONE_NUMBER"),
                ):
                    for match in pattern.finditer(masked_text):
                        token_counter += 1
                        token = f"__DLP_TOKEN_{token_counter:04d}__"
                        original = match.group(0)
                        masked_text = masked_text.replace(original, token, 1)
                        token_rows.append(
                            {
                                "job_id": job_id,
                                "chunk_index": chunk_index,
                                "token": token,
                                "original_value": original,
                                "info_type": info_type,
                            }
                        )
                masked_chunks[chunk_index] = masked_text

        return DlpResult(masked_chunks=masked_chunks, token_rows=token_rows, dlp_provider=provider)

