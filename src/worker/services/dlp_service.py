"""PII masking service using Google Cloud DLP with regex fallback."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

from src.config.constants import settings


class DlpProvider(StrEnum):
    """Provider used to perform PII detection."""

    GOOGLE_CLOUD_DLP = "google_cloud_dlp"
    REGEX_FALLBACK = "regex_fallback"


logger = logging.getLogger(__name__)

# Comprehensive list of Google Cloud DLP built-in info types.
# https://cloud.google.com/sensitive-data-protection/docs/infotypes-reference
_DLP_INFO_TYPES: list[str] = [
    # ── Contact ──────────────────────────────────────────────────────────────
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "STREET_ADDRESS",
    # ── Identity ─────────────────────────────────────────────────────────────
    "PERSON_NAME",
    "DATE_OF_BIRTH",
    "AGE",
    # ── Financial ────────────────────────────────────────────────────────────
    "CREDIT_CARD_NUMBER",
    "CREDIT_CARD_TRACK_NUMBER",
    "IBAN_CODE",
    "SWIFT_CODE",
    "US_BANK_ROUTING_MICR",
    # ── US government identifiers ─────────────────────────────────────────────
    "US_SOCIAL_SECURITY_NUMBER",
    "US_PASSPORT",
    "US_DRIVERS_LICENSE_NUMBER",
    "US_INDIVIDUAL_TAXPAYER_IDENTIFICATION_NUMBER",
    "US_EMPLOYER_IDENTIFICATION_NUMBER",
    # ── UK ───────────────────────────────────────────────────────────────────
    "UK_NATIONAL_INSURANCE_NUMBER",
    "UK_TAXPAYER_REFERENCE",
    "UK_PASSPORT",
    "UK_DRIVERS_LICENSE_NUMBER",
    # ── Europe ───────────────────────────────────────────────────────────────
    "FRANCE_NIR",
    "FRANCE_PASSPORT",
    "FRANCE_CNI",
    "FRANCE_TAX_IDENTIFICATION_NUMBER",
    "GERMANY_IDENTITY_CARD_NUMBER",
    "GERMANY_PASSPORT",
    "GERMANY_TAX_IDENTIFICATION_NUMBER",
    "SPAIN_NIE_NUMBER",
    "SPAIN_NIF_NUMBER",
    "ITALY_FISCAL_CODE",
    "ITALY_PASSPORT",
    "NETHERLANDS_BSN_NUMBER",
    "NETHERLANDS_PASSPORT",
    "POLAND_PESEL_NUMBER",
    "POLAND_NATIONAL_ID_NUMBER",
    "PORTUGAL_CDC_NUMBER",
    "BELGIUM_NATIONAL_SUBJECT_CODE",
    "AUSTRIA_DRIVERS_LICENSE_NUMBER",
    "AUSTRIA_NATIONAL_ID_NUMBER",
    "AUSTRIA_PASSPORT",
    "SWEDEN_NATIONAL_ID_NUMBER",
    "SWEDEN_PASSPORT",
    "DENMARK_CPR_NUMBER",
    "FINLAND_NATIONAL_ID_NUMBER",
    "NORWAY_NI_NUMBER",
    "CZECH_PERSONAL_NUMBER",
    "CROATIA_PERSONAL_ID_NUMBER",
    "ROMANIA_PERSONAL_NUMERIC_CODE",
    "HUNGARY_TAX_IDENTIFICATION_NUMBER",
    "GREECE_TAX_REFERENCE_NUMBER",
    # ── Asia-Pacific ─────────────────────────────────────────────────────────
    "INDIA_AADHAAR_INDIVIDUAL",
    "INDIA_PAN_INDIVIDUAL",
    "INDIA_GST_INDIVIDUAL",
    "INDIA_DRIVERS_LICENSE_NUMBER",
    "INDIA_VOTER_ID",
    "INDIA_PASSPORT",
    "AUSTRALIA_TAX_FILE_NUMBER",
    "AUSTRALIA_DRIVERS_LICENSE",
    "AUSTRALIA_PASSPORT",
    "AUSTRALIA_MEDICARE_NUMBER",
    "CHINA_RESIDENT_ID_NUMBER",
    "CHINA_PASSPORT",
    "JAPAN_INDIVIDUAL_NUMBER",
    "JAPAN_BANK_ACCOUNT",
    "JAPAN_DRIVERS_LICENSE_NUMBER",
    "JAPAN_PASSPORT",
    "SOUTH_KOREA_RESIDENT_REGISTRATION_NUMBER",
    "SOUTH_KOREA_PASSPORT",
    "TAIWAN_PASSPORT",
    "SINGAPORE_NATIONAL_REGISTRATION_ID_NUMBER",
    "SINGAPORE_PASSPORT",
    "HONG_KONG_HKID_NUMBER",
    "THAILAND_NATIONAL_ID_NUMBER",
    "INDONESIA_NIK_NUMBER",
    "MALAYSIA_NATIONAL_REGISTRATION_INFORMATION_CARD",
    # ── Americas ─────────────────────────────────────────────────────────────
    "CANADA_SOCIAL_INSURANCE_NUMBER",
    "CANADA_PASSPORT",
    "CANADA_DRIVERS_LICENSE_NUMBER",
    "BRAZIL_CPF_NUMBER",
    "BRAZIL_RG_NUMBER",
    "BRAZIL_PASSPORT",
    "MEXICO_CURP_NUMBER",
    "MEXICO_PASSPORT",
    "MEXICO_VOTER_ID",
    "ARGENTINA_DNI_NUMBER",
    "CHILE_CDI_NUMBER",
    "COLOMBIA_CC_NUMBER",
    "VENEZUELA_CDI_NUMBER",
    "PERU_DNI_NUMBER",
    # ── Network / Technical ───────────────────────────────────────────────────
    "IP_ADDRESS",
    "MAC_ADDRESS",
    # ── Medical / Healthcare ──────────────────────────────────────────────────
    "MEDICAL_RECORD_NUMBER",
    "US_HEALTHCARE_NPI",
    "US_DEA_NUMBER",
    "FDA_CODE",
    # ── Credentials / Secrets ─────────────────────────────────────────────────
    "AUTH_TOKEN",
    "BASIC_AUTH_HEADER",
    "ENCRYPTION_KEY",
    "GCP_CREDENTIALS",
    "GCP_API_KEY",
    "AWS_CREDENTIALS",
    "AZURE_AUTH_TOKEN",
    "JSON_WEB_TOKEN",
    "HTTP_COOKIE",
    "OAUTH_CLIENT_SECRET",
    "PASSWORD",
    "WEAK_PASSWORD_HASH",
    "SSL_CERTIFICATE",
    "XSRF_TOKEN",
]

# Regex fallback patterns applied when Google Cloud DLP is unavailable or fails.
# (compiled_pattern, info_type_name) — ordered so broader patterns come after narrower ones.
_REGEX_FALLBACK: list[tuple[re.Pattern[str], str]] = [
    # ── Contact ──────────────────────────────────────────────────────────────
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "EMAIL_ADDRESS",
    ),
    # E.164 compact international phone: +12025551234
    (
        re.compile(r"\+[1-9]\d{6,14}\b"),
        "PHONE_NUMBER",
    ),
    # International phone with formatting: +1 (212) 555-1212 or +44 20 7946 0958
    (
        re.compile(r"\+[1-9][\d\s().\-]{6,20}\d\b"),
        "PHONE_NUMBER",
    ),
    # US/CA without country code: (xxx) xxx-xxxx  or  xxx-xxx-xxxx
    (
        re.compile(r"\b(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}\b"),
        "PHONE_NUMBER",
    ),
    # ── Network ──────────────────────────────────────────────────────────────
    (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "IP_ADDRESS",
    ),
    (
        re.compile(r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b"),
        "MAC_ADDRESS",
    ),
    # ── Financial ────────────────────────────────────────────────────────────
    # Visa / Mastercard / Amex / Discover / JCB / Diners
    (
        re.compile(
            r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}"
            r"|3(?:0[0-5]|[68]\d)\d{11}|6(?:011|5\d{2})\d{12}"
            r"|(?:2131|1800|35\d{3})\d{11})\b"
        ),
        "CREDIT_CARD_NUMBER",
    ),
    # IBAN: 2-letter country code, 2 check digits, up to 30 alphanumeric
    (
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"),
        "IBAN_CODE",
    ),
    # SWIFT / BIC code (8 or 11 chars)
    (
        re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
        "SWIFT_CODE",
    ),
    # ── US government identifiers ─────────────────────────────────────────────
    # SSN: 000-00-0000, excluding invalid prefixes
    (
        re.compile(r"\b(?!000|666|9\d{2})\d{3}[-\s](?!00)\d{2}[-\s](?!0000)\d{4}\b"),
        "US_SOCIAL_SECURITY_NUMBER",
    ),
    # US Passport: one letter + 8 digits
    (
        re.compile(r"\b[A-Z]\d{8}\b"),
        "US_PASSPORT",
    ),
    # EIN: XX-XXXXXXX
    (
        re.compile(r"\b\d{2}-\d{7}\b"),
        "US_EMPLOYER_IDENTIFICATION_NUMBER",
    ),
    # ── UK ───────────────────────────────────────────────────────────────────
    # NI number: XX 999999 X  (excluding known invalid prefixes)
    (
        re.compile(r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z]{2}\s?\d{6}\s?[A-D]\b"),
        "UK_NATIONAL_INSURANCE_NUMBER",
    ),
    # ── India ─────────────────────────────────────────────────────────────────
    # Aadhaar: 12 digits, first digit 2-9
    (
        re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
        "INDIA_AADHAAR_INDIVIDUAL",
    ),
    # PAN: AAAAA9999A
    (
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        "INDIA_PAN_INDIVIDUAL",
    ),
    # ── Credentials ──────────────────────────────────────────────────────────
    # JWT: header.payload.signature (starts with eyJ base64)
    (
        re.compile(r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+"),
        "JSON_WEB_TOKEN",
    ),
    # HTTP Basic Auth header value
    (
        re.compile(r"\bBasic\s+[A-Za-z0-9+/]{10,}={0,2}"),
        "BASIC_AUTH_HEADER",
    ),
    # HTTP Bearer token
    (
        re.compile(r"\bBearer\s+[A-Za-z0-9\-_.]{10,}"),
        "AUTH_TOKEN",
    ),
    # GCP API key (AIza...)
    (
        re.compile(r"\bAIza[A-Za-z0-9\-_]{35}\b"),
        "GCP_API_KEY",
    ),
    # AWS Access Key ID
    (
        re.compile(r"\b(?:AKIA|ASIA|AROA|ANPA|ANVA|AIPA)[A-Z0-9]{16}\b"),
        "AWS_CREDENTIALS",
    ),
]


@dataclass(slots=True)
class DlpResult:
    """Masked output payload."""

    masked_chunks: list[str]
    token_rows: list[dict]
    dlp_provider: DlpProvider


class DlpService:
    """Apply deterministic placeholder masking for PII patterns.

    Attempts Google Cloud DLP API first (covering all built-in info types);
    falls back to regex when the library is not installed or the API call fails.
    """

    # Class-level caches shared across instances.
    _dlp_client: object = None
    _dlp_available: bool | None = None  # None = not yet probed
    _valid_info_types: list[str] | None = None  # filtered to what this region supports

    def select_provider(self) -> DlpProvider:
        return DlpProvider.GOOGLE_CLOUD_DLP

    def _get_dlp_client(self) -> object | None:
        """Return a lazily-initialised DLP client, or None if unavailable."""
        if DlpService._dlp_available is False:
            return None
        if DlpService._dlp_client is not None:
            return DlpService._dlp_client
        try:
            from google.cloud import dlp_v2  # noqa: PLC0415

            DlpService._dlp_client = dlp_v2.DlpServiceClient()
            DlpService._dlp_available = True
            logger.info("[DLP] Google Cloud DLP client initialised successfully")
            return DlpService._dlp_client
        except ImportError:
            logger.warning(
                "[DLP] google-cloud-dlp is not installed; using regex fallback"
            )
            DlpService._dlp_available = False
            return None
        except Exception as exc:
            logger.warning(
                "[DLP] Client init failed (%s: %s); using regex fallback",
                type(exc).__name__,
                exc,
            )
            DlpService._dlp_available = False
            return None

    def _get_valid_info_types(self, client) -> list[str]:
        """Return the subset of _DLP_INFO_TYPES supported at this project/location.

        Result is cached at class level so the list_info_types call happens only once.
        """
        if DlpService._valid_info_types is not None:
            return DlpService._valid_info_types

        try:
            from google.cloud import dlp_v2  # noqa: PLC0415

            response = client.list_info_types(
                request=dlp_v2.ListInfoTypesRequest(parent="locations/global")
            )
            available = {it.name for it in response.info_types}
            valid = [t for t in _DLP_INFO_TYPES if t in available]
            skipped = [t for t in _DLP_INFO_TYPES if t not in available]
            logger.info(
                "[DLP] Info types available in this region: %d/%d (skipped %d: %s)",
                len(valid),
                len(_DLP_INFO_TYPES),
                len(skipped),
                skipped if skipped else "none",
            )
            DlpService._valid_info_types = valid
        except Exception as exc:
            logger.warning(
                "[DLP] Could not validate info types (%s); using full list", exc
            )
            DlpService._valid_info_types = list(_DLP_INFO_TYPES)

        return DlpService._valid_info_types

    def _iter_chunk_windows(
        self, chunks: list[str], provider: DlpProvider | str, max_chars_per_request: int
    ) -> list[tuple[int, int]]:
        """Return index windows that respect provider request character budgets."""
        if provider != DlpProvider.GOOGLE_CLOUD_DLP:
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

    def _mask_with_google_dlp(
        self,
        text: str,
        job_id: str,
        chunk_index: int,
        token_counter: int,
    ) -> tuple[str, list[dict], int]:
        """Call Google Cloud DLP API to find and replace PII with tokens.

        Replaces right-to-left so byte offsets from the API remain valid.
        Raises on any API or library error — caller handles fallback.
        """
        from google.cloud import dlp_v2  # noqa: PLC0415

        client = self._get_dlp_client()
        parent = f"projects/{settings.GOOGLE_CLOUD_PROJECT}/locations/global"
        likelihood = getattr(
            dlp_v2.Likelihood,
            settings.GOOGLE_DLP_MIN_LIKELIHOOD,
            dlp_v2.Likelihood.UNLIKELY,
        )
        info_types = self._get_valid_info_types(client)
        response = client.inspect_content(
            request=dlp_v2.InspectContentRequest(
                parent=parent,
                inspect_config=dlp_v2.InspectConfig(
                    info_types=[dlp_v2.InfoType(name=t) for t in info_types],
                    min_likelihood=likelihood,
                    include_quote=True,
                    limits=dlp_v2.InspectConfig.FindingLimits(
                        max_findings_per_request=0,  # 0 = unlimited
                    ),
                ),
                item=dlp_v2.ContentItem(value=text),
            )
        )

        findings = list(response.result.findings)
        if not findings:
            return text, [], token_counter

        findings.sort(key=lambda f: f.location.byte_range.start, reverse=True)
        text_bytes = text.encode("utf-8")
        token_rows: list[dict] = []
        for finding in findings:
            b_start = finding.location.byte_range.start
            b_end = finding.location.byte_range.end
            original = text_bytes[b_start:b_end].decode("utf-8", errors="replace")
            if not original.strip():
                continue
            token_counter += 1
            token = f"__DLP_TOKEN_{token_counter:04d}__"
            text_bytes = (
                text_bytes[:b_start] + token.encode("utf-8") + text_bytes[b_end:]
            )
            token_rows.append(
                {
                    "job_id": job_id,
                    "chunk_index": chunk_index,
                    "token": token,
                    "original_value": original,
                    "info_type": finding.info_type.name,
                }
            )
        token_rows.reverse()  # restore document order
        return text_bytes.decode("utf-8", errors="replace"), token_rows, token_counter

    def _mask_with_regex(
        self,
        text: str,
        job_id: str,
        chunk_index: int,
        token_counter: int,
    ) -> tuple[str, list[dict], int]:
        """Detect and mask PII using compiled regex patterns.

        Deduplicates overlapping matches (first match wins), then replaces
        right-to-left to keep character positions valid.
        """
        raw: list[tuple[int, int, str, str]] = []
        for pattern, info_type in _REGEX_FALLBACK:
            for m in pattern.finditer(text):
                raw.append((m.start(), m.end(), m.group(0), info_type))

        if not raw:
            return text, [], token_counter

        # Sort by start position; skip any match that overlaps the previous one
        raw.sort(key=lambda x: x[0])
        resolved: list[tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, original, info_type in raw:
            if start >= last_end:
                resolved.append((start, end, original, info_type))
                last_end = end

        token_rows: list[dict] = []
        for start, end, original, info_type in reversed(resolved):
            token_counter += 1
            token = f"__DLP_TOKEN_{token_counter:04d}__"
            text = text[:start] + token + text[end:]
            token_rows.append(
                {
                    "job_id": job_id,
                    "chunk_index": chunk_index,
                    "token": token,
                    "original_value": original,
                    "info_type": info_type,
                }
            )
        token_rows.reverse()  # restore document order
        return text, token_rows, token_counter

    def mask_chunks(
        self,
        *,
        job_id: str,
        chunks: list[str],
        source_language: str,
        token_counter_start: int = 0,
    ) -> DlpResult:
        client = self._get_dlp_client()
        use_google_dlp = client is not None and bool(
            getattr(settings, "GOOGLE_DLP_ENABLED", True)
        )
        provider = (
            DlpProvider.GOOGLE_CLOUD_DLP
            if use_google_dlp
            else DlpProvider.REGEX_FALLBACK
        )
        logger.info(
            "[DLP] Provider selected: %s | job=%s | chunks=%d | enabled=%s | client=%s",
            provider.value,
            job_id,
            len(chunks),
            settings.GOOGLE_DLP_ENABLED,
            "ready" if use_google_dlp else "unavailable",
        )
        token_counter = max(0, int(token_counter_start))
        masked_chunks: list[str] = list(chunks)
        all_token_rows: list[dict] = []

        for chunk_index, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                continue

            if use_google_dlp:
                try:
                    masked, rows, token_counter = self._mask_with_google_dlp(
                        chunk, job_id, chunk_index, token_counter
                    )
                    masked_chunks[chunk_index] = masked
                    all_token_rows.extend(rows)
                    continue
                except Exception:
                    logger.warning(
                        "Google Cloud DLP failed for chunk %d (job=%s); falling back to regex",
                        chunk_index,
                        job_id,
                        exc_info=True,
                    )
                    provider = DlpProvider.REGEX_FALLBACK
                    logger.info("[DLP] Switched to: %s", provider.value)

            masked, rows, token_counter = self._mask_with_regex(
                chunk, job_id, chunk_index, token_counter
            )
            masked_chunks[chunk_index] = masked
            all_token_rows.extend(rows)

        return DlpResult(
            masked_chunks=masked_chunks,
            token_rows=all_token_rows,
            dlp_provider=provider,
        )
