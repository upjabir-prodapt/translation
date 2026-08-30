"""Tests for src/worker/services/token_verification_service.py.

implementation_plan.md Phase D.6.1 (EC-01/EC-02/EC-14): best-effort,
non-blocking verification that protected tokens (digits, currency, URLs,
emails, ticket IDs, CIDR blocks, CLI flags) present in the source text
still appear verbatim in the translated output.
"""

from __future__ import annotations

import logging

from src.worker.services.token_verification_service import TokenVerificationResult
from src.worker.services.token_verification_service import (
    log_token_verification_warning,
)
from src.worker.services.token_verification_service import verify_protected_tokens


class TestVerifyProtectedTokens:
    def test_empty_source_or_translated_returns_empty_result(self):
        assert verify_protected_tokens("", "anything").checked_token_count == 0
        assert verify_protected_tokens("anything", "").checked_token_count == 0
        assert verify_protected_tokens("", "").passed is True

    def test_prose_with_no_protected_tokens_passes_trivially(self):
        result = verify_protected_tokens(
            "This is an ordinary sentence with no special tokens.",
            "Ceci est une phrase ordinaire sans jetons speciaux.",
        )
        assert result.checked_token_count == 0
        assert result.passed is True

    def test_url_preserved_passes(self):
        source = "Visit https://example.com/docs for more info."
        translated = "Visitez https://example.com/docs pour plus d'informations."
        result = verify_protected_tokens(source, translated)
        assert result.passed is True
        assert result.checked_token_count >= 1

    def test_url_dropped_is_detected(self):
        source = "Visit https://example.com/docs for more info."
        translated = "Visitez notre site pour plus d'informations."
        result = verify_protected_tokens(source, translated)
        assert result.passed is False
        assert "https://example.com/docs" in result.missing_by_category["url"]

    def test_email_dropped_is_detected(self):
        source = "Contact support@example.com for help."
        translated = "Contactez notre support pour de l'aide."
        result = verify_protected_tokens(source, translated)
        assert result.passed is False
        assert "support@example.com" in result.missing_by_category["email"]

    def test_currency_preserved_passes(self):
        source = "The total is $1,234.56 due on receipt."
        translated = "Le total est de $1,234.56 a payer a reception."
        result = verify_protected_tokens(source, translated)
        assert result.passed is True

    def test_currency_dropped_is_detected(self):
        source = "The total is $1,234.56 due on receipt."
        translated = "Le montant est du a reception."
        result = verify_protected_tokens(source, translated)
        assert result.passed is False
        assert "currency" in result.missing_by_category

    def test_cidr_block_preserved_passes(self):
        source = "Allow access from 10.0.0.0/8 only."
        translated = "Autoriser l'acces depuis 10.0.0.0/8 uniquement."
        result = verify_protected_tokens(source, translated)
        assert result.passed is True

    def test_cidr_block_dropped_is_detected(self):
        source = "Allow access from 10.0.0.0/8 only."
        translated = "Autoriser l'acces depuis le reseau local uniquement."
        result = verify_protected_tokens(source, translated)
        assert result.passed is False
        assert "10.0.0.0/8" in result.missing_by_category["cidr"]

    def test_ticket_id_preserved_passes(self):
        source = "See ticket PROJ-1234 for details, also #4567."
        translated = "Voir le ticket PROJ-1234 pour plus de details, aussi #4567."
        result = verify_protected_tokens(source, translated)
        assert result.passed is True

    def test_ticket_id_dropped_is_detected(self):
        source = "See ticket PROJ-1234 for details."
        translated = "Voir le ticket pour plus de details."
        result = verify_protected_tokens(source, translated)
        assert result.passed is False
        assert "PROJ-1234" in result.missing_by_category["ticket_id"]

    def test_cli_flag_preserved_passes(self):
        source = "Run the command with --verbose enabled."
        translated = "Executez la commande avec --verbose active."
        result = verify_protected_tokens(source, translated)
        assert result.passed is True

    def test_cli_flag_dropped_is_detected(self):
        source = "Run the command with --verbose enabled."
        translated = "Executez la commande en mode detaille."
        result = verify_protected_tokens(source, translated)
        assert result.passed is False
        assert "--verbose" in result.missing_by_category["cli_flag"]

    def test_bare_digits_preserved_passes(self):
        source = "Order 42 items by Friday."
        translated = "Commandez 42 articles avant vendredi."
        result = verify_protected_tokens(source, translated)
        assert result.passed is True

    def test_bare_digits_dropped_is_detected(self):
        source = "Order 42 items by Friday."
        translated = "Commandez plusieurs articles avant vendredi."
        result = verify_protected_tokens(source, translated)
        assert result.passed is False
        assert "42" in result.missing_by_category["digits"]

    def test_dlp_masking_token_syntax_is_never_flagged(self):
        """__DLP_TOKEN_NNNN__ is masked PII, not a protected token this
        checker should ever look for -- it has its own dedicated
        verbatim-copy instruction to the LLM elsewhere in the pipeline."""
        source = "Contact __DLP_TOKEN_0001__ for help."
        translated = "Contactez __DLP_TOKEN_0001__ pour de l'aide."
        result = verify_protected_tokens(source, translated)
        # No category should ever be "dlp_token" -- confirms the DLP
        # token pattern was deliberately excluded from _TOKEN_PATTERNS.
        assert "dlp_token" not in result.missing_by_category

    def test_duplicate_tokens_in_source_are_deduplicated(self):
        source = "Visit https://example.com twice: https://example.com again."
        translated = "No links here at all."
        result = verify_protected_tokens(source, translated)
        assert result.missing_by_category["url"] == ["https://example.com"]

    def test_multiple_categories_missing_simultaneously(self):
        source = "Email admin@example.com or call re: ticket INC-9999, $50 owed."
        translated = "Contactez-nous pour plus d'informations."
        result = verify_protected_tokens(source, translated)
        assert result.passed is False
        assert set(result.missing_by_category.keys()) >= {
            "email",
            "ticket_id",
            "currency",
        }

    def test_result_to_dict_shape(self):
        result = verify_protected_tokens(
            "Visit https://example.com now.", "No link here."
        )
        as_dict = result.to_dict()
        assert as_dict["passed"] is False
        assert as_dict["checked_token_count"] == 1
        assert as_dict["missing_count"] == 1
        assert "url" in as_dict["missing_by_category"]


class TestLogTokenVerificationWarning:
    def test_no_warning_logged_when_passed(self, caplog):
        result = TokenVerificationResult()
        with caplog.at_level(logging.WARNING):
            log_token_verification_warning(result, job_id="job-1", attempt_index=1)
        assert caplog.text == ""

    def test_warning_logged_when_tokens_missing(self, caplog):
        result = verify_protected_tokens(
            "Visit https://example.com now.", "No link here."
        )
        with caplog.at_level(logging.WARNING):
            log_token_verification_warning(result, job_id="job-1", attempt_index=2)
        assert "Protected-token verification" in caplog.text
        assert "job-1" in caplog.text
        assert "attempt=2" in caplog.text
