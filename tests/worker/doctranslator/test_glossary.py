"""Tests for shared glossary primitives (`TermLanguageDropTracker`).

Shared between the DOCX (`format/docx/term_extractor.py`) and PDF
(`format/pdf/document_il/midend/automatic_term_extractor.py`) term
extractors so a term's self-reported `src_lang` is validated/dropped
identically in both, and a systemic drop-rate regression is visible above
debug level.
"""

import logging

from src.worker.doctranslator.glossary import TermLanguageDropTracker


class TestResolve:
    def test_supported_language_normalizes_and_is_not_counted_as_dropped(self):
        tracker = TermLanguageDropTracker()
        assert tracker.resolve("en", source_term="cat") == "en"
        assert tracker.candidate_count == 1
        assert tracker.dropped_count == 0

    def test_alias_normalizes_to_canonical_code(self):
        tracker = TermLanguageDropTracker()
        assert tracker.resolve("English", source_term="cat") == "en"

    def test_unrecognized_language_is_dropped(self):
        tracker = TermLanguageDropTracker()
        assert tracker.resolve("ru", source_term="договор") is None
        assert tracker.candidate_count == 1
        assert tracker.dropped_count == 1

    def test_missing_src_lang_is_dropped(self):
        tracker = TermLanguageDropTracker()
        assert tracker.resolve("", source_term="cat") is None
        assert tracker.dropped_count == 1


class TestWarnIfHighDropRate:
    def test_no_drops_logs_nothing(self, caplog):
        tracker = TermLanguageDropTracker()
        tracker.resolve("en", source_term="a")
        tracker.resolve("fr", source_term="b")
        with caplog.at_level(logging.WARNING):
            tracker.warn_if_high_drop_rate(extractor_name="test")
        assert caplog.text == ""

    def test_low_drop_rate_stays_below_debug_only(self, caplog):
        """A handful of drops per document is normal (an occasional
        hallucinated code) and must not escalate to WARNING."""
        tracker = TermLanguageDropTracker(high_drop_rate=0.5)
        for _ in range(9):
            tracker.resolve("en", source_term="a")
        tracker.resolve("ru", source_term="b")
        with caplog.at_level(logging.WARNING):
            tracker.warn_if_high_drop_rate(extractor_name="test")
        assert caplog.text == ""

    def test_high_drop_rate_escalates_to_warning(self, caplog):
        tracker = TermLanguageDropTracker(high_drop_rate=0.5)
        for _ in range(6):
            tracker.resolve("ru", source_term="x")
        for _ in range(4):
            tracker.resolve("en", source_term="y")
        with caplog.at_level(logging.WARNING):
            tracker.warn_if_high_drop_rate(extractor_name="DOCX term extraction")
        assert "DOCX term extraction" in caplog.text
        assert "6/10" in caplog.text

    def test_no_candidates_logs_nothing(self, caplog):
        tracker = TermLanguageDropTracker()
        with caplog.at_level(logging.WARNING):
            tracker.warn_if_high_drop_rate(extractor_name="test")
        assert caplog.text == ""
