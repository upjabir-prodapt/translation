import pytest
from src.worker.services.dlp_service import DlpProvider
from src.worker.services.dlp_service import DlpService
from src.worker.services.dlp_service import _expand_to_word_boundaries
from src.worker.services.dlp_tokens import build_token_map
from src.worker.services.dlp_tokens import restore_tokens


@pytest.fixture
def service():
    return DlpService()


class TestDlpService:
    def test_select_provider(self, service):
        assert service.select_provider() == DlpProvider.GOOGLE_CLOUD_DLP

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


class TestGermanMaskingBoundaries:
    """A placeholder must never be emitted welded to surrounding word
    characters: German inflection and compounding make DLP report a finding
    that covers only part of a word, and a welded placeholder is read by the
    model as a German word and reformatted, breaking restoration."""

    def _mask(self, service, text):
        return DlpService._mask_with_regex(service, text, "job", 0, 0)

    def test_span_is_widened_to_whole_word(self):
        text = "Unterschrift des Mustermanns hier"
        start = text.index("Mustermann")
        assert _expand_to_word_boundaries(text, start, start + len("Mustermann")) == (
            start,
            start + len("Mustermanns"),
        )

    def test_span_is_widened_across_a_compound_hyphen(self):
        text = "Herr Müller-Schmidt kommt"
        start = text.index("Müller")
        new_start, new_end = _expand_to_word_boundaries(
            text, start, start + len("Müller")
        )
        assert text[new_start:new_end] == "Müller-Schmidt"

    def test_a_trailing_hyphen_is_not_swallowed(self):
        text = "Müller- und Schmidt-Gruppe"
        new_start, new_end = _expand_to_word_boundaries(text, 0, len("Müller"))
        assert text[new_start:new_end] == "Müller"

    def test_masked_token_is_never_adjacent_to_a_letter(self, service):
        masked, rows, _ = self._mask(
            service, "Die IBAN DE89370400440532013000 ist gesperrt."
        )
        assert "__DLP_TOKEN_0001__" in masked
        assert rows[0]["original_value"] == "DE89370400440532013000"

    def test_masking_round_trips_exactly(self, service):
        text = "Rufen Sie +49 30 901820 an oder mailen Sie max@example.de."
        masked, rows, _ = self._mask(service, text)
        restored, _count = restore_tokens(masked, build_token_map(rows))
        assert restored == text


class TestSwiftCodeFalsePositives:
    """The BIC regex is structurally just "an 8- or 11-character ALL-CAPS
    run", which matches ordinary capitalised German words."""

    def _info_types(self, service, text):
        _masked, rows, _ = DlpService._mask_with_regex(service, text, "job", 0, 0)
        return [row["info_type"] for row in rows]

    def test_german_all_caps_word_is_not_masked_as_a_bic(self, service):
        assert "SWIFT_CODE" not in self._info_types(
            service, "Siehe ANTRAGES-Formular und VERTRAGS-Klausel."
        )

    def test_bic_with_swift_context_is_still_masked(self, service):
        assert "SWIFT_CODE" in self._info_types(
            service, "SWIFT-Code DEUTDEFF für die Überweisung."
        )

    def test_bic_with_bic_context_is_still_masked(self, service):
        assert "SWIFT_CODE" in self._info_types(service, "BIC: COBADEFFXXX")
