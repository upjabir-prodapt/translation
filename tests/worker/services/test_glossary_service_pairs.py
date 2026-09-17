"""Loader behaviour for the language-pair keying and the read-side hygiene filter.

Two defects are pinned here, both from TRANSLATION_FIX_PLAN.md:

RC-3, cross-language contamination
    The loader read *every* source section and filtered only by target, so a
    French->Italian job inherited terms a Spanish->Italian job had written.

The unreachable-curated-glossary defect
    The pipeline passes canonical codes (``es``), while hand-authored sections
    key their translations by display name (``"Spanish"``). ``.get("es")``
    never matched ``{"Spanish": ...}``, so no curated term ever reached a
    translation and the only entries that could match were the auto-extracted
    ones -- i.e. exclusively the polluted ones.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from src.worker.services.glossary_service import GlossaryService


@pytest.fixture
def service():
    with patch("src.worker.services.glossary_service.get_storage_client"):
        return GlossaryService()


def _load(service, payload, **kwargs):
    with patch.object(service, "_download_glossary_json", return_value=payload):
        return service.load_domain_glossary(domain="commercial", **kwargs)


MIXED_KEYING = {
    "glossary": {
        # Hand-authored: display-name keys, the shape the loader could not read.
        "English": {
            "terms": [
                {
                    "source_term": "cross connect",
                    "translations": {"Spanish": "crossconexión"},
                }
            ],
            "preserve_as_is": ["Colt", "SLA"],
        },
        # Auto-extracted: canonical codes.
        "en": {
            "terms": [
                {
                    "source_term": "dark fibre",
                    "translations": {"es": "fibra oscura"},
                }
            ]
        },
        # A different source language, which must not leak into an en->es job.
        "fr": {
            "terms": [
                {
                    "source_term": "fibre noire",
                    "translations": {"es": "fibra oscura"},
                }
            ]
        },
    }
}


class TestLanguagePairKeying:
    def test_reads_display_name_and_code_keyed_translations(self, service):
        glossaries = _load(
            service, MIXED_KEYING, source_language="en", target_language_name="es"
        )
        sources = {e.source for e in glossaries[0].entries}
        assert "cross connect" in sources, "display-name keyed term still unreachable"
        assert "dark fibre" in sources, "code-keyed term not loaded"

    def test_other_source_language_does_not_leak(self, service):
        glossaries = _load(
            service, MIXED_KEYING, source_language="en", target_language_name="es"
        )
        sources = {e.source for e in glossaries[0].entries}
        assert "fibre noire" not in sources, (
            "a French source term reached an en->es job"
        )

    def test_source_section_selected_by_display_name_too(self, service):
        glossaries = _load(
            service, MIXED_KEYING, source_language="fr", target_language_name="es"
        )
        sources = {e.source for e in glossaries[0].entries}
        assert "fibre noire" in sources
        assert "cross connect" not in sources

    def test_preserve_as_is_becomes_do_not_translate_entries(self, service):
        glossaries = _load(
            service, MIXED_KEYING, source_language="en", target_language_name="es"
        )
        by_source = {e.source: e.target for e in glossaries[0].entries}
        # An identity pair is rejected everywhere else; here it is the intent.
        assert by_source.get("Colt") == "Colt"
        assert by_source.get("SLA") == "SLA"


class TestReadSideHygiene:
    """Pollution already in GCS must be inert without waiting for a purge."""

    POLLUTED = {
        "glossary": {
            "en": {
                "terms": [
                    {"source_term": "der", "translations": {"es": "der"}},
                    {"source_term": "integrity", "translations": {"es": "integrity"}},
                    {
                        "source_term": "__DLP_TOKEN_0001__",
                        "translations": {"es": "__DLP_TOKEN_0001__"},
                    },
                    {"source_term": "Elon Musk", "translations": {"es": "Elon Musk"}},
                    {
                        "source_term": "multilingual terminologist",
                        "translations": {"es": "terminólogo multilingüe"},
                    },
                    {
                        "source_term": "force majeure",
                        "translations": {"es": "fuerza mayor"},
                    },
                ]
            }
        }
    }

    def test_only_the_real_term_survives(self, service):
        glossaries = _load(
            service, self.POLLUTED, source_language="en", target_language_name="es"
        )
        assert [(e.source, e.target) for e in glossaries[0].entries] == [
            ("force majeure", "fuerza mayor")
        ]

    def test_a_wholly_polluted_glossary_loads_as_nothing(self, service):
        payload = {
            "glossary": {
                "en": {
                    "terms": [
                        {"source_term": "der", "translations": {"es": "der"}},
                        {"source_term": "los", "translations": {"es": "los"}},
                    ]
                }
            }
        }
        assert (
            _load(service, payload, source_language="en", target_language_name="es")
            == []
        )


def test_missing_source_section_yields_nothing(service):
    assert (
        _load(service, MIXED_KEYING, source_language="ja", target_language_name="es")
        == []
    )
