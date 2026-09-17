"""Regression tests for the glossary hygiene gate.

Each case in `REJECTED` is a term pair taken verbatim from the glossary JSONs
that were live during the UAT round analysed in TRANSLATION_FIX_PLAN.md, or a
near variant of one. They are the defects the reviewers actually reported --
"residual source-language words" was the single most common complaint of the
round, and every one of those words reached the model as a standing glossary
instruction to leave it alone.

`ACCEPTED` exists to keep the gate honest in the other direction: it is easy to
reject junk by rejecting everything.
"""

from __future__ import annotations

import pytest
from src.config.glossary_hygiene import Trust
from src.config.glossary_hygiene import evaluate_term_pair
from src.config.glossary_hygiene import is_acceptable_term_pair
from src.config.glossary_hygiene import sanitize_term_pairs

# (source, target, expected_reason)
REJECTED = [
    # Identity pairs -- the direct cause of the residual-word findings.
    ("der", "der", "identity"),
    ("los", "los", "identity"),
    ("que", "que", "identity"),
    ("des", "des", "identity"),
    ("vos", "vos", "identity"),
    ("fecha", "fecha", "identity"),
    ("Sehr", "Sehr", "identity"),
    ("Damen", "Damen", "identity"),
    ("integrity", "integrity", "identity"),
    ("Colt", "Colt", "identity"),
    ("API", "API", "identity"),
    ("network", "network", "identity"),
    # Function words, even where the pair is not an identity.
    ("una", "eine", "too_short"),
    ("prima del", "vor dem", "function_word"),
    ("a pesar de ello", "trotzdem", "function_word"),
    # Generic document furniture, from test and injection documents.
    ("password", "パスワード", "generic_non_term"),
    ("manager", "gestor", "generic_non_term"),
    # Masked customer data. `finance.json` shipped a Japanese company name with
    # a DLP token embedded in it.
    ("__DLP_TOKEN_0001__", "__DLP_TOKEN_0001__", "placeholder"),
    ("DLP_TOKEN_0028", "トークン", "placeholder"),
    ("株式会社__DLP_TOKEN_0008__", "Cable Television", "placeholder"),
    # The extractor reading its own prompt: `legal.json` opens with seven of these.
    ("multilingual terminologist", "terminologue multilingue", "prompt_echo"),
    ("named entities", "entités nommées", "prompt_echo"),
    ("reference glossary", "glossaire de référence", "prompt_echo"),
    # Personal data, from prompt-injection test documents and signature blocks.
    ("Elon Musk", "イーロン・マスク", "person_name"),
    ("Mukul Gupta", "Mukul Gupta", "identity"),
    ("Priya Fenwick", "Priya Fenwick", "identity"),
    # Document instances rather than terminology.
    ("28th December 2022", "28. Dezember 2022", "date_literal"),
    ("01 July 2026", "1er juillet 2026", "date_literal"),
    ("LON-PE-01", "LON-PE-02", "instance_value"),
    ("LLD v2.0.1", "LLD v2.1.0", "instance_value"),
    ("v16.10 to v16.12 upgrade", "Upgrade", "instance_value"),
    # Headings and enumerations, not terms.
    ("policy, guidance, template and process set", "x", "heading"),
    (
        "Colt AI Build Platform Integration and Governance Advancement Statement",
        "y",
        "too_many_words",
    ),
    # Degenerate output.
    ("", "algo", "empty"),
    ("term", "", "empty"),
    ("bq", "bq", "identity"),
]

ACCEPTED = [
    ("force majeure", "fuerza mayor"),
    ("liquidated damages", "penale contrattuale"),
    ("Network Address Translation", "Traducción de Direcciones de Red"),
    ("cross connect", "crossconexión"),
    ("incident management", "gestión de incidencias"),
    ("low latency", "baja latencia"),
    ("dark fibre", "fibra oscura"),
    ("Legal Entity", "リーガルエンティティ"),
    ("Account Executive", "referente commerciale"),
    ("split percentage", "分割率"),
    ("managed router", "router gestionado"),
    ("load sharing", "balanceo de carga"),
    ("cable landing station", "estación de amarre de cable submarino"),
    # Acronyms are terminology and are exempt from the length rules.
    ("IVR", "SVI"),
    ("NAT", "Traducción de Direcciones de Red"),
    ("MRC", "canone mensile ricorrente"),
    # CJK is denser than Latin script; the ratio test must not punish that.
    ("不可抗力", "force majeure"),
    ("準拠法", "governing law"),
    # A German ordinal is not a sentence break.
    ("Abrechnung nach dem 95. Perzentil", "95th percentile billing"),
]


@pytest.mark.parametrize(("source", "target", "reason"), REJECTED)
def test_rejects_known_bad_pairs(source, target, reason):
    verdict = evaluate_term_pair(source, target)
    assert not verdict.accepted, f"{source!r} -> {target!r} should be rejected"
    assert verdict.reason == reason, (
        f"{source!r} -> {target!r} rejected for {verdict.reason!r}, expected {reason!r}"
    )


@pytest.mark.parametrize(("source", "target"), ACCEPTED)
def test_accepts_real_terminology(source, target):
    verdict = evaluate_term_pair(source, target)
    assert verdict.accepted, (
        f"{source!r} -> {target!r} wrongly rejected ({verdict.reason})"
    )


def test_identity_allowed_only_for_curated_preserve_lists():
    """`preserve_as_is` means "never translate this", which is deliberate."""
    assert not is_acceptable_term_pair("Colt", "Colt")
    assert is_acceptable_term_pair("Colt", "Colt", trust=Trust.PRESERVE)


def test_sanitize_reports_counts_and_deduplicates():
    kept, rejected = sanitize_term_pairs(
        [
            ("force majeure", "fuerza mayor"),
            ("der", "der"),
            ("los", "los"),
            ("Force Majeure", "fuerza mayor"),  # case variant of the first
            ("incident management", "gestión de incidencias"),
        ]
    )
    assert kept == [
        ("force majeure", "fuerza mayor"),
        ("incident management", "gestión de incidencias"),
    ]
    assert rejected == {"identity": 2, "duplicate": 1}


def test_sanitize_tolerates_empty_input():
    assert sanitize_term_pairs([]) == ([], {})
