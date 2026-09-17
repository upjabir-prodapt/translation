#!/usr/bin/env python
"""Validate glossary data against the runtime hygiene gate.

The glossary files are the one place a bad term can reach the model without
passing through any code, so this runs the *same* gate the pipeline applies at
translation time. A term that would be silently dropped in production shows up
here instead.

Run it before every upload to the bucket::

    python scripts/validate_glossary.py glossary_seed/*.json
    python scripts/validate_glossary.py assets/glossaries/commercial.json --verbose

Exit codes: 0 clean, 1 rejected entries found, 2 could not read the source.
Wire it into CI ahead of any glossary publish.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config.glossary_hygiene import HygieneContext  # noqa: E402
from src.config.glossary_hygiene import Trust  # noqa: E402
from src.config.glossary_hygiene import evaluate_term_pair  # noqa: E402
from src.config.linguistic_data import log_coverage  # noqa: E402
from src.config.translation_routing import SUPPORTED_DOMAINS  # noqa: E402
from src.config.translation_routing import normalize_language  # noqa: E402

Row = tuple[str, str, str, str, str, str]
"""domain, source_language, target_language, source_term, target_term, origin."""


def _rows_from_domain_json(payload: dict, label: str) -> tuple[list[Row], list[str]]:
    """Rows from the nested per-source-language shape the runtime reads."""
    rows: list[Row] = []
    preserve: list[str] = []
    domain = str(payload.get("domain") or label)
    for source_language, section in (payload.get("glossary") or {}).items():
        if not isinstance(section, dict):
            continue
        preserve.extend(str(t) for t in (section.get("preserve_as_is") or []))
        for term in section.get("terms") or []:
            source = str(term.get("source_term") or "")
            origin = str(term.get("origin") or "")
            for target_language, target in (term.get("translations") or {}).items():
                rows.append(
                    (
                        domain,
                        source_language,
                        str(target_language),
                        source,
                        str(target),
                        origin,
                    )
                )
    return rows, preserve


def load_path(path: Path) -> tuple[list[Row], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(payload, dict) and "glossary" in payload):
        raise ValueError(
            f"{path}: not a domain glossary. Expected a 'glossary' object keyed "
            "by source language, which is the shape the runtime reads."
        )
    return _rows_from_domain_json(payload, path.stem)


def check(rows: list[Row], preserve: list[str], *, verbose: bool) -> Counter:
    """Evaluate every row; returns a {reason: count} tally."""
    # Trusted vocabulary for the person-name heuristic, from the do-not-translate
    # list. Never from the rows being judged: that would whitelist them.
    context = HygieneContext.from_terms(preserve)
    tally: Counter = Counter()

    for domain, source_language, target_language, source, target, origin in rows:
        # Trust is carried by the data, not assumed by the validator: an entry
        # that does not declare itself curated is checked as machine-written.
        trust = Trust.from_origin(origin)
        problems = []
        if domain and domain not in SUPPORTED_DOMAINS:
            problems.append(f"unknown domain {domain!r}")
        for label, code in (("source", source_language), ("target", target_language)):
            try:
                if code and normalize_language(code) != code:
                    problems.append(
                        f"{label}_language {code!r} is not canonical "
                        f"(expected {normalize_language(code)!r}) -- the pipeline "
                        "looks up by canonical code, so this row is unreachable"
                    )
            except ValueError:
                problems.append(f"{label}_language {code!r} is not a supported code")
        if source_language and source_language == target_language:
            problems.append("source_language == target_language")

        verdict = evaluate_term_pair(source, target, trust=trust, context=context)
        if not verdict.accepted:
            problems.append(f"hygiene: {verdict.reason}")

        for problem in problems:
            tally[problem.split(":")[0].split(" ")[0]] += 1
            if verbose:
                print(
                    f"  REJECT [{domain}/{source_language}->{target_language}] "
                    f"{source!r} -> {target!r}: {problem}"
                )

    for term in preserve:
        verdict = evaluate_term_pair(term, term, trust=Trust.PRESERVE)
        if not verdict.accepted:
            tally[f"preserve:{verdict.reason}"] += 1
            if verbose:
                print(f"  REJECT preserve_as_is {term!r}: {verdict.reason}")

    return tally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="domain glossary JSON")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print each reject"
    )
    args = parser.parse_args()

    log_coverage()

    rows: list[Row] = []
    preserve: list[str] = []
    for path in args.paths:
        try:
            path_rows, path_preserve = load_path(path)
        except Exception as exc:
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            return 2
        print(f"{path}: {len(path_rows)} pairs, {len(path_preserve)} preserve entries")
        rows += path_rows
        preserve += path_preserve

    tally = check(rows, preserve, verbose=args.verbose)
    total = sum(tally.values())
    print(f"\nchecked {len(rows)} pairs + {len(preserve)} preserve entries")
    if not total:
        print("clean: every entry passes the gate the runtime applies")
        return 0
    print(f"{total} rejected:")
    for reason, count in tally.most_common():
        print(f"  {count:6}  {reason}")
    if not args.verbose:
        print("\nre-run with --verbose to see each one")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
