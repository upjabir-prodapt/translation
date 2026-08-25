#!/usr/bin/env python3
"""Seed/update Firestore user_entitlements documents for a list of emails.

Grants access to one or more services (translation_access, sales_agent_access)
for every email listed, without touching any other fields already present on
each user's document (uses Firestore's `update`/merge semantics via `set`
with `merge=True`, so existing entitlements for other services are preserved).

Usage:
    python3 scripts/seed_entitlements.py \
        --project aicoeprod \
        --grant translation_access \
        --emails-file emails.txt

    python3 scripts/seed_entitlements.py \
        --project aicoeprod \
        --grant translation_access --grant sales_agent_access \
        user1@colt.net user2@internal.colt.net

Requires: google-cloud-firestore (already a dependency of this repo) and
Application Default Credentials with Firestore write access (the same
identity you use for `gcloud` commands works if you've run
`gcloud auth application-default login`, or run this from an environment
with a service account that has roles/datastore.user).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from google.cloud import firestore

ENTITLEMENTS_COLLECTION = "user_entitlements"
VALID_GRANTS = {"translation_access", "sales_agent_access"}


def normalize_email(raw: str) -> str:
    return raw.strip().lower()


def load_emails(args: argparse.Namespace) -> list[str]:
    emails: list[str] = []
    if args.emails_file:
        with Path(args.emails_file).open(encoding="utf-8") as f:
            emails.extend(line.strip() for line in f if line.strip())
    emails.extend(args.emails)

    if not emails:
        print(
            "ERROR: no emails provided (use --emails-file or positional args)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Normalize + dedupe while preserving first-seen order (helpful for review output)
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in emails:
        email = normalize_email(raw)
        if not email or "@" not in email:
            print(f"WARNING: skipping invalid-looking email: {raw!r}", file=sys.stderr)
            continue
        if email in seen:
            print(f"NOTE: duplicate email skipped: {email}", file=sys.stderr)
            continue
        seen.add(email)
        normalized.append(email)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--project", required=True, help="GCP project ID (e.g. aicoeprod)"
    )
    parser.add_argument(
        "--grant",
        action="append",
        required=True,
        choices=sorted(VALID_GRANTS),
        help="Entitlement field to set to true for every email (repeatable)",
    )
    parser.add_argument("--emails-file", help="Path to a file with one email per line")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written, without writing",
    )
    parser.add_argument(
        "emails", nargs="*", help="Additional emails as positional arguments"
    )
    args = parser.parse_args()

    emails = load_emails(args)
    grants = dict.fromkeys(args.grant, True)

    print(f"Project: {args.project}")
    print(f"Granting: {grants}")
    print(f"Emails ({len(emails)}):")
    for email in emails:
        print(f"  - {email}")

    if args.dry_run:
        print("\n[dry-run] No writes performed.")
        return

    client = firestore.Client(project=args.project)
    collection = client.collection(ENTITLEMENTS_COLLECTION)

    for email in emails:
        doc_ref = collection.document(email)
        doc_ref.set(grants, merge=True)
        print(f"  OK: {email} -> {grants}")

    print(
        f"\nDone. Wrote {len(emails)} document(s) to {ENTITLEMENTS_COLLECTION!r} in project {args.project!r}."
    )


if __name__ == "__main__":
    main()
