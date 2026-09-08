"""Translate a document end-to-end on this machine, with no cloud plumbing.

This runs the *same* code the deployed service runs -- the API-boundary
validators and the worker's `PipelineOrchestrator`, untouched -- but with
BigQuery, Cloud Tasks and GCS replaced by local stand-ins:

  * no API server and no worker server are started;
  * no BigQuery job row is written (an in-memory store logs what would be);
  * no Cloud Task is enqueued (the orchestrator is awaited in-process);
  * no GCS traffic (input/output are copied inside a local scratch tree).

Everything else is the production path, so every check the endpoint applies
still applies here:

  * request-schema validation (filename extension, base64, domain, language
    codes, source != target)                    -- src/api/schemas/requests.py
  * file validation: size, PDF magic bytes, encryption / password /
    copy-permission, text-layer probe; DOCX OOXML structure; TXT UTF-8
                        -- src/api/utils/{pdf_validator,document_validator}.py
  * source-language detection + the mixed-language and wrong-language guards
                        -- PipelineOrchestrator._assert_language_matches
  * the domain-consistency guard  -- PipelineOrchestrator._assert_domain_matches
  * DLP masking, model-chain routing, translation, the quality judge,
    per-chunk cost attribution and the job cost ceiling.

The LLM itself is still Vertex AI, so credentials and network access are
required for the translation to actually happen.

Usage
-----
    python scripts/local_translate.py index.pdf --source en --target fr --domain legal

    # a whole directory, and drop the results beside the sources
    python scripts/local_translate.py ./docs --source en --target de --domain hr

    # human-readable logs instead of the server's Cloud Logging JSON
    python scripts/local_translate.py index.pdf -s en -t fr -d legal --log-format text
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Formats the pipeline knows how to translate; mirrors settings.ALLOWED_EXTENSIONS.
_FORMAT_BY_SUFFIX = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_translate",
        description=(
            "Run the full translation pipeline locally: no server, no "
            "BigQuery job row, no Cloud Task, no GCS."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Document to translate, or a directory of documents (.pdf/.docx/.txt).",
    )
    parser.add_argument(
        "-s",
        "--source",
        "--source-language",
        dest="source_language",
        required=True,
        help="Source language name or code (e.g. 'en', 'English'). Required, "
        "exactly as the API requires it -- it is what the detected language "
        "is checked against.",
    )
    parser.add_argument(
        "-t",
        "--target",
        "--target-language",
        dest="target_language",
        required=True,
        help="Target language name or code (e.g. 'fr', 'French').",
    )
    parser.add_argument(
        "-d",
        "--domain",
        required=True,
        help="Document domain: commercial, legal, finance, hr, operations.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write translated output (default: alongside each input).",
    )
    parser.add_argument(
        "--user-id",
        default="local-cli",
        help="Cost-attribution user id recorded on the job (default: local-cli).",
    )
    parser.add_argument(
        "--business-unit",
        default="local",
        help="Cost-attribution business unit (default: local).",
    )
    parser.add_argument(
        "--organization",
        default="local",
        help="Cost-attribution organization (default: local).",
    )

    checks = parser.add_argument_group(
        "checks",
        "All guards are ON by default, matching the deployed defaults. Each "
        "flag below turns one off for this run only (via its env var).",
    )
    checks.add_argument(
        "--no-dlp",
        action="store_true",
        help="Skip Google DLP masking (GOOGLE_DLP_ENABLED=false).",
    )
    checks.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Skip the domain-consistency guard (DOMAIN_MISMATCH_CHECK_ENABLED=false).",
    )
    checks.add_argument(
        "--no-language-check",
        action="store_true",
        help=(
            "Skip the mixed-language and wrong-language guards "
            "(LANGUAGE_MISMATCH_CHECK_ENABLED=false). Detection still runs "
            "and still logs the full distribution -- only the rejection is "
            "suppressed."
        ),
    )
    checks.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM quality judge (QUALITY_JUDGE_ENABLED=false).",
    )
    checks.add_argument(
        "--max-secondary-share",
        type=float,
        default=None,
        metavar="SHARE",
        help=(
            "Tolerated share of a second detected language before the "
            "mixed-language guard fires (LANGUAGE_MIXED_MAX_SECONDARY_SHARE; "
            "default 0.0, i.e. any second language fails)."
        ),
    )

    parser.add_argument(
        "--credentials",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Service-account JSON to authenticate Vertex AI (and DLP) with, "
            "exported as GOOGLE_APPLICATION_CREDENTIALS. Omit if Application "
            "Default Credentials are already configured "
            "(`gcloud auth application-default login`)."
        ),
    )
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        action="append",
        default=None,
        metavar="PEM",
        help=(
            "Extra root CA(s) to trust, for a corporate TLS-inspecting proxy. "
            "Repeatable. Merged with certifi into one bundle and exported as "
            "SSL_CERT_FILE, REQUESTS_CA_BUNDLE and "
            "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH -- gRPC reads only the last of "
            "those, which is why Vertex still fails with "
            "CERTIFICATE_VERIFY_FAILED when only the usual two are set."
        ),
    )
    parser.add_argument(
        "--log-format",
        choices=("json", "text"),
        default="json",
        help="json = the Cloud Logging format the server emits (default); "
        "text = one readable line per record.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override LOG_LEVEL for this run (e.g. DEBUG).",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep the local stand-in GCS tree instead of deleting it.",
    )
    return parser


def apply_env_overrides(args: argparse.Namespace) -> None:
    """Set the settings env vars the flags map to.

    Must run *before* `src.config.constants` is imported: `Settings` is
    built once at import time, so an override applied afterwards would be
    read by nothing.
    """
    if args.no_dlp:
        os.environ["GOOGLE_DLP_ENABLED"] = "false"
    if args.no_domain_check:
        os.environ["DOMAIN_MISMATCH_CHECK_ENABLED"] = "false"
    if args.no_language_check:
        os.environ["LANGUAGE_MISMATCH_CHECK_ENABLED"] = "false"
    if args.no_judge:
        os.environ["QUALITY_JUDGE_ENABLED"] = "false"
    if args.max_secondary_share is not None:
        os.environ["LANGUAGE_MIXED_MAX_SECONDARY_SHARE"] = str(
            args.max_secondary_share
        )
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level.upper()
    if args.credentials:
        path = args.credentials.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"No such credentials file: {path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    if args.ca_bundle:
        bundle = _merged_ca_bundle(args.ca_bundle)
        os.environ["SSL_CERT_FILE"] = bundle
        os.environ["REQUESTS_CA_BUNDLE"] = bundle
        os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = bundle


def _merged_ca_bundle(extra_cas: list[Path]) -> str:
    """Write certifi's roots plus every `extra_cas` PEM to one file.

    Pointing the TLS env vars straight at a corporate CA file would trust
    *only* those roots; merging keeps the public ones working too. The
    merged file is cached in the temp dir under a name derived from each
    source's path, size and mtime, so repeated runs reuse it and a
    refreshed corporate CA produces a new one.
    """
    import certifi

    sources: list[Path] = []
    for candidate in extra_cas:
        path = candidate.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"No such CA bundle: {path}")
        sources.append(path)

    fingerprint = "|".join(
        f"{path}:{path.stat().st_size}:{int(path.stat().st_mtime)}" for path in sources
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    merged = Path(tempfile.gettempdir()) / f"local-translate-ca-{digest}.pem"

    if not merged.exists():
        chunks = [Path(certifi.where()).read_text(encoding="utf-8")]
        chunks.extend(path.read_text(encoding="utf-8") for path in sources)
        merged.write_text("\n".join(chunks), encoding="utf-8")
    return str(merged)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_env_overrides(args)

    # Imported only now, so the overrides above are visible to Settings.
    from scripts._local_pipeline import run_cli

    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
