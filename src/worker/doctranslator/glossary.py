import csv
import io
import logging
import re
import threading
import time
from pathlib import Path
from typing import NamedTuple

import chardet
import hyperscan
import regex

from src.config.translation_routing import normalize_language
from src.worker.doctranslator.utils.common import batched

logger = logging.getLogger(__name__)


class GlossaryEntry:
    def __init__(self, source: str, target: str, target_language: str | None = None):
        self.source = source
        self.target = target
        self.target_language = target_language

    def __repr__(self):
        return f"GlossaryEntry(source='{self.source}', target='{self.target}', target_language='{self.target_language}')"


class ExtractedGlossaryTerm(NamedTuple):
    """One (source_term, target_term) pair from automatic term extraction,
    plus the language the source term was actually extracted from.

    `source_language` is reported by the extraction LLM itself (see
    `schemas.ExtractedTerm.src_lang`), not assumed from the job's declared
    `source_language` -- the extraction prompt now allows pulling terms
    from any language in `language_mapper.json`, so a mixed-language
    document can contribute terms in more than one language from a single
    extraction pass, each correctly attributed.

    This is the field `GlossaryService` buckets on when merging into the
    shared GCS domain glossary (`merge_new_terms_into_domain_glossary`) and
    filters on when loading it back for a later job
    (`load_domain_glossary`'s `source_languages` argument) -- see those
    docstrings for why per-term attribution, not a single value for the
    whole extraction batch, is what makes that filtering meaningful.
    Callers that only have a (source, target) pair and no reliable
    per-term language (e.g. a term whose reported language did not
    normalize to a supported code) should not construct one of these --
    drop the term instead of guessing.
    """

    source: str
    target: str
    source_language: str


# Above this fraction of a batch's candidate terms being dropped for an
# unrecognized `src_lang`, `TermLanguageDropTracker.warn_if_high_drop_rate`
# escalates from per-term debug logging to a batch-level WARNING -- see its
# docstring.
_HIGH_SRC_LANG_DROP_RATE = 0.5


class TermLanguageDropTracker:
    """Shared validate-and-drop + drop-rate tracking for extracted terms.

    Both the DOCX (`format/docx/term_extractor.py`) and PDF
    (`format/pdf/document_il/midend/automatic_term_extractor.py`) extractors
    ask the LLM to self-report each term's source language (`src_lang`) and
    must drop any term whose reported value doesn't normalize to a
    `language_mapper.json` code -- previously this validate-and-drop block
    was hand-duplicated in both files with only per-term debug logging, so a
    systemic regression (the model silently no longer following the
    `src_lang` instruction, e.g. after a prompt or structured-output change)
    had no operator-visible signal: extraction volume could collapse toward
    zero across many jobs with nothing above debug level to notice it.

    One instance is meant to live for the lifetime of one extraction call
    (a document/job, not a single batch) so the drop rate reflects the whole
    extraction pass, not one batch's sampling noise. Both extractors run
    their batches concurrently across a `ThreadPoolExecutor`, so `resolve()`
    is internally lock-protected.
    """

    def __init__(self, *, high_drop_rate: float = _HIGH_SRC_LANG_DROP_RATE) -> None:
        self.candidate_count = 0
        self.dropped_count = 0
        self._high_drop_rate = high_drop_rate
        self._lock = threading.Lock()

    def resolve(self, raw_src_lang: str, *, source_term: str) -> str | None:
        """Normalize one term's reported `src_lang`, or None to drop it.

        `normalize_language` only ever returns a value in
        `get_supported_languages()` by construction, so a `ValueError` here
        means the model didn't follow the extraction prompt's `src_lang`
        rule, not that a rarer supported language was reported.
        """
        try:
            normalized = normalize_language(str(raw_src_lang))
        except ValueError:
            with self._lock:
                self.candidate_count += 1
                self.dropped_count += 1
            logger.debug(
                "Dropping extracted term %r: src_lang %r is not one of the "
                "supported languages",
                source_term,
                raw_src_lang,
            )
            return None
        with self._lock:
            self.candidate_count += 1
        return normalized

    def warn_if_high_drop_rate(self, *, extractor_name: str) -> None:
        """Escalate to WARNING once dropped terms are the majority of a pass.

        Call once after all of an extraction call's batches have been
        resolved. A handful of drops per document is normal (an occasional
        hallucinated code); the whole pass losing its majority to
        unrecognized codes is not, and is worth a signal above debug.
        """
        if not self.candidate_count or not self.dropped_count:
            return
        rate = self.dropped_count / self.candidate_count
        if rate < self._high_drop_rate:
            return
        logger.warning(
            "%s: dropped %d/%d candidate terms (%.0f%%) for an unrecognized "
            "src_lang -- if this persists across documents it likely means "
            "the model has stopped following the src_lang instruction (a "
            "prompt or structured-output regression), not that unusual "
            "languages were encountered.",
            extractor_name,
            self.dropped_count,
            self.candidate_count,
            rate * 100,
        )


TERM_NORM_PATTERN = re.compile(r"\s+", regex.UNICODE)


class Glossary:
    def __init__(self, name: str, entries: list[GlossaryEntry]):
        self.name = name

        # Deduplicate entries based on normalized source
        unique_entries = []
        seen_normalized_sources = set()
        for entry in entries:
            normalized_source = self.normalize_source(entry.source)
            if normalized_source not in seen_normalized_sources:
                unique_entries.append(entry)
                seen_normalized_sources.add(normalized_source)
        self.entries = unique_entries

        self.normalized_lookup: dict[str, tuple[str, str]] = {}
        self.id_lookup: list[tuple[str, str]] = []
        self.hs_dbs: list[hyperscan.Database] | None = None
        self._build_regex_and_lookup()

    @staticmethod
    def normalize_source(source_term: str) -> str:
        """Normalizes a source term by lowercasing and standardizing whitespace."""
        term = source_term.lower()
        term = TERM_NORM_PATTERN.sub(
            " ", term
        )  # Replace multiple whitespace with single space
        return term.strip()

    def _build_regex_and_lookup(self):
        logger.debug(
            f"start build regex for glossary {self.name} with {len(self.entries)} entries"
        )
        """
        Builds a combined regex for all source terms and a lookup dictionary
        from normalized source terms to (original_source, original_target).
        Regex patterns are sorted by length in descending order to prioritize longer matches.
        """
        self.normalized_lookup = {}

        if not self.entries:
            self.source_terms_regex = None
            return

        self.hs_dbs = []
        hs_pattern = []
        start = time.time()
        for idx, entry in enumerate(self.entries):
            normalized_key = self.normalize_source(entry.source)
            self.normalized_lookup[normalized_key] = (entry.source, entry.target)
            self.id_lookup.append((entry.source, entry.target))

            hs_pattern.append((re.escape(entry.source).encode("utf-8"), idx))

        chunk_size = 20000
        for i, pattern_chunk in enumerate(
            batched(hs_pattern, chunk_size, strict=False)
        ):
            logger.debug(
                f"building hs_db chunk {i + 1} / {len(self.entries) // chunk_size + 1}"
            )
            expressions, ids = zip(*pattern_chunk, strict=False)

            hs_db = hyperscan.Database()
            hs_db.compile(
                expressions=expressions,
                ids=ids,
                elements=len(pattern_chunk),
                flags=hyperscan.HS_FLAG_CASELESS | hyperscan.HS_FLAG_SINGLEMATCH,
                # | hyperscan.HS_FLAG_UTF8
                # | hyperscan.HS_FLAG_UCP,
            )
            self.hs_dbs.append(hs_db)

        end = time.time()
        logger.debug(
            f"finished building regex for glossary {self.name} in {end - start:.2f} seconds"
        )
        logger.debug(
            f"build hs database for glossary {self.name} with {len(self.entries)} entries, hs_info: {self.hs_dbs[0].info()}"
        )
        if not self.hs_dbs:
            self.hs_dbs = None

    @classmethod
    def from_csv(cls, file_path: Path, target_lang_out: str) -> "Glossary":
        """
        Loads glossary entries from a CSV file.
        CSV format: source,target,tgt_lng (tgt_lng is optional)
        Filters entries based on tgt_lng matching target_lang_out.
        The glossary name is derived from the CSV filename.
        """
        glossary_name = file_path.stem
        loaded_entries: list[GlossaryEntry] = []

        # Normalize target_lang_out once for comparison
        normalized_target_lang_out = target_lang_out.lower().replace("-", "_")

        try:
            with file_path.open("rb") as f:
                content = f.read()
                encoding = chardet.detect(content)["encoding"]
                buffer = io.StringIO(content.decode(encoding))
                reader = csv.DictReader(buffer, doublequote=True)
                if not all(col in reader.fieldnames for col in ["source", "target"]):
                    raise ValueError(
                        f"CSV file {file_path} must contain 'source' and 'target' columns."
                    )

                for row in reader:
                    source = row["source"]
                    target = row["target"]
                    tgt_lng = row.get("tgt_lng", None)  # Handle optional tgt_lng

                    if tgt_lng and tgt_lng.strip():
                        normalized_entry_tgt_lng = (
                            tgt_lng.strip().lower().replace("-", "_")
                        )
                        if normalized_entry_tgt_lng != normalized_target_lang_out:
                            continue  # Skip if language doesn't match

                    loaded_entries.append(GlossaryEntry(source, target, tgt_lng))
        except FileNotFoundError:
            # Or handle as per your project's error strategy, e.g., log and return empty Glossary
            raise
        except Exception as e:
            # Or handle as per your project's error strategy
            raise ValueError(
                f"Error reading or parsing CSV file {file_path}: {e}"
            ) from e

        return cls(name=glossary_name, entries=loaded_entries)

    def to_csv(self) -> str:
        """Exports the glossary entries to a CSV formatted string."""
        dict_data = [
            {
                "source": x.source,
                "target": x.target,
                "tgt_lng": x.target_language if x.target_language else "",
            }
            for x in self.entries
        ]
        buffer = io.StringIO()
        dict_writer = csv.DictWriter(
            buffer, fieldnames=["source", "target", "tgt_lng"], doublequote=True
        )
        dict_writer.writeheader()
        dict_writer.writerows(dict_data)
        return buffer.getvalue()

    def __repr__(self):
        return f"Glossary(name='{self.name}', num_entries={len(self.entries)})"

    def get_active_entries_for_text(self, text: str) -> list[tuple[str, str]]:
        """Returns a list of (original_source, target_text) tuples for terms found in the given text."""
        if not self.hs_dbs or not text:
            return []

        text = TERM_NORM_PATTERN.sub(" ", text)  # Normalize whitespace in the text
        if not text:
            return []

        active_entries = []

        def on_match(
            idx: int, _from: int, _to: int, _flags: int, _context=None
        ) -> bool | None:
            active_entries.append(self.id_lookup[idx])
            return False

        for hs_db in self.hs_dbs:
            # Scan the text with the hyperscan database
            scratch = hyperscan.Scratch(hs_db)
            hs_db.scan(text.encode("utf-8"), on_match, scratch=scratch)
        return active_entries
