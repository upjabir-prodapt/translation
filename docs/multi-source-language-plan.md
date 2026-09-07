# Multi-Source-Language Translation Plan

**Status:** proposed
**Scope:** worker translation pipeline (PDF + DOCX/TXT)
**In-scope languages:** `en`, `es`, `fr`, `de`, `it`, `ja`

---

## 1. Goal

Today a document whose language mix has no ≥70% dominant language is **rejected**:
`MixedLanguageError` is raised during detection and the job fails before a single
model attempt runs.

Target behaviour:

```
chunk/paragraph
   └─> language metadata (exact: which language, which span, what share)
         └─> if the unit's top language ≥ 70%  ->  assign it as that unit's source language
         └─> group units by source language
               └─> one LLM call per language group, sending that language as lang_in
                     └─> translated output, reassembled by unit id
```

The `%`-of-language architecture is kept. What changes is **where the 70% test is
applied** (per unit, not per document) and **what a failure means** (route
differently, never cancel the job).

---

## 2. Current behaviour worth knowing before changing anything

| Fact | Location | Why it matters |
|---|---|---|
| The batch prompt **never mentions the source language** — only `$lang_out` is substituted | [`PROMPT_TEMPLATE`](../src/worker/doctranslator/format/pdf/document_il/midend/il_translator_llm_only.py), [`_build_prompt`](../src/worker/doctranslator/format/docx/paragraph_translator.py) | Per-unit source language is a **prompt-data** change, not an engine rebuild |
| `lang_in` is used only by: model routing, `DlpConfig.dlp_source_language`, the CJK token multiplier, the Redis cache key, and the **single-unit fallback** prompt | [`build_translation_prompt`](../src/worker/doctranslator/translator/prompts.py), [`build_cache_key`](../src/worker/doctranslator/translator/translation_cache.py) | The fallback prompt *does* say "translate from `{lang_in}`" — it will lie for a minority-language unit |
| Per-unit detection already exists, but **only to drop text** | [`_is_unsupported_or_already_target_language`](../src/worker/doctranslator/format/pdf/document_il/midend/il_translator_llm_only.py) (PDF), [same](../src/worker/doctranslator/format/docx/paragraph_translator.py) (DOCX) | With `SKIP_UNSUPPORTED_LANGUAGE_UNITS=True`, a Portuguese/Dutch/Polish paragraph is **silently passed through untranslated today** |
| Detection extracts text via pymupdf `get_text("blocks")`, translation uses `ParagraphFinder` over the IL | [`_iter_page_text_chunks`](../src/worker/services/processor_service.py) | The two unit sets are **not reconcilable**. Do not try to map one onto the other |
| `langdetect` returns **one label** and collapses to ~1.0 even on 50/50 mixed text | measured | You **cannot** get a per-unit distribution from `detect_langs()` directly — this is why §3 exists |
| PDF batches are page-local; `translate_paragraph` takes a single `page_font_map` | [`process_page`](../src/worker/doctranslator/format/pdf/document_il/midend/il_translator_llm_only.py) | Language groups **cannot span pages** — see §5 |

### Removing `zh`

`zh` is currently in the supported set via [`language_mapper.json`](../src/config/language_mapper.json)
(`"zh-cn": "zh"`, `"chinese": "zh"`). Taking Chinese out of scope means:

- delete those two entries from `language_mapper.json` (this is the single source of truth for
  `get_supported_languages()`), and the `"zh": "Chinese"` display entry in
  [`translation_routing.py`](../src/config/translation_routing.py);
- drop the `"zh-cn"` / `"zh-tw"` rows from `DETECTED_LANGUAGE_ALIASES` in
  [`language_detection_core.py`](../src/worker/services/language_detection_core.py);
- **leave `_is_cjk_language_code` alone** — it matches `zh|ja|ko`, and `ja` keeps the
  CJK token multiplier alive. Removing `zh` there changes nothing and risks nothing.

Chinese text encountered in a document is then simply an out-of-set language and is
handled by the same path as any other (see Phase 0).

---

## 3. The unit language profile

`langdetect` cannot report a mixture, so the distribution has to be **built** by
segmenting the unit first, then char-weighting per-segment verdicts. This is the
core new component.

### New module: `src/worker/services/language_profiler.py`

**Algorithm, per unit:**

1. **Script segmentation** — walk the text, group into maximal Unicode-script runs
   (Latin / Japanese kana+CJK / Cyrillic / Arabic / Devanagari / Greek / Hebrew / Thai).
   Non-Latin runs get their language *from the script* — deterministic, no
   thresholds, exact character offsets, **100% reliable**.
2. **Sentence segmentation** inside each Latin run: `(?<=[.!?;:])\s+|\n+`.
3. **Detection** per sentence of at least `PROFILER_MIN_SENTENCE_CHARS` (40).
   Reuse the existing `case_normalize_for_detection()` and the 0.80 confidence
   floor from `language_detection_core`. Anything shorter or less confident is
   added to `undetected_chars` and **never guessed**.
4. **Char-weight** the verdicts into a `Counter`, the same weighting the
   document-level pass already uses.

### Data model

```python
@dataclass(frozen=True, slots=True)
class LanguageSpan:
    start: int
    end: int
    language: str
    method: Literal["script", "sentence"]

@dataclass(frozen=True, slots=True)
class UnitLanguageProfile:
    unit_ref: str                    # PdfParagraph.debug_id | str(TranslatableUnit.unit_id)
    char_count: int
    char_counts: dict[str, int]      # language -> detected chars
    shares: dict[str, float]         # highest first
    undetected_chars: int
    undetected_ratio: float
    scripts: frozenset[str]
    spans: tuple[LanguageSpan, ...]
    decision: Literal["assign", "mixed", "inherit"]
    source_language: str | None      # set only when decision == "assign"
    confidence: Literal["high", "low"]
```

### Measured behaviour of the prototype

Run against representative units (English-dominant document, `doc_dominant="en"`):

| Unit | Decision | Source | Shares |
|---|---|---|---|
| pure EN paragraph | `assign` | `en` | en 100% |
| pure FR paragraph | `assign` | `fr` | fr 100% |
| pure DE paragraph | `assign` | `de` | de 100% |
| 50/50 EN/FR paragraph | **`mixed`** | — | fr 51%, en 49% |
| EN + JA cross-script | `assign` | `en` | en 75%, ja 25% |
| `"Status: Live"` (table cell) | `inherit` | `en` | — (11 undetected chars) |
| `"Haftungsausschluss"` (heading) | `inherit` | `en` | — (18 undetected chars) |

The two `inherit` rows are the important ones: bare `detect_langs("Status: Live")`
returns **Estonian at 0.857**, which clears the 0.80 confidence floor. Refusing to
decide is the correct outcome, and the sentence-length floor is what produces it.

**Cost: 2.32 ms/unit measured** (2,000 units profiled in 4.64 s, single-threaded,
pure Python). A 2,000-paragraph document costs ~4.6 s of profiling. Acceptable,
parallelisable, and cacheable by text hash.

---

## 4. The 70% decision

`UNIT_DOMINANT_SHARE_FLOOR = 0.70`, mirroring the existing
`LANGUAGE_DETECTION_MIN_DOMINANT_SHARE` so document- and unit-level rules agree.

```
total_detected == 0                  -> INHERIT  (no evidence; use document dominant)
top_share >= 0.70                    -> ASSIGN   top language
top_share <  0.70                    -> MIXED    (do not assign one source)
```

Two modifiers on `ASSIGN`:

- **`undetected_ratio > PROFILER_MAX_UNDETECTED_RATIO` (0.35)** → `confidence="low"`.
  Still assigned, but the batch carries the multi-source hint. This covers the
  measured "80/20 EN/FR" case, where a 24-character French sentence fell below the
  40-char floor and the unit scored `en 100%` with 24 undetected chars — the
  assignment is defensible but should not be presented to the model as certain.
- **Multi-script units always annotate.** If two or more language-bearing scripts are
  present and the minority run is at least `MIN_SCRIPT_RUN_CHARS` (12), record it as a
  minority language **even when the 70% test passes**. Script evidence is
  *categorical, not statistical* — in the measured "EN + JA" case the unit assigns
  `en` at 75%, but the 25% Japanese is certain and must not be silently treated as
  English.

> **A sub-70% unit is never skipped.** It routes to the mixed lane (§5). Dropping it
> would reintroduce exactly the silent content loss that Phase 0 removes.

---

## 5. Grouping and batching — the scalability core

This is where a naive implementation falls over, so it gets designed first.

### The cost model

From the fitted model in [`batching.py`](../src/worker/doctranslator/batching.py) and the
deployed configuration:

- `latency(s) ≈ 16.6 + 12.7 × (payload_tokens / 1000)`
- `TRANSLATION_POOL_MAX_WORKERS = 12`, `LLM_MAX_INFLIGHT_CALLS = 16`
- `LLM_TRANSLATION_BATCH_MAX_TOKENS = 1200`, adaptive band `[600, 2500]`

So a full 1,200-token batch costs ≈ **31.8 s**, while a 100-token batch costs ≈
**17.9 s** — 56% of the price for 8% of the work. **Per-language splitting is only
free when each language group is large.**

Naively splitting every page's batch by language on a 4-language document turns
~10 batches into ~40: one wave becomes four, ~32 s becomes ~128 s on identical content.
That regression is the single biggest risk in this plan.

### Three lanes

Units are grouped into lanes by their decision *and their group size*:

| Lane | Contents | `lang_in` sent | Per-item annotation | Used when |
|---|---|---|---|---|
| **Homogeneous** | all units `ASSIGN`ed the same language | that language | none | group payload ≥ `MIN_LANGUAGE_GROUP_TOKENS` (600) |
| **Annotated** | leftovers from small `ASSIGN` groups + `low`-confidence units | document dominant | `"source_language": "<code>"` | group payload < 600 |
| **Mixed** | `MIXED` units | document dominant | `"languages_present": ["en","fr"]` | always |

`INHERIT` units join the homogeneous group of the document dominant language.

**The annotated lane is the scalability lever.** It delivers per-unit source fidelity
without paying a 16.6 s fixed overhead per language. Each JSON item already carries
`layout_label` and optionally `formula_placeholders_hint`
([`_build_json_format_input`](../src/worker/doctranslator/format/pdf/document_il/midend/il_translator_llm_only.py));
`source_language` is one additional key and the response contract (`id` + `output`)
is unchanged, so `_unwrap_llm_output` and `_parse_translation_results` need no edits.

### Batch planning per lane

Call `compute_batch_plan()` **once per lane**, passing that lane's
`total_payload_tokens` and a worker share proportional to its payload:

```python
lane_workers = max(1, round(POOL_MAX_WORKERS * lane_tokens / total_tokens))
```

so the document as a whole still targets one wave, instead of one wave per language.

### PDF grouping must stay page-local

`translate_paragraph` receives a single `page_font_map`, and
`pre_translate_paragraph` resolves fonts from it. The existing cross-page path works
only because it merges two *adjacent* pages' maps (`{**curr_font_map, **next_font_map}`).
Font maps are keyed by `font_id` strings (`"F1"`, `"F2"`), which **collide across
arbitrary pages** — merging N pages would corrupt font resolution and produce
garbled glyphs.

→ Group within a page. Absorb the resulting small groups with the annotated lane.
DOCX has no font-map constraint and can group document-wide.

---

## 6. Phases

### Phase 0 — Stop silently dropping foreign text *(prerequisite)*

`SKIP_UNSUPPORTED_LANGUAGE_UNITS=True` currently drops any unit confidently detected
outside the supported set — precisely the content this plan exists to translate.

- Split into `SKIP_ALREADY_TARGET_LANGUAGE_UNITS` (keep `True`) and
  `SKIP_UNSUPPORTED_LANGUAGE_UNITS` (default **`False`**).
- Count and log instead of dropping.
- Files: [`constants.py`](../src/config/constants.py),
  [`il_translator_llm_only.py`](../src/worker/doctranslator/format/pdf/document_il/midend/il_translator_llm_only.py),
  [`paragraph_translator.py`](../src/worker/doctranslator/format/docx/paragraph_translator.py).

**Ships and is verifiable on its own.**

### Phase 1 — Profiler *(pure, no wiring)*

- New `language_profiler.py` per §3, built on the existing primitives in
  `language_detection_core.py` (`case_normalize_for_detection`,
  `MIN_DETECTION_CONFIDENCE`, `normalize_detected_language`).
- `lru_cache` on `profile_text(text: str)` keyed by the text itself — repeated
  clauses in long agreements are common and profiling is deterministic.
- No caller changes. Fully unit-testable in isolation.

### Phase 2 — Document level: stop failing

- Add `resolve_language_profile(counter, *, source_label) -> DocumentLanguageProfile`
  that **never raises for mixing** (only for "nothing detectable at all").
- `resolve_dominant_language()` becomes a thin wrapper raising `MixedLanguageError`
  only when `not settings.MULTI_SOURCE_LANGUAGE_ENABLED and profile.is_multi_source`.
  Keep the class and its message — that is the rollback path.
- **The source == target guard must change.** At
  [`pipeline_orchestrator.py`](../src/worker/services/pipeline_orchestrator.py), dominant ==
  target no longer means "no translation needed". New condition: fail only when
  `shares` has exactly one significant language and it equals the target.

### Phase 3 — Lane grouping and batching

- New `LanguageLanePlanner` that takes `list[(unit, UnitLanguageProfile)]` plus the
  document dominant and returns `list[Lane]`.
- PDF: rework `process_page` to profile → plan lanes → emit one `BatchParagraph` per
  batch per lane, carrying `lane.source_language`.
- DOCX: same in `_batch_units` / `translate_all`, grouping document-wide.
- Cross-page and cross-column paths keep their current behaviour (2 paragraphs,
  merged font maps) but tag the batch with the profile of the *first* paragraph.

### Phase 4 — Prompts, per-unit `lang_in`, cache key

- Add `$source_language_block` to both batch prompt templates:
  - homogeneous → `"The source text is in {lang_in}."` *(missing entirely today)*
  - annotated → `"Items may be in different languages. Each item declares its own
    source_language. Translate every item into {lang_out} regardless of its source.
    Never copy an item through untranslated because it is not in {lang_in}."`
  - mixed → `"Some items contain more than one language. Translate all content in
    every item into {lang_out}."`
- Thread the unit's `source_language` into the **single-unit fallback**
  (`build_translation_prompt`), which otherwise states the wrong source language.
- **`build_cache_key` must receive the same per-unit `lang_in`.** It already includes
  `lang_in`; if the fallback prompt varies per unit and the key does not, a French
  unit's translation gets served to an identical-text English unit. These two changes
  **must land in the same commit**, and `PROMPT_VERSION` must be bumped.
- Do **not** rebuild the translator per unit — `create_translator` constructs a Vertex
  client once per attempt in `_build_translation_config`. Per-unit language is prompt
  data, not client configuration.

### Phase 5 — Metadata persistence

- Persist alongside the existing `detected_languages`: `is_multi_source`,
  `minority_source_languages`, per-lane unit counts, and a **capped** sample of unit
  profiles (200 max, or counts only — a 2,000-page manual would otherwise blow up the
  BigQuery row).
- Extend `SplitPoint` in [`split_manager.py`](../src/worker/doctranslator/format/pdf/split_manager.py)
  with `source_languages: dict[str, float]` and `dominant_source_language: str | None`,
  populated by mapping unit page indices into `[start_page, end_page]`. Safe:
  `SplitPoint` is a plain dataclass with defaults and `compute_per_chunk_costs` reads
  only `token_count` / `chunk_index`.

### Phase 6 — Observability

One structured line per job, mirroring the existing `batch_plan` line:

```
language_plan job=<id> dominant=fr dominant_share=0.58 multi_source=true
  lanes=homogeneous:fr(41 units,4 batches),homogeneous:en(18,2),annotated(9,1),mixed(3,1)
  inherit_units=22 low_confidence_units=6 profile_ms=4612
```

Without this, diagnosing "why did this document produce 40 LLM calls" is guesswork.

### Phase 7 — Tests

- `test_language_profiler.py`: script runs (Latin/Japanese/Cyrillic); sentence
  segmentation; the seven measured cases in §3 as a table test; `undetected_ratio`;
  multi-script annotation firing at 75/25.
- `test_language_lane_planner.py`: small groups fold into the annotated lane; lane
  worker shares sum sanely; PDF lanes never span pages.
- `test_language_detection_core.py`: multi-source no longer raises with the flag on,
  still raises with it off.
- `test_pipeline_language_distribution.py`: a 55/45 fr/en document **completes**;
  `is_multi_source` and `minority_source_languages` persist.
- **Existing test that will break:** the unsupported-unit skip tests, due to the
  Phase 0 default flip. Update them, don't delete them.
- Cache-key regression: same text, different per-unit `lang_in` → different key.

---

## 7. Configuration

| Setting | Default | Purpose |
|---|---|---|
| `SKIP_UNSUPPORTED_LANGUAGE_UNITS` | `False` *(flipped)* | stop dropping out-of-set languages |
| `SKIP_ALREADY_TARGET_LANGUAGE_UNITS` | `True` *(new)* | preserve the genuinely useful half of the old flag |
| `MULTI_SOURCE_LANGUAGE_ENABLED` | `True` *(new)* | master switch; `False` restores `MixedLanguageError` |
| `UNIT_DOMINANT_SHARE_FLOOR` | `0.70` *(new)* | the per-unit 70% rule |
| `PROFILER_MIN_SENTENCE_CHARS` | `40` *(new)* | below this a sentence is undetected, not guessed |
| `PROFILER_MAX_UNDETECTED_RATIO` | `0.35` *(new)* | above this an assignment is `low` confidence |
| `MIN_SCRIPT_RUN_CHARS` | `12` *(new)* | minimum minority-script run to annotate |
| `MIN_LANGUAGE_GROUP_TOKENS` | `600` *(new)* | below this a group folds into the annotated lane |
| `SUBUNIT_LANGUAGE_DETECTION_ENABLED` | `True` *(new)* | disables §3 steps 2–4, leaving script-only profiling |

Every phase is reversible by configuration alone except Phase 4's cache-key change,
which is covered by the `PROMPT_VERSION` bump.

---

## 8. Risks

1. **Cache poisoning — highest severity.** Per-unit `lang_in` in the fallback prompt and
   in `build_cache_key` must ship together. Bump `PROMPT_VERSION`.
2. **Batch fragmentation.** Quantified in §5; the annotated lane and per-lane
   `compute_batch_plan` are the mitigations. Gate rollout on the `language_plan` log line
   showing batch counts within ~1.3× of the pre-change baseline.
3. **`PdfParagraph` is `@dataclass(slots=True)`**, generated from
   [`il_version_1.xsd`](../src/worker/doctranslator/format/pdf/document_il/il_version_1.xsd) — you
   cannot attach a profile to it ad hoc. Keep a side map `dict[debug_id, UnitLanguageProfile]`
   on `SharedContextCrossSplitPart` rather than regenerating the IL model.
4. **Font-map collisions** if PDF language groups are ever allowed to span pages (§5).
5. **BigQuery row size** if unit profiles are persisted unbounded (Phase 5).
6. **Profiling cost on pathological documents.** 2.32 ms/unit is fine at 2,000 units
   (~4.6 s) but a 50,000-unit document costs ~2 minutes. Bound it: reuse the existing
   `MAX_DETECTION_CHARS` budget, and fall back to script-only profiling past it.

---

## 9. Known limits

- **Foreign fragments under 40 characters inside a paragraph are invisible** to the
  profiler — they land in `undetected_chars`, not in a minority language. Measured:
  a 24-char French sentence inside an English paragraph scored `en 100%`. The
  `low`-confidence path exists to make this visible rather than to fix it.
- **Same-script short lines cannot be classified.** `"Status: Live"` reads as Estonian
  at 0.857 confidence. Headings, table cells and short bullets will land on `inherit`,
  which is correct but means they follow the document dominant language.
- **Sentence segmentation is regex-based** and will mis-split on abbreviations
  (`"Art. 5"`, `"e.g."`). Impact is bounded — a mis-split only produces a shorter
  segment, which either still detects correctly or falls below the floor and is
  excluded.
- If genuine character-level span accuracy is ever required, `lingua-py`'s
  `detect_multiple_languages_of()` returns real `(start, end, language)` spans and is
  built for short text. It is **not** currently a dependency (`langdetect>=1.0.9` is the
  only detector), and adopting it would replace §3 steps 2–4 rather than extend them.
