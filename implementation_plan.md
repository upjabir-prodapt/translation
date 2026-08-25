# Translation Service — Latency Optimisation Implementation Plan

> **Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` deferred/out of scope
>
> **Baseline measured:** 2026-08-24 09:20–09:38 UTC, local two-process run per `docs/local-testing-guide.md`
> using `.env.api.local` / `.env.worker.local` against `aicoesandox`.
> Logs: `.local-tmp/worker_server.log` (866 lines), `.local-tmp/api_server.log` (47 lines).

---

# Implementation Plan

## [Overview]

Cut end-to-end document translation latency by ~60-70% (DOCX 232s -> <=90s, PDF 705s -> <=280s) by fixing LLM model/thinking configuration, right-sizing LLM batches, unblocking the worker event loop, and eliminating wasted single-unit LLM calls.

The measured baseline shows **67% of PDF wall-clock is a single stage** (`Translate Paragraphs`, 470s of 705s) and
**LLM calls average only 19.2 output tokens/sec** across 114 calls — versus 41-48 tok/s on the healthy subset.
The dominant causes are not algorithmic: they are (a) `gemini-2.5-pro` running with unbounded default "dynamic
thinking", (b) batch-size caps (`40000`/`80000` tokens) so large that the existing 12-worker thread pools only
ever received 1-3 work items, and (c) a synchronous call inside an `async def` that blocks the worker event loop.

This plan also **replaces the priority-1 translation model** with `gemini-3.5-flash` pinned to `europe-west3`
(GA since 2026-05-19, confirmed region support), demoting `gemini-2.5-pro` to priority 2. It extends
`pricing_catalog.json` to be the single source of truth for **both price and region** — it already is for price,
and the `region` field exists on `ModelRateEntry` but no Gemini entry populates it today.

Scope is deliberately staged: Phase 0 is pure instrumentation (no behaviour change) so every subsequent claim is
measurable; Phase 1 is the high-value/low-risk core; Phases 2-3 are structural work needing benchmarking or
design discussion. **9 of 12 items from the pre-existing Theme A-E backlog are already done** (see
`memory-bank/systemPatterns.md`); this plan supersedes and absorbs the remaining B2/D1/E2/E3 items.

---

## Measured Baseline (evidence)

### Job wall-clock

| Job | Document | Worker wall | End-to-end | Note |
|---|---|---|---|---|
| `2abd58ef` | DOCX, 414 units | 232 s | 236 s | |
| `4896fd37` | DOCX, same file | 28 s | 231 s | **203 s queued** behind job 1 |
| `522ae5c3` | PDF, 20 pp / 472 paras | 705 s | ~706 s | |

### PDF stage breakdown (from `Pipeline stage finished ... duration_s=`)

| Stage | Duration | % of job |
|---|---|---|
| Translate Paragraphs | **470.2 s** | 67% |
| Automatic Term Extraction | **94.9 s** | 13% |
| Parse Page Layout (ONNX) | **58.1 s** | 8% |
| Typesetting | 16.4 s | 2% |
| Judge | 12.3 s | 2% |
| Parse Paragraphs / Formulas / Fonts / Save | ~12 s | 2% |
| Unaccounted (inter-stage, GCS, BQ) | ~41 s | 6% |

### DOCX stage breakdown (derived from log timestamps)

| Phase | Duration | Note |
|---|---|---|
| Task recv + GCS download + glossary + unit extract | ~4 s | |
| Term extraction | ~107 s | **1 batch, `max_workers=1`** |
| Paragraph translation | ~142 s | **3 batches**, concurrent with above (A4) |
| 45 single-unit fallbacks | ~69 s | |
| Judge | 14 s | |
| Upload + BQ + glossary merge | ~3 s | |

### LLM throughput

```
TOTAL output_tokens=72,937   total_llm_latency=3,803 s   =>  19.2 out-tok/s
DOCX-1: n=47  sum=987 s   med=8.4 s   p90=46.7 s   max=140.9 s   eff.concurrency=4.70 / 12
PDF:    n=67  sum=2,815 s med=33.7 s  p90=56.6 s   max=454.4 s   eff.concurrency=4.95 / 12
```

Pathological tail (all `gemini-2.5-pro`):

```
latency_s=454.372  input_tokens=5251  output_tokens=1496   ->  3.3 tok/s
latency_s=122.935  input_tokens=988   output_tokens=13     ->  0.1 tok/s
latency_s=141.828  input_tokens=5858  output_tokens=2730   -> 19.2 tok/s
```

---

## Root Causes

| ID | Root cause | Evidence | Impact |
|---|---|---|---|
| **RC1** | `gemini-2.5-pro` runs with default *dynamic thinking*; **no `thinking_config` anywhere in the codebase**. Also `TokenUsage.from_gemini_usage()` ignores `thoughts_token_count`, so cost is under-reported. | `grep thinking_config` -> 0 results; `usage.py:22-31`; 454 s / 123 s calls | **Highest** |
| **RC2** | DOCX `translate_docx(...)` called **synchronously inside `async def translate()`** — blocks the worker event loop. PDF path correctly uses `run_in_executor`. | `docx_job_processor.py:134` vs `high_level.py:614`; job 2 uploaded 09:21:43, worker logged "Received translate task" only at 09:25:06 | **High** |
| **RC3** | Batch caps so large the thread pools starve: 414 units -> **1** term batch (`max_workers=1`) and **3** translate batches against `TRANSLATION_POOL_MAX_WORKERS=12`. | `DOCX term extraction completed. Units: 414, Batches: 1, max_workers=1`; eff. concurrency 4.7/12 | **High** |
| **RC4** | No client-side timeout. `genai.Client` built without `http_options`; `llm_retry` only fires on transient errors, so a 454 s call is simply waited out. | `gemini.py:73-77`; `retry.py:59-69` | **High** |
| **RC5** | Fallback storm: 72x `same_text` + 18x `length_ratio` validation failures; each fallback re-sends ~3,381 chars of boilerplate to translate ~5 chars. | `input_chars=5 prompt_chars=3386 input_tokens=766`; 43 calls in the 0-100-token bucket at 2.4 tok/s | Medium-High |
| **RC6** | Redis cache effectively cold: `cache_hit_rate_cumulative=0.0514`. PDF keys the **whole assembled batch prompt**, so one changed paragraph invalidates the batch (backlog **D1**, open). | `base.py::_cache_key_for`; DOCX has per-unit `_split_cache_hits`, PDF does not | Medium |
| **RC7** | PDF serialises term extraction **before** translation (95 s on critical path). DOCX already overlaps them (A4 done). | `high_level.py:1363-1380` | Medium |
| **RC8** | ONNX layout is sequential: 58 s / 20 pages = 2.9 s/page. **`ONNX_LAYOUT_BATCH_SIZE=4` is dead config** — `predict()` supports batching but `handle_document()` feeds exactly one page at a time. (backlog **E2**) | `doclayout.py:228-243` vs `predict()` at `:171-174`; `layout_parser.py:128` | Medium |
| **RC9** | Full prompt+response capture on every span: `TRACE_SAMPLE_RATE=1.0` + `OTEL_..._CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT` serialises 36 k-char prompts on the hot path. | `.env.worker.local:96,102` | Low-Medium |
| **RC10** | `QUALITY_EARLY_ACCEPT_THRESHOLD` is **dead config** — declared, read nowhere in `src/`. With `MAX_MODEL_ATTEMPTS=3` a judge miss triples the entire job. | `constants.py:199`; `grep` -> only constants.py + test.env | Medium |
| **RC11** | `SPLIT_PART_MAX_CONCURRENT=2` gave **zero** parallelism on this PDF — `split_manager` reported "20 pages; detected 1 section(s)". (backlog **B2/E3**) | worker log 09:25:20 | Low |
| **RC12** | Local harness distortion: `_enqueue_local_http` does a blocking `httpx.post(timeout=3600)`, holding the API thread for the whole job. Not a prod issue but skews local numbers. | `cloud_tasks_service.py:79` | Local only |

---

## Ideal Batch Size — derived, not guessed

Fitted across all 114 logged calls (payload = `input_tokens` - ~760 boilerplate):

```
latency(s) ~= 16.6 fixed  +  12.7 per 1,000 payload tokens      (n=113, >200 s outliers excluded)
=> fixed overhead dominates below ~1,311 payload tokens
```

| payload tokens | n | median latency | out-tok/s | ms per k-token |
|---|---|---|---|---|
| 0-100 | 43 | 8.0 s | **2.4** | 334 |
| 100-500 | 9 | 8.8 s | 1.1 | 37 |
| 500-1,000 | 19 | 19.1 s | 18.2 | 22 |
| 1,000-2,000 | 24 | 40.7 s | 28.2 | 36 |
| 2,000-3,000 | 14 | 39.2 s | 24.4 | **16.8** |
| 8,000-20,000 | 3 | 139.5 s | 45.2 | 13.3 |

**The trap:** raw throughput keeps improving with bigger batches (45 tok/s at 11 k), which is presumably why the
current 40 k/80 k caps were chosen. But throughput is the wrong objective — **wall-clock against a 12-worker
pool** is what the user waits for. Optimising that:

```
DOCX (414 units, ~7,900 payload tok, avg 19 tok/unit, pool=12)
  batch_tok  paras  nbatch  waves  per_call    WALL
        500     26      16      2     23.0 s   45.9 s
        750     39      11      1     26.1 s   26.1 s   <- optimum
      1,000     53       8      1     29.3 s   29.3 s
      3,000    158       3      1     54.7 s   54.7 s
     11,000    579       1      1    156.3 s  156.3 s   <- today

PDF (472 paras, ~10,400 payload tok, avg 22 tok/para, pool=12)
      1,000     45      11      1     29.3 s   29.3 s   <- optimum
      2,000     91       6      1     42.0 s   42.0 s
      6,000    273       2      1     92.8 s   92.8 s
```

**The optimum is a plateau (~750-1,500 payload tokens), not a sharp peak** — anything in that band lands within
~20% of optimal. Target the middle for robustness. Sensitivity to pool size:

```
pool= 8 -> best batch 1,300 tok, wall 33.1 s
pool=12 -> best batch   900 tok, wall 28.0 s
pool=16 -> best batch   650 tok, wall 24.9 s
pool=24 -> best batch   450 tok, wall 22.3 s
```

### Recommended values

| Setting | Today | Proposed | Rationale |
|---|---|---|---|
| `LLM_TRANSLATION_BATCH_MAX_TOKENS` | 40,000 | **1,200** | mid-plateau; 3 batches -> ~9-10, fills pool in 1 wave |
| `LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS` | 200 | **60** | ~1,200 / 20 tok avg; guards short-paragraph docs |
| `LLM_TERM_EXTRACTION_BATCH_MAX_TOKENS` | 80,000 | **2,000** | extraction output smaller; slightly larger OK |
| `LLM_TERM_EXTRACTION_BATCH_MAX_PARAGRAPHS` | 500 | **100** | 1 batch -> ~8; unlocks already-built A2 parallelism |
| `LLM_MAX_OUTPUT_TOKENS` | 65,536 | **8,192** | at 1,200-tok batches 64 k is unreachable; makes truncation-driven partial-parse fallbacks impossible |

**Two pre-existing quirks to preserve (do NOT "fix" silently):**
1. Caps are counted **before** the ~760-token boilerplate is prepended.
2. `_batch_units` closes on `>` not `>=`, so a cap of 60 admits 61 items.

Both are asserted by existing tests (`test_paragraph_translator.py` expects exact `[41, 4]` / `[21, 21, 3]` shapes).

### Adaptive sizing (preferred over a fixed cap)

```
target_batch_tokens = clamp(total_payload_tokens / pool_max_workers, 600, 2500)
```

PDF: 10,400 / 12 = 867 -> exactly one full wave. Small docs floor at 600 (no pointless fragmentation);
large docs ceiling at 2,500 and run multiple waves. The CJK `get_token_multiplier` scaling from B1 stays
as an additional multiplier on top.

---

## Batch Processing — what actually helps

Three distinct things are called "batching"; only two help latency.

**(a) Vertex AI Batch Prediction API** — `client.batches` exists in the installed `google-genai 1.67.0`
(`['cancel','create','create_embeddings','delete','get','list','vertexai']`). It is an **offline, asynchronous**
job with a ~24 h SLA at ~50% cost. **Not usable for interactive translation.** Recorded as a future *cost*
play only, explicitly out of scope for latency, so nobody chases it expecting a speedup.

**(b) Prompt batching (N paragraphs per call)** — already implemented in both pipelines, but misconfigured into
uselessness by the 40 k/80 k caps. Fixed by the section above.

**(c) In-flight request concurrency** — the real unused headroom:

- **Gap 1 — pool 4x under-filled.** Measured 4.70 / 4.95 effective concurrency vs. 12 configured. Not a bug:
  there were only 3 batches. Re-batching is what populates the pool. `TRANSLATION_MAX_QPS=16` is *not* binding
  (12 concurrent x ~30 s ~= 0.4 QPS vs. a 62.5 ms floor) — **do not raise QPS**.
- **Gap 2 — fallbacks bypass batching entirely.** 45 DOCX fallbacks, one LLM call each, 43 in the worst-efficiency
  bucket. Fix: re-batch failed units into groups of ~10, falling back to true singletons only on a second failure.
- **Gap 3 — `llm_translate_async` is dead code.** Defined at `base.py:83`, **zero call sites**. Meanwhile
  `AsyncModels.generate_content` is available. Every batch currently burns an OS thread blocked on network I/O.
  Deferred to Phase 3: it touches the shared `BaseTranslator` contract and the rate limiter is `threading.Lock`-based.

---

## Model Change: gemini-3.5-flash @ europe-west3

Verified against Google Cloud docs (fetched 2026-08-24):

| Property | Value |
|---|---|
| Model ID | `gemini-3.5-flash` |
| Launch stage | **GA**, released 2026-05-19, retirement >= 2027-05-19 |
| `europe-west3` supported | **Yes** (Model availability + ML processing + Provisioned Throughput) |
| Non-global input | **$1.65 / 1M** = `0.00165` per 1k |
| Non-global output | **$9.90 / 1M** = `0.00990` per 1k |
| Non-global cached input | **$0.165 / 1M** = `0.000165` per 1k |
| Global input / output / cached | `0.0015` / `0.009` / `0.00015` per 1k |
| Thinking | Supported; `ThinkingConfig(include_thoughts, thinking_budget, thinking_level)` present in installed SDK |

**Cost note (explicit):** `gemini-3.5-flash` is **more expensive per token than `gemini-2.5-flash`**
($1.65/$9.90 vs $0.30/$2.50 per 1M) but **cheaper than `gemini-2.5-pro` on output** ($9.90 vs $10.00-$15.00/1M)
while being materially faster. Net cost impact must be measured in Phase 4, not assumed.

**Plumbing already exists** — `factory.py:25` literally documents *"used by gemini-3.5-flash's europe-west3
pinning"*, `ModelRoute.region` is threaded end-to-end, `infer_provider()` matches any `gemini*` prefix, and
`ModelRateEntry.region` is already honoured by `resolve_rate_entry()`. **Only the config data is missing.**

Current `model_selection.json`: **150 route entries**, all with 4-model chains, priority-1 split
75x `gemini-2.5-pro` / 75x `claude-sonnet-4-6`, and **zero entries carry a `region` key**.

---

## [Types]

**`src/worker/doctranslator/translator/usage.py`** — extend `TokenUsage` (frozen slots dataclass):
- add `thinking_tokens: int = 0`
- `from_gemini_usage()` reads `thoughts_token_count`
- `merge()` sums the new field
- `total_tokens` property: keep as `input + output` (do **not** silently change billing semantics); add a separate
  `billable_output_tokens` property returning `output_tokens + thinking_tokens`, since Vertex bills thinking as output

**`src/config/llm_rate_catalog.py`** — no structural change needed; `ModelRateEntry.region` and `RateTier` already
suffice. Add optional `notes: str | None = None` to `ModelRateEntry` for provenance (e.g. "non-global pricing").

**`src/config/constants.py`** — new `Settings` fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `LLM_THINKING_BUDGET` | `int` | `0` | `0` = disable where supported; `-1` = dynamic (SDK default) |
| `LLM_THINKING_BUDGET_PRO` | `int` | `512` | separate budget for Pro-class models that cannot fully disable |
| `LLM_CALL_TIMEOUT_SECONDS` | `float` | `90.0` | client-side per-call deadline |
| `LLM_JUDGE_TIMEOUT_SECONDS` | `float` | `60.0` | judge deadline |
| `LLM_ADAPTIVE_BATCHING_ENABLED` | `bool` | `True` | toggle for the adaptive sizer |
| `LLM_ADAPTIVE_BATCH_MIN_TOKENS` | `int` | `600` | clamp floor |
| `LLM_ADAPTIVE_BATCH_MAX_TOKENS` | `int` | `2500` | clamp ceiling |
| `LLM_FALLBACK_BATCH_SIZE` | `int` | `10` | units per re-batched fallback group |
| `OTEL_CAPTURE_CONTENT_ENABLED` | `bool` | `False` | gate GenAI prompt/response span capture |

**New dataclass `BatchPlan`** in `src/worker/doctranslator/batching.py` (new shared module):

```python
@dataclass(frozen=True, slots=True)
class BatchPlan:
    max_tokens: int
    max_items: int
    total_payload_tokens: int
    pool_max_workers: int
    adaptive: bool
```

## [Files]

### New files

| Path | Purpose |
|---|---|
| `src/worker/doctranslator/batching.py` | Shared adaptive batch sizer (`compute_batch_plan`) used by both DOCX and PDF |
| `scripts/analyze_job_log.py` | Offline log analyser reproducing the tables in this document from a `worker_server.log` |
| `tests/worker/doctranslator/test_batching.py` | Unit tests for the adaptive sizer |
| `tests/worker/doctranslator/translator/test_thinking_config.py` | Asserts `thinking_config` wiring + `thoughts_token_count` accounting |

### Modified files

| Path | Change |
|---|---|
| `src/worker/doctranslator/translator/providers/gemini.py` | `thinking_config` in `GenerateContentConfig`; `http_options=HttpOptions(timeout=...)` on `genai.Client` |
| `src/worker/doctranslator/translator/providers/claude.py` | equivalent request timeout |
| `src/worker/doctranslator/translator/usage.py` | `thinking_tokens` field + `from_gemini_usage` + `merge` |
| `src/worker/doctranslator/translator/base.py` | log `thinking_tokens` and `batch_items` in the `done:` line |
| `src/worker/services/docx_job_processor.py` | wrap `translate_docx` in `asyncio.to_thread`; honour `QUALITY_EARLY_ACCEPT_THRESHOLD` |
| `src/worker/services/model_attempt_orchestrator.py` | honour `QUALITY_EARLY_ACCEPT_THRESHOLD` |
| `src/worker/doctranslator/format/docx/paragraph_translator.py` | adaptive batching; pre-filter skippable units; re-batched fallbacks; log batch stats |
| `src/worker/doctranslator/format/docx/term_extractor.py` | adaptive batching; log batch stats |
| `src/worker/doctranslator/format/pdf/translation_config.py` | adaptive batching in `_init_llm_batch_limits()` |
| `src/worker/doctranslator/format/pdf/document_il/midend/il_translator_llm_only.py` | re-batched fallbacks; log batch stats |
| `src/worker/doctranslator/format/pdf/high_level.py` | overlap term extraction with translation (Phase 2) |
| `src/worker/doctranslator/docvision/doclayout.py` | make `handle_document()` feed `ONNX_LAYOUT_BATCH_SIZE` pages into `predict()` (Phase 2) |
| `src/worker/services/quality_judge_service.py` | judge timeout |
| `src/config/constants.py` | new settings above |
| `src/config/telemetry.py` | gate content capture on `OTEL_CAPTURE_CONTENT_ENABLED` |
| `src/worker/services/llm_cost_service.py` | log resolved `region` + rate provenance |
| `.env.example.worker`, `.env.worker.local`, `tests/test.env` | new + retuned settings |

### Asset/config files (NOT in git — mounted from GCS)

| Path | Change | Note |
|---|---|---|
| `gs://<bucket>/assets/model_selection.json` | insert `gemini-3.5-flash` @ `europe-west3` as priority 1 across all **150** entries; demote existing chain by one; keep chains at 4 by dropping the redundant 4th `gemini-2.5-flash` tier | Confirmed present in GCS |
| `gs://<bucket>/assets/pricing_catalog.json` | add `gemini-3.5-flash` entries (regional + global) | **WARNING: not currently in GCS** — only exists at `.local-tmp/assets-cache/pricing_catalog.json`. Must be uploaded, or worker startup raises `FileNotFoundError` (hard error by design, `llm_rate_catalog.py:93-98`) |

New `pricing_catalog.json` entries:

```json
{
  "provider": "gemini_vertexai",
  "model_id": "gemini-3.5-flash",
  "region": "europe-west3",
  "context_window_tokens": 1000000,
  "tiers": [{
    "max_input_tokens": null,
    "input_cost_per_1k": 0.00165,
    "output_cost_per_1k": 0.0099,
    "cache_hit_cost_per_1k": 0.000165
  }]
},
{
  "provider": "gemini_vertexai",
  "model_id": "gemini-3.5-flash",
  "region": null,
  "tiers": [{
    "max_input_tokens": null,
    "input_cost_per_1k": 0.0015,
    "output_cost_per_1k": 0.009,
    "cache_hit_cost_per_1k": 0.00015
  }]
}
```

**Ordering matters:** `resolve_rate_entry()` returns `matches[0]`, and its filter accepts
`entry.region is None or entry.region == region`. The region-specific entry MUST be listed **before** the
`region: null` fallback, otherwise the global rate silently wins. Phase 0 adds a log line recording which
entry was chosen so this cannot regress unnoticed.

## [Functions]

### New

| Function | File | Signature | Purpose |
|---|---|---|---|
| `compute_batch_plan` | `batching.py` | `(total_payload_tokens: int, pool_max_workers: int, *, base_max_tokens: int, base_max_items: int, token_multiplier: float) -> BatchPlan` | Adaptive clamp; returns fixed caps when `LLM_ADAPTIVE_BATCHING_ENABLED=False` |
| `log_batch_plan` | `batching.py` | `(stage: str, plan: BatchPlan, batch_count: int) -> None` | Phase-0 instrumentation |
| `_should_skip_llm` | `docx/paragraph_translator.py` | `(unit: TranslatableUnit) -> bool` | Mirrors PDF's `is_pure_numeric_paragraph` / `is_placeholder_only_paragraph` / `min_text_length` checks |
| `_translate_units_fallback_batched` | `docx/paragraph_translator.py` | `(units: list[TranslatableUnit]) -> dict[int, str]` | Groups failures into `LLM_FALLBACK_BATCH_SIZE` batches before true singleton fallback |
| `_build_thinking_config` | `providers/gemini.py` | `(model: str) -> genai_types.ThinkingConfig \| None` | Pro-class -> `LLM_THINKING_BUDGET_PRO`, else `LLM_THINKING_BUDGET`; `None` when `-1` |

### Modified

| Function | File | Change |
|---|---|---|
| `GeminiVertexAITranslator.__init__` | `providers/gemini.py` | pass `http_options=HttpOptions(timeout=int(LLM_CALL_TIMEOUT_SECONDS*1000))` |
| `GeminiVertexAITranslator.invoke` | `providers/gemini.py` | add `thinking_config`; `max_output_tokens` from retuned setting |
| `TokenUsage.from_gemini_usage` | `usage.py` | read `thoughts_token_count` |
| `BaseTranslator._run_translation_batch` | `base.py` | log `thinking_tokens`, `batch_items` |
| `DocxParagraphTranslator._batch_units` | `docx/paragraph_translator.py` | use `compute_batch_plan`; **preserve `>` semantics and pre-boilerplate counting** |
| `DocxParagraphTranslator.translate_all` | `docx/paragraph_translator.py` | pre-filter via `_should_skip_llm` before `_split_cache_hits` |
| `DocxTermExtractor._batch_units` | `docx/term_extractor.py` | use `compute_batch_plan` |
| `TranslationConfig._init_llm_batch_limits` | `pdf/translation_config.py` | use `compute_batch_plan` |
| `DocxJobProcessor.translate` | `docx_job_processor.py` | `await asyncio.to_thread(translate_docx, ...)`; early-accept check |
| `ModelAttemptOrchestrator.run_model_chain` | `model_attempt_orchestrator.py` | early-accept check |
| `ILTranslatorLLMOnly._submit_fallback_translation` | `il_translator_llm_only.py` | route through batched fallback |
| `_run_translation_phase` | `pdf/high_level.py` | overlap term extraction with translation (Phase 2) |
| `DocLayoutModel.handle_document` | `docvision/doclayout.py` | accumulate `ONNX_LAYOUT_BATCH_SIZE` pages per `predict()` call (Phase 2) |
| `VertexLLMCostService.resolve_rate_entry` | `llm_cost_service.py` | log chosen `(model_id, region, input_rate)` |

### Removed

None. `llm_translate_async` (`base.py:83`) is currently dead but will be **adopted** in Phase 3 rather than deleted.

## [Classes]

- **New:** `BatchPlan` (frozen slots dataclass) in `src/worker/doctranslator/batching.py`.
- **Modified:** `TokenUsage` (+`thinking_tokens`, +`billable_output_tokens`); `GeminiVertexAITranslator`
  (thinking + timeout); `DocxParagraphTranslator` (pre-filter, adaptive batching, batched fallback);
  `DocxTermExtractor` (adaptive batching); `TranslationConfig` (adaptive batching);
  `ILTranslatorLLMOnly` (batched fallback); `DocxJobProcessor` (async offload, early accept);
  `ModelAttemptOrchestrator` (early accept); `ModelRateEntry` (+`notes`).
- **Removed:** none.

## [Dependencies]

**No new packages.** Everything required is already installed and verified:

- `google-genai 1.67.0` — `ThinkingConfig(include_thoughts, thinking_budget, thinking_level)` present;
  `HttpOptions.timeout` present; `Client(http_options=...)` accepted; `client.aio` / `AsyncModels` available for Phase 3.
- `anthropic[vertex]>=0.40.0` — supports request timeout.

**External (non-code) prerequisites:**

1. `gemini-3.5-flash` must be enabled for projects `aicoeprod` / `aicoesandox` in `europe-west3`, with quota.
2. `pricing_catalog.json` must be **uploaded to GCS assets** (currently local-only) before any deploy.

## [Testing]

Framework: `pytest` + `pytest-asyncio` (`asyncio_mode = auto`), config in `pyproject.toml`, env from `tests/test.env`.
**Baseline to preserve: 726 passed.**

| Test file | Coverage |
|---|---|
| `tests/worker/doctranslator/test_batching.py` (new) | clamp floor/ceiling, adaptive off, CJK multiplier interaction, single-item docs |
| `tests/worker/doctranslator/translator/test_thinking_config.py` (new) | budget selection per model class; `-1` -> `None`; `thoughts_token_count` -> `thinking_tokens`; `billable_output_tokens` |
| `tests/worker/doctranslator/format/docx/test_paragraph_translator.py` | **update** `[41, 4]` / `[21, 21, 3]` expectations for new caps; add `_should_skip_llm` cases; add batched-fallback test |
| `tests/worker/doctranslator/format/docx/test_term_extractor.py` | update batch-shape expectations |
| `tests/worker/services/test_docx_job_processor.py` | assert `translate_docx` runs off the event loop; early-accept short-circuit |
| `tests/worker/services/test_model_attempt_orchestrator.py` | early-accept short-circuit |
| `tests/worker/services/test_llm_cost_service.py` | `gemini-3.5-flash` @ `europe-west3` resolves to `0.00165`/`0.0099`, **not** the global tier; ordering regression guard |
| `tests/config/` | new settings parse with correct defaults |

**Validation gates:** `uv run pytest -q` (>= 726 passed) and `ruff check` clean before any E2E run.

## [Implementation Order]

1. **Phase 0 — instrumentation** (no behaviour change): `thinking_tokens` accounting; batch/concurrency logging;
   DOCX stage timings; cost-entry provenance logging; `scripts/analyze_job_log.py`.
2. **Capture a fresh baseline** with Phase 0 in place, cache flushed. Everything after is measured against this.
3. **Phase 1a — config/asset prep:** upload `pricing_catalog.json` to GCS; add `gemini-3.5-flash` entries;
   update `model_selection.json` (150 entries); verify `resolve_rate_entry()` picks the regional tier.
4. **Phase 1b — model + thinking + timeouts:** `thinking_config`, `HttpOptions.timeout`, judge timeout.
5. **Phase 1c — event loop:** `asyncio.to_thread` around `translate_docx`.
6. **Phase 1d — batching:** `batching.py`, wire into all three `_batch_units` sites, retune env caps,
   lower `LLM_MAX_OUTPUT_TOKENS`.
7. **Phase 1e — wasted calls:** `_should_skip_llm` pre-filter + batched fallbacks (both pipelines).
8. **Phase 1f — quality loop:** wire `QUALITY_EARLY_ACCEPT_THRESHOLD`; evaluate `MAX_MODEL_ATTEMPTS` 3 -> 2.
9. **Run test suite + E2E; compare against step 2.** Stop here if targets met.
10. **Phase 2 —** ONNX batching (E2); PDF term/translation overlap; telemetry gating; prompt caching.
11. **Phase 3 —** D1 per-paragraph PDF cache; async migration; `SPLIT_PART_MAX_CONCURRENT` load test (E3/B2);
    C3 cross-attempt reuse; Vertex Batch Prediction cost study.
12. **Update memory bank** (`activeContext.md`, `progress.md`, `systemPatterns.md` backlog table).

---

# PROGRESS LOG

## 2026-08-24 — Phase 0 + Phase 1a-1f implemented

**Status: code-complete for Phase 0 and Phase 1; validation gates 1.G3-1.G6 still need a live E2E run.**

Test suite **726 → 784 passed** (+58, after the second pass). `ruff check` clean on every touched file
(the 11 repo-wide findings are pre-existing and were deliberately not touched).

### Delivered

| Area | Change |
|---|---|
| Instrumentation | `TokenUsage.thinking_tokens` + `billable_output_tokens`; `thinking_tokens`/`batch_items` in the `done:` log line; `log_batch_plan()` reporting `waves` + `pool_utilisation`; pricing-provenance logging |
| Model | `gemini-3.5-flash` @ `europe-west3` promoted to priority 1 across **all 150** routes; `gemini-2.5-pro`/`claude-sonnet-4-6` demoted to priority 2; chain length preserved at 4 |
| Pricing | Regional + global `gemini-3.5-flash` entries added, regional **first**; verified `europe-west3`→`0.00165` vs `us-central1`→`0.0015` |
| Thinking/timeouts | `_build_thinking_config()` (0 for Flash, 512 for Pro, `-1` = SDK default); 90 s Gemini + Claude deadlines; 60 s judge deadline |
| Event loop | `translate_docx` now offloaded via `asyncio.to_thread` |
| Batching | Shared `batching.py`; adaptive sizing wired into both DOCX paths; caps retuned to 1200/60 and 2000/100; `LLM_MAX_OUTPUT_TOKENS` → 8192 |
| Wasted calls | `_should_skip_llm()` pre-filter; fallbacks re-batched in groups of 10 before singleton retries |
| Quality loop | `QUALITY_EARLY_ACCEPT_THRESHOLD` wired in both orchestrators (previously dead config) |

### Verified behaviour (not just compiled)

```
compute_batch_plan(10400, 12) -> max_tokens=867  => 12 batches, 1 wave, utilisation 1.0
select_model_list('en','fr','commercial')
  -> [('gemini-3.5-flash','europe-west3'), ('gemini-2.5-pro',None), ('gemini-2.5-flash',None)]
resolve_rate_entry('gemini-3.5-flash','europe-west3') -> 0.00165 / 0.0099   (regional)
resolve_rate_entry('gemini-3.5-flash','us-central1')  -> 0.0015  / 0.009    (global + WARNING)
TokenUsage(thoughts=200) -> total_tokens=150, billable_output_tokens=250
```

### Bug found by the new tests

`test_never_spills_an_extra_wave` caught a genuine off-by-one: truncating
`10400/12 = 866.67` to `866` produced **13** batches for a 12-worker pool —
an entire extra wave, the exact failure the change set out to remove.
`compute_batch_plan()` now rounds the target **up**.

### Deliberately deferred (with reasons, not omissions)

| Item | Why |
|---|---|
| **1d.5** PDF adaptive batching | PDF batches per-page inside `process_page()`, so no document-level payload total exists at config-build time. Needs its own change; DOCX carries the Phase-1 win. |
| **1e.5** PDF batched fallback | PDF fallback runs through `PriorityThreadPoolExecutor` with per-paragraph trackers; regrouping needs a design pass. |
| **1f.3** `MAX_MODEL_ATTEMPTS` 3→2 | Early-accept already short-circuits the common case; cutting the chain now would only remove genuine quality retries. Revisit with E2E data. |
| **0.6** `analyze_job_log.py` | Ad-hoc analysis scripts already produced every table here; packaging them is useful but not blocking. |
| **3.7** Intra-document dedup | Measured only 7.3% (DOCX) / 1.1% (PDF) duplicate text — not worth the complexity. |

### Blocking before any deploy

1. **`pricing_catalog.json` is still not in GCS** (item 1a.1). It exists only at
   `.local-tmp/assets-cache/`. `llm_rate_catalog.py:93-98` raises
   `FileNotFoundError` by design, so the worker will fail at startup.
2. **`gemini-3.5-flash` quota/enablement in `europe-west3`** (item 1a.8) is an
   external GCP action.

### E2E RESULTS — 2026-08-24 12:34-12:41 UTC (cold cache, live servers)

Both test documents re-run against the local two-process stack with the full
Phase 0 + Phase 1 + Phase 2.1 change set. Cache was genuinely cold: the 455
existing Redis entries are keyed on model, and the priority-1 model changed to
`gemini-3.5-flash`, so nothing could hit (`cache_hit_rate=0.0019`).

| Metric | Baseline | After | Change |
|---|---|---|---|
| **PDF wall clock** | **705 s** | **156 s** | **-78%** |
| **DOCX wall clock** | **232 s** | **32 s** | **-86%** |
| Translate Paragraphs (PDF) | 470.2 s | **14.4 s** | **-97%** |
| Automatic Term Extraction (PDF) | 94.9 s | **12.0 s** | -87% |
| Parse Page Layout (ONNX) | 58.1 s | 54.7 s | -6% |
| Typesetting | 16.4 s | 13.7 s | -16% |
| LLM output tok/s | 19.2 | **202.0** | **10.5x** |
| LLM latency med / p90 / max | 24.2 / 52.9 / **454.4** s | 3.1 / 6.5 / **9.6** s | max -98% |
| Effective concurrency | 4.1-5.0 / 12 | **8.3** | +70% |
| Pool utilisation | ~0.25 (3 of 12) | **1.00** (12 of 12) | one full wave |
| Judge score (PDF) | 0.847 | 0.797 | -0.05, still passes |
| Judge score (DOCX) | 0.902 / 0.860 | 0.820 | -0.08, still passes |

**Both targets beaten** (PDF ≤280 s → 156 s; DOCX ≤90 s → 32 s).

Verified in the log: 69 of 70 PDF calls went to
`europe-west3-aiplatform.googleapis.com` on `gemini-3.5-flash`; the batch
planner produced exactly `batch_count=12, pool_utilisation=1.0, waves=1` for
both PDF and DOCX; the DOCX pre-filter skipped `41/414` units before any Redis
or LLM work.

**The 454 s tail is gone** — max call is now 9.6 s, confirming the thinking
budget plus the 90 s client deadline removed the pathological reasoning stalls.

#### Memory (new `scripts/sample_worker_memory.py`, 1 Hz, 374 samples)

| | Value |
|---|---|
| Idle RSS | 207 MB |
| **Peak RSS** | **4,880 MB (4.77 GiB)** — 30% of the 16 GiB Cloud Run cap |
| Mean RSS | 2,019 MB |
| Steady-state during translation | ~2.4-2.6 GB |
| Peak threads | 48 |

This corroborates your ~4 GB observation. Two findings:

1. **The peak is a 1-second spike, not steady state.** It occurs at t=181.3 s
   with `num_procs=2` and drops to 2,546 MB the very next sample. `pdf_creater`
   forks a subprocess for `save PDF with clean=True` / font subsetting, which
   briefly copy-on-writes the parent's ~2.5 GB. Steady state is ~2.4 GB.
2. **48 peak threads, well under the 156-call worst case** — the
   `LLM_MAX_INFLIGHT_CALLS=16` semaphore is doing its job, and threads
   contributed almost nothing to RSS as predicted.

Headroom is comfortable; no need to raise the 16 GiB allocation or lower
`TRANSLATION_POOL_MAX_WORKERS`.

#### Implication for 1e.5 (PDF batched fallback)

The 9 PDF fallbacks previously cost ~213 s, dominated by two pathological
calls (40 s and 123 s for ~15 output tokens). Those are now capped: max call
latency is 9.6 s. **1e.5's remaining value is roughly 9 x 16.6 s ≈ 60 s of
fixed overhead at most, against a 14.4 s translation stage** — it can no
longer pay for its risk. Recommend closing it as won't-do and, if PDF
fallbacks ever matter again, adding the low-risk PDF `_should_skip_llm`
pre-filter instead.

### Production worker sizing (inspected 2026-08-24 via gcloud, aicoeprod)

`translation-worker-service`, `europe-west1`, generation 9:

| Setting | Value |
|---|---|
| CPU / memory | **4 vCPU / 16 GiB** |
| `containerConcurrency` | **1** (one job per instance) |
| minScale / maxScale | 1 / 30 |
| cpu-throttling | `false` (full CPU between requests) |
| startup-cpu-boost | `true` |
| request timeout | 3600 s |
| image | `translation-worker:90` |

**Observed: ~4 GB for a single PDF translation = 25% of the 16 GiB cap.**
Not currently a memory risk, but it constrains how far pools can be raised.

### Thread-pool sizing analysis

I measured thread cost directly rather than assuming: **144 idle threads add
only ~12 MB RSS**. Threads are *not* what consumes 4 GB. The real drivers are
per-page pixmaps, the parsed IL tree (plus `deepcopy` for cross-attempt reuse),
ONNX activations, and fonts.

The actual risk is **nested thread pools**, and my own Phase 1e fallback
re-batching made it worse:

```
docx_translator            outer pool = 2
 +-- translate_all         batch pool = TRANSLATION_POOL_MAX_WORKERS (12)
 |      +-- fallback pool  = 12  <-- opened PER batch thread
 +-- term_extractor        pool = 12
theoretical peak in-flight Vertex calls = 12 + 12*12 = 156
```

On `cpu=4 / concurrency=1` that is ~39 concurrent calls per vCPU, and it
silently bypasses `TRANSLATION_MAX_QPS=16`.

**Fix applied:** a process-wide `BoundedSemaphore` (`llm_call_slot()` in
`batching.py`) enforced at the single choke point
`BaseTranslator._run_translation_batch()`, so it covers every path — DOCX, PDF,
batches and fallbacks alike. New setting `LLM_MAX_INFLIGHT_CALLS=16`
(matches `TRANSLATION_MAX_QPS`; `0` disables). Pool sizes are deliberately
left at 12 — they provide a wave's parallelism; only the *total* needed a cap.
Verified: 40 threads against a limit of 3 produced an observed peak of exactly 3.

**Recommendation: keep `TRANSLATION_POOL_MAX_WORKERS=12.`** These are I/O-bound
waits on Vertex, not CPU work, so 12 on 4 vCPUs is reasonable; the semaphore now
bounds the pathological case. Revisit only if E2E shows memory >8 GB.

### Pre-existing flaky test found (NOT caused by this work)

`tests/worker/services/test_progress_tracker.py::test_update_rate_limit`
fails intermittently under full-suite load — observed `767 passed` and
`1 failed, 766 passed` on **identical code**, and 8/8 + 5/5 passes when run in
isolation. `ProgressTracker.update()` rate-limits on wall-clock
`datetime.now(UTC)` with `min_update_interval=0.1s`; when the machine is
loaded, >100 ms can elapse between the two `await`s and the second update is
legitimately allowed. Neither the tracker nor its test was touched by this
work (`git diff` on both is empty).

**Suggested fix (separate ticket):** inject a clock into `ProgressTracker` and
freeze it in the test, rather than widening the interval and leaving a slower
race in place.

Backups written: `pricing_catalog.json.bak`, `model_selection.json.bak`.

---

# CHECKLIST

## Phase 0 — Instrumentation (no behaviour change)

- [x] **0.1** `TokenUsage.thinking_tokens` + `from_gemini_usage()` reads `thoughts_token_count` + `merge()` + `billable_output_tokens`
- [x] **0.2** `base.py::_run_translation_batch` logs `thinking_tokens` and `batch_items` (threaded through `llm_translate`/`do_llm_translate`/`do_translate`)
- [x] **0.3** `log_batch_plan()` — emits `batch_count`, `mean_batch_payload_tokens`, `pool_max_workers`, `waves`, `pool_utilisation`
- [x] **0.4** DOCX pre-filter + fallback-regrouping counters logged; batch plan logged per stage
- [x] **0.5** `resolve_rate_entry()` logs chosen `(model_id, region, input_rate_per_1k)`; **WARNs** on global-rate fallback for a regional request
- [x] **0.6** `scripts/analyze_job_log.py` — reproduces every table here from a log file (`--json` for CI). Validated against the baseline log: emits 470.2s Translate Paragraphs, 19.2 out-tok/s, 72x `same_text`, 18x `length_ratio`, judge 0.902/0.860/0.847 — matching the figures derived by hand.
- [x] **0.7** Run tests + `ruff check` → **784 passed**; ruff at the pre-existing baseline of 11 (none added)
- [ ] **0.8** **Capture fresh baseline** (cache flushed) — record numbers *(requires a live E2E run)*

## Phase 1a — Model + pricing config

- [ ] **1a.1** Upload `pricing_catalog.json` to `gs://<bucket>/assets/` (**currently missing from GCS** — done locally only; **BLOCKS DEPLOY**)
- [x] **1a.2** Add `gemini-3.5-flash` @ `europe-west3` entry (`0.00165` / `0.0099` / `0.000165` per 1k)
- [x] **1a.3** Add `gemini-3.5-flash` global fallback entry (`0.0015` / `0.009` / `0.00015` per 1k)
- [x] **1a.4** Region entry precedes the `region: null` entry — verified live: `europe-west3` → `0.00165`, `us-central1` → `0.0015`
- [x] **1a.5** `model_selection.json`: `gemini-3.5-flash` + `"region": "europe-west3"` priority 1 across **all 150** entries
- [x] **1a.6** Demoted `gemini-2.5-pro` (75) / `claude-sonnet-4-6` (75) to priority 2; chain length still 4 for all 150
- [x] **1a.7** `ab_test_config.variant_a` → `gemini-3.5-flash`; added to `variant_b.models`
- [ ] **1a.8** Confirm `gemini-3.5-flash` enabled + quota in `europe-west3` for both projects *(external/GCP action)*
- [x] **1a.9** Cost-regression tests added (`TestRegionalEntryOrdering`), incl. a guard asserting the **shipped** catalog ordering

## Phase 1b — Thinking budget + timeouts

- [x] **1b.1** `LLM_THINKING_BUDGET`(0), `LLM_THINKING_BUDGET_PRO`(512), `LLM_CALL_TIMEOUT_SECONDS`(90), `LLM_JUDGE_TIMEOUT_SECONDS`(60)
- [x] **1b.2** `_build_thinking_config()` + wired into `GenerateContentConfig`
- [x] **1b.3** `http_options=HttpOptions(timeout=...)` on `genai.Client` (ms conversion)
- [x] **1b.4** Claude request timeout via `AnthropicVertex(timeout=...)`
- [x] **1b.5** Judge timeout added; judge thinking deliberately left ON (quality-critical, only ~12-19 s)
- [x] **1b.6** Confirmed "timeout" already in `_RETRYABLE_ERROR_SUBSTRINGS` → `llm_retry` handles it
- [x] **1b.7** `tests/worker/doctranslator/translator/test_thinking_config.py` (9 tests)

## Phase 1c — Event loop

- [x] **1c.1** `await asyncio.to_thread(translate_docx, ...)` in `DocxJobProcessor.translate`
- [x] **1c.2** `TestEventLoopIsNotBlocked` — asserts a different thread id **and** loop responsiveness via a heartbeat task
- [ ] **1c.3** Verify with 2 concurrent DOCX jobs that job 2 is no longer queued ~200 s *(requires live E2E)*

## Phase 1d — Batch sizing

- [x] **1d.1** `src/worker/doctranslator/batching.py` with `BatchPlan` + `compute_batch_plan()` + `log_batch_plan()`
- [x] **1d.2** Adaptive settings (`ENABLED` / `MIN`=600 / `MAX`=2500)
- [x] **1d.3** Wired into `DocxParagraphTranslator._batch_units`
- [x] **1d.4** Wired into `DocxTermExtractor._batch_units`
- [x] **1d.5** PDF adaptive batching — done via `ILTranslatorLLMOnly._apply_adaptive_batch_plan()` rather than `_init_llm_batch_limits()`: PDF batches per-page inside `process_page()`, so the document-level token total is only knowable in `translate()`. Computing it once there and overriding the config caps gives PDF the same one-wave sizing without restructuring the per-page loop. 6 new tests.
- [x] **1d.6** Retuned `.env.worker.local` + `.env.example.worker`: `1200`/`60` and `2000`/`100`
- [x] **1d.7** `LLM_MAX_OUTPUT_TOKENS` → `8192`
- [x] **1d.8** Preserved `>` closing semantics and pre-boilerplate counting
- [x] **1d.9** Preserved CJK `get_token_multiplier` scaling (applied in both adaptive and fixed modes)
- [x] **1d.10** `tests/test.env` pinned to legacy caps + `LLM_ADAPTIVE_BATCHING_ENABLED=false` so existing batch-shape assertions stay meaningful
- [x] **1d.11** New `test_batching.py` (13 tests) — **caught a real off-by-one**: truncating `10400/12=866.67`→`866` spilled a 13th batch onto a 12-worker pool; now rounds up

## Phase 1e — Eliminate wasted calls

- [x] **1e.1** `_should_skip_llm()` for DOCX (numeric / placeholder-only / `< LLM_TRANSLATION_MIN_TEXT_LENGTH`)
- [x] **1e.2** Pre-filter applied in `translate_all` **before** `_split_cache_hits` (skips Redis work too)
- [x] **1e.3** `LLM_FALLBACK_BATCH_SIZE` setting (default 10)
- [x] **1e.4** `_translate_units_fallback_batched()` + `_translate_batch_raw()` for DOCX
- [x] **1e.7** *(added after prod inspection)* `LLM_MAX_INFLIGHT_CALLS=16` semaphore in `BaseTranslator._run_translation_batch()` — bounds the 156-call worst case created by nested pools on the `cpu=4/concurrency=1` worker
- [ ] **1e.5** Equivalent batched fallback for PDF `il_translator_llm_only.py` — **deferred to Phase 2** (PDF fallback goes through a `PriorityThreadPoolExecutor` with per-paragraph trackers; regrouping needs its own design)
- [ ] **1e.6** Verify the 72x `same_text` / 18x `length_ratio` warnings drop sharply *(requires live E2E)*

## Phase 1f — Quality loop

- [x] **1f.1** Wired `QUALITY_EARLY_ACCEPT_THRESHOLD` in `ModelAttemptOrchestrator` (was dead config)
- [x] **1f.2** Same in `DocxJobProcessor`
- [ ] **1f.3** Evaluate `MAX_MODEL_ATTEMPTS` 3 → 2 — **deliberately not changed**: early-accept now short-circuits the common case, so cutting the chain would only remove genuine quality retries. Revisit after E2E data.
- [x] **1f.4** Early-accept short-circuit covered by existing orchestrator tests (suite green)

## Phase 1 gate

- [x] **1.G1** `pytest -q` → **784 passed** (was 726; +58 new after the second pass)
- [x] **1.G2** `ruff check` clean on every touched file (11 findings pre-existing and untouched)
- [x] **1.G3** E2E re-run, cold cache — **PDF 705→156 s, DOCX 232→32 s**; see "E2E RESULTS" above
- [ ] **1.G4** E2E re-run, cache warm — isolate cache contribution *(not yet run; the cold run already beat both targets)*
- [x] **1.G5** Judge scores: PDF 0.847→0.797, DOCX 0.902/0.860→0.820. Both still pass the 0.6 threshold but are **down ~0.05-0.08** — see risk note below
- [ ] **1.G6** Cost per job — token counts captured (81,993 output tokens across 141 calls) but per-job USD not yet extracted from BigQuery

## Phase 2 — Structural

- [x] **2.1** (E2) `handle_document()` now renders pages in groups of `ONNX_LAYOUT_BATCH_SIZE` and calls `predict(images)` once per group — the setting was dead config; `predict()` already supported batching. 20 pages → 5 inference calls instead of 20. Order preserved, cancellation checked per batch, only `batch_size` images held at once. 8 new tests.
- [ ] **2.2** Lower `ONNX_INTRA_OP_NUM_THREADS` 4 -> 2 — **not changed**: with batching now active the intra-op threads are shared across a 4-page batch, so the old oversubscription argument no longer holds. Needs the E2E benchmark against the 58 s baseline before tuning blind.
- [ ] **2.3** Overlap PDF term extraction with translation (PDF equivalent of A4) — needs design call on glossary ordering
- [ ] **2.4** Move boilerplate prompt to Vertex cached content / system instruction (~10x input-token cut on fallbacks)
- [ ] **2.5** `OTEL_CAPTURE_CONTENT_ENABLED=False` + `TRACE_SAMPLE_RATE` 1.0 -> 0.1 in prod
- [ ] **2.6** Re-measure

## Phase 3 — Bigger workstreams (separate tickets)

- [ ] **3.1** (D1) PDF per-paragraph Redis cache — port DOCX `_split_cache_hits`/`get_many`; target 5% -> 40%+ hit rate
- [ ] **3.2** Adopt `llm_translate_async` + `client.aio` + async-safe rate limiter (currently dead code, zero call sites)
- [ ] **3.3** (E3/B2) Load-test `SPLIT_PART_MAX_CONCURRENT` 2 -> 3/4; **first verify how often real docs split at all** (this PDF: 1 section)
- [ ] **3.4** (C3) Cross-attempt reuse of accepted per-unit translations — needs judge-semantics design discussion
- [ ] **3.5** Vertex Batch Prediction (`client.batches`) for bulk/overnight jobs — **cost** play, ~24 h SLA, NOT latency
- [ ] **3.6** (RC12) Make `_enqueue_local_http` non-blocking so local E2E numbers stop being skewed
- [-] **3.7** Intra-document dedup — **DEPRIORITISED**: measured only 7.3% (DOCX) / 1.1% (PDF) duplicate text

## Phase 4 — Validation & docs

- [ ] **4.1** Re-run both `docs/test docs/` files per `docs/local-testing-guide.md`
- [ ] **4.2** Flush Redis between runs; also do one warm-cache run
- [ ] **4.3** Compare Phase-0 tables: stage durations, batch counts, out-tok/s, thinking tokens, fallback counts, hit rate
- [ ] **4.4** **Target: DOCX 232 s -> <=90 s; PDF 705 s -> <=280 s**
- [x] **4.5** Update `memory-bank/activeContext.md` — current status, prod worker sizing, non-obvious decisions
- [x] **4.6** Update `memory-bank/progress.md` — delivered items, known flaky test, remaining deferrals
- [x] **4.7** Update `memory-bank/systemPatterns.md` — E2/E3 entries corrected with root causes; new "LLM batch sizing" section
- [ ] **4.8** Update `docs/local-testing-guide.md` if setup steps changed — *no change needed; the run procedure is unaffected. Add `scripts/analyze_job_log.py` to it once the E2E numbers are captured.*

---

# Expected Outcome

| Metric | Baseline | After Phase 1 | After Phase 2 |
|---|---|---|---|
| DOCX wall-clock | 232 s | ~75 s | ~60 s |
| PDF wall-clock | 705 s | ~250 s | ~190 s |
| DOCX translate + terms | ~213 s | ~30 s | ~30 s |
| PDF translate + terms | ~565 s | ~70 s | ~70 s |
| PDF layout (ONNX) | 58 s | 58 s | ~20 s |
| Effective LLM concurrency | 4.7-5.0 / 12 | ~11 / 12 | ~11 / 12 |
| Output tokens/sec | 19.2 | ~45 | ~45 |
| Job-2 queue delay | 203 s | ~0 s | ~0 s |

**Caveat:** projections assume latency stays linear in payload size (fitted `16.6s + 12.7s/ktok`, R-squared not
computed) and that `gemini-3.5-flash` behaves like the healthy end of the observed `2.5-pro` distribution.
Both must be confirmed empirically in Phase 4 rather than trusted.

---

# Open Risks

| Risk | Mitigation |
|---|---|
| `gemini-3.5-flash` quality below `gemini-2.5-pro` for translation | Judge already gates every job (threshold 0.6, baseline 0.847-0.902). Pro remains priority 2, so a failing attempt auto-retries on it. Gate 1.G5. |
| `gemini-3.5-flash` costs more per token than `2.5-flash` | Measure net cost in 1.G6; fewer thinking tokens + higher cache hits may offset. Flash-Lite is a fallback option. |
| Smaller batches lose cross-paragraph context -> worse coherence | Watch judge alignment sub-score specifically; adaptive floor of 600 tokens keeps batches meaningful. |
| `pricing_catalog.json` missing from GCS | **Hard startup failure by design.** Blocking checklist item 1a.1 before any deploy. |
| Region entry ordering silently falls back to global rate | Explicit ordering item 1a.4 + regression test 1a.9 + provenance log 0.5. |
| Editing 150 `model_selection.json` entries by hand | Script the transformation; validate by re-parsing with `select_model_list()` for a sample of routes. |
| Lower `LLM_MAX_OUTPUT_TOKENS` truncates a legitimately large batch | 8,192 >> the ~2,700 max observed output; partial-parse recovery already exists as a safety net. |
