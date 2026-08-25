# Production env delta — latency optimisation (2026-08-24)

> **Do NOT copy `.env.worker.local` / `.env.api.local` into production.**
> They contain `IS_LOCAL=true`, `WORKER_SKIP_OIDC_VERIFICATION=true`,
> `CLOUD_TASKS_WORKER_URL=http://localhost:8001/...`, the **aicoesandox**
> project/bucket/dataset, and machine-specific `ASSETS_ROOT` / `TEMP_DIR`
> paths. Copying them wholesale would point prod at the sandbox and disable
> OIDC verification on the worker's task endpoint.

This file lists **only** the settings that must change in the
`translation-worker-env` secret (project `aicoeprod`) to reproduce the
validated latency improvement.

## What the E2E actually ran with

The local servers were started as:

```bash
DOTENV_PATH=.env.worker.local PORT=8001 python main_worker.py
DOTENV_PATH=.env.api.local    PORT=8000 python main_api.py
```

Important: **9 of the new settings were not present in either env file** —
they took their defaults from `src/config/constants.py`. The effective values
during the run were confirmed by loading `Settings` directly:

```
LLM_THINKING_BUDGET           = 0
LLM_THINKING_BUDGET_PRO       = 512
LLM_CALL_TIMEOUT_SECONDS      = 90.0
LLM_JUDGE_TIMEOUT_SECONDS     = 60.0
LLM_ADAPTIVE_BATCHING_ENABLED = True
LLM_ADAPTIVE_BATCH_MIN_TOKENS = 600
LLM_ADAPTIVE_BATCH_MAX_TOKENS = 2500
LLM_FALLBACK_BATCH_SIZE       = 10
LLM_MAX_INFLIGHT_CALLS        = 16
```

Because these have safe defaults in code, **prod will pick them up on deploy
even if you add nothing**. Adding them explicitly is still recommended so the
values are visible and tunable without a code change.

## 1. Changed values (already exist in the prod secret)

| Key | Prod now | Set to | Why |
|---|---|---|---|
| `LLM_TRANSLATION_BATCH_MAX_TOKENS` | `4000` | **`1200`** | 3 batches → ~12; fills the 12-worker pool in one wave |
| `LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS` | `40` | **`60`** | ≈1200 ÷ ~20 tok/para |
| `LLM_TERM_EXTRACTION_BATCH_MAX_TOKENS` | `6000` | **`2000`** | 1 batch (`max_workers=1`) → ~12 |
| `LLM_TERM_EXTRACTION_BATCH_MAX_PARAGRAPHS` | `60` | **`100`** | matches the token cap |

Everything else already matches: `LLM_MAX_OUTPUT_TOKENS=8192`,
`TRANSLATION_POOL_MAX_WORKERS=12`, `TERM_EXTRACTION_POOL_MAX_WORKERS=12`,
`TRANSLATION_MAX_QPS=16`, `ONNX_LAYOUT_BATCH_SIZE=4`,
`ONNX_INTRA_OP_NUM_THREADS=4`, `MAX_MODEL_ATTEMPTS=3`,
`QUALITY_EARLY_ACCEPT_THRESHOLD=0.92`, `GEMINI_MODEL=gemini-2.5-flash`.

Note `GEMINI_MODEL` is only the **fallback** when no `model_selection.json`
route matches; the priority-1 model comes from that asset file, not from here.

## 2. New keys to append (optional — defaults already apply)

```bash
# --- Thinking / reasoning budget ---
# Gemini defaults to *dynamic* thinking, which produced 454s single calls.
# 0 disables where supported; -1 restores the SDK default (quality escape hatch).
LLM_THINKING_BUDGET=0
LLM_THINKING_BUDGET_PRO=512

# --- Client-side deadlines (seconds) ---
# "timeout" is already in _RETRYABLE_ERROR_SUBSTRINGS so tenacity retries these.
LLM_CALL_TIMEOUT_SECONDS=90
LLM_JUDGE_TIMEOUT_SECONDS=60

# --- Adaptive batch sizing (src/worker/doctranslator/batching.py) ---
# Targets one full wave across the pool, clamped to this band.
LLM_ADAPTIVE_BATCHING_ENABLED=true
LLM_ADAPTIVE_BATCH_MIN_TOKENS=600
LLM_ADAPTIVE_BATCH_MAX_TOKENS=2500

# --- Fallback / concurrency guards ---
LLM_FALLBACK_BATCH_SIZE=10
# Caps total in-flight Vertex calls. Nested pools (batch -> per-batch fallback)
# can otherwise reach 12 + 12*12 = 156 concurrent calls on a cpu=4 instance.
LLM_MAX_INFLIGHT_CALLS=16
```

## 3. Asset files — NOT in the secret

Both are read from the GCS assets mount, not from env:

| File | Change | Status |
|---|---|---|
| `model_selection.json` | `gemini-3.5-flash` + `"region": "europe-west3"` as priority 1 across all 150 routes; existing chain demoted to priority 2 | Done locally in `.local-tmp/assets-cache/`; **must be uploaded** |
| `pricing_catalog.json` | `gemini-3.5-flash` entries — **regional entry first**, then global | Done locally; **not present in GCS at all** |

`pricing_catalog.json` is currently **missing from `gs://<bucket>/assets/`**.
`llm_rate_catalog.py:93-98` raises `FileNotFoundError` by design when it is
absent, so **the worker will fail at startup** without it. Hard deploy blocker.

Ordering matters in `pricing_catalog.json`: `resolve_rate_entry()` returns
`matches[0]` and accepts `region is None` as a match, so a global entry listed
before the regional one silently wins and every cost figure is wrong. Guarded
by `TestRegionalEntryOrdering` and a runtime WARNING.

## 4. Also required outside config

- `gemini-3.5-flash` must be **enabled with quota in `europe-west3`** for
  `aicoeprod`. The E2E confirmed model + region work in `aicoesandox`.

## 5. Cloud Run service settings — no change needed

Inspected on `translation-worker-service`: `cpu=4`,
`memory=8Gi`, `containerConcurrency=1`, `minScale=1`, `maxScale=30`,
`cpu-throttling=false`, `timeout=3600`. Peak RSS measured during the E2E was
**4.77 GiB — ~60% of the 8 GiB cap**, providing optimal right-sizing with a ~3.2 GiB safety buffer.

## 6. Rollback

Each lever is independently revertible without a code deploy:

| Symptom | Revert |
|---|---|
| Quality regression | `LLM_THINKING_BUDGET=-1` (restores dynamic thinking) |
| Suspect batching hurt coherence | `LLM_ADAPTIVE_BATCHING_ENABLED=false` |
| Full latency rollback | restore the four batch caps to 4000/40/6000/60 |
| Model regression | revert `model_selection.json` from `model_selection.json.bak` |

## 7. Known caveats before promoting

1. **Judge scores dropped ~0.05-0.08** (PDF 0.847→0.797, DOCX 0.902/0.860→
   0.820). Still well above the 0.6 threshold, but consistent across both
   formats. Not yet isolated between model change and smaller batches — one
   run with `LLM_ADAPTIVE_BATCHING_ENABLED=false` would tell you which.
2. **Cost is unmeasured.** `gemini-3.5-flash` is *more* expensive per token
   than `gemini-2.5-flash` ($1.65/$9.90 vs $0.30/$2.50 per 1M). Latency is
   78% better; net cost has not been extracted from BigQuery yet.
3. Consider a **canary**: deploy the four batch-cap changes first (pure
   latency, no model change), then the `model_selection.json` swap separately,
   so any judge-score effect is attributable to one or the other.
