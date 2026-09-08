"""
Application Settings - Centralized configuration using Pydantic Settings.

All configuration values are loaded from environment variables (via .env).
There are no hardcoded defaults on Settings fields — every value must be
defined in the active .env file.

Source priority (first wins):
  1. Process environment variables
  2. .env file (IS_LOCAL=true  → <repo-root>/.env)
              (IS_LOCAL=false → /secrets/.env, mounted by Cloud Run)

Bootstrap path resolution (before .env load):
  IS_LOCAL may be set in the process environment to choose which .env file to load.
  If unset, local mode is assumed and <repo-root>/.env is used.

Escape hatches (checked before IS_LOCAL):
  DOTENV_DISABLE=true   – skip loading any .env file (useful in tests/CI)
  DOTENV_PATH=/path     – load that exact file instead of the auto-selected one
"""

import logging
import os
from pathlib import Path

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import PydanticBaseSettingsSource
from pydantic_settings import SettingsConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository root detection
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/config → src → repo root

LOCAL_ENV_FILE = _REPO_ROOT / ".env"
CLOUD_RUN_ENV_FILE = Path("/secrets/.env")


# ---------------------------------------------------------------------------
# Runtime detection helpers
# ---------------------------------------------------------------------------


def _env_flag(name: str) -> bool | None:
    """Return True/False for recognised truthy/falsy strings, None if absent/empty."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() in ("1", "true", "yes")


def is_local_runtime() -> bool:
    """Return True when running in local-dev mode (default when IS_LOCAL is unset)."""
    explicit = _env_flag("IS_LOCAL")
    return explicit if explicit is not None else True


def resolve_dotenv_path() -> Path | None:
    """Resolve which .env file to load.

    Priority:
      1. DOTENV_DISABLE=true  → None (skip loading)
      2. DOTENV_PATH=<path>   → that exact path
      3. IS_LOCAL=true        → <repo-root>/.env
      4. IS_LOCAL=false       → /secrets/.env
    """
    if _env_flag("DOTENV_DISABLE"):
        return None
    raw = os.getenv("DOTENV_PATH", "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (_REPO_ROOT / p).resolve()
    return LOCAL_ENV_FILE if is_local_runtime() else CLOUD_RUN_ENV_FILE


def load_dotenv_file(path: Path | None = None) -> Path | None:
    """Load the resolved .env file into os.environ via python-dotenv.

    A missing file is silently ignored (returns None).
    Called once at module-import time so os.environ is populated before
    Settings() is instantiated.
    """
    from dotenv import load_dotenv

    target = resolve_dotenv_path() if path is None else path
    if target is None:
        return None
    if target.is_file():
        load_dotenv(target)
        logger.debug("Loaded .env from %s", target)
        return target
    logger.debug(".env file not found at %s (skipped)", target)
    return None


# Load once at module-import time — must happen before Settings() is built.
_DOTENV_FILE = load_dotenv_file()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def find_project_root(start_path: Path) -> Path:
    """Find project root by locating pyproject.toml or .git."""
    for parent in [start_path] + list(start_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found")


class Settings(BaseSettings):
    """Application settings loaded exclusively from env vars and the active .env file."""

    model_config = SettingsConfigDict(
        env_file=_DOTENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -----------------------------
    # Bootstrap
    # -----------------------------

    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str
    IS_LOCAL: bool

    # -----------------------------
    # GCS
    # -----------------------------

    GCS_BUCKET_NAME: str
    GCS_ASSETS_PREFIX: str
    GCS_TRANSLATION_PREFIX: str
    GCS_INPUT_FOLDER: str
    GCS_OUTPUT_FOLDER: str
    TRACKING_SAMPLE_PERCENTAGE: int = 10

    # -----------------------------
    # BigQuery
    # -----------------------------

    BIGQUERY_DATASET: str
    BIGQUERY_LOCATION: str
    BIGQUERY_TABLE: str
    BIGQUERY_COST_TABLE: str
    BIGQUERY_DLP_TABLE: str
    BIGQUERY_REVIEWS_TABLE: str

    # -----------------------------
    # Redis (Memorystore via PSC) — LLM translation cache
    # -----------------------------

    REDIS_HOST: str = ""
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TLS_ENABLED: bool = True
    REDIS_PASSWORD: str = ""
    REDIS_SOCKET_TIMEOUT_SECONDS: float = 2.0
    REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS: float = 2.0
    REDIS_CACHE_TTL_SECONDS: int = 604800  # 7 days
    # Namespaces all cache keys so this service's keys do not collide with
    # other services sharing the same Redis Cluster (no DB/AUTH isolation
    # is available in Redis Cluster mode). Keep the trailing separator.
    REDIS_KEY_PREFIX: str = "translation-cache:"

    # --- Cross-instance job lease (see services/job_lease.py) ------------
    # How long a lease survives without a heartbeat. Must comfortably exceed
    # the refresh interval so a slow GC pause or a stalled event loop cannot
    # let a healthy owner's lease lapse, while staying short enough that a
    # genuinely dead instance frees the job promptly.
    JOB_LEASE_TTL_SECONDS: int = 300
    JOB_LEASE_REFRESH_SECONDS: float = 60.0

    API_USE_BACKGROUND_PIPELINE: bool

    # -----------------------------
    # App role / Cloud Tasks
    # -----------------------------

    # api | worker | "" (empty = treat as worker for asset dirs — safe for tests/local)
    APP_ROLE: str = ""

    CLOUD_TASKS_PROJECT: str = ""
    CLOUD_TASKS_LOCATION: str = ""
    CLOUD_TASKS_QUEUE: str = ""
    # Optional second queue for latency-sensitive work. Cloud Tasks has no
    # per-task priority field, so separate queues with independent dispatch
    # budgets are the only supported way to prioritise. Empty = feature off;
    # every job then uses CLOUD_TASKS_QUEUE. See docs/cloud-tasks-queues.md.
    CLOUD_TASKS_QUEUE_HIGH: str = ""
    # Document formats auto-promoted to the high-priority queue. Server-side
    # only -- clients cannot request priority (see ProcessingOptions.priority).
    HIGH_PRIORITY_FORMATS: list[str] = ["txt"]
    HIGH_PRIORITY_ROUTING_ENABLED: bool = True
    CLOUD_TASKS_WORKER_URL: str = ""
    CLOUD_TASKS_OIDC_SERVICE_ACCOUNT: str = ""
    # Cloud Tasks caps the dispatch deadline for HTTP targets at 30 minutes
    # (the documented interval is [15s, 1800s]); larger values are rejected
    # by the API. Keep this in sync with the worker's Cloud Run --timeout,
    # otherwise a job still running past the deadline is re-dispatched onto
    # a second instance while the first keeps going, doubling the load.
    CLOUD_TASKS_DISPATCH_DEADLINE_SECONDS: int = 1800
    # Worker-only: override OIDC audience if different from CLOUD_TASKS_WORKER_URL
    WORKER_OIDC_AUDIENCE: str = ""
    # Local/dev only — never enable in production
    WORKER_SKIP_OIDC_VERIFICATION: bool = False

    # -----------------------------
    # Gemini / Judge
    # -----------------------------

    QUALITY_JUDGE_ENABLED: bool = True
    JUDGE_MODEL: str
    JUDGE_MODEL_REGION: str = ""
    QUALITY_THRESHOLD: float
    QUALITY_EARLY_ACCEPT_THRESHOLD: float

    # --- Chunked judging -------------------------------------------------
    # The judge used to send the entire document in one call, which made
    # every judgement all-or-nothing: one timeout or one unparseable
    # response collapsed the score to a fallback and burned the whole model
    # chain. It now scores aligned (source, translation) segment pairs in
    # chunks, so a single failure costs coverage rather than the verdict.
    #
    # Chunks are grouped on *source* characters, never splitting a pair.
    # Source is the invariant across attempts, so identical chunk
    # boundaries make attempt-to-attempt score comparison valid.
    QUALITY_JUDGE_CHUNK_CHARS: int = 8000
    # Cap on chunks actually sent. 16 == LLM_MAX_INFLIGHT_CALLS, i.e. one
    # wave, so judge latency and cost are O(1) in document size rather than
    # linear. Above the cap a seeded stratified sample spanning the whole
    # document is used; raise this to trade latency for coverage.
    QUALITY_JUDGE_MAX_CHUNKS: int = 16
    # Safety net for the whole judging pass: LLM_JUDGE_TIMEOUT_SECONDS x
    # LLM_RETRY_MAX_ATTEMPTS x two waves is a ~12 min theoretical worst
    # case, which on its own would push a job past the Cloud Tasks dispatch
    # deadline. On expiry the judge stops issuing chunks and aggregates
    # whatever completed.
    QUALITY_JUDGE_TOTAL_BUDGET_SECONDS: float = 420.0
    # Below this fraction of the document judged, the score is reported as
    # a fallback: it is a sample too thin to defend.
    QUALITY_JUDGE_MIN_COVERAGE_RATIO: float = 0.5
    MAX_MODEL_ATTEMPTS: int
    GEMINI_MODEL: str
    GEMINI_MODEL_REGION: str = ""

    # -----------------------------
    # Input consistency guards (declared source language / domain)
    # -----------------------------
    # Two pre-translation guards that fail a job before any translation is
    # attempted, when the document contradicts what the submitter declared.
    # Both are fail-closed: a guard that cannot reach a verdict fails the
    # job rather than letting an unchecked document through.
    #
    # IMPORTANT: a failed job is terminal (src/shared/job_status.py) and the
    # Cloud Tasks handler returns 200 for it, so there is no automatic retry.
    # A Vertex/DLP outage therefore fails jobs permanently. These two
    # *_CHECK_ENABLED flags are the kill switches for that situation and must
    # stay settable from the environment without a redeploy.

    # Guard 1 -- declared source language vs. detected source language.
    LANGUAGE_MISMATCH_CHECK_ENABLED: bool = True
    # Maximum share of detected characters any NON-dominant language may hold
    # before the document is rejected as mixed-language. Mixed-language
    # translation is out of scope, so the default is 0.0: any second detected
    # language fails the job.
    #
    # Detection is per text block with a 0.80 confidence floor, but stray
    # blocks still happen in practice (party addresses, "force majeure",
    # tables of names). If monolingual documents are being rejected, raise
    # this to tolerate that noise -- e.g. 0.10 allows up to 10% of detected
    # characters in other languages. The full distribution is logged on
    # every job specifically so this value can be tuned from real data.
    LANGUAGE_MIXED_MAX_SECONDARY_SHARE: float = 0.0

    # Guard 2 -- declared domain vs. LLM-classified document domain
    # (e.g. an HR policy submitted as `legal`). Costs one small LLM call per
    # source document, cached per source_hash so a multi-target batch pays
    # for it once.
    DOMAIN_MISMATCH_CHECK_ENABLED: bool = True
    # Falls back to JUDGE_MODEL when empty. Keeping the default aligned with
    # the judge means no new pricing_catalog.json entry is required; a
    # different model needs one or cost resolution will raise.
    DOMAIN_CLASSIFIER_MODEL: str = ""
    DOMAIN_CLASSIFIER_REGION: str = ""
    # Only a confident contradiction fails the job. Domains genuinely overlap
    # (an HR policy is full of contractual language; a finance document is
    # full of regulatory language), so this floor -- not the on/off flag --
    # is the real false-positive control. Lower it only after looking at a
    # confusion matrix on real documents.
    DOMAIN_CLASSIFIER_MIN_CONFIDENCE: float = 0.70
    # Characters of document text sampled for classification. Sampled across
    # the whole document rather than the first N chars: a cover page and
    # letterhead are a poor domain signal.
    DOMAIN_CLASSIFIER_SAMPLE_CHARS: int = 4000
    DOMAIN_CLASSIFIER_TIMEOUT_SECONDS: float = 60.0

    # -----------------------------
    # Claude / Anthropic (Vertex AI Model Garden)
    # -----------------------------

    CLAUDE_MODEL: str
    CLAUDE_VERTEX_REGION: str

    # -----------------------------
    # LLM / Translation Performance
    # -----------------------------

    TRANSLATION_POOL_MAX_WORKERS: int
    TRANSLATION_MAX_QPS: int
    TERM_EXTRACTION_POOL_MAX_WORKERS: int
    SPLIT_PART_MAX_CONCURRENT: int
    TYPESETTING_MAX_WORKERS: int

    LLM_TRANSLATION_BATCH_MAX_TOKENS: int
    LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS: int
    LLM_TERM_EXTRACTION_BATCH_MAX_TOKENS: int
    LLM_TERM_EXTRACTION_BATCH_MAX_PARAGRAPHS: int

    LLM_TOKEN_MULTIPLIER_CJK: float
    LLM_TOKEN_MULTIPLIER_DEFAULT: float

    LLM_MAX_CONTEXT_LENGTH: int
    LLM_MAX_OUTPUT_TOKENS: int
    LLM_TEMPERATURE: float
    LLM_TRANSLATION_MIN_TEXT_LENGTH: int
    LLM_DISABLE_SAME_TEXT_FALLBACK: bool

    # --- Thinking / reasoning budget -------------------------------------
    # Gemini defaults to *dynamic* thinking, which produced 450s+ single
    # calls in the 2026-08-24 baseline. 0 disables thinking where the model
    # supports it; -1 restores the SDK's dynamic default. Pro-class models
    # cannot fully disable thinking, so they get their own (small) budget.
    LLM_THINKING_BUDGET: int = 0
    LLM_THINKING_BUDGET_PRO: int = 512

    # --- Client-side deadlines -------------------------------------------
    # Without these a hung/slow generation is simply waited out; "timeout"
    # is already in the retryable-substring list so tenacity picks it up.
    LLM_CALL_TIMEOUT_SECONDS: float = 90.0
    # Raised 60 -> 120 for headroom on 429 backoff. Judge payloads are now
    # chunked to QUALITY_JUDGE_CHUNK_CHARS, so this deadline is rarely the
    # binding constraint; when it is, the cost is one chunk's coverage
    # rather than the whole verdict.
    LLM_JUDGE_TIMEOUT_SECONDS: float = 120.0

    # --- Adaptive batch sizing (see doctranslator/batching.py) -----------
    LLM_ADAPTIVE_BATCHING_ENABLED: bool = True
    LLM_ADAPTIVE_BATCH_MIN_TOKENS: int = 600
    LLM_ADAPTIVE_BATCH_MAX_TOKENS: int = 2500

    # Failed units are re-batched in groups of this size before falling back
    # to genuinely one-at-a-time translation.
    LLM_FALLBACK_BATCH_SIZE: int = 10

    # Process-wide ceiling on simultaneously in-flight LLM calls, enforced in
    # BaseTranslator._run_translation_batch(). The DOCX path nests pools
    # (batch pool -> per-batch fallback pool), so at
    # TRANSLATION_POOL_MAX_WORKERS=12 a single job can theoretically put
    # 12 + 12*12 = 156 concurrent Vertex requests in flight on a
    # cpu=4 / containerConcurrency=1 instance. 16 matches TRANSLATION_MAX_QPS
    # and keeps one full wave (12) running while leaving headroom for
    # fallbacks. Set to 0 to disable the guard entirely.
    LLM_MAX_INFLIGHT_CALLS: int = 16

    # Skip (pass through unchanged, never send to the LLM) any DOCX/PDF
    # unit confidently (>= language_detection_core.MIN_DETECTION_CONFIDENCE)
    # detected in a language outside the configured set
    # (language_mapper.json), or already confidently in the target
    # language (implementation_plan.md Phase C.4). Default True; can be
    # turned off without a redeploy if it ever needs to be disabled.
    SKIP_UNSUPPORTED_LANGUAGE_UNITS: bool = True

    ONNX_LAYOUT_BATCH_SIZE: int
    ONNX_INTRA_OP_NUM_THREADS: int
    ONNX_INTER_OP_NUM_THREADS: int

    # -----------------------------
    # DocTranslator Assets
    # -----------------------------

    WATERMARK_VERSION: str
    DOCLAYOUT_MODEL_FILENAME: str
    TABLE_DETECTION_MODEL_FILENAME: str
    FONT_METADATA_FILENAME: str
    CMAP_METADATA_FILENAME: str
    FONTS_DIR: str
    CMAP_DIR: str
    MODELS_DIR: str
    METADATA_DIR: str
    TIKTOKEN_DIR: str
    GLOSSARIES_DIR: str
    MODEL_SELECTION_FILENAME: str
    PRICING_CATALOG_FILENAME: str = "pricing_catalog.json"

    # -----------------------------
    # Job Config
    # -----------------------------

    JOB_TTL_HOURS: int
    MAX_CONCURRENT_JOBS: int
    TEMP_JOBS_ROOT: str
    GCS_GLOSSARIES_PREFIX: str

    # -----------------------------
    # API Config
    # -----------------------------

    API_TITLE: str
    API_VERSION: str
    API_PREFIX: str
    STARTUP_WARMUP_ENABLED: bool
    STARTUP_WARMUP_STRICT: bool
    STARTUP_BACKGROUND_WARMUP_ENABLED: bool
    STARTUP_PREFLIGHT_TIMEOUT_SECONDS: int
    STARTUP_PREFETCH_GLOSSARIES: list[str]
    MODEL_SELECTION_CACHE_TTL_SECONDS: int
    GLOSSARY_CACHE_TTL_SECONDS: int
    WARMUP_SYNC_CONCURRENCY: int
    WARMUP_SYNC_PHASE_PREFIXES: list[str]

    # -----------------------------
    # Security
    # -----------------------------

    ALLOWED_HOSTS: list[str]
    CORS_ORIGINS: list[str]
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int
    IAP_AUDIENCE: str = ""
    HUB_IAP_AUDIENCE: str = ""
    # Entra security group required for Translation entitlement (checked against IAP JWT `groups` claim).
    TRANSLATION_REQUIRED_GROUP: str = ""
    # Hard ceiling on a sliding session's total lifetime, measured from the
    # `auth_time` claim stamped at the original IAP login and preserved
    # unchanged across every renewal. POST /auth/refresh refuses to mint past
    # this point, so 8 hours after signing in the user must authenticate with
    # IAP again regardless of how continuously active they have been.
    SESSION_ABSOLUTE_MAX_MINUTES: int = 480
    # When False (rollout default), a token carrying NO `scopes` claim is
    # accepted for backward compatibility with sessions minted before scopes
    # existed; a token that *has* `scopes` must still include this service's
    # own scope. Flip to True once every legacy token has expired -- that is
    # the step that actually closes the cross-service bypass.
    REQUIRE_SCOPE_CLAIM: bool = False

    # -----------------------------
    # File Limits
    # -----------------------------

    MAX_FILE_SIZE: int
    ALLOWED_EXTENSIONS: set[str]
    # Number of leading pages sampled by PDFValidator to detect an
    # image-only/no-text-layer PDF (implementation_plan.md Phase B).
    # Capped so a 500-page scanned file is still cheap to reject.
    PDF_TEXT_PROBE_PAGES: int = 5

    # -----------------------------
    # Duplicate-submission idempotency (implementation_plan.md D.5, EC-15)
    # -----------------------------
    # A double-click / accidental resubmit of the identical
    # (user, source document, target language, domain) combination within
    # this many seconds returns the existing job instead of creating a new
    # one. 0 disables the check entirely (every submission always creates a
    # new job, matching the pre-D.5 behaviour).
    DUPLICATE_SUBMISSION_WINDOW_SECONDS: int = 30

    # -----------------------------
    # Logging
    # -----------------------------

    LOG_LEVEL: str

    # -----------------------------
    # Telemetry / Tracing
    # -----------------------------

    TRACE_ENABLED: bool
    TRACE_SAMPLE_RATE: float
    OTEL_SERVICE_NAME: str
    OTEL_EXPORTER_OTLP_ENDPOINT: str
    OTEL_EXPORTER_OTLP_PROTOCOL: str
    OTEL_RESOURCE_ATTRIBUTES: str
    OTEL_SEMCONV_STABILITY_OPT_IN: str
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: str
    OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED: bool
    APP_VERSION: str

    # -----------------------------
    # Retry / Detection
    # -----------------------------

    DOWNLOAD_MAX_ATTEMPTS: int
    DOWNLOAD_RETRY_MIN_SECONDS: int
    DOWNLOAD_RETRY_MAX_SECONDS: int
    DOWNLOAD_RETRY_MULTIPLIER: int
    LLM_RETRY_MAX_ATTEMPTS: int
    LLM_RETRY_MIN_SECONDS: int
    LLM_RETRY_MAX_SECONDS: int
    LLM_RETRY_MULTIPLIER: int
    GCS_RETRY_MAX_ATTEMPTS: int
    GCS_RETRY_MIN_SECONDS: int
    GCS_RETRY_MAX_SECONDS: int
    GCS_RETRY_MULTIPLIER: int
    LANGUAGE_DETECTION_MAX_CHARS: int

    # Language-detection tuning (see language_detection_core.py). Layer 1
    # discards text that carries no reliable language signal; layer 2 decides
    # whether the document is dominated by one language or genuinely
    # mixed. Defaults are set here so the knobs can be tuned per
    # environment without a code change.
    LANGUAGE_DETECTION_MIN_UNIT_CHARS: int = 50
    LANGUAGE_DETECTION_MIN_UNIT_WORDS: int = 8
    LANGUAGE_DETECTION_NOISE_SHARE: float = 0.05
    LANGUAGE_DETECTION_MIN_DOMINANT_SHARE: float = 0.70
    LANGUAGE_DETECTION_MIN_MIXED_DECISION_CHARS: int = 500

    # Secondary-language prompt hint (see language_prompts.py). When the
    # document is not purely monolingual, the minority languages detection
    # found are named in the translation prompt and the model is told to
    # translate each segment from whichever of them it is actually written
    # in, instead of translating every segment from the dominant language.
    #
    # Only reachable on jobs that survive the mixed-language guard: with
    # LANGUAGE_MISMATCH_CHECK_ENABLED=true and
    # LANGUAGE_MIXED_MAX_SECONDARY_SHARE=0.0 (the defaults), such a
    # document is rejected before translation and this never renders.
    MIXED_LANGUAGE_PROMPT_HINT_ENABLED: bool = True
    # Far below LANGUAGE_DETECTION_NOISE_SHARE on purpose: the languages
    # this hint exists to rescue are the ones the mixed-language decision
    # treats as incidental (a real English cover page measured 1.8%).
    MIXED_LANGUAGE_PROMPT_MIN_SHARE: float = 0.005
    MIXED_LANGUAGE_PROMPT_MAX_LANGUAGES: int = 4

    GOOGLE_DLP_MAX_CHARS_PER_REQUEST: int
    GOOGLE_DLP_ENABLED: bool
    GOOGLE_DLP_MIN_LIKELIHOOD: str

    # -----------------------------
    # Runtime Paths
    # -----------------------------

    ASSETS_ROOT: str
    TEMP_DIR: str

    # -----------------------------
    # Memory pressure
    # -----------------------------

    # Fraction of the container's cgroup memory limit at which the
    # translation pipeline logs a structured WARNING. Purely observational:
    # nothing is aborted, it exists so an OOM SIGKILL is preceded by a log
    # line naming the job and phase instead of appearing out of nowhere.
    # Set to 0 to disable.
    MEMORY_PRESSURE_WARN_FRACTION: float = 0.85

    PROJECT_ROOT: Path = Field(
        default_factory=lambda: find_project_root(Path(__file__).resolve())
    )

    @property
    def assets_root_path(self) -> Path:
        """Canonical root for persisted asset cache files."""
        return Path(self.ASSETS_ROOT)

    @property
    def temp_root_path(self) -> Path:
        """Canonical root for runtime temporary/job execution files."""
        path = Path(self.TEMP_DIR)
        if not path.exists():
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        return path

    # --------------------------------------------------
    # Post Initialization
    # --------------------------------------------------

    @model_validator(mode="after")
    def sync_otel_environment(self) -> "Settings":
        """Push OTEL / GenAI SDK env vars for auto-instrumentation (read at instrument() time)."""
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = self.OTEL_EXPORTER_OTLP_ENDPOINT
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = self.OTEL_EXPORTER_OTLP_PROTOCOL
        resource_attrs = self.OTEL_RESOURCE_ATTRIBUTES.strip()
        if not resource_attrs:
            resource_attrs = (
                f"service.name={self.OTEL_SERVICE_NAME},"
                f"gcp.project_id={self.GOOGLE_CLOUD_PROJECT}"
            )
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = resource_attrs
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = self.OTEL_SEMCONV_STABILITY_OPT_IN
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            self.OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
        )
        os.environ["OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED"] = (
            "true" if self.OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED else "false"
        )
        return self

    @property
    def is_api_role(self) -> bool:
        return self.APP_ROLE.strip().lower() == "api"

    @property
    def is_worker_role(self) -> bool:
        return self.APP_ROLE.strip().lower() == "worker"

    @model_validator(mode="after")
    def setup_directories(self) -> "Settings":
        if self.JWT_ALGORITHM.upper() != "HS256":
            raise ValueError("JWT_ALGORITHM must be HS256")
        if not self.IS_LOCAL and not self.JWT_SECRET_KEY:
            raise ValueError("JWT_SECRET_KEY is required when IS_LOCAL is false")
        # Worker does not use IAP; API and unset role still require it in cloud.
        if not self.IS_LOCAL and not self.is_worker_role and not self.IAP_AUDIENCE:
            raise ValueError("IAP_AUDIENCE is required when IS_LOCAL is false")

        # Asset cache dirs are worker-owned (GCS FUSE). Skip when APP_ROLE=api.
        ensure_assets = not self.is_api_role
        if ensure_assets:
            cache_folder = self.assets_root_path
            cache_folder.mkdir(parents=True, exist_ok=True)
            for subdir in (
                self.MODELS_DIR,
                self.METADATA_DIR,
                self.FONTS_DIR,
                self.CMAP_DIR,
                self.TIKTOKEN_DIR,
                self.GLOSSARIES_DIR,
            ):
                (cache_folder / subdir).mkdir(parents=True, exist_ok=True)
            temp_dir = self.temp_root_path
            temp_dir.mkdir(parents=True, exist_ok=True)

        self._log_config_sources()
        return self

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def create_temp_dir(self) -> Path:
        """Create a temporary working directory."""
        import tempfile

        return Path(tempfile.mkdtemp(dir=self.temp_root_path))

    def _log_config_sources(self) -> None:
        """Log a startup summary of where configuration was loaded from."""
        if self.IS_LOCAL:
            source = str(LOCAL_ENV_FILE)
        else:
            source = str(CLOUD_RUN_ENV_FILE)
        logger.info(
            f"Settings loaded | IS_LOCAL={self.IS_LOCAL} | source={source}"
            f" | project={self.GOOGLE_CLOUD_PROJECT} | location={self.GOOGLE_CLOUD_LOCATION}"
            f" | assets_root={self.assets_root_path} | temp_root={self.temp_root_path}"
        )

    # --------------------------------------------------
    # Pydantic Settings Sources
    # --------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        _settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


# Singleton
settings = Settings()  # type: ignore[call-arg]
