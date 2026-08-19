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

    # -----------------------------
    # BigQuery
    # -----------------------------

    BIGQUERY_DATASET: str
    BIGQUERY_LOCATION: str
    BIGQUERY_TABLE: str
    BIGQUERY_COST_TABLE: str
    BIGQUERY_DLP_TABLE: str
    BIGQUERY_REVIEWS_TABLE: str

    API_USE_BACKGROUND_PIPELINE: bool

    # -----------------------------
    # App role / Cloud Tasks
    # -----------------------------

    # api | worker | "" (empty = treat as worker for asset dirs — safe for tests/local)
    APP_ROLE: str = ""

    CLOUD_TASKS_PROJECT: str = ""
    CLOUD_TASKS_LOCATION: str = ""
    CLOUD_TASKS_QUEUE: str = ""
    CLOUD_TASKS_WORKER_URL: str = ""
    CLOUD_TASKS_OIDC_SERVICE_ACCOUNT: str = ""
    CLOUD_TASKS_DISPATCH_DEADLINE_SECONDS: int = 3600
    # Worker-only: override OIDC audience if different from CLOUD_TASKS_WORKER_URL
    WORKER_OIDC_AUDIENCE: str = ""
    # Local/dev only — never enable in production
    WORKER_SKIP_OIDC_VERIFICATION: bool = False

    # -----------------------------
    # Gemini / Judge
    # -----------------------------

    GEMINI_INPUT_COST_PER_1K: float
    GEMINI_OUTPUT_COST_PER_1K: float
    GEMINI_2_5_FLASH_INPUT_COST_PER_1K: float
    GEMINI_2_5_FLASH_OUTPUT_COST_PER_1K: float
    GEMINI_2_5_FLASH_CACHE_HIT_COST_PER_1K: float
    GEMINI_2_5_FLASH_LITE_INPUT_COST_PER_1K: float
    GEMINI_2_5_FLASH_LITE_OUTPUT_COST_PER_1K: float
    GEMINI_2_5_FLASH_LITE_CACHE_HIT_COST_PER_1K: float
    GEMINI_2_5_PRO_SHORT_INPUT_COST_PER_1K: float
    GEMINI_2_5_PRO_SHORT_OUTPUT_COST_PER_1K: float
    GEMINI_2_5_PRO_SHORT_CACHE_HIT_COST_PER_1K: float
    GEMINI_2_5_PRO_LONG_INPUT_COST_PER_1K: float
    GEMINI_2_5_PRO_LONG_OUTPUT_COST_PER_1K: float
    GEMINI_2_5_PRO_LONG_CACHE_HIT_COST_PER_1K: float
    LLM_RATE_CATALOG_OVERRIDE_JSON: str
    JUDGE_MODEL: str
    QUALITY_THRESHOLD: float
    QUALITY_EARLY_ACCEPT_THRESHOLD: float
    MAX_MODEL_ATTEMPTS: int
    GEMINI_MODEL: str

    # -----------------------------
    # Claude / Anthropic (Vertex AI Model Garden)
    # -----------------------------

    CLAUDE_MODEL: str
    CLAUDE_INPUT_COST_PER_1K: float
    CLAUDE_OUTPUT_COST_PER_1K: float
    CLAUDE_CACHE_HIT_COST_PER_1K: float
    CLAUDE_CACHE_WRITE_5M_COST_PER_1K: float
    CLAUDE_CACHE_WRITE_1H_COST_PER_1K: float
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


    # -----------------------------
    # File Limits
    # -----------------------------

    MAX_FILE_SIZE: int
    ALLOWED_EXTENSIONS: set[str]

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
    GOOGLE_DLP_MAX_CHARS_PER_REQUEST: int
    GOOGLE_DLP_ENABLED: bool
    GOOGLE_DLP_MIN_LIKELIHOOD: str

    # -----------------------------
    # Runtime Paths
    # -----------------------------

    ASSETS_ROOT: str
    TEMP_DIR: str

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
