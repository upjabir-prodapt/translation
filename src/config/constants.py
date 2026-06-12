"""
Application Settings - Centralized configuration using Pydantic Settings.

Source priority (first wins):
  1. Process environment variables
  2. .env file (IS_LOCAL=true  → <repo-root>/.env)
              (IS_LOCAL=false → /secrets/.env, mounted by Cloud Run)
  3. Field defaults

Bootstrap variables are always read from the process environment.
For local dev they live in <repo-root>/.env.
For Cloud Run they are either set via --set-env-vars (IS_LOCAL, ASSETS_ROOT,
TEMP_DIR, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION) or come from the
mounted /secrets/.env file.

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
    """Application settings loaded from env vars or a mounted .env file."""

    model_config = SettingsConfigDict(
        env_file=_DOTENV_FILE,  # pydantic also reads the file as a fallback
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -----------------------------
    # Bootstrap (always from env)
    # -----------------------------

    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str
    IS_LOCAL: bool = Field(default=True)

    # -----------------------------
    # GCS
    # -----------------------------

    GCS_BUCKET_NAME: str
    GCS_ASSETS_PREFIX: str = "assets"
    GCS_TRANSLATION_PREFIX: str = "translation-service"
    GCS_INPUT_FOLDER: str = "input"
    GCS_OUTPUT_FOLDER: str = "output"

    # -----------------------------
    # BigQuery
    # -----------------------------

    BIGQUERY_DATASET: str
    BIGQUERY_LOCATION: str = "europe-west1"
    BIGQUERY_TABLE: str = "translation_jobs"
    BIGQUERY_COST_TABLE: str = "translation_costs"
    BIGQUERY_DLP_TABLE: str = "dlp_mappings"
    BIGQUERY_DLQ_TABLE: str = "translation_dlq"
    BIGQUERY_REVIEWS_TABLE: str = "translation_reviews"

    API_USE_BACKGROUND_PIPELINE: bool = True

    # -----------------------------
    # Gemini / Judge
    # -----------------------------

    GEMINI_INPUT_COST_PER_1K: float = 0.0003
    GEMINI_OUTPUT_COST_PER_1K: float = 0.0025
    JUDGE_MODEL: str = "gemini-2.5-flash"
    QUALITY_THRESHOLD: float = 0.6
    QUALITY_EARLY_ACCEPT_THRESHOLD: float = 0.92
    MAX_MODEL_ATTEMPTS: int = 3
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # -----------------------------
    # Claude / Anthropic (Vertex AI Model Garden)
    # -----------------------------

    CLAUDE_MODEL: str = "claude-opus-4-7"
    CLAUDE_INPUT_COST_PER_1K: float = 0.0
    CLAUDE_OUTPUT_COST_PER_1K: float = 0.0
    CLAUDE_VERTEX_REGION: str = "global"

    # -----------------------------
    # Qwen (Vertex AI OpenAI-compatible endpoint)
    # -----------------------------

    QWEN_MODEL: str = "qwen/qwen3-235b-a22b-instruct-maas"
    QWEN_INPUT_COST_PER_1K: float = 0.0
    QWEN_OUTPUT_COST_PER_1K: float = 0.0
    QWEN_VERTEX_LOCATION: str = "us-central1"

    # -----------------------------
    # LLM / Translation Performance
    # -----------------------------

    # Concurrency
    TRANSLATION_POOL_MAX_WORKERS: int = 12
    TRANSLATION_MAX_QPS: int = 16
    TERM_EXTRACTION_POOL_MAX_WORKERS: int = 12
    SPLIT_PART_MAX_CONCURRENT: int = 2
    TYPESETTING_MAX_WORKERS: int = 4

    # LLM Context Budget (tokens)
    LLM_TRANSLATION_BATCH_MAX_TOKENS: int = 4000
    LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS: int = 40
    LLM_TERM_EXTRACTION_BATCH_MAX_TOKENS: int = 6000
    LLM_TERM_EXTRACTION_BATCH_MAX_PARAGRAPHS: int = 60

    # Language-specific token multipliers
    LLM_TOKEN_MULTIPLIER_CJK: float = 0.5
    LLM_TOKEN_MULTIPLIER_DEFAULT: float = 1.0

    # LLM provider/runtime configuration
    LLM_PROVIDER: str = "gemini_vertexai"
    LLM_MAX_CONTEXT_LENGTH: int = 1000000
    LLM_MAX_OUTPUT_TOKENS: int = 8192
    LLM_TEMPERATURE: float = 0.0
    LLM_TRANSLATION_MIN_TEXT_LENGTH: int = 5
    LLM_DISABLE_SAME_TEXT_FALLBACK: bool = False

    # ONNX runtime
    ONNX_LAYOUT_BATCH_SIZE: int = 4
    ONNX_INTRA_OP_NUM_THREADS: int = 4
    ONNX_INTER_OP_NUM_THREADS: int = 1

    # -----------------------------
    # DocTranslator Assets
    # -----------------------------

    WATERMARK_VERSION: str = "1.0"
    DOCLAYOUT_MODEL_FILENAME: str = "doclayout_yolo_docstructbench_imgsz1024.onnx"
    TABLE_DETECTION_MODEL_FILENAME: str = "ch_PP-OCRv4_det_infer.onnx"
    FONT_METADATA_FILENAME: str = "font_metadata.json"
    CMAP_METADATA_FILENAME: str = "cmap_metadata.json"
    FONTS_DIR: str = "fonts"
    CMAP_DIR: str = "cmap"
    MODELS_DIR: str = "models"
    METADATA_DIR: str = "metadata"
    TIKTOKEN_DIR: str = "tiktoken"
    GLOSSARIES_DIR: str = "glossaries"
    MODEL_SELECTION_FILENAME: str = "model_selection.json"

    # -----------------------------
    # Job Config
    # -----------------------------

    JOB_TTL_HOURS: int = 720
    MAX_CONCURRENT_JOBS: int = 10
    TEMP_JOBS_ROOT: str = "jobs"
    GCS_GLOSSARIES_PREFIX: str = "glossaries"

    # -----------------------------
    # API Config
    # -----------------------------

    API_TITLE: str = "DocTranslator Translation API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    STARTUP_WARMUP_ENABLED: bool = True
    STARTUP_WARMUP_STRICT: bool = False
    STARTUP_BACKGROUND_WARMUP_ENABLED: bool = False
    STARTUP_PREFLIGHT_TIMEOUT_SECONDS: int = 20
    STARTUP_PREFETCH_GLOSSARIES: list[str] = Field(default_factory=list)
    MODEL_SELECTION_CACHE_TTL_SECONDS: int = 300
    GLOSSARY_CACHE_TTL_SECONDS: int = 1800
    WARMUP_SYNC_CONCURRENCY: int = 8
    WARMUP_SYNC_PHASE_PREFIXES: list[str] = Field(
        default_factory=lambda: ["metadata", "models", "cmap", "fonts", "glossaries"]
    )

    # -----------------------------
    # Security
    # -----------------------------

    ALLOWED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])
    JWT_SECRET_KEY: str | None = None
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # -----------------------------
    # File Limits
    # -----------------------------

    MAX_FILE_SIZE: int = 5242880  # 5 MB
    ALLOWED_EXTENSIONS: set[str] = Field(
        default_factory=lambda: {".pdf", ".docx", ".doc"}
    )

    # -----------------------------
    # Logging
    # -----------------------------

    LOG_LEVEL: str = "INFO"

    # -----------------------------
    # Telemetry / Tracing
    # -----------------------------

    TRACE_ENABLED: bool = True
    TRACE_SAMPLE_RATE: float = 1.0
    OTEL_SERVICE_NAME: str = "translation_service"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "https://telemetry.googleapis.com/v1/traces"
    OTEL_EXPORTER_OTLP_PROTOCOL: str = "http/protobuf"
    OTEL_RESOURCE_ATTRIBUTES: str = ""
    OTEL_SEMCONV_STABILITY_OPT_IN: str = "gen_ai_latest_experimental"
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: str = "SPAN_AND_EVENT"
    OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED: bool = False
    APP_VERSION: str = "0.5.23"

    # -----------------------------
    # Retry / Detection
    # -----------------------------

    DOWNLOAD_MAX_ATTEMPTS: int = 3
    DOWNLOAD_RETRY_MIN_SECONDS: int = 2
    DOWNLOAD_RETRY_MAX_SECONDS: int = 10
    DOWNLOAD_RETRY_MULTIPLIER: int = 1
    LLM_RETRY_MAX_ATTEMPTS: int = 3
    LLM_RETRY_MIN_SECONDS: int = 1
    LLM_RETRY_MAX_SECONDS: int = 8
    LLM_RETRY_MULTIPLIER: int = 1
    GCS_RETRY_MAX_ATTEMPTS: int = 5
    GCS_RETRY_MIN_SECONDS: int = 10
    GCS_RETRY_MAX_SECONDS: int = 300
    GCS_RETRY_MULTIPLIER: int = 2
    LANGUAGE_DETECTION_MAX_CHARS: int = 10000
    GOOGLE_DLP_MAX_CHARS_PER_REQUEST: int = 300000
    GOOGLE_DLP_ENABLED: bool = True
    GOOGLE_DLP_MIN_LIKELIHOOD: str = "UNLIKELY"

    # -----------------------------
    # Runtime Paths (resolved in setup_directories)
    # -----------------------------

    PROJECT_ROOT: Path = find_project_root(Path(__file__).resolve())
    ASSETS_ROOT: str | None = None
    TEMP_DIR: Path | None = None

    @property
    def assets_root_path(self) -> Path:
        """Canonical root for persisted asset cache files."""
        return (
            Path(self.ASSETS_ROOT) if self.ASSETS_ROOT else self.PROJECT_ROOT / "assets"
        )

    @property
    def temp_root_path(self) -> Path:
        """Canonical root for runtime temporary/job execution files."""
        if self.TEMP_DIR:
            path = Path(self.TEMP_DIR)
        elif self.IS_LOCAL:
            path = self.PROJECT_ROOT / "tmp"
        else:
            import tempfile

            path = Path(tempfile.gettempdir()) / "translation-api"

        # Ensure the directory exists with restricted permissions (owner only)
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

    @model_validator(mode="after")
    def setup_directories(self) -> "Settings":
        if self.JWT_ALGORITHM.upper() != "HS256":
            raise ValueError("JWT_ALGORITHM must be HS256")
        if not self.IS_LOCAL and not self.JWT_SECRET_KEY:
            raise ValueError("JWT_SECRET_KEY is required when IS_LOCAL is false")

        cache_folder = self.assets_root_path
        cache_folder.mkdir(parents=True, exist_ok=True)

        temp_dir = self.temp_root_path
        temp_dir.mkdir(parents=True, exist_ok=True)
        for subdir in (
            self.MODELS_DIR,
            self.METADATA_DIR,
            self.FONTS_DIR,
            self.CMAP_DIR,
            self.TIKTOKEN_DIR,
            self.GLOSSARIES_DIR,
        ):
            (cache_folder / subdir).mkdir(parents=True, exist_ok=True)

        self.TEMP_DIR = temp_dir

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
