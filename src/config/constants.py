"""
Application Settings - Centralized configuration using Pydantic Settings.

Source priority (first wins):
  1. Process environment variables
  2. GCP Secret Manager JSON  (IS_LOCAL=false + APP_CONFIG_SECRET_NAME set)
  3. .env file                (IS_LOCAL=true / local development)
  4. Field defaults

Bootstrap variables are always read from process env and must be passed via
``--set-env-vars`` on Cloud Run:
  GOOGLE_CLOUD_PROJECT_ID, GOOGLE_CLOUD_LOCATION,
  APP_CONFIG_SECRET_NAME, APP_CONFIG_SECRET_VERSION, IS_LOCAL
"""

import json
import logging
import os
from pathlib import Path

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import PydanticBaseSettingsSource
from pydantic_settings import SettingsConfigDict

DEFAULT_SECRET_VERSION = "latest"  # noqa: S105

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Project Root Detection
# --------------------------------------------------


def find_project_root(start_path: Path) -> Path:
    """Find project root by locating pyproject.toml or .git."""
    for parent in [start_path] + list(start_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found")


def get_secrets(project_id: str, secret_name: str, version: str = "latest") -> dict:
    """Fetch and JSON-decode a secret payload from GCP Secret Manager."""
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    response = client.access_secret_version(name=name)
    return json.loads(response.payload.data.decode("utf-8"))


# --------------------------------------------------
# Custom Settings Source: GCP Secret Manager
# --------------------------------------------------


class SecretManagerSettingsSource(PydanticBaseSettingsSource):
    """Loads settings from a GCP Secret Manager secret when IS_LOCAL is falsy.

    The secret payload must be a flat JSON object whose keys match Settings
    field names (same names as environment variables).  Env vars always win because
    ``env_settings`` is placed before this source in
    ``settings_customise_sources``.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data: dict = {}

        is_local_raw = os.environ.get("IS_LOCAL", "true").strip().lower()
        is_local = is_local_raw not in ("false", "0", "no")
        secret_name = os.environ.get("APP_CONFIG_SECRET_NAME", "").strip()
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "").strip()
        version = os.environ.get("APP_CONFIG_SECRET_VERSION", "latest").strip()

        if not is_local and secret_name and project_id:
            try:
                self._data = get_secrets(project_id, secret_name, version)
                logger.info(
                    "Loaded %d config keys from Secret Manager secret '%s' (version=%s)",
                    len(self._data),
                    secret_name,
                    version,
                )
            except Exception as exc:
                logger.exception(
                    f"Failed to load config from Secret Manager secret {secret_name} with exception {exc}"
                )

    def get_field_value(self, field: object, field_name: str) -> tuple:  # type: ignore[override]
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict:
        return {k: v for k, v in self._data.items() if v is not None}


# --------------------------------------------------
# Settings
# --------------------------------------------------


class Settings(BaseSettings):
    """Application settings loaded from env vars, Secret Manager, or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -----------------------------
    # Bootstrap (always from env, set via Cloud Run --set-env-vars)
    # -----------------------------

    GOOGLE_CLOUD_PROJECT_ID: str
    GOOGLE_CLOUD_LOCATION: str
    IS_LOCAL: bool = Field(default=True)
    APP_CONFIG_SECRET_NAME: str | None = None
    APP_CONFIG_SECRET_VERSION: str = DEFAULT_SECRET_VERSION

    # -----------------------------
    # GCS
    # -----------------------------

    GCS_BUCKET_NAME: str
    GCS_ASSETS_PREFIX: str = "assets"
    GCS_TRANSLATION_PREFIX: str = "translation"
    GCS_INPUT_FOLDER: str = "input"
    GCS_OUTPUT_FOLDER: str = "output"

    # -----------------------------
    # BigQuery
    # -----------------------------

    BIGQUERY_DATASET: str
    BIGQUERY_LOCATION: str = "US"
    BIGQUERY_TABLE: str = "translation_jobs"
    BIGQUERY_COST_TABLE: str = "cost_attribution"
    BIGQUERY_DLP_TABLE: str = "dlp_tokens"

    API_USE_BACKGROUND_PIPELINE: bool = True

    # -----------------------------
    # Gemini / Judge
    # -----------------------------

    GEMINI_INPUT_COST_PER_1K: float = 0.0
    GEMINI_OUTPUT_COST_PER_1K: float = 0.0
    JUDGE_MODEL: str = "gemini-2.5-flash"
    QUALITY_THRESHOLD: float = 0.6
    QUALITY_EARLY_ACCEPT_THRESHOLD: float = 0.92
    MAX_MODEL_ATTEMPTS: int = 3
    GEMINI_MODEL: str = "gemini-2.5-flash"

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

    JOB_TTL_HOURS: int = 24
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
    STARTUP_BACKGROUND_WARMUP_ENABLED: bool = True
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

    MAX_FILE_SIZE: int = 5 * 1024 * 1024
    ALLOWED_EXTENSIONS: set[str] = Field(default_factory=set)

    # -----------------------------
    # Logging
    # -----------------------------

    LOG_LEVEL: str = "DEBUG"

    # -----------------------------
    # Telemetry / Tracing
    # -----------------------------

    TRACE_ENABLED: bool = True
    TRACE_SAMPLE_RATE: float = 1.0
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "telemetry.googleapis.com:443"
    OTEL_SERVICE_NAME: str = "translation_service"
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
    LANGUAGE_DETECTION_MAX_CHARS: int = 10000
    GOOGLE_DLP_MAX_CHARS_PER_REQUEST: int = 300000

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
            source = ".env"
            secret_info = ""
        else:
            source = "Secret Manager"
            secret_info = (
                f" (secret={self.APP_CONFIG_SECRET_NAME!r},"
                f" version={self.APP_CONFIG_SECRET_VERSION!r})"
                if self.APP_CONFIG_SECRET_NAME
                else " (no secret name provided)"
            )
        logger.info(
            f"Settings loaded | IS_LOCAL={self.IS_LOCAL} | source={source}{secret_info}"
            f" | project={self.GOOGLE_CLOUD_PROJECT_ID} | location={self.GOOGLE_CLOUD_LOCATION}"
            f" | assets_root={self.assets_root_path} | temp_root={self.temp_root_path}"
        )

    # --------------------------------------------------
    # Pydantic Settings Sources
    # --------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            SecretManagerSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


# Singleton
settings = Settings()  # type: ignore[call-arg]
