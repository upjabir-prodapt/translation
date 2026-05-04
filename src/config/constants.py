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
    APP_CONFIG_SECRET_VERSION: str = "latest"

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
    MAX_MODEL_ATTEMPTS: int = 3
    GEMINI_MODEL: str = "gemini-2.5-flash"

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
    # Retry / Detection
    # -----------------------------

    DOWNLOAD_MAX_ATTEMPTS: int = 3
    DOWNLOAD_RETRY_MIN_SECONDS: int = 2
    DOWNLOAD_RETRY_MAX_SECONDS: int = 10
    DOWNLOAD_RETRY_MULTIPLIER: int = 1
    LANGUAGE_DETECTION_MAX_CHARS: int = 10000
    GOOGLE_DLP_MAX_CHARS_PER_REQUEST: int = 300000

    # -----------------------------
    # Runtime Paths (resolved in setup_directories)
    # -----------------------------

    PROJECT_ROOT: Path = find_project_root(Path(__file__).resolve())
    ASSETS_ROOT: str | None = None
    TEMP_DIR: Path | None = None
    CACHE_FOLDER: Path | None = None

    # --------------------------------------------------
    # Post Initialization
    # --------------------------------------------------

    @model_validator(mode="after")
    def setup_directories(self) -> "Settings":
        cache_folder = (
            Path(self.ASSETS_ROOT) if self.ASSETS_ROOT else self.PROJECT_ROOT / "assets"
        )
        cache_folder.mkdir(parents=True, exist_ok=True)

        temp_dir = cache_folder / "tmp"
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
        self.CACHE_FOLDER = cache_folder
        self.ASSETS_ROOT = str(cache_folder)

        self._log_config_sources()
        return self

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def create_temp_dir(self) -> Path:
        """Create a temporary working directory."""
        import tempfile

        return Path(tempfile.mkdtemp(dir=self.TEMP_DIR))

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
            "Settings loaded | IS_LOCAL=%s | source=%s%s"
            " | project=%s | location=%s",
            self.IS_LOCAL,
            source,
            secret_info,
            self.GOOGLE_CLOUD_PROJECT_ID,
            self.GOOGLE_CLOUD_LOCATION,
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
