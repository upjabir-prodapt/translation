"""BabelDOC Constants - Centralized configuration using Pydantic Settings."""

from pathlib import Path

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings


def find_project_root(start_path: Path) -> Path:
    """Find project root by looking for pyproject.toml or .git."""
    for parent in [start_path] + list(start_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found")


# Calculate project root and .env path at module level
_PROJECT_ROOT = find_project_root(Path(__file__).resolve())
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Centralized application settings."""

    # Google Cloud Configuration
    GOOGLE_CLOUD_PROJECT_ID: str = Field()
    GOOGLE_CLOUD_LOCATION: str = Field()

    # GCS Configuration
    GCS_BUCKET_NAME: str = Field()
    GCS_ASSETS_PREFIX: str = Field()
    GCS_TRANSLATION_PREFIX: str = Field(default="translation")
    GCS_INPUT_FOLDER: str = Field(default="input")
    GCS_OUTPUT_FOLDER: str = Field(default="output")

    # Firestore Configuration
    FIRESTORE_DATABASE: str = Field(default="(default)")
    FIRESTORE_COLLECTION: str = Field()

    # BigQuery Configuration
    BIGQUERY_DATASET: str = Field()
    BIGQUERY_LOCATION: str = Field()

    # Cloud Tasks Configuration
    CLOUD_TASKS_QUEUE: str = Field()
    CLOUD_TASKS_LOCATION: str = Field()
    CLOUD_TASKS_DEADLINE_SECONDS: int = Field()
    WORKER_URL: str = Field()

    # OpenAI Configuration
    OPENAI_API_KEY: str = Field()
    OPENAI_MODEL: str = Field()
    OPENAI_QPS: int = Field()

    # BabelDOC Asset Configuration
    WATERMARK_VERSION: str = Field()
    DOCLAYOUT_MODEL_FILENAME: str = Field()
    TABLE_DETECTION_MODEL_FILENAME: str = Field()
    FONT_METADATA_FILENAME: str = Field()
    CMAP_METADATA_FILENAME: str = Field()

    # Job Configuration
    JOB_TTL_HOURS: int = Field()

    # Logging Configuration
    LOG_LEVEL: str = Field()

    # API Configuration
    api_title: str = "BabelDOC Translation API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Security Configuration
    allowed_hosts: list[str] = ["*"]
    cors_origins: list[str] = ["*"]

    # File Configuration
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024)  # 100MB
    allowed_extensions: set[str] = {".pdf"}

    # Model Configuration
    cache_ttl_seconds: int = 3600  # 1 hour
    max_concurrent_jobs: int = 10

    # Download Retry Configuration
    download_max_attempts: int = Field(default=3)
    download_retry_min_seconds: int = Field(default=2)
    download_retry_max_seconds: int = Field(default=10)
    download_retry_multiplier: int = Field(default=1)

    @model_validator(mode="after")
    def setup_cache_folder(self) -> "Settings":
        """Setup tiktoken cache directory after settings are loaded."""

        CACHE_FOLDER = _PROJECT_ROOT / "assests"
        CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
        return self

    class Config:
        env_file = str(_ENV_FILE)
        case_sensitive = False
        populate_by_name = True
        extra = "ignore"  # Allow extra fields in .env without validation errors


# Global settings instance (tiktoken cache setup runs automatically in validator)
settings = Settings()  # type: ignore[call-arg]
