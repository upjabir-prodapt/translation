"""
Application Settings - Centralized configuration using Pydantic Settings.
"""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# --------------------------------------------------
# Project Root Detection
# --------------------------------------------------


def find_project_root(start_path: Path) -> Path:
    """Find project root by locating pyproject.toml or .git."""
    for parent in [start_path] + list(start_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    # -----------------------------
    # GCP
    # -----------------------------

    GOOGLE_CLOUD_PROJECT_ID: str
    GOOGLE_CLOUD_LOCATION: str

    # -----------------------------
    # GCS
    # -----------------------------

    GCS_BUCKET_NAME: str
    GCS_ASSETS_PREFIX: str
    GCS_TRANSLATION_PREFIX: str = "translation"
    GCS_INPUT_FOLDER: str = "input"
    GCS_OUTPUT_FOLDER: str = "output"

    # -----------------------------
    # Firestore
    # -----------------------------

    FIRESTORE_DATABASE: str = "(default)"
    FIRESTORE_COLLECTION: str

    # -----------------------------
    # BigQuery
    # -----------------------------

    BIGQUERY_DATASET: str
    BIGQUERY_LOCATION: str

    # -----------------------------
    # Cloud Tasks
    # -----------------------------

    CLOUD_TASKS_QUEUE: str
    CLOUD_TASKS_LOCATION: str
    CLOUD_TASKS_DEADLINE_SECONDS: int
    WORKER_URL: str

    # -----------------------------
    # OpenAI
    # -----------------------------

    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    OPENAI_QPS: int = 10
    OPENAI_INPUT_COST_PER_1K: float = 0.0
    OPENAI_OUTPUT_COST_PER_1K: float = 0.0

    # -----------------------------
    # Gemini / Judge
    # -----------------------------

    GEMINI_INPUT_COST_PER_1K: float = 0.0
    GEMINI_OUTPUT_COST_PER_1K: float = 0.0
    JUDGE_MODEL: str = "gemini-2.5-flash"
    QUALITY_THRESHOLD: float = 0.8
    MAX_MODEL_ATTEMPTS: int = 2

    # -----------------------------
    # BabelDOC Assets
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

    # -----------------------------
    # Job Config
    # -----------------------------

    JOB_TTL_HOURS: int = 24
    MAX_CONCURRENT_JOBS: int = 10

    # -----------------------------
    # API Config
    # -----------------------------

    API_TITLE: str = "BabelDOC Translation API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # -----------------------------
    # Security
    # -----------------------------

    ALLOWED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[str] = ["*"]

    # -----------------------------
    # File Limits
    # -----------------------------

    MAX_FILE_SIZE: int = 100 * 1024 * 1024
    ALLOWED_EXTENSIONS: set[str] = {".pdf"}

    # -----------------------------
    # Logging
    # -----------------------------

    LOG_LEVEL: str = "INFO"

    # -----------------------------
    # Retry
    # -----------------------------

    DOWNLOAD_MAX_ATTEMPTS: int = 3
    DOWNLOAD_RETRY_MIN_SECONDS: int = 2
    DOWNLOAD_RETRY_MAX_SECONDS: int = 10
    DOWNLOAD_RETRY_MULTIPLIER: int = 1

    # -----------------------------
    # Runtime Paths (initialized later)
    # -----------------------------

    PROJECT_ROOT: Path = PROJECT_ROOT
    TEMP_DIR: Path | None = None
    CACHE_FOLDER: Path | None = None

    # --------------------------------------------------
    # Post Initialization
    # --------------------------------------------------

    @model_validator(mode="after")
    def setup_directories(self):
        cache_folder = self.PROJECT_ROOT / "assets"
        cache_folder.mkdir(parents=True, exist_ok=True)

        temp_dir = cache_folder / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.TEMP_DIR = temp_dir
        self.CACHE_FOLDER = cache_folder

        return self

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def create_temp_dir(self) -> Path:
        """Create a temporary working directory."""
        import tempfile

        return Path(tempfile.mkdtemp(dir=self.TEMP_DIR))

    # --------------------------------------------------
    # Pydantic Config
    # --------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        case_sensitive=False,
        extra="ignore",
    )


# Singleton
settings = Settings()  # type: ignore[call-arg]
