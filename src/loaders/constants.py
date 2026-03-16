"""Constants for asset hashes and directory names."""

from config.constants import settings

# SHA3-256 hashes for model files
DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256 = (
    "60be061226930524958b5465c8c04af3d7c03bcb0beb66454f5da9f792e3cf2a"
)

TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256 = (
    "062f4619afe91b33147c033acadecbb53f2a7b99ac703d157b96d5b10948da5e"
)

# Tiktoken cache hashes
TIKTOKEN_CACHES = {
    "fb374d419588a4632f3f557e76b4b70aebbca790": (
        "cb04bcda5782cfbbe77f2f991d92c0ea785d9496ef1137c91dfc3c8c324528d6"
    )
}

# Directory names for different asset types
FONTS_DIR = "fonts"
CMAP_DIR = "cmap"
MODELS_DIR = "models"
METADATA_DIR = "metadata"
TIKTOKEN_DIR = "tiktoken"

# Metadata filenames
FONT_METADATA_FILENAME = settings.FONT_METADATA_FILENAME
CMAP_METADATA_FILENAME = settings.CMAP_METADATA_FILENAME

# Model filenames (from settings)
DOCLAYOUT_MODEL_FILENAME = settings.DOCLAYOUT_MODEL_FILENAME
TABLE_DETECTION_MODEL_FILENAME = settings.TABLE_DETECTION_MODEL_FILENAME


# Retry configuration - now centralized in config/constants.py
# These are re-exported here for backward compatibility
DOWNLOAD_MAX_ATTEMPTS = settings.download_max_attempts
DOWNLOAD_RETRY_MIN_SECONDS = settings.download_retry_min_seconds
DOWNLOAD_RETRY_MAX_SECONDS = settings.download_retry_max_seconds
DOWNLOAD_RETRY_MULTIPLIER = settings.download_retry_multiplier
