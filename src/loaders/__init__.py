"""Loaders module - Asset loading and management.

This module provides a clean API for downloading, caching, and accessing
assets like fonts, models, and CMap files.

Example usage:
    from src.loaders import get_font_family, warmup

    # Get font family for a language
    fonts = get_font_family("CN")

    # Warmup all assets
    warmup()
"""

# Public API exports
from src.loaders.assets import get_cmap_data
from src.loaders.assets import get_cmap_file_path
from src.loaders.assets import get_doclayout_onnx_model_path
from src.loaders.assets import get_font_and_metadata
from src.loaders.assets import get_font_family
from src.loaders.assets import get_table_detection_rapidocr_model_path
from src.loaders.constants import DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256
from src.loaders.constants import TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256
from src.loaders.constants import TIKTOKEN_CACHES
from src.loaders.exceptions import AssetDownloadError

# Exceptions
from src.loaders.exceptions import AssetError
from src.loaders.exceptions import AssetIntegrityError
from src.loaders.exceptions import MetadataNotFoundError
from src.loaders.exceptions import WarmupError
from src.loaders.models import AssetMetadata
from src.loaders.models import CMapMetadata

# Models
from src.loaders.models import FontFamilyConfig
from src.loaders.models import FontMetadata
from src.loaders.models.font_families import ALL_FONT_FAMILIES as ALL_FONT_FAMILY

# Backward compatibility - re-export from canonical sources
from src.loaders.repositories.metadata_repository import CMAP_METADATA
from src.loaders.repositories.metadata_repository import EMBEDDING_FONT_METADATA
from src.loaders.repositories.metadata_repository import get_font_names as font_names
from src.loaders.services.warmup_service import async_warmup
from src.loaders.services.warmup_service import warmup
from src.loaders.utils.path_helpers import get_cache_file_path

__all__ = [
    # Public API
    "get_cache_file_path",
    "get_doclayout_onnx_model_path",
    "get_table_detection_rapidocr_model_path",
    "get_font_and_metadata",
    "get_font_family",
    "get_cmap_file_path",
    "get_cmap_data",
    "warmup",
    "async_warmup",
    # Exceptions
    "AssetError",
    "AssetDownloadError",
    "AssetIntegrityError",
    "MetadataNotFoundError",
    "WarmupError",
    # Models
    "FontFamilyConfig",
    "FontMetadata",
    "CMapMetadata",
    "AssetMetadata",
    # Backward compatibility
    "EMBEDDING_FONT_METADATA",
    "CMAP_METADATA",
    "font_names",
    "ALL_FONT_FAMILY",
    "DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256",
    "TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256",
    "TIKTOKEN_CACHES",
]
