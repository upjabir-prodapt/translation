"""Loaders module - Asset loading and management.

This module provides a clean API for downloading, caching, and accessing
assets like fonts, models, and CMap files.

Example usage:
    from loaders import get_font_family, warmup

    # Get font family for a language
    fonts = get_font_family("CN")

    # Warmup all assets
    warmup()
"""

# Public API exports
from loaders.assets import async_warmup
from loaders.assets import get_cache_file_path
from loaders.assets import get_cmap_data
from loaders.assets import get_cmap_file_path
from loaders.assets import get_doclayout_onnx_model_path
from loaders.assets import get_font_and_metadata
from loaders.assets import get_font_family
from loaders.assets import get_table_detection_rapidocr_model_path
from loaders.assets import warmup
from loaders.constants import DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256
from loaders.constants import TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256
from loaders.constants import TIKTOKEN_CACHES
from loaders.exceptions import AssetDownloadError

# Exceptions
from loaders.exceptions import AssetError
from loaders.exceptions import AssetIntegrityError
from loaders.exceptions import MetadataNotFoundError
from loaders.exceptions import WarmupError
from loaders.models import AssetMetadata
from loaders.models import CMapMetadata

# Models
from loaders.models import FontFamilyConfig
from loaders.models import FontMetadata
from loaders.models.font_families import ALL_FONT_FAMILIES as ALL_FONT_FAMILY

# Backward compatibility - re-export from canonical sources
from loaders.repositories.metadata_repository import CMAP_METADATA
from loaders.repositories.metadata_repository import EMBEDDING_FONT_METADATA
from loaders.repositories.metadata_repository import get_font_names as FONT_NAMES

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
    "FONT_NAMES",
    "ALL_FONT_FAMILY",
    "DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256",
    "TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256",
    "TIKTOKEN_CACHES",
]
