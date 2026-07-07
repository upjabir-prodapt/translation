"""Optimized asset loader with retry logic and robust error handling.

This module provides a clean public API for accessing cached assets.
Implementation is delegated to the loaders.services and loaders.repositories modules.
"""

import json
from pathlib import Path
from typing import Any

from config.constants import settings
from loaders.constants import DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256
from loaders.constants import TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256
from loaders.exceptions import AssetIntegrityError
from loaders.exceptions import MetadataNotFoundError
from loaders.models import FontFamilyConfig
from loaders.models.font_families import get_font_family as _get_font_family
from loaders.repositories.cache_repository import verify_or_delete
from loaders.repositories.metadata_repository import get_cmap_metadata_by_name
from loaders.repositories.metadata_repository import get_font_metadata_by_name
from loaders.services.download_service import download_and_verify
from loaders.services.download_service import get_or_download_model
from loaders.utils.path_helpers import get_cache_file_path

# ============================================================================
# Model Access Functions
# ============================================================================


def get_doclayout_onnx_model_path() -> Path:
    """Get the path to the DocLayout ONNX model, downloading if necessary.

    Returns:
        Path to the verified model file

    Raises:
        AssetIntegrityError: If download or verification fails
    """
    try:
        return get_or_download_model(
            filename=settings.DOCLAYOUT_MODEL_FILENAME,
            expected_hash=DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256,
            model_name="DocLayout",
            subdir=settings.MODELS_DIR,
        )
    except Exception as e:
        raise AssetIntegrityError(
            f"Failed to get DocLayout model: {e}",
            asset_name=settings.DOCLAYOUT_MODEL_FILENAME,
        ) from e


def get_table_detection_rapidocr_model_path() -> Path:
    """Get the path to the RapidOCR table detection model, downloading if necessary.

    Returns:
        Path to the verified model file

    Raises:
        AssetIntegrityError: If download or verification fails
    """
    try:
        return get_or_download_model(
            filename=settings.TABLE_DETECTION_MODEL_FILENAME,
            expected_hash=TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256,
            model_name="Table Detection",
            subdir=settings.MODELS_DIR,
        )
    except Exception as e:
        raise AssetIntegrityError(
            f"Failed to get Table Detection model: {e}",
            asset_name=settings.TABLE_DETECTION_MODEL_FILENAME,
        ) from e


# ============================================================================
# Font Access Functions
# ============================================================================


def get_font_and_metadata(font_file_name: str) -> tuple[Path, dict[str, Any]]:
    """Get font file path and metadata, downloading if necessary.

    Args:
        font_file_name: Name of the font file (e.g., "SourceHanSansCN-Regular.ttf")

    Returns:
        Tuple of (font_path, metadata_dict)

    Raises:
        MetadataNotFoundError: If font not found in metadata
        AssetIntegrityError: If download verification fails
    """
    metadata = get_font_metadata_by_name(font_file_name)
    if metadata is None:
        raise MetadataNotFoundError(
            f"Font {font_file_name} not found in metadata",
            asset_name=font_file_name,
        )

    font_path = get_cache_file_path(font_file_name, settings.FONTS_DIR)

    # Check if already valid
    if verify_or_delete(font_path, metadata.sha3_256):
        return font_path, {
            "sha3_256": metadata.sha3_256,
            "size": metadata.size,
            "url": metadata.url,
            "font_name": metadata.font_name,
            "subset_font_path": metadata.subset_font_path,
        }

    # Download and verify
    download_and_verify(
        blob_path=f"{settings.FONTS_DIR}/{font_file_name}",
        local_path=font_path,
        expected_hash=metadata.sha3_256,
        asset_name=font_file_name,
    )

    return font_path, {
        "sha3_256": metadata.sha3_256,
        "size": metadata.size,
        "url": metadata.url,
        "font_name": metadata.font_name,
        "subset_font_path": metadata.subset_font_path,
    }


def get_font_family(lang_code: str) -> FontFamilyConfig:
    """Get the appropriate font family for the language code.

    Args:
        lang_code: Language code (e.g., "CN", "JA", "EN")

    Returns:
        FontFamilyConfig containing script, normal, fallback, and base fonts
    """
    return _get_font_family(lang_code)


# ============================================================================
# CMap Access Functions
# ============================================================================


def get_cmap_file_path(name: str) -> Path:
    """Get CMap file path, downloading if necessary.

    Args:
        name: CMap name (with or without .json extension)

    Returns:
        Path to the CMap file

    Raises:
        MetadataNotFoundError: If CMap not found in metadata
        AssetIntegrityError: If download verification fails
    """
    file_name = f"{name}.json" if not name.endswith(".json") else name

    metadata = get_cmap_metadata_by_name(file_name)
    if metadata is None:
        raise MetadataNotFoundError(
            f"CMap {file_name} not found in metadata",
            asset_name=file_name,
        )

    cmap_path = get_cache_file_path(file_name, settings.CMAP_DIR)

    # Check if already valid
    if verify_or_delete(cmap_path, metadata.sha3_256):
        return cmap_path

    # Download and verify
    download_and_verify(
        blob_path=f"{settings.CMAP_DIR}/{file_name}",
        local_path=cmap_path,
        expected_hash=metadata.sha3_256,
        asset_name=file_name,
    )

    return cmap_path


def get_cmap_data(name: str) -> dict[str, Any]:
    """Get CMap JSON data.

    Args:
        name: CMap name (with or without .json extension)

    Returns:
        Parsed CMap JSON data

    Raises:
        MetadataNotFoundError: If CMap not found
        json.JSONDecodeError: If JSON is invalid
    """
    path = get_cmap_file_path(name)
    return json.loads(path.read_text())
