"""Data models for asset metadata."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssetMetadata:
    """Base metadata for any downloadable asset."""

    name: str
    sha3_256: str
    size: int | None = None
    url: str | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "AssetMetadata":
        """Create metadata from a dictionary."""
        return cls(
            name=name,
            sha3_256=data.get("sha3_256", ""),
            size=data.get("size"),
            url=data.get("url"),
        )


@dataclass(frozen=True)
class FontMetadata(AssetMetadata):
    """Metadata for a font file."""

    font_name: str = ""
    subset_font_path: str | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "FontMetadata":
        """Create font metadata from a dictionary."""
        return cls(
            name=name,
            sha3_256=data.get("sha3_256", ""),
            size=data.get("size"),
            url=data.get("url"),
            font_name=data.get("font_name", ""),
            subset_font_path=data.get("subset_font_path"),
        )


@dataclass(frozen=True)
class CMapMetadata(AssetMetadata):
    """Metadata for a CMap file."""

    cmap_name: str = ""

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "CMapMetadata":
        """Create CMap metadata from a dictionary."""
        return cls(
            name=name,
            sha3_256=data.get("sha3_256", ""),
            size=data.get("size"),
            url=data.get("url"),
            cmap_name=data.get("cmap_name", ""),
        )


@dataclass(frozen=True)
class ModelMetadata(AssetMetadata):
    """Metadata for an ML model file."""

    model_type: str = ""
    description: str = ""
