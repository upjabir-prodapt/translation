"""Custom exceptions for the loaders module."""


class AssetError(Exception):
    """Base exception for all asset-related errors."""

    def __init__(self, message: str, asset_name: str | None = None) -> None:
        super().__init__(message)
        self.asset_name = asset_name


class AssetDownloadError(AssetError):
    """Raised when an asset fails to download after all retry attempts."""

    def __init__(
        self,
        message: str,
        asset_name: str | None = None,
        blob_path: str | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message, asset_name)
        self.blob_path = blob_path
        self.attempts = attempts


class AssetIntegrityError(AssetError):
    """Raised when an asset fails integrity verification."""

    def __init__(
        self,
        message: str,
        asset_name: str | None = None,
        expected_hash: str | None = None,
        actual_hash: str | None = None,
    ) -> None:
        super().__init__(message, asset_name)
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class MetadataNotFoundError(AssetError):
    """Raised when metadata cannot be loaded or is invalid."""

    def __init__(
        self,
        message: str,
        asset_name: str | None = None,
        metadata_file: str | None = None,
    ) -> None:
        super().__init__(message, asset_name)
        self.metadata_file = metadata_file


class FontFamilyError(AssetError):
    """Raised when a font family configuration is invalid."""

    def __init__(
        self,
        message: str,
        lang_code: str | None = None,
        font_family: dict | str | None = None,
    ) -> None:
        super().__init__(message)
        self.lang_code = lang_code
        self.font_family = font_family


class WarmupError(AssetError):
    """Raised when the asset warmup process fails."""

    def __init__(
        self,
        message: str,
        failed_assets: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_assets = failed_assets or []
