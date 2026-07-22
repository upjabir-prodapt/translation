"""Shared domain exceptions used across API and worker."""


class TranslationDomainError(Exception):
    """Base class for translation domain errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(TranslationDomainError):
    """Invalid or missing configuration."""


class StorageError(TranslationDomainError):
    """GCS / artifact storage failure."""
