__version__ = "0.5.23"


def async_translate(*args, **kwargs):
    """Lazy proxy to avoid importing heavy PDF pipeline at module import time."""
    from src.doctranslator.format.pdf.high_level import async_translate as _async_translate

    return _async_translate(*args, **kwargs)


__all__ = [
    "async_translate",
    "__version__",
]
