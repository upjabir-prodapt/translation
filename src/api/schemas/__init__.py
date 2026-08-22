"""API schemas package."""

from .common import BaseJobSchema
from .common import FileInfo
from .common import OutputFiles
from .common import ProgressInfo
from .requests import AuthTokenRequest
from .requests import JobCancelRequest
from .requests import JobListRequest
from .requests import MultiJobStatusRequest
from .requests import TranslateRequest
from .requests import TranslationTargetsInput
from .responses import AuthTokenResponse
from .responses import DownloadResponse
from .responses import ErrorResponse
from .responses import HealthResponse
from .responses import JobListResponse
from .responses import JobStatusResponse
from .responses import MultiJobStatusItemResponse
from .responses import MultiJobStatusResponse
from .responses import MultiTranslateJobResponse
from .responses import MultiTranslateResponse
from .responses import TranslateResponse

__all__ = [
    # Requests
    "TranslateRequest",
    "AuthTokenRequest",
    "JobCancelRequest",
    "JobListRequest",
    "TranslationTargetsInput",
    "MultiJobStatusRequest",
    # Responses
    "TranslateResponse",
    "MultiTranslateJobResponse",
    "MultiTranslateResponse",
    "MultiJobStatusItemResponse",
    "MultiJobStatusResponse",
    "AuthTokenResponse",
    "JobStatusResponse",
    "JobListResponse",
    "DownloadResponse",
    "HealthResponse",
    "ErrorResponse",
    # Common
    "BaseJobSchema",
    "FileInfo",
    "ProgressInfo",
    "OutputFiles",
]
