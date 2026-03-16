__version__ = "0.5.23"

# New async translation service API
from babeldoc.format.pdf.high_level import async_translate
from babeldoc.new_main import cancel_job
from babeldoc.new_main import cleanup_job
from babeldoc.new_main import cleanup_job_directory
from babeldoc.new_main import create_translation_job
from babeldoc.new_main import get_job_status
from babeldoc.new_main import list_jobs
from babeldoc.new_main import run_translation_job
from babeldoc.new_main import translate_pdf
from schemas.babeldoc_schemas import TranslationConfigSchema
from schemas.babeldoc_schemas import TranslationJobStatus
from schemas.babeldoc_schemas import TranslationModelSchema

__all__ = [
    # New async service
    "create_translation_job",
    "run_translation_job",
    "get_job_status",
    "list_jobs",
    "cancel_job",
    "cleanup_job",
    "cleanup_job_directory",
    "translate_pdf",
    "async_translate",
    # Schemas
    "TranslationConfigSchema",
    "TranslationModelSchema",
    "TranslationJobStatus",
    # Legacy
    "__version__",
]
