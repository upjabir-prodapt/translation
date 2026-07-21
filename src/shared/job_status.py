"""Shared job status constants."""

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
HUMAN_REVIEW_REQUIRED = "human_review_required"

TERMINAL_STATUSES = frozenset(
    {COMPLETED, FAILED, CANCELLED, HUMAN_REVIEW_REQUIRED}
)
