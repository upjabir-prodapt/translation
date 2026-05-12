"""Cover page metadata support."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime


class CoverPageService:
    """Build cover-page metadata persisted with final output."""

    def build(
        self, *, model_used: str, confidence_score: float, reviewer: str
    ) -> dict[str, str | float]:
        return {
            "disclaimer": "This document was translated using AI assistance and should be reviewed before legal or operational use.",
            "translation_date": datetime.now(UTC).isoformat(),
            "model_used": model_used,
            "confidence_score": confidence_score,
            "assigned_reviewer": reviewer,
        }
