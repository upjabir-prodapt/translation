"""Single source of truth for the "run another model attempt?" decision.

`ModelAttemptOrchestrator` (PDF) and `DocxJobProcessor` (DOCX) both loop over
the model chain and both have to decide, after each attempt, whether to stop
or to burn another full translation pass. Those two loops were hand-mirrored
-- `docx_job_processor` literally carries a "Mirror ModelAttemptOrchestrator"
comment -- and they had already drifted: the DOCX loop guards both of its exit
conditions on `quality_result is not None`, so a judge failure there silently
falls through to "keep going" and runs the entire chain.

Centralising the rule here means a change to the stop condition cannot apply
to one pipeline and not the other, and one parametrised test covers both.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from src.config.constants import settings
from src.worker.services.quality_judge_service import QualityJudgeResult

logger = logging.getLogger(__name__)


class AttemptDecision(StrEnum):
    """What the model-attempt loop should do after scoring an attempt."""

    #: Score met QUALITY_THRESHOLD -- accept and stop.
    PASS = "pass"  # noqa: S105 - a judge verdict, not a credential
    #: Score is below the threshold but at/above QUALITY_EARLY_ACCEPT_THRESHOLD;
    #: chasing a marginal gain is not worth another full pass.
    EARLY_ACCEPT = "early_accept"
    #: The judge could not form an opinion. Re-translating on a different model
    #: cannot fix a *measurement* failure, so accept rather than paying 2-3x.
    INCONCLUSIVE_ACCEPT = "inconclusive_accept"
    #: Genuinely poor score -- try the next model in the chain.
    CONTINUE = "continue"

    @property
    def should_stop(self) -> bool:
        return self is not AttemptDecision.CONTINUE


def decide_after_attempt(
    quality_result: QualityJudgeResult | None,
    *,
    attempt_index: int,
    pipeline: str = "pdf",
) -> AttemptDecision:
    """Return whether the model-attempt loop should stop after this attempt.

    `quality_result is None` means the judge is disabled; with no signal to
    act on there is nothing to improve by retrying, so the first successful
    attempt is accepted.
    """
    if quality_result is None:
        return AttemptDecision.INCONCLUSIVE_ACCEPT

    if getattr(quality_result, "inconclusive", False):
        logger.warning(
            f"[{pipeline}] Quality judge was inconclusive for attempt "
            f"{attempt_index} (reasons={quality_result.reasons}); accepting this "
            "attempt instead of re-translating the document on another model. "
            "A judge failure is a measurement problem, not a translation problem."
        )
        return AttemptDecision.INCONCLUSIVE_ACCEPT

    if quality_result.pass_fail:
        return AttemptDecision.PASS

    early_accept = float(settings.QUALITY_EARLY_ACCEPT_THRESHOLD)
    final_score = float(quality_result.final_score)
    if 0 < early_accept <= final_score:
        logger.info(
            f"[{pipeline}] Early-accepting attempt {attempt_index} "
            f"(score={final_score:.3f} >= "
            f"QUALITY_EARLY_ACCEPT_THRESHOLD={early_accept}); "
            "skipping remaining model attempts"
        )
        return AttemptDecision.EARLY_ACCEPT

    return AttemptDecision.CONTINUE
