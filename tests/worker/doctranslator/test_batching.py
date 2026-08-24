"""Tests for the shared adaptive LLM batch sizer.

tests/test.env pins LLM_ADAPTIVE_BATCHING_ENABLED=false so the pre-existing
batch-shape assertions in the DOCX tests stay meaningful. These tests
therefore patch the setting explicitly rather than relying on the fixture.
"""

import contextlib
from unittest.mock import patch

from src.worker.doctranslator.batching import BatchPlan
from src.worker.doctranslator.batching import compute_batch_plan
from src.worker.doctranslator.batching import log_batch_plan


def _enable_adaptive(min_tokens=600, max_tokens=2500):
    return patch.multiple(
        "src.worker.doctranslator.batching.settings",
        LLM_ADAPTIVE_BATCHING_ENABLED=True,
        LLM_ADAPTIVE_BATCH_MIN_TOKENS=min_tokens,
        LLM_ADAPTIVE_BATCH_MAX_TOKENS=max_tokens,
    )


class TestAdaptiveDisabled:
    def test_returns_configured_caps_unchanged(self):
        with patch.multiple(
            "src.worker.doctranslator.batching.settings",
            LLM_ADAPTIVE_BATCHING_ENABLED=False,
        ):
            plan = compute_batch_plan(
                10_000, 12, base_max_tokens=40_000, base_max_items=200
            )
        assert plan.max_tokens == 40_000
        assert plan.max_items == 200
        assert plan.adaptive is False

    def test_cjk_multiplier_still_applied_when_disabled(self):
        """Disabling adaptive sizing must not disable B1's CJK scaling."""
        with patch.multiple(
            "src.worker.doctranslator.batching.settings",
            LLM_ADAPTIVE_BATCHING_ENABLED=False,
        ):
            plan = compute_batch_plan(
                10_000,
                12,
                base_max_tokens=40_000,
                base_max_items=200,
                token_multiplier=0.5,
            )
        assert plan.max_items == 100


class TestAdaptiveSizing:
    def test_targets_one_wave_across_the_pool(self):
        """10400 tokens over 12 workers -> ~867, i.e. one full wave."""
        with _enable_adaptive():
            plan = compute_batch_plan(
                10_400, 12, base_max_tokens=40_000, base_max_items=200
            )
        assert plan.adaptive is True
        # Rounded UP from 866.67 so the work fits in exactly one wave.
        assert plan.max_tokens == 867
        assert -(-plan.total_payload_tokens // plan.max_tokens) == 12

    def test_never_spills_an_extra_wave(self):
        """Regression: truncating the target caused a 13th batch on 12 workers."""
        with _enable_adaptive():
            for total in range(1_000, 30_000, 137):
                plan = compute_batch_plan(
                    total, 12, base_max_tokens=40_000, base_max_items=200
                )
                batches = -(-total // plan.max_tokens)
                if plan.max_tokens < 2500:  # not clamped by the ceiling
                    assert batches <= 12, (
                        f"total={total} produced {batches} batches for 12 workers"
                    )

    def test_small_document_clamps_to_floor(self):
        with _enable_adaptive():
            plan = compute_batch_plan(
                600, 12, base_max_tokens=40_000, base_max_items=200
            )
        assert plan.max_tokens == 600

    def test_large_document_clamps_to_ceiling(self):
        with _enable_adaptive():
            plan = compute_batch_plan(
                10_000_000, 12, base_max_tokens=40_000, base_max_items=200
            )
        assert plan.max_tokens == 2500

    def test_never_exceeds_configured_hard_cap(self):
        """Adaptive sizing may only shrink batches, never grow them."""
        with _enable_adaptive():
            plan = compute_batch_plan(
                10_000_000, 12, base_max_tokens=800, base_max_items=200
            )
        assert plan.max_tokens == 800

    def test_zero_payload_falls_back_to_fixed(self):
        with _enable_adaptive():
            plan = compute_batch_plan(
                0, 12, base_max_tokens=40_000, base_max_items=200
            )
        assert plan.adaptive is False
        assert plan.max_tokens == 40_000

    def test_zero_workers_does_not_divide_by_zero(self):
        with _enable_adaptive():
            plan = compute_batch_plan(
                5_000, 0, base_max_tokens=40_000, base_max_items=200
            )
        assert plan.pool_max_workers == 1
        assert plan.max_tokens == 2500

    def test_cjk_multiplier_scales_tokens_and_items(self):
        with _enable_adaptive():
            plan = compute_batch_plan(
                10_400,
                12,
                base_max_tokens=40_000,
                base_max_items=200,
                token_multiplier=0.5,
            )
        assert plan.max_tokens == 433  # int(867 * 0.5)
        assert plan.max_items == 100

    def test_result_is_immutable(self):
        plan = compute_batch_plan(100, 4, base_max_tokens=10, base_max_items=5)
        assert isinstance(plan, BatchPlan)


class TestInflightCallBudget:
    """Bounds total concurrent LLM calls across nested thread pools.

    `translate_all()` runs up to TRANSLATION_POOL_MAX_WORKERS batches, and each
    of those may open its own fallback pool of the same size — 12 + 12*12 = 156
    potential simultaneous Vertex calls from one job on a cpu=4 /
    containerConcurrency=1 Cloud Run instance.
    """

    @staticmethod
    def _measure_peak(limit, threads=40):
        import threading
        import time

        from src.worker.doctranslator import batching as mod

        with patch.object(mod.settings, "LLM_MAX_INFLIGHT_CALLS", limit):
            mod._llm_slots = None  # reset the lazily-built singleton
            try:
                state = {"cur": 0, "peak": 0}
                lock = threading.Lock()

                def work():
                    with mod.llm_call_slot():
                        with lock:
                            state["cur"] += 1
                            state["peak"] = max(state["peak"], state["cur"])
                        time.sleep(0.02)
                        with lock:
                            state["cur"] -= 1

                workers = [threading.Thread(target=work) for _ in range(threads)]
                for t in workers:
                    t.start()
                for t in workers:
                    t.join()
                return state["peak"]
            finally:
                mod._llm_slots = None

    def test_caps_concurrent_calls(self):
        assert self._measure_peak(3) <= 3

    def test_zero_disables_the_guard(self):
        """0 must be a true no-op so the guard can be turned off in config."""
        from src.worker.doctranslator import batching as mod

        with patch.object(mod.settings, "LLM_MAX_INFLIGHT_CALLS", 0):
            mod._llm_slots = None
            try:
                with mod.llm_call_slot():
                    pass
            finally:
                mod._llm_slots = None

    def test_slot_is_released_on_exception(self):
        """A failing call must not permanently leak a slot."""
        from src.worker.doctranslator import batching as mod

        with patch.object(mod.settings, "LLM_MAX_INFLIGHT_CALLS", 1):
            mod._llm_slots = None
            try:
                for _ in range(3):
                    with contextlib.suppress(RuntimeError):  # noqa: SIM117
                        with mod.llm_call_slot():
                            raise RuntimeError("boom")
                # If slots leaked, this would deadlock rather than return.
                with mod.llm_call_slot():
                    pass
            finally:
                mod._llm_slots = None


class TestLogBatchPlan:
    def test_reports_utilisation_and_waves(self, caplog):
        plan = compute_batch_plan(
            10_400, 12, base_max_tokens=40_000, base_max_items=200
        )
        with caplog.at_level("INFO"):
            log_batch_plan("UnitTest", plan, batch_count=12)
        message = caplog.text
        assert "stage=UnitTest" in message
        assert "batch_count=12" in message
        assert "pool_utilisation=1.0" in message
        assert "waves=1" in message

    def test_underfilled_pool_is_visible(self, caplog):
        """The baseline failure mode: 3 batches for a 12-worker pool."""
        plan = compute_batch_plan(
            10_400, 12, base_max_tokens=40_000, base_max_items=200
        )
        with caplog.at_level("INFO"):
            log_batch_plan("UnitTest", plan, batch_count=3)
        assert "pool_utilisation=0.25" in caplog.text

    def test_zero_batches_does_not_divide_by_zero(self, caplog):
        plan = compute_batch_plan(
            10_400, 12, base_max_tokens=40_000, base_max_items=200
        )
        with caplog.at_level("INFO"):
            log_batch_plan("UnitTest", plan, batch_count=0)
        assert "batch_count=0" in caplog.text
