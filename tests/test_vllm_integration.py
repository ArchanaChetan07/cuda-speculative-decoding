"""Integration tests for the vLLM shim (OptimizedDraftManager)."""

import numpy as np
import pytest

from src.vllm_integration import OptimizedDraftManager, RollbackTracker


class TestOptimizedDraftManager:

    def test_full_step_cycle(self):
        """Rank -> lock -> plan eviction -> record -> unlock."""
        mgr = OptimizedDraftManager(num_drafts=4, num_blocks=64)

        order = mgr.rank_for_step()
        assert len(order) == 4

        # Primary draft locks its blocks
        mgr.lock_blocks([0, 1, 2, 3])

        # Secondary draft wants overlapping eviction
        plan = np.zeros(64, dtype=bool)
        plan[[2, 3, 10, 11]] = True
        safe, penalty = mgr.plan_eviction(plan)

        assert not safe[2] and not safe[3], "locked blocks must be filtered"
        assert safe[10] and safe[11], "unlocked blocks must survive"
        assert penalty == pytest.approx(0.5), "2 of 4 desired evictions conflicted"

        mgr.record_outcome(int(order[0]), accepted=True)
        mgr.unlock_all()
        assert not mgr.locked_blocks.any()

    def test_confidence_shifts_ranking(self):
        """Repeated acceptance should promote a draft to primary."""
        mgr = OptimizedDraftManager(num_drafts=3)

        for _ in range(50):
            mgr.record_outcome(2, accepted=True)
            mgr.record_outcome(0, accepted=False)
            mgr.record_outcome(1, accepted=False)

        order = mgr.rank_for_step()
        assert order[0] == 2, "consistently accepted draft becomes primary"

    def test_acceptance_summary(self):
        mgr = OptimizedDraftManager(num_drafts=2)
        mgr.record_outcome(0, accepted=True)
        summary = mgr.acceptance_summary()
        assert summary['total_generated'][0] == 1
        assert summary['accepted'][0] == 1


class TestRollbackTracker:

    def test_rollback_rate(self):
        tracker = RollbackTracker(window=100)
        for i in range(100):
            tracker.record(rolled_back=(i % 4 == 0))  # 25% rollback
        assert tracker.rollback_rate() == pytest.approx(0.25)

    def test_sliding_window(self):
        tracker = RollbackTracker(window=10)
        for _ in range(10):
            tracker.record(True)
        for _ in range(10):
            tracker.record(False)
        assert tracker.rollback_rate() == 0.0, "old events must slide out"

    def test_empty(self):
        assert RollbackTracker().rollback_rate() == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
