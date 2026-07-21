"""
vLLM integration shim for the Speculative Decoding Optimizer.

Wraps vLLM's speculative decoding proposer/scorer flow with:
- deterministic draft ranking (drop-in ordering for proposal batches)
- soft-lock conflict resolution before KV block eviction

Usage (inside a vLLM fork or plugin):

    from spec_decode_opt.vllm_integration import OptimizedDraftManager

    manager = OptimizedDraftManager(num_drafts=4)
    order = manager.rank_for_step()           # deterministic proposal order
    safe, penalty = manager.plan_eviction(eviction_plan)  # conflict-safe plan
    manager.record_outcome(draft_id, accepted=True)       # feed back results
"""

import numpy as np
from typing import List, Optional, Tuple

from . import reference
from . import ops


class OptimizedDraftManager:
    """Manages draft ranking and KV conflict resolution for one sequence group."""

    def __init__(self, num_drafts: int = 4, num_blocks: int = 1024, ema_alpha: float = 0.1):
        self.num_drafts = num_drafts
        self.ema_alpha = ema_alpha
        self.stats = reference.DraftStats(num_drafts)
        self.locked_blocks = np.zeros(num_blocks, dtype=bool)

    def rank_for_step(self) -> np.ndarray:
        """Deterministic draft order for the next proposal step.

        Highest-confidence draft first; it gets first claim on KV blocks.
        """
        return ops.rank_drafts_by_confidence(self.stats, self.num_drafts)

    def lock_blocks(self, block_indices: List[int]) -> None:
        """Soft-lock blocks claimed by the primary draft."""
        self.locked_blocks[block_indices] = True

    def unlock_all(self) -> None:
        """Release all soft locks (end of verification step)."""
        self.locked_blocks[:] = False

    def plan_eviction(
        self,
        eviction_plan: np.ndarray,
        is_primary: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Filter an eviction plan against current soft locks.

        Returns (safe_plan, conflict_penalty). A high penalty signals the
        scheduler to defer this draft's eviction rather than rolling back.
        """
        return ops.detect_conflict(eviction_plan, self.locked_blocks, is_primary)

    def record_outcome(self, draft_id: int, accepted: bool) -> None:
        """Feed verification results back into confidence tracking (EMA)."""
        self.stats.update(draft_id, accepted, alpha=self.ema_alpha)

    def acceptance_summary(self) -> dict:
        """Current acceptance statistics per draft."""
        return {
            'acceptance_rate': self.stats.acceptance_rate.tolist(),
            'total_generated': self.stats.total_generated.tolist(),
            'accepted': self.stats.accepted.tolist(),
        }


class RollbackTracker:
    """Tracks rollback frequency to validate the 23% -> 8% target in production."""

    def __init__(self, window: int = 1000):
        self.window = window
        self.events: List[bool] = []

    def record(self, rolled_back: bool) -> None:
        self.events.append(rolled_back)
        if len(self.events) > self.window:
            self.events.pop(0)

    def rollback_rate(self) -> float:
        if not self.events:
            return 0.0
        return sum(self.events) / len(self.events)
