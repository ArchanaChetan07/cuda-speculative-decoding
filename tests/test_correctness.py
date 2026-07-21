"""
Correctness tests for Speculative Decoding Optimizer.

9 unit tests covering:
- Draft ranking logic
- Conflict detection
- Eviction scoring
- Block selection
- Confidence tracking
- End-to-end composition

All tests use NumPy reference implementation (CPU-only, no GPU required).
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from reference import (
    DraftStats,
    rank_drafts_by_confidence,
    detect_conflict,
    eviction_scores_ref,
    select_eviction_candidates_ref,
    update_draft_confidence,
    end_to_end_pipeline_reference,
)


# ============================================================================
# TEST SUITE 1: CORRECTNESS (9 tests)
# ============================================================================

class TestDraftRanking:
    """Tests for rank_drafts_by_confidence()"""

    def test_rank_drafts_no_tie(self):
        """Higher acceptance rate → lower index (higher priority)"""
        stats = DraftStats(3)
        stats.acceptance_rate = np.array([0.9, 0.5, 0.7], dtype=np.float32)
        result = rank_drafts_by_confidence(stats, 3)

        assert result[0] == 0, "Draft 0 (90%) should be primary"
        assert result[1] == 2, "Draft 2 (70%) should be secondary"
        assert result[2] == 1, "Draft 1 (50%) should be tertiary"

    def test_rank_drafts_tie_broken_by_activity(self):
        """Tie in acceptance rate broken by total_generated (prefer active)"""
        stats = DraftStats(2)
        stats.acceptance_rate = np.array([0.8, 0.8], dtype=np.float32)  # Tie
        stats.total_generated = np.array([100, 200], dtype=np.int64)     # Draft 1 more active

        result = rank_drafts_by_confidence(stats, 2)

        assert result[0] == 1, "Draft 1 (more active) wins tie"
        assert result[1] == 0

    def test_rank_drafts_single_draft(self):
        """Single draft is trivially ranked first"""
        stats = DraftStats(1)
        stats.acceptance_rate[0] = 0.5
        result = rank_drafts_by_confidence(stats, 1)

        assert len(result) == 1
        assert result[0] == 0


class TestConflictDetection:
    """Tests for detect_conflict()"""

    def test_conflict_full_overlap(self):
        """All desired evictions are locked"""
        eviction = np.array([True, True, True], dtype=bool)
        locked = np.array([True, True, True], dtype=bool)

        safe, penalty = detect_conflict(eviction, locked, is_primary=False)

        assert safe.sum() == 0, "No safe evictions"
        assert penalty == 1.0, "100% conflict"

    def test_conflict_no_overlap(self):
        """No overlap between eviction and locked"""
        eviction = np.array([True, True, False], dtype=bool)
        locked = np.array([False, False, True], dtype=bool)

        safe, penalty = detect_conflict(eviction, locked, is_primary=False)

        assert safe.sum() == 2, "All 2 evictions are safe"
        assert penalty == 0.0, "0% conflict"

    def test_conflict_partial_overlap(self):
        """50% of desired evictions conflict"""
        eviction = np.array([True, True, False], dtype=bool)  # Want to evict 2
        locked = np.array([False, True, False], dtype=bool)   # 1 locked

        safe, penalty = detect_conflict(eviction, locked, is_primary=False)

        assert safe.sum() == 1, "1 of 2 evictions is safe"
        assert abs(penalty - 0.5) < 1e-6, "50% conflict"

    def test_conflict_empty_eviction_plan(self):
        """No blocks to evict → 0% conflict"""
        eviction = np.array([False, False, False], dtype=bool)
        locked = np.array([True, True, True], dtype=bool)

        safe, penalty = detect_conflict(eviction, locked, is_primary=False)

        assert safe.sum() == 0
        assert penalty == 0.0, "Can't conflict if no evictions desired"


class TestEvictionScoring:
    """Tests for eviction_scores_ref()"""

    def test_scoring_basic(self):
        """Scores should be in a reasonable range"""
        num_blocks = 10
        recency = np.random.rand(num_blocks).astype(np.float32)
        frequency = np.random.rand(num_blocks).astype(np.float32)
        attn = np.random.rand(num_blocks, 16).astype(np.float32)
        valid = np.ones((num_blocks, 16), dtype=bool)

        scores = eviction_scores_ref(recency, frequency, attn, valid)

        assert scores.shape == (num_blocks,)
        assert scores.dtype == np.float32
        assert not np.any(np.isnan(scores)), "No NaN in scores"
        assert not np.any(np.isinf(scores)), "No Inf in scores"

    def test_scoring_empty_blocks(self):
        """Blocks with no valid tokens → mean_attn = 0"""
        num_blocks = 3
        recency = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        frequency = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        attn = np.ones((num_blocks, 4), dtype=np.float32)
        valid = np.array([
            [True, True, True, True],   # All valid
            [False, False, False, False],  # All invalid (empty block)
            [True, False, True, False],    # Partial
        ], dtype=bool)

        scores = eviction_scores_ref(recency, frequency, attn, valid)

        # Empty block should have lower score (more attractive to evict)
        # score = 0.4*rec + 0.3*freq + 0.3*mean_attn
        # Valid block: mean_attn = 1.0
        # Empty block: mean_attn = 0.0
        # Partial: mean_attn = 0.5
        assert scores[1] < scores[0], "Empty block has lower score (better eviction candidate)"


class TestBlockSelection:
    """Tests for select_eviction_candidates_ref()"""

    def test_selection_basic(self):
        """Select k lowest-scoring blocks"""
        scores = np.array([0.5, 0.3, 0.8, 0.2, 0.6], dtype=np.float32)
        selected = select_eviction_candidates_ref(scores, 2)

        assert len(selected) == 2
        assert list(selected) == [3, 1], f"Expected [3, 1], got {list(selected)}"
        assert scores[selected[0]] <= scores[selected[1]]

    def test_selection_deterministic_ties(self):
        """Ties broken by lower block index"""
        scores = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)  # All tied
        selected = select_eviction_candidates_ref(scores, 2)

        assert list(selected) == [0, 1], "Ties broken by index: [0, 1]"

    def test_selection_k_equals_num_blocks(self):
        """Select all blocks"""
        scores = np.array([0.3, 0.1, 0.5], dtype=np.float32)
        selected = select_eviction_candidates_ref(scores, 3)

        assert len(selected) == 3
        assert set(selected) == {0, 1, 2}


class TestConfidenceTracking:
    """Tests for update_draft_confidence()"""

    def test_confidence_single_acceptance(self):
        """One acceptance moves rate from 0.5 → 0.55"""
        rate = 0.5
        updated = update_draft_confidence(rate, accepted=True, alpha=0.1)

        expected = 0.1 * 1.0 + 0.9 * 0.5
        assert abs(updated - expected) < 1e-6

    def test_confidence_single_rejection(self):
        """One rejection moves rate from 0.5 → 0.45"""
        rate = 0.5
        updated = update_draft_confidence(rate, accepted=False, alpha=0.1)

        expected = 0.1 * 0.0 + 0.9 * 0.5
        assert abs(updated - expected) < 1e-6

    def test_confidence_converges_to_one(self):
        """Repeated acceptances converge toward 1.0"""
        rate = 0.5
        for _ in range(100):
            rate = update_draft_confidence(rate, accepted=True, alpha=0.1)

        assert rate > 0.95, f"Expected >0.95, got {rate}"

    def test_confidence_converges_to_zero(self):
        """Repeated rejections converge toward 0.0"""
        rate = 0.5
        for _ in range(100):
            rate = update_draft_confidence(rate, accepted=False, alpha=0.1)

        assert rate < 0.05, f"Expected <0.05, got {rate}"


class TestComposition:
    """Test that components compose correctly"""

    def test_end_to_end_pipeline(self):
        """Full pipeline runs without error"""
        result = end_to_end_pipeline_reference(
            num_seqs=32,
            num_blocks=4096,
            block_size=16,
            num_drafts=4,
            eviction_fraction=0.15
        )

        assert result['num_seqs'] == 32
        assert result['num_blocks'] == 4096
        assert 0 <= result['rollback_rate'] <= 1.0
        assert result['evicted_blocks'] > 0

    def test_integration_score_select_rank(self):
        """Scoring, selection, and ranking work together"""
        num_blocks = 100
        recency = np.random.rand(num_blocks).astype(np.float32)
        frequency = np.random.rand(num_blocks).astype(np.float32)
        attn = np.random.rand(num_blocks, 16).astype(np.float32)
        valid = np.ones((num_blocks, 16), dtype=bool)

        # Score
        scores = eviction_scores_ref(recency, frequency, attn, valid)

        # Select
        selected = select_eviction_candidates_ref(scores, 15)

        # Verify: selected blocks should have lower scores than non-selected
        selected_scores = scores[selected]
        non_selected_scores = scores[[i for i in range(num_blocks) if i not in selected]]

        assert selected_scores.max() <= non_selected_scores.min() + 1e-5, \
            "Selected blocks should have lower scores"


if __name__ == '__main__':
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
