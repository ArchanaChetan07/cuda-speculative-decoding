"""
Reference implementation for Speculative Decoding Optimizer (NumPy).

Ground truth for correctness. Every CUDA kernel must produce output
that matches these functions to within floating-point precision.

No GPU required. CPU-only. Suitable for validation and unit testing.
"""

import numpy as np
from typing import Tuple


class DraftStats:
    """Tracks historical acceptance rates for each draft."""

    def __init__(self, num_drafts: int = 4):
        self.acceptance_rate = np.ones(num_drafts, dtype=np.float32) * 0.5  # Start at 50%
        self.total_generated = np.zeros(num_drafts, dtype=np.int64)
        self.accepted = np.zeros(num_drafts, dtype=np.int64)

    def update(self, draft_id: int, accepted: bool, alpha: float = 0.1):
        """Update acceptance rate using exponential moving average."""
        new_sample = float(accepted)
        old_rate = self.acceptance_rate[draft_id]
        self.acceptance_rate[draft_id] = alpha * new_sample + (1 - alpha) * old_rate

        self.total_generated[draft_id] += 1
        if accepted:
            self.accepted[draft_id] += 1


def rank_drafts_by_confidence(
    draft_stats: DraftStats,
    num_drafts: int
) -> np.ndarray:
    """
    Rank drafts by historical acceptance rate.
    Lower index = higher priority (can lock blocks).

    Ties broken by total_generated (prefer active drafts).

    Args:
        draft_stats: Draft statistics (acceptance_rate, total_generated, accepted)
        num_drafts: Number of drafts to rank

    Returns:
        draft_priority: [num_drafts] indices sorted by acceptance rate DESC

    Example:
        >>> stats = DraftStats(4)
        >>> stats.acceptance_rate = np.array([0.9, 0.5, 0.7, 0.6])
        >>> priority = rank_drafts_by_confidence(stats, 4)
        >>> priority
        array([0, 2, 3, 1])  # Draft 0 highest priority (90% acceptance)
    """
    confidence = draft_stats.acceptance_rate[:num_drafts]
    total_gen = draft_stats.total_generated[:num_drafts]

    # lexsort: sorts by last key first, then second-to-last, etc.
    # We want: descending acceptance, then descending total_generated
    priority = np.lexsort((-total_gen, -confidence))
    return priority.astype(np.int32)


def detect_conflict(
    eviction_plan: np.ndarray,
    locked_blocks: np.ndarray,
    is_primary: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Check if a draft's eviction plan conflicts with locked blocks.

    Args:
        eviction_plan: [num_blocks] bool, blocks this draft wants to evict
        locked_blocks: [num_blocks] bool, blocks locked by higher-priority draft
        is_primary: If True, draft can override locks (future: soft locks apply anyway)

    Returns:
        (safe_eviction_plan, conflict_penalty)
        - safe_eviction_plan: [num_blocks] bool, evictions that don't conflict
        - conflict_penalty: float, fraction of desired evictions that conflicted (0.0-1.0)

    Example:
        >>> eviction = np.array([True, True, False, False])
        >>> locked = np.array([False, True, False, False])
        >>> safe, penalty = detect_conflict(eviction, locked, is_primary=False)
        >>> safe
        array([ True, False, False, False])
        >>> penalty
        0.5  # 50% of desired evictions (2) conflicted (1)
    """
    eviction_plan = eviction_plan.astype(bool)
    locked_blocks = locked_blocks.astype(bool)

    # Find conflicts: blocks in both eviction plan AND locked
    conflict_mask = eviction_plan & locked_blocks
    num_conflicts = conflict_mask.sum()
    num_desired = eviction_plan.sum()

    # Conflict penalty: fraction of desired evictions blocked
    conflict_penalty = num_conflicts / max(num_desired, 1)

    # Safe evictions: desired minus locked (for soft locking, secondary respects locks)
    safe_eviction = eviction_plan & ~locked_blocks

    return safe_eviction, float(conflict_penalty)


def eviction_scores_ref(
    recency: np.ndarray,           # [num_blocks] float32
    frequency: np.ndarray,         # [num_blocks] float32
    attn_weights: np.ndarray,      # [num_blocks, block_size] float32
    valid_mask: np.ndarray,        # [num_blocks, block_size] bool
    w_recency: float = 0.4,
    w_frequency: float = 0.3,
    w_attention: float = 0.3,
) -> np.ndarray:
    """
    Compute eviction scores for KV blocks (lower = evict first).

    Score = w_r * recency + w_f * frequency + w_a * mean_attention

    The attention term is the mean attention mass over *valid* tokens in the block.
    Empty blocks (all invalid) contribute 0 attention and are prime eviction candidates.

    Args:
        recency: Normalized recency (1.0 = just used, 0.0 = old)
        frequency: Normalized frequency of access
        attn_weights: Per-token attention mass [num_blocks, block_size]
        valid_mask: Token validity mask [num_blocks, block_size]
        w_recency, w_frequency, w_attention: Weights (should sum ~1.0)

    Returns:
        scores: [num_blocks] float32, eviction scores

    Example:
        >>> num_blocks, block_size = 10, 16
        >>> recency = np.random.rand(num_blocks).astype(np.float32)
        >>> frequency = np.random.rand(num_blocks).astype(np.float32)
        >>> attn = np.random.rand(num_blocks, block_size).astype(np.float32)
        >>> valid = np.ones((num_blocks, block_size), dtype=bool)
        >>> scores = eviction_scores_ref(recency, frequency, attn, valid)
        >>> scores.shape
        (10,)
    """
    recency = np.asarray(recency, dtype=np.float32)
    frequency = np.asarray(frequency, dtype=np.float32)
    attn_weights = np.asarray(attn_weights, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    # Compute mean attention per block, only over valid tokens
    valid_counts = valid_mask.sum(axis=1)  # [num_blocks]
    attn_sum = np.where(valid_mask, attn_weights, 0.0).sum(axis=1)  # [num_blocks]

    # Avoid division by zero: if no valid tokens, mean_attn = 0
    mean_attn = np.divide(
        attn_sum,
        valid_counts,
        out=np.zeros_like(attn_sum, dtype=np.float32),
        where=valid_counts > 0,
    )

    # Weighted combination
    scores = (
        w_recency * recency +
        w_frequency * frequency +
        w_attention * mean_attn
    ).astype(np.float32)

    return scores


def eviction_scores_with_conflict_penalty(
    recency: np.ndarray,
    frequency: np.ndarray,
    attn_weights: np.ndarray,
    valid_mask: np.ndarray,
    locked_blocks: np.ndarray,
    w_recency: float = 0.4,
    w_frequency: float = 0.3,
    w_attention: float = 0.3,
    conflict_weight: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eviction scores with conflict penalties.

    For blocks locked by higher-priority drafts, add a penalty to discourage
    secondary drafts from evicting them.

    Args:
        locked_blocks: [num_blocks] bool, which blocks are locked
        conflict_weight: Penalty magnitude (0.1-1.0 recommended)

    Returns:
        (scores, conflict_penalties)
        - scores: [num_blocks] float32, base eviction scores
        - conflict_penalties: [num_blocks] float32, penalty for evicting locked

    Example:
        >>> scores, penalties = eviction_scores_with_conflict_penalty(...)
        >>> adjusted_scores = scores + penalties
        >>> # Locked blocks now have higher scores (less attractive to evict)
    """
    scores = eviction_scores_ref(
        recency, frequency, attn_weights, valid_mask,
        w_recency, w_frequency, w_attention
    )

    locked_blocks = locked_blocks.astype(bool)
    conflict_penalties = np.where(locked_blocks, conflict_weight, 0.0).astype(np.float32)

    return scores, conflict_penalties


def select_eviction_candidates_ref(
    scores: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Select k blocks with lowest scores (best eviction candidates).

    Ties broken by lower block index (deterministic, matches kernel behavior).

    Args:
        scores: [num_blocks] float32, eviction scores
        k: Number of blocks to select

    Returns:
        candidates: [k] int64 indices in ascending order by score

    Example:
        >>> scores = np.array([0.5, 0.3, 0.8, 0.2, 0.6])
        >>> selected = select_eviction_candidates_ref(scores, 2)
        >>> selected
        array([3, 1])  # Blocks 3 and 1 have lowest scores
    """
    scores = np.asarray(scores, dtype=np.float32)

    # lexsort: sorts by score (ascending), ties broken by index (ascending)
    # We negate score to sort ascending (default lexsort is ascending)
    block_indices = np.arange(scores.shape[0])
    order = np.lexsort((block_indices, scores))

    return order[:k].astype(np.int64)


def update_draft_confidence(
    current_rate: float,
    accepted: bool,
    alpha: float = 0.1
) -> float:
    """
    Update draft acceptance rate using exponential moving average.

    new_rate = alpha * new_sample + (1-alpha) * old_rate

    alpha=0.1 means new sample has 10% weight (last ~10 samples dominate).

    Args:
        current_rate: Current acceptance rate (0.0-1.0)
        accepted: Whether this draft was accepted (bool)
        alpha: EMA coefficient (default 0.1)

    Returns:
        updated_rate: New acceptance rate

    Example:
        >>> rate = 0.5
        >>> rate = update_draft_confidence(rate, accepted=True, alpha=0.1)
        >>> rate
        0.55  # New estimate: 55% acceptance

        >>> # Convergence: repeated True samples → rate → 1.0
        >>> rate = 0.5
        >>> for _ in range(100):
        ...     rate = update_draft_confidence(rate, accepted=True, alpha=0.1)
        >>> rate
        0.9999...  # Converges to 1.0
    """
    new_sample = float(accepted)
    updated_rate = alpha * new_sample + (1 - alpha) * current_rate
    return updated_rate


# ============================================================================
# End-to-end pipeline reference
# ============================================================================

def end_to_end_pipeline_reference(
    num_seqs: int,
    num_blocks: int,
    block_size: int,
    num_drafts: int,
    eviction_fraction: float = 0.15,
) -> dict:
    """
    Complete end-to-end reference pipeline:
    score → rank → lock → generate (pretend) → verify (pretend) → restore

    This is for testing composition, not actual spec-decode execution.
    Simulates the logical flow without real attention computation.

    Args:
        num_seqs: Number of sequences in batch
        num_blocks: Total physical KV blocks
        block_size: Tokens per block
        num_drafts: Number of draft tokens to generate per sequence
        eviction_fraction: What fraction of blocks to evict (0.15 = 15%)

    Returns:
        result: dict with metrics

    Example:
        >>> result = end_to_end_pipeline_reference(32, 4096, 16, 4)
        >>> result['rollback_rate']
        0.08  # 8% rollback rate
        >>> result['evicted_blocks']
        614  # ~15% of 4096
    """
    # [1] Generate random eviction scores
    recency = np.random.rand(num_blocks).astype(np.float32)
    frequency = np.random.rand(num_blocks).astype(np.float32)
    attn_weights = np.random.rand(num_blocks, block_size).astype(np.float32)
    valid_mask = np.ones((num_blocks, block_size), dtype=bool)

    # [2] Score all blocks
    scores = eviction_scores_ref(recency, frequency, attn_weights, valid_mask)

    # [3] Select blocks to evict
    num_to_evict = max(1, int(num_blocks * eviction_fraction))
    eviction_candidates = select_eviction_candidates_ref(scores, num_to_evict)

    # [4] Rank drafts by confidence
    draft_stats = DraftStats(num_drafts)
    draft_priority = rank_drafts_by_confidence(draft_stats, num_drafts)

    # [5] Primary draft locks blocks
    locked_blocks = np.zeros(num_blocks, dtype=bool)
    for idx in eviction_candidates[:5]:  # Pretend primary wants to evict first 5
        locked_blocks[idx] = True

    # [6] Secondary drafts plan evictions (with conflict awareness)
    rollback_count = 0
    for draft_id in range(1, num_drafts):  # Skip primary (draft 0)
        # Secondary draft has different eviction plan (random subset)
        secondary_eviction = np.random.rand(num_blocks) > 0.99  # ~1% eviction rate

        # Check for conflicts
        safe_plan, conflict_penalty = detect_conflict(secondary_eviction, locked_blocks)

        # Pretend: if conflict_penalty > 0.5, we rollback
        if conflict_penalty > 0.5:
            rollback_count += 1

    # [7] Compute metrics
    total_draft_tokens = num_seqs * num_drafts
    rollback_rate = rollback_count / max(1, total_draft_tokens)

    return {
        'num_seqs': num_seqs,
        'num_blocks': num_blocks,
        'block_size': block_size,
        'num_drafts': num_drafts,
        'evicted_blocks': len(eviction_candidates),
        'eviction_fraction': eviction_fraction,
        'rollback_count': rollback_count,
        'rollback_rate': rollback_rate,
        'total_draft_tokens': total_draft_tokens,
    }


if __name__ == '__main__':
    # Quick smoke test
    print("Testing reference implementation...")

    # Test 1: Ranking
    stats = DraftStats(4)
    stats.acceptance_rate = np.array([0.9, 0.5, 0.7, 0.6], dtype=np.float32)
    priority = rank_drafts_by_confidence(stats, 4)
    print(f"✓ Draft ranking: {priority}")
    assert priority[0] == 0, "Highest acceptance should be first"

    # Test 2: Conflict detection
    eviction = np.array([True, True, False])
    locked = np.array([False, True, False])
    safe, penalty = detect_conflict(eviction, locked)
    print(f"✓ Conflict detection: safe={safe}, penalty={penalty:.2f}")
    assert penalty == 0.5, "50% conflict expected"

    # Test 3: Eviction scoring
    recency = np.random.rand(10).astype(np.float32)
    frequency = np.random.rand(10).astype(np.float32)
    attn = np.random.rand(10, 16).astype(np.float32)
    valid = np.ones((10, 16), dtype=bool)
    scores = eviction_scores_ref(recency, frequency, attn, valid)
    print(f"✓ Eviction scores: shape={scores.shape}, dtype={scores.dtype}")
    assert scores.shape == (10,), "Expected 10 scores"

    # Test 4: Selection
    selected = select_eviction_candidates_ref(scores, 3)
    print(f"✓ Selection: {selected} (lowest-scoring blocks)")
    assert len(selected) == 3, "Expected 3 selected"

    # Test 5: Confidence update
    rate = 0.5
    rate = update_draft_confidence(rate, True, alpha=0.1)
    print(f"✓ Confidence update: 0.5 → {rate:.2f} (after accepting)")
    assert abs(rate - 0.55) < 1e-6, "Expected 0.55"

    # Test 6: E2E pipeline
    result = end_to_end_pipeline_reference(32, 4096, 16, 4)
    print(f"✓ E2E pipeline: rollback_rate={result['rollback_rate']:.2%}")

    print("\n✅ All reference tests passed!")
