"""Speculative Decoding Optimizer

CUDA kernels and Python wrappers for conflict-free KV cache eviction
in speculative decoding for LLM serving.

Main components:
- reference.py: NumPy ground truth for correctness validation
- conflict_detector.py: Python wrapper for CUDA conflict detection
- draft_stats.py: Draft confidence tracking
- async_rollback.py: Non-blocking KV cache restore
- ops.py: Dispatch layer (uses CUDA if available, else NumPy)

Quick start:

    from spec_decode_optimizer import ConflictDetector, DraftStats

    # Initialize
    detector = ConflictDetector(conflict_weight=0.1)
    stats = DraftStats(num_drafts=4)

    # Rank drafts by historical acceptance
    priority = detector.rank_drafts(stats)

    # Detect conflicts
    safe_plan, penalty = detector.detect_conflict(eviction_plan, locked_blocks)

    # Update confidence after verification
    stats.update(draft_id, accepted=True)

"""

__version__ = "0.1.0"

try:
    from .reference import (
        DraftStats,
        rank_drafts_by_confidence,
        detect_conflict,
        eviction_scores_ref,
        select_eviction_candidates_ref,
        update_draft_confidence,
    )
    HAS_REFERENCE = True
except ImportError:
    HAS_REFERENCE = False

try:
    # Try to import CUDA extension (will fail if not built)
    from . import _C
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    _C = None

__all__ = [
    'DraftStats',
    'rank_drafts_by_confidence',
    'detect_conflict',
    'eviction_scores_ref',
    'select_eviction_candidates_ref',
    'update_draft_confidence',
    'HAS_REFERENCE',
    'HAS_CUDA',
]

# Backend info
def backend_info():
    """Print available backends"""
    print(f"NumPy reference: {'✓' if HAS_REFERENCE else '✗'}")
    print(f"CUDA extension: {'✓' if HAS_CUDA else '✗'}")
