"""
Dispatch layer for Speculative Decoding Optimizer.

Routes operations to CUDA kernels when available, falling back to
NumPy reference implementations otherwise. Guarantees identical results
across backends (within floating-point precision).
"""

import numpy as np
from typing import Tuple, Optional

from . import reference

# Backend detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from . import _C  # Compiled CUDA extension
    HAS_CUDA = _C is not None
except ImportError:
    HAS_CUDA = False
    _C = None

if not HAS_CUDA:
    # Opt-in JIT compile: SPEC_DECODE_JIT_CUDA=1 (needs GPU + nvcc)
    from ._jit import load_extension
    _C = load_extension("spec_decode_opt_C", "SPEC_DECODE_JIT_CUDA")
    HAS_CUDA = _C is not None


def _use_cuda(force: Optional[bool] = None) -> bool:
    """Decide whether to use CUDA backend."""
    if force is not None:
        return force and HAS_CUDA
    return HAS_CUDA and HAS_TORCH


def rank_drafts_by_confidence(
    draft_stats: reference.DraftStats,
    num_drafts: int,
    use_cuda: Optional[bool] = None,
) -> np.ndarray:
    """Rank drafts by confidence. Dispatches to CUDA or reference.

    Args:
        draft_stats: Draft statistics
        num_drafts: Number of drafts to rank
        use_cuda: Force backend (None = auto-detect)

    Returns:
        Priority indices (int32)
    """
    if _use_cuda(use_cuda):
        acceptance = torch.from_numpy(
            draft_stats.acceptance_rate[:num_drafts].copy()
        ).cuda()
        generated = torch.from_numpy(
            draft_stats.total_generated[:num_drafts].astype(np.int32).copy()
        ).cuda()
        priority = _C.rank_drafts(acceptance, generated)
        return priority.cpu().numpy()

    return reference.rank_drafts_by_confidence(draft_stats, num_drafts)


def detect_conflict(
    eviction_plan: np.ndarray,
    locked_blocks: np.ndarray,
    is_primary: bool = False,
    use_cuda: Optional[bool] = None,
) -> Tuple[np.ndarray, float]:
    """Detect conflicts between eviction plan and locked blocks.

    Args:
        eviction_plan: Boolean/bitmask array of blocks to evict
        locked_blocks: Boolean/bitmask array of locked blocks
        is_primary: Whether this is the primary draft
        use_cuda: Force backend (None = auto-detect)

    Returns:
        (safe_eviction, conflict_penalty)
    """
    if _use_cuda(use_cuda):
        # Pack booleans into uint32 bitmasks for CUDA
        plan_packed = _pack_bitmask(eviction_plan)
        locked_packed = _pack_bitmask(locked_blocks)

        plan_t = torch.from_numpy(plan_packed.astype(np.int32)).cuda()
        locked_t = torch.from_numpy(locked_packed.astype(np.int32)).cuda()

        safe_packed, penalty = _C.detect_conflict(plan_t, locked_t)
        safe = _unpack_bitmask(safe_packed.cpu().numpy(), len(eviction_plan))
        return safe.astype(bool), float(penalty.cpu().item())

    return reference.detect_conflict(eviction_plan, locked_blocks, is_primary)


def eviction_scores(
    recency: np.ndarray,
    frequency: np.ndarray,
    attn_weights: np.ndarray,
    valid_mask: np.ndarray,
    w_recency: float = 0.4,
    w_frequency: float = 0.3,
    w_attention: float = 0.3,
    use_cuda: Optional[bool] = None,
) -> np.ndarray:
    """Compute eviction scores for KV blocks.

    Args:
        recency: Recency scores per block
        frequency: Frequency scores per block
        attn_weights: Attention weights [num_blocks, seq_len]
        valid_mask: Valid entry mask [num_blocks, seq_len]
        w_recency, w_frequency, w_attention: Weights
        use_cuda: Force backend (None = auto-detect)

    Returns:
        Eviction scores per block (float32)
    """
    if _use_cuda(use_cuda):
        recency_t = torch.from_numpy(recency.astype(np.float32)).cuda()
        frequency_t = torch.from_numpy(frequency.astype(np.float32)).cuda()
        attn_t = torch.from_numpy(attn_weights.astype(np.float32)).cuda()
        mask_t = torch.from_numpy(valid_mask.astype(bool)).cuda()

        scores = _C.eviction_scores(
            recency_t, frequency_t, attn_t, mask_t,
            w_recency, w_frequency, w_attention
        )
        return scores.cpu().numpy()

    return reference.eviction_scores_ref(
        recency, frequency, attn_weights, valid_mask,
        w_recency, w_frequency, w_attention
    )


def _pack_bitmask(bool_array: np.ndarray) -> np.ndarray:
    """Pack boolean array into uint32 bitmask."""
    n = len(bool_array)
    num_words = (n + 31) // 32
    packed = np.zeros(num_words, dtype=np.uint32)
    for i in range(n):
        if bool_array[i]:
            packed[i // 32] |= (1 << (i % 32))
    return packed


def _unpack_bitmask(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack uint32 bitmask into boolean array."""
    result = np.zeros(n, dtype=bool)
    packed_u = packed.astype(np.uint32)
    for i in range(n):
        result[i] = bool(packed_u[i // 32] & (1 << (i % 32)))
    return result


def backend_status() -> dict:
    """Return current backend availability."""
    return {
        'has_torch': HAS_TORCH,
        'has_cuda': HAS_CUDA,
        'active_backend': 'cuda' if _use_cuda() else 'reference',
    }
