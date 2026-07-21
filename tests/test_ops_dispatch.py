"""
Dispatch-layer tests: verify ops routes correctly and backends agree.

These run on any machine; CUDA-vs-reference parity checks are skipped
when no compiled extension is present.
"""

import numpy as np
import pytest

from src import reference
from src import ops


class TestDispatchReference:
    """Ops layer must work with reference backend forced."""

    def test_rank_drafts_dispatch(self):
        stats = reference.DraftStats(4)
        stats.acceptance_rate = np.array([0.9, 0.5, 0.7, 0.6], dtype=np.float32)

        result = ops.rank_drafts_by_confidence(stats, 4, use_cuda=False)
        expected = reference.rank_drafts_by_confidence(stats, 4)
        np.testing.assert_array_equal(result, expected)

    def test_detect_conflict_dispatch(self):
        eviction = np.array([True, True, False, True])
        locked = np.array([True, False, False, False])

        safe, penalty = ops.detect_conflict(eviction, locked, use_cuda=False)
        safe_ref, penalty_ref = reference.detect_conflict(eviction, locked)

        np.testing.assert_array_equal(safe, safe_ref)
        assert penalty == pytest.approx(penalty_ref)

    def test_eviction_scores_dispatch(self):
        np.random.seed(7)
        num_blocks, seq_len = 16, 128
        recency = np.random.rand(num_blocks).astype(np.float32)
        frequency = np.random.rand(num_blocks).astype(np.float32)
        attn = np.random.rand(num_blocks, seq_len).astype(np.float32)
        mask = np.random.rand(num_blocks, seq_len) > 0.2

        scores = ops.eviction_scores(recency, frequency, attn, mask, use_cuda=False)
        scores_ref = reference.eviction_scores_ref(recency, frequency, attn, mask)
        np.testing.assert_allclose(scores, scores_ref, rtol=1e-6)

    def test_backend_status(self):
        status = ops.backend_status()
        assert 'active_backend' in status
        assert status['active_backend'] in ('cuda', 'reference')


class TestBitmaskPacking:
    """Bitmask helpers must round-trip exactly."""

    def test_pack_unpack_roundtrip(self):
        np.random.seed(11)
        for n in [1, 31, 32, 33, 64, 100]:
            original = np.random.rand(n) > 0.5
            packed = ops._pack_bitmask(original)
            unpacked = ops._unpack_bitmask(packed, n)
            np.testing.assert_array_equal(original, unpacked)


@pytest.mark.skipif(not ops.HAS_CUDA, reason="compiled CUDA extension not available")
class TestCUDAParity:
    """CUDA kernels must match reference exactly (when compiled)."""

    def test_conflict_parity(self):
        np.random.seed(3)
        eviction = np.random.rand(256) > 0.5
        locked = np.random.rand(256) > 0.7

        safe_cuda, pen_cuda = ops.detect_conflict(eviction, locked, use_cuda=True)
        safe_ref, pen_ref = reference.detect_conflict(eviction, locked)

        np.testing.assert_array_equal(safe_cuda, safe_ref)
        assert pen_cuda == pytest.approx(pen_ref, abs=1e-5)

    def test_eviction_scores_parity(self):
        np.random.seed(5)
        num_blocks, seq_len = 64, 512
        recency = np.random.rand(num_blocks).astype(np.float32)
        frequency = np.random.rand(num_blocks).astype(np.float32)
        attn = np.random.rand(num_blocks, seq_len).astype(np.float32)
        mask = np.random.rand(num_blocks, seq_len) > 0.1

        scores_cuda = ops.eviction_scores(recency, frequency, attn, mask, use_cuda=True)
        scores_ref = reference.eviction_scores_ref(recency, frequency, attn, mask)
        np.testing.assert_allclose(scores_cuda, scores_ref, rtol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
