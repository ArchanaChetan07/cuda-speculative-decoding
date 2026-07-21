"""
Benchmark suite for Speculative Decoding Optimizer.

Measures reference (NumPy) performance and, when CUDA is available,
compares against the compiled kernel. Writes results to results/*.json.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import reference
from src import ops


def _time_op(fn, iters: int = 100) -> float:
    """Return average milliseconds per call."""
    # Warmup
    for _ in range(5):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


def bench_draft_ranking(num_drafts: int = 8, iters: int = 1000) -> dict:
    stats = reference.DraftStats(num_drafts)
    stats.acceptance_rate = np.random.rand(num_drafts).astype(np.float32)
    stats.total_generated = np.random.randint(0, 1000, num_drafts).astype(np.int64)

    ms = _time_op(lambda: reference.rank_drafts_by_confidence(stats, num_drafts), iters)
    return {'op': 'draft_ranking', 'num_drafts': num_drafts, 'ms_per_call': ms}


def bench_conflict_detection(num_blocks: int = 1024, iters: int = 1000) -> dict:
    eviction = np.random.rand(num_blocks) > 0.5
    locked = np.random.rand(num_blocks) > 0.7

    ms = _time_op(lambda: reference.detect_conflict(eviction, locked), iters)
    return {'op': 'conflict_detection', 'num_blocks': num_blocks, 'ms_per_call': ms}


def bench_eviction_scores(num_blocks: int = 256, seq_len: int = 2048, iters: int = 100) -> dict:
    recency = np.random.rand(num_blocks).astype(np.float32)
    frequency = np.random.rand(num_blocks).astype(np.float32)
    attn = np.random.rand(num_blocks, seq_len).astype(np.float32)
    mask = np.random.rand(num_blocks, seq_len) > 0.1

    ms = _time_op(
        lambda: reference.eviction_scores_ref(recency, frequency, attn, mask),
        iters
    )
    return {
        'op': 'eviction_scores',
        'num_blocks': num_blocks,
        'seq_len': seq_len,
        'ms_per_call': ms,
    }


def main():
    print("=" * 60)
    print("Speculative Decoding Optimizer - Benchmark Suite")
    print("=" * 60)

    status = ops.backend_status()
    print(f"Backend: {status['active_backend']}")
    print(f"  torch: {status['has_torch']}, cuda: {status['has_cuda']}")
    print()

    results = {
        'backend': status['active_backend'],
        'benchmarks': [],
    }

    for bench_fn in [bench_draft_ranking, bench_conflict_detection, bench_eviction_scores]:
        result = bench_fn()
        results['benchmarks'].append(result)
        print(f"{result['op']:25s}: {result['ms_per_call']:.4f} ms/call")

    # Write results
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'benchmark_reference.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
