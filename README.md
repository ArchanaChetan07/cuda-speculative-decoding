# CUDA-Accelerated Speculative Decoding Optimizer for LLM Inference

[![Tests](https://img.shields.io/badge/tests-31%20passing-brightgreen)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()

Custom CUDA kernels and scheduling logic that eliminate draft-token conflicts in
speculative decoding for LLM serving. Deterministic draft ranking, soft-lock KV-cache
conflict resolution, and asynchronous cache restoration — designed as a drop-in
optimization layer for [vLLM](https://github.com/vllm-project/vllm)-style engines.

---

## Verified Results

All numbers below are measured in this repository and reproducible with the commands shown.

| Check | Result | Hardware |
|---|---|---|
| Test suite (GPU mode, `SPEC_DECODE_JIT_CUDA=1`) | **31 / 31 passing, 0 skipped** | NVIDIA T1000, CUDA 12.5 |
| Test suite (CPU-only mode) | 29 passing, 2 GPU-gated skips | any machine, no GPU needed |
| CUDA `rank_drafts` kernel vs NumPy reference | **exact match** (incl. tie-breaking) | T1000 |
| CUDA `detect_conflict` kernel vs NumPy reference | **exact match** (bitmask + penalty) | T1000 |
| CUDA `eviction_scores` kernel vs NumPy reference | **exact match** (rtol 1e-4), verified at 16 and 64 blocks | T1000 |
| Draft ranking (reference impl) | 0.0017 ms / call | CPU |
| Conflict detection, 1024 blocks (reference impl) | 0.0047 ms / call | CPU |
| Eviction scoring, 256 blocks × 2048 seq (reference impl) | 0.84 ms / call | CPU |

**Design targets** (pending end-to-end vLLM A/B validation): rollback rate 23% → 8%,
P99 latency 187 ms → 164 ms, throughput +18–25% on multi-draft speculative workloads.

---

## The Problem

Speculative decoding generates draft tokens with a small model and verifies them in
parallel with the target model. In multi-draft and multi-tenant serving, drafts compete
for the same paged KV-cache blocks:

- **Draft conflicts** — concurrent drafts evict each other's KV blocks, forcing rollbacks
- **Rollback overhead** — each rollback costs 8–12 ms of cache invalidation and re-decode
- **Non-determinism** — unordered draft scheduling makes tail latency unpredictable

## The Solution

Three composable mechanisms, each implemented twice — a NumPy ground-truth reference
and a CUDA kernel verified bit-exact against it:

1. **Deterministic draft ranking** — drafts ordered by exponential-moving-average
   acceptance confidence; ties broken by activity, then index. Identical stats always
   produce identical schedules, on both backends.
2. **Soft-lock conflict resolution** — the primary draft locks its KV blocks in a GPU
   bitmask (`atomicOr`/`atomicAnd`); lower-priority drafts get their eviction plans
   *filtered* (`plan & ~locked`) instead of triggering rollbacks. A conflict-penalty
   signal lets the scheduler defer a heavily-conflicting draft for one step.
3. **Async KV restoration** — suspended draft contexts restore on a background CUDA
   stream; the decode stream never waits.

```
┌────────────────────────────────────────────────────────┐
│                   vLLM Engine Loop                     │
│   (propose  →  verify  →  commit / rollback)           │
└──────────────────────┬─────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────┐
│   vllm_integration.py — OptimizedDraftManager          │
│   rank_for_step() · lock/unlock · plan_eviction()      │
└──────────────────────┬─────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────┐
│   ops.py — dispatch (CUDA ext ⇄ NumPy reference)       │
└──────────┬──────────────────────────────┬──────────────┘
           │                              │
┌──────────▼──────────┐      ┌────────────▼─────────────┐
│  reference.py       │ must │  csrc/*.cu + bindings    │
│  NumPy ground truth │◄────►│  6 CUDA kernels          │
└─────────────────────┘ match└──────────────────────────┘
```

---

## Quick Start

### CPU-only (no GPU required)

```bash
git clone https://github.com/ArchanaChetan07/CUDA-Accelerated-Speculative-Decoding-Optimizer-for-LLM-Inference-PyTorch-vLLM-.git
cd CUDA-Accelerated-Speculative-Decoding-Optimizer-for-LLM-Inference-PyTorch-vLLM-
pip install -e .
pytest tests/ -q            # 29 passed, 2 skipped
python benchmarks/bench_speculative.py
```

### GPU mode (CUDA GPU + nvcc + PyTorch with CUDA)

The extension JIT-compiles on import and is cached afterward. On Windows, run from
PowerShell or cmd (nvcc's host-compiler subprocess fails under Git Bash/MSYS):

```powershell
$env:SPEC_DECODE_JIT_CUDA = "1"
pytest tests/ -q            # 31 passed, 0 skipped — CUDA parity tests active
```

Or build the extension permanently:

```bash
PAGEDKV_FORCE_CUDA=1 pip install -e .
```

### Usage

```python
from src.vllm_integration import OptimizedDraftManager

mgr = OptimizedDraftManager(num_drafts=4, num_blocks=1024)

order = mgr.rank_for_step()               # deterministic proposal order
mgr.lock_blocks(primary_blocks)           # primary draft claims its KV blocks
safe, penalty = mgr.plan_eviction(plan)   # conflict-filtered eviction for others
mgr.record_outcome(draft_id, accepted=True)  # EMA confidence feedback
mgr.unlock_all()
```

---

## Repository Structure

```
├── src/
│   ├── reference.py           NumPy ground truth (every kernel must match this)
│   ├── ops.py                 backend dispatch + bitmask packing
│   ├── vllm_integration.py    OptimizedDraftManager, RollbackTracker
│   └── _jit.py                opt-in JIT compile of the CUDA extension
├── csrc/
│   ├── speculative_decode_kernel.cu   6 kernels: rank, conflict, eviction
│   │                                  scores, soft lock/unlock, async restore
│   └── bindings.cpp                   PyTorch pybind11 bindings
├── tests/                     31 tests: correctness, dispatch parity, integration
├── benchmarks/                micro-benchmarks (results committed in results/)
├── docs/ARCHITECTURE.md       component map, data flow, kernel notes, invariants
└── results/                   benchmark JSON artifacts
```

## Testing Philosophy

Every CUDA kernel has a NumPy twin that serves as executable specification.
The test pyramid:

1. **Correctness** (`test_correctness.py`) — 18 tests against the reference:
   ranking determinism, tie-breaking, conflict filtering, eviction scoring,
   full-pipeline composition
2. **Dispatch parity** (`test_ops_dispatch.py`) — bitmask round-trips; CUDA
   output must match the reference exactly when the extension is present
3. **Integration** (`test_vllm_integration.py`) — full step cycle
   (rank → lock → plan → record → unlock), confidence-driven re-ranking,
   rollback-rate tracking

## Roadmap

- [x] NumPy reference implementation + full test suite
- [x] CUDA kernels, bit-exact vs reference on real hardware
- [x] JIT build path + packaging
- [ ] vLLM fork integration (attach `OptimizedDraftManager` at the proposer)
- [ ] End-to-end A/B validation of rollback/latency/throughput targets
- [ ] Multi-GPU draft coordination

## License

Apache-2.0
