# Getting Started

Developer guide for the CUDA-Accelerated Speculative Decoding Optimizer.
For the project overview and verified results, see [README.md](README.md).

---

## Install & verify (2 minutes, no GPU required)

```bash
git clone https://github.com/ArchanaChetan07/CUDA-Accelerated-Speculative-Decoding-Optimizer-for-LLM-Inference-PyTorch-vLLM-.git
cd CUDA-Accelerated-Speculative-Decoding-Optimizer-for-LLM-Inference-PyTorch-vLLM-
pip install -e .
pytest tests/ -q                      # 29 passed, 2 GPU-gated skips
python benchmarks/bench_speculative.py
```

## GPU mode (CUDA GPU + nvcc)

The extension JIT-compiles on first import and is cached afterward. On Windows,
run from PowerShell or cmd — nvcc's host-compiler subprocess fails under
Git Bash/MSYS shells:

```powershell
$env:SPEC_DECODE_JIT_CUDA = "1"
pytest tests/ -q                      # 31 passed, 0 skipped — CUDA parity active
```

Permanent build instead of JIT: `PAGEDKV_FORCE_CUDA=1 pip install -e .`

Docker (CPU test suite): `docker build -t spec-decode . && docker run spec-decode`

---

## Codebase tour

| Path | What it is |
|---|---|
| `src/reference.py` | NumPy ground truth — the executable specification every kernel must match |
| `src/ops.py` | Backend dispatch: routes each op to the CUDA extension or the reference |
| `src/vllm_integration.py` | `OptimizedDraftManager` (rank / lock / plan / record cycle), `RollbackTracker` |
| `src/_jit.py` | Opt-in JIT compilation of the CUDA extension |
| `csrc/speculative_decode_kernel.cu` | 6 kernels: draft ranking, conflict detection, eviction scoring, soft lock/unlock, async restore |
| `csrc/bindings.cpp` | PyTorch pybind11 bindings |
| `tests/` | 31 tests — correctness (18), dispatch parity (8), integration (5) |
| `docs/ARCHITECTURE.md` | Component map, data flow, kernel notes, invariants |

## Development workflow

1. Change behavior? **Change `src/reference.py` first** — it is the spec.
2. Make the CUDA kernel match; `tests/test_ops_dispatch.py` parity tests enforce it.
3. Numbers that go in the README come from `benchmarks/` runs committed to `results/`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guidelines.

## Current state & next milestones

**Done:** reference implementation, 6 CUDA kernels (bit-exact vs reference on
NVIDIA T1000), dispatch layer, vLLM shim, benchmarks, CI.

**Next:** vLLM fork integration at the proposer, end-to-end A/B validation of the
rollback/latency targets, multi-GPU draft coordination.
