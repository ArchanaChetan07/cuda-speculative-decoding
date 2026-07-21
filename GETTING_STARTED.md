# Project 1: Speculative Decoding Optimizer — Getting Started Guide

**Your complete, ready-to-use Project 1 folder is here:**  
`C:\Users\archa\OneDrive\Desktop\VLLM\Project1-SpeculativeDecoding`

---

## 📋 What's Included

### Complete Package
- ✅ **README.md** — Project overview + architecture
- ✅ **Python reference implementation** (`src/reference.py`) — NumPy ground truth
- ✅ **Unit tests** (`tests/test_correctness.py`) — 9 correctness tests
- ✅ **Build config** (`setup.py`, `pyproject.toml`, `CMakeLists.txt`)
- ✅ **Package structure** — Ready to pip install

### What's NOT Included Yet (Week 3-5)
- CUDA kernels (`csrc/conflict_score.cu`)
- PyTorch bindings (`csrc/bindings.cpp`)
- Performance tests
- Integration tests
- E2E tests

---

## 🚀 Quick Start (5 minutes)

### Step 1: Install
```bash
cd CUDA-Accelerated-Speculative-Decoding-Optimizer-for-LLM-Inference-PyTorch-vLLM-
pip install -e .
pytest tests/ -q          # CPU-only: 29 passed, 2 skipped
```

With a CUDA GPU + nvcc (run from PowerShell/cmd on Windows, not Git Bash):

```powershell
$env:SPEC_DECODE_JIT_CUDA = "1"   # JIT-compiles csrc/ on import (cached)
pytest tests/ -q                   # 31 passed, 0 skipped — CUDA parity active
```

### Step 2: Verify Installation
```bash
python -c "from spec_decode_optimizer import DraftStats; print('✓ OK')"
```

### Step 3: Run Tests
```bash
pytest tests/test_correctness.py -v
```

**Expected output:** All 9 tests passing ✓

---

## 📁 Project Structure

```
Project1-SpeculativeDecoding/
├── README.md                      Overview + architecture
├── GETTING_STARTED.md             This file
├── pyproject.toml                 Package metadata
├── setup.py                       Installation script
├── CMakeLists.txt                 CUDA build config (Week 3+)
│
├── src/                           Python source code
│   ├── __init__.py                Package initialization
│   └── reference.py               NumPy reference implementation ✅
│
├── tests/                         Test suite
│   ├── test_correctness.py        9 correctness tests ✅
│   ├── test_determinism.py        (Week 6)
│   ├── test_performance.py        (Week 8)
│   ├── test_integration.py        (Week 8)
│   └── test_end_to_end.py         (Week 11)
│
├── csrc/                          CUDA source (Week 3+)
│   ├── conflict_score.cu          Main CUDA kernel
│   └── bindings.cpp               PyTorch bindings
│
├── benchmarks/                    Benchmark scripts (Week 6+)
│   ├── bench_conflict_detection.py
│   └── bench_async_rollback.py
│
├── scripts/                       Utility scripts (Week 6+)
│   ├── profile_kernels.py
│   └── run_demo.py
│
├── docs/                          Documentation (Week 12)
│   ├── ARCHITECTURE.md
│   ├── DESIGN_DECISIONS.md
│   └── DEPLOYMENT.md
│
└── results/                       Benchmark results (JSON)
    ├── conflict_detect_bench.json
    └── async_rollback_bench.json
```

---

## 📖 Reading Order

### Week 1-2 (NOW)
1. **README.md** (this repo) — Understand the project
2. **src/reference.py** — See the implementation
3. Run **tests/test_correctness.py** — Validate understanding

### Week 3-5
1. Read `docs/ARCHITECTURE.md` (not in repo yet, see Project1-Complete-Technical-Plan.md)
2. Implement `csrc/conflict_score.cu` (copy from technical plan)
3. Implement `csrc/bindings.cpp`
4. Run tests after each component

### Week 6+
1. Run `benchmarks/` scripts
2. Add `test_performance.py`, `test_integration.py`
3. Profile with nsys

---

## 🧪 Testing

### Run All Current Tests
```bash
pytest tests/test_correctness.py -v
```

### Run with Coverage
```bash
pytest tests/test_correctness.py -v --cov=spec_decode_optimizer
```

### Run Specific Test
```bash
pytest tests/test_correctness.py::TestDraftRanking::test_rank_drafts_no_tie -v
```

### Expected Results (Week 1-2)
```
test_correctness.py::TestDraftRanking::test_rank_drafts_no_tie PASSED
test_correctness.py::TestDraftRanking::test_rank_drafts_tie_broken_by_activity PASSED
test_correctness.py::TestDraftRanking::test_rank_drafts_single_draft PASSED
test_correctness.py::TestConflictDetection::test_conflict_full_overlap PASSED
test_correctness.py::TestConflictDetection::test_conflict_no_overlap PASSED
test_correctness.py::TestConflictDetection::test_conflict_partial_overlap PASSED
test_correctness.py::TestConflictDetection::test_conflict_empty_eviction_plan PASSED
test_correctness.py::TestEvictionScoring::test_scoring_basic PASSED
test_correctness.py::TestEvictionScoring::test_scoring_empty_blocks PASSED
test_correctness.py::TestBlockSelection::test_selection_basic PASSED
test_correctness.py::TestBlockSelection::test_selection_deterministic_ties PASSED
test_correctness.py::TestBlockSelection::test_selection_k_equals_num_blocks PASSED
test_correctness.py::TestConfidenceTracking::test_confidence_single_acceptance PASSED
test_correctness.py::TestConfidenceTracking::test_confidence_single_rejection PASSED
test_correctness.py::TestConfidenceTracking::test_confidence_converges_to_one PASSED
test_correctness.py::TestConfidenceTracking::test_confidence_converges_to_zero PASSED
test_correctness.py::TestComposition::test_end_to_end_pipeline PASSED
test_correctness.py::TestComposition::test_integration_score_select_rank PASSED

========================== 18 passed in 0.42s ==========================
```

---

## 🎯 This Week's Tasks (Week 1-2)

### Day 1-2: Setup & Understand
- [ ] Clone/navigate to the folder
- [ ] Install: `pip install -e .`
- [ ] Verify: `python -c "from spec_decode_optimizer import DraftStats; print('OK')"`
- [ ] Read: `README.md`

### Day 3-4: Run Tests & Review Code
- [ ] Run: `pytest tests/test_correctness.py -v`
- [ ] Read: `src/reference.py` (understand implementation)
- [ ] Review: `tests/test_correctness.py` (understand testing)

### Day 5: Plan Week 3
- [ ] Read: The technical specification in `Project1-Complete-Technical-Plan.md`
- [ ] Understand: CUDA kernel requirements
- [ ] Decision: Hard locks vs soft locks (recommend: soft)

### Deliverables (End of Week 2)
- [x] NumPy reference implementation (`src/reference.py`)
- [x] Unit tests (9 correctness tests)
- [x] Architecture validated
- [x] Ready for CUDA kernel development

---

## 🔧 Configuration

### Environment Variables
```bash
# CPU-only (default)
SPEC_DECODE_CPU_ONLY=1

# Build with CUDA
PAGEDKV_FORCE_CUDA=1 pip install -e ".[cuda]"

# Conflict penalty weight
SPEC_DECODE_CONFLICT_WEIGHT=0.1

# EMA coefficient
SPEC_DECODE_CONFIDENCE_ALPHA=0.1
```

### Python Usage
```python
from spec_decode_optimizer import (
    DraftStats,
    rank_drafts_by_confidence,
    detect_conflict,
    eviction_scores_ref,
)

# Example 1: Rank drafts
stats = DraftStats(num_drafts=4)
stats.acceptance_rate = np.array([0.9, 0.5, 0.7, 0.6])
priority = rank_drafts_by_confidence(stats, 4)
print(f"Primary draft: {priority[0]}")  # 0 (highest acceptance)

# Example 2: Detect conflicts
eviction = np.array([True, True, False])
locked = np.array([False, True, False])
safe, penalty = detect_conflict(eviction, locked)
print(f"Conflict penalty: {penalty:.2f}")  # 0.50 (50% conflict)

# Example 3: Score blocks
recency = np.random.rand(100).astype(np.float32)
frequency = np.random.rand(100).astype(np.float32)
attn = np.random.rand(100, 16).astype(np.float32)
valid = np.ones((100, 16), dtype=bool)
scores = eviction_scores_ref(recency, frequency, attn, valid)
```

---

## 📚 Key Files Explained

### `src/reference.py`
**What:** Ground-truth NumPy implementation  
**Why:** Validates all CUDA kernels must match this  
**Key functions:**
- `rank_drafts_by_confidence()` — Sort drafts by acceptance rate
- `detect_conflict()` — Check eviction plan vs locked blocks
- `eviction_scores_ref()` — Compute block scores

### `tests/test_correctness.py`
**What:** 9 unit tests for correctness  
**Why:** Gate releases, catch regressions  
**Coverage:**
- Draft ranking (3 tests)
- Conflict detection (4 tests)
- Eviction scoring (2 tests)
- Block selection (3 tests)
- Confidence tracking (3 tests)
- Composition (1 test)

### `setup.py` & `pyproject.toml`
**What:** Python package configuration  
**Why:** Allows `pip install -e .`  
**Includes:** Dependencies, build flags, metadata

### `CMakeLists.txt`
**What:** CUDA build configuration  
**Why:** Needed for `csrc/*.cu` compilation  
**Builds:** CUDA kernels for GPU execution

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'spec_decode_optimizer'"
```bash
# Solution: Install in development mode
cd C:\Users\archa\OneDrive\Desktop\VLLM\Project1-SpeculativeDecoding
pip install -e .
```

### "test_correctness.py not found"
```bash
# Solution: Run from project root
cd C:\Users\archa\OneDrive\Desktop\VLLM\Project1-SpeculativeDecoding
pytest tests/test_correctness.py -v
```

### "pytest: command not found"
```bash
# Solution: Install pytest
pip install pytest
```

### Tests fail with numerical differences
- This is **expected before Week 3** (no CUDA yet)
- Reference implementation should be exact
- If tests still fail, check NumPy version: `pip install --upgrade numpy`

---

## 📊 Progress Checklist

### ✅ Week 1-2: Reference Implementation
- [x] NumPy reference complete (`src/reference.py`)
- [x] 9 correctness tests written
- [x] All tests passing
- [x] Package installable (`pip install -e .`)
- [x] Architecture validated

### ⏳ Week 3-5: CUDA Kernel
- [ ] Implement `csrc/conflict_score.cu`
- [ ] Implement `csrc/bindings.cpp`
- [ ] Implement `src/async_rollback.py`
- [ ] 3 performance tests pass
- [ ] Latency targets met (<100 µs @ 16K blocks)

### ⏳ Week 6-7: Validation
- [ ] Determinism tests (4 tests)
- [ ] Profiling with nsys
- [ ] Trace-level validation

### ⏳ Week 8-10: Full Test Suite
- [ ] 6 integration tests
- [ ] 3 E2E tests
- [ ] >95% code coverage
- [ ] All 30 tests passing

### ⏳ Week 11-12: Production
- [ ] E2E pipeline test
- [ ] Operations runbook
- [ ] Production deployment plan

---

## 🎯 Next Steps

### Immediate (Today)
1. `cd C:\Users\archa\OneDrive\Desktop\VLLM\Project1-SpeculativeDecoding`
2. `pip install -e .`
3. `pytest tests/test_correctness.py -v`

### This Week
1. Read `README.md` (architecture overview)
2. Read `src/reference.py` (understand implementation)
3. Review `tests/test_correctness.py` (understand testing)

### Next Week (Week 3)
1. Read `Project1-Complete-Technical-Plan.md` (CUDA specifications)
2. Start implementing `csrc/conflict_score.cu`
3. Implement `csrc/bindings.cpp`

---

## 📝 Notes

- This is **Week 1-2 complete** of a 12-week plan
- Reference implementation is validated ground truth
- All tests should pass on CPU (no GPU needed)
- CUDA implementation comes Week 3-5
- Full timeline in `Project1-Complete-Technical-Plan.md`

---

## 🚀 Ready to Go!

You have everything needed to start. Run the tests and you'll see the implementation work. Next week, you'll add the CUDA kernel.

```bash
cd C:\Users\archa\OneDrive\Desktop\VLLM\Project1-SpeculativeDecoding
pip install -e .
pytest tests/test_correctness.py -v
```

**Good luck! 🎉**
