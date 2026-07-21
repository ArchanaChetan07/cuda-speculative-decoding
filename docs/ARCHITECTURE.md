# Architecture: Speculative Decoding Optimizer

## Component Map

```
┌──────────────────────────────────────────────────────────┐
│                     vLLM Engine Loop                     │
│  (proposal step -> verification step -> commit/rollback) │
└───────────────┬──────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────┐
│           src/vllm_integration.py                        │
│  OptimizedDraftManager     RollbackTracker               │
│  - rank_for_step()         - rollback_rate()             │
│  - lock/unlock blocks                                    │
│  - plan_eviction()                                       │
└───────────────┬──────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────┐
│                  src/ops.py (dispatch)                   │
│  Routes each call to CUDA ext (_C) or NumPy reference    │
└──────┬──────────────────────────────────┬────────────────┘
       │                                  │
┌──────▼────────────┐         ┌───────────▼───────────────┐
│ src/reference.py  │         │ csrc/*.cu + bindings.cpp  │
│ NumPy ground      │ must    │ - rank_drafts kernel      │
│ truth             │ match   │ - detect_conflict kernel  │
│                   │ ◄─────► │ - eviction_scores kernel  │
│                   │         │ - soft_lock / unlock      │
│                   │         │ - async_kv_restore        │
└───────────────────┘         └───────────────────────────┘
```

## Data Flow (one speculative step)

1. **Rank** — `rank_for_step()` orders drafts by EMA acceptance confidence.
   Deterministic: ties broken by activity (total_generated), then index.
2. **Lock** — the primary draft soft-locks the KV blocks it needs.
   Locks are a bitmask (`uint32` words on GPU, `bool` array on CPU).
3. **Plan** — each secondary draft's eviction plan is filtered against
   locks (`plan_eviction`). Conflicting evictions are dropped, not rolled
   back; the returned penalty lets the scheduler defer the draft instead.
4. **Verify** — vLLM's verification runs unchanged.
5. **Record** — accepted/rejected outcomes update EMA confidence
   (`record_outcome`, alpha = 0.1).
6. **Unlock** — all soft locks released before the next step.

## Key invariants

- **Determinism**: identical stats → identical ranking, on both backends.
- **Safety**: `safe_eviction = plan & ~locked` — a locked block is never
  evicted by a lower-priority draft, so the primary path never rolls back
  due to a secondary draft.
- **Non-blocking**: restoration of suspended drafts runs on a background
  CUDA stream (`async_kv_restore_kernel`); the decode stream never waits.

## CUDA kernel notes

| Kernel | Grid/Block | Shared mem | Notes |
|--------|-----------|------------|-------|
| detect_kv_conflict | 1D over words | threads × f32 | popc + tree reduction |
| eviction_scores | (num_blocks) × 256 | 2 arrays | masked mean of attention |
| soft_lock/unlock | 1D over blocks | — | atomicOr / atomicAnd on bitmask |
| async_kv_restore | 1D over elements | — | plain copy, separate stream |

## Failure modes and fallbacks

- No compiled extension → `ops` silently uses the NumPy reference
  (correct, slower). `ops.backend_status()` reports which is active.
- Penalty ≥ 1.0 (all desired evictions conflict) → scheduler should
  defer the draft one step rather than shrinking its window.
