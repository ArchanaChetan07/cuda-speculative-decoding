/*
 * Speculative Decoding Optimizer: CUDA Kernels
 *
 * Implements:
 * - Deterministic draft ranking via confidence scoring
 * - Soft locking mechanism for KV block conflict resolution
 * - Async KV restoration via CUDA background streams
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

/*
 * rank_drafts_confidence_kernel
 *
 * Ranks draft tokens by acceptance confidence (EMA-based).
 * Input: acceptance_rate[num_drafts], total_generated[num_drafts]
 * Output: priority[num_drafts] (indices sorted by confidence DESC,
 *         ties broken by total_generated DESC, then index ASC —
 *         must match numpy.lexsort((-total_gen, -confidence)))
 *
 * num_drafts is tiny (typically <= 8), so a single-thread insertion
 * sort is faster than any parallel scheme once launch overhead counts.
 * Launch with <<<1, 1>>>.
 */
__global__ void rank_drafts_confidence_kernel(
    const float* acceptance_rate,
    const int* total_generated,
    int num_drafts,
    int* output_priority
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int i = 0; i < num_drafts; i++) {
        output_priority[i] = i;
    }

    // Insertion sort by (confidence DESC, generated DESC, index ASC)
    for (int i = 1; i < num_drafts; i++) {
        int key = output_priority[i];
        float kc = acceptance_rate[key];
        int kg = total_generated[key];
        int j = i - 1;
        while (j >= 0) {
            int cur = output_priority[j];
            float cc = acceptance_rate[cur];
            int cg = total_generated[cur];
            bool key_wins = (kc > cc) ||
                            (kc == cc && kg > cg) ||
                            (kc == cc && kg == cg && key < cur);
            if (!key_wins) break;
            output_priority[j + 1] = cur;
            j--;
        }
        output_priority[j + 1] = key;
    }
}

/*
 * detect_kv_conflict_kernel
 *
 * Detects conflicts between eviction plan and locked KV blocks.
 * Input: eviction_plan[num_blocks], locked_blocks[num_blocks]
 * Output: safe_eviction[num_blocks], conflict_penalty (scalar)
 */
__global__ void detect_kv_conflict_kernel(
    const uint32_t* eviction_plan,
    const uint32_t* locked_blocks,
    int num_words,               // number of uint32 bitmask words
    uint32_t* safe_eviction,
    float* conflict_penalty
) {
    // Single-block launch: a grid-stride loop over words with one shared
    // reduction gives correct totals without cross-block atomics. Word
    // counts are tiny (1024 KV blocks -> 32 words), so one block is plenty.
    int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    int* s_conflicts = (int*)shared_mem;
    int* s_desired = (int*)(shared_mem + blockDim.x * sizeof(int));

    int local_conflicts = 0;
    int local_desired = 0;

    for (int i = tid; i < num_words; i += blockDim.x) {
        uint32_t plan = eviction_plan[i];
        uint32_t conflict = plan & locked_blocks[i];
        safe_eviction[i] = plan & ~locked_blocks[i];
        local_conflicts += __popc(conflict);
        local_desired += __popc(plan);
    }

    s_conflicts[tid] = local_conflicts;
    s_desired[tid] = local_desired;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_conflicts[tid] += s_conflicts[tid + s];
            s_desired[tid] += s_desired[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int num_desired = s_desired[0];
        *conflict_penalty = (num_desired > 0)
            ? (float)s_conflicts[0] / num_desired : 0.0f;
    }
}

/*
 * eviction_scores_kernel
 *
 * Computes eviction scores for KV blocks based on recency, frequency, attention.
 * Input: recency[num_blocks], frequency[num_blocks], attn_weights[num_blocks, seq_len]
 * Output: scores[num_blocks]
 */
__global__ void eviction_scores_kernel(
    const float* recency,
    const float* frequency,
    const float* attn_weights,
    const bool* valid_mask,
    int num_blocks,
    int seq_len,
    float w_recency,
    float w_frequency,
    float w_attention,
    float* scores
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (bid >= num_blocks) return;

    // Each block processes one KV block's score.
    // Shared arrays are sized by blockDim.x (one slot per thread) —
    // sizing by num_blocks would read out of bounds whenever
    // num_blocks < blockDim.x.
    extern __shared__ char shared_mem[];
    float* attn_sum = (float*)shared_mem;
    int* valid_count = (int*)(shared_mem + blockDim.x * sizeof(float));

    // Initialize per-thread accumulators
    attn_sum[tid] = 0.0f;
    valid_count[tid] = 0;

    // Compute statistics for this block
    for (int s = tid; s < seq_len; s += blockDim.x) {
        int idx = bid * seq_len + s;
        if (valid_mask[idx]) {
            attn_sum[tid] += attn_weights[idx];
            valid_count[tid]++;
        }
    }

    __syncthreads();

    // Reduce across threads
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            attn_sum[tid] += attn_sum[tid + s];
            valid_count[tid] += valid_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean_attn = (valid_count[0] > 0) ? attn_sum[0] / valid_count[0] : 0.0f;
        scores[bid] = w_recency * recency[bid] +
                      w_frequency * frequency[bid] +
                      w_attention * mean_attn;
    }
}

/*
 * async_kv_restore_kernel
 *
 * Asynchronously restore KV cache from pinned memory.
 * This kernel is enqueued on a separate stream for pipelining.
 */
__global__ void async_kv_restore_kernel(
    const float* pinned_kv,
    float* gpu_kv,
    int num_tokens,
    int head_dim,
    int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * head_dim;

    if (idx < total) {
        gpu_kv[offset + idx] = pinned_kv[idx];
    }
}

/*
 * soft_lock_kv_kernel
 *
 * Apply soft locks to selected KV blocks for ranking.
 */
__global__ void soft_lock_kv_kernel(
    uint32_t* locked_blocks,
    const int* priority_blocks,
    int num_priority_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_priority_blocks) {
        int block_idx = priority_blocks[idx];
        int byte_idx = block_idx / 32;
        int bit_idx = block_idx % 32;
        atomicOr(&locked_blocks[byte_idx], 1u << bit_idx);
    }
}

/*
 * soft_unlock_kv_kernel
 *
 * Release soft locks after ranking.
 */
__global__ void soft_unlock_kv_kernel(
    uint32_t* locked_blocks,
    const int* priority_blocks,
    int num_priority_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_priority_blocks) {
        int block_idx = priority_blocks[idx];
        int byte_idx = block_idx / 32;
        int bit_idx = block_idx % 32;
        atomicAnd(&locked_blocks[byte_idx], ~(1u << bit_idx));
    }
}

/*
 * Wrapper functions for CPU interface
 */

extern "C" {

cudaError_t cu_rank_drafts(
    const float* acceptance_rate,
    const int* total_generated,
    int num_drafts,
    int* output_priority,
    cudaStream_t stream
) {
    // Serial insertion sort in-kernel: num_drafts is tiny (<= 8 typical)
    rank_drafts_confidence_kernel<<<1, 1, 0, stream>>>(
        acceptance_rate, total_generated, num_drafts, output_priority
    );

    return cudaGetLastError();
}

cudaError_t cu_detect_conflict(
    const uint32_t* eviction_plan,
    const uint32_t* locked_blocks,
    int num_words,
    uint32_t* safe_eviction,
    float* conflict_penalty,
    cudaStream_t stream
) {
    // Single block: the kernel grid-strides over words and reduces
    // conflict/desired totals in shared memory (no cross-block atomics)
    int threads = 256;
    int shared_size = 2 * threads * sizeof(int);

    detect_kv_conflict_kernel<<<1, threads, shared_size, stream>>>(
        eviction_plan, locked_blocks, num_words, safe_eviction, conflict_penalty
    );

    return cudaGetLastError();
}

cudaError_t cu_eviction_scores(
    const float* recency,
    const float* frequency,
    const float* attn_weights,
    const bool* valid_mask,
    int num_blocks,
    int seq_len,
    float w_recency,
    float w_frequency,
    float w_attention,
    float* scores,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = num_blocks;
    int shared_size = threads * (sizeof(float) + sizeof(int));

    eviction_scores_kernel<<<blocks, threads, shared_size, stream>>>(
        recency, frequency, attn_weights, valid_mask,
        num_blocks, seq_len,
        w_recency, w_frequency, w_attention,
        scores
    );

    return cudaGetLastError();
}

cudaError_t cu_async_restore_kv(
    const float* pinned_kv,
    float* gpu_kv,
    int num_tokens,
    int head_dim,
    int offset,
    cudaStream_t stream
) {
    int threads = 256;
    int total = num_tokens * head_dim;
    int blocks = (total + threads - 1) / threads;

    async_kv_restore_kernel<<<blocks, threads, 0, stream>>>(
        pinned_kv, gpu_kv, num_tokens, head_dim, offset
    );

    return cudaGetLastError();
}

cudaError_t cu_soft_lock(
    uint32_t* locked_blocks,
    const int* priority_blocks,
    int num_priority_blocks,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_priority_blocks + threads - 1) / threads;

    soft_lock_kv_kernel<<<blocks, threads, 0, stream>>>(
        locked_blocks, priority_blocks, num_priority_blocks
    );

    return cudaGetLastError();
}

cudaError_t cu_soft_unlock(
    uint32_t* locked_blocks,
    const int* priority_blocks,
    int num_priority_blocks,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_priority_blocks + threads - 1) / threads;

    soft_unlock_kv_kernel<<<blocks, threads, 0, stream>>>(
        locked_blocks, priority_blocks, num_priority_blocks
    );

    return cudaGetLastError();
}

} // extern "C"
