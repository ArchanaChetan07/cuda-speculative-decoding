/*
 * PyTorch C++ bindings for Speculative Decoding CUDA kernels
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA wrapper functions
extern "C" {
cudaError_t cu_rank_drafts(const float*, const int*, int, int*, cudaStream_t);
cudaError_t cu_detect_conflict(const uint32_t*, const uint32_t*, int, uint32_t*, float*, cudaStream_t);
cudaError_t cu_eviction_scores(const float*, const float*, const float*, const bool*,
                                int, int, float, float, float, float*, cudaStream_t);
cudaError_t cu_async_restore_kv(const float*, float*, int, int, int, cudaStream_t);
cudaError_t cu_soft_lock(uint32_t*, const int*, int, cudaStream_t);
cudaError_t cu_soft_unlock(uint32_t*, const int*, int, cudaStream_t);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor rank_drafts(
    torch::Tensor acceptance_rate,
    torch::Tensor total_generated
) {
    CHECK_INPUT(acceptance_rate);
    CHECK_INPUT(total_generated);

    int num_drafts = acceptance_rate.size(0);
    auto output = torch::empty({num_drafts},
                               torch::TensorOptions().dtype(torch::kInt32).device(acceptance_rate.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_rank_drafts(
        acceptance_rate.data_ptr<float>(),
        total_generated.data_ptr<int>(),
        num_drafts,
        output.data_ptr<int>(),
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "rank_drafts failed: ", cudaGetErrorString(err));
    return output;
}

std::vector<torch::Tensor> detect_conflict(
    torch::Tensor eviction_plan,
    torch::Tensor locked_blocks
) {
    CHECK_INPUT(eviction_plan);
    CHECK_INPUT(locked_blocks);

    int num_blocks = eviction_plan.size(0);
    auto safe_eviction = torch::empty_like(eviction_plan);
    auto conflict_penalty = torch::zeros({1},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(eviction_plan.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_detect_conflict(
        (const uint32_t*)eviction_plan.data_ptr<int32_t>(),
        (const uint32_t*)locked_blocks.data_ptr<int32_t>(),
        num_blocks,
        (uint32_t*)safe_eviction.data_ptr<int32_t>(),
        conflict_penalty.data_ptr<float>(),
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "detect_conflict failed: ", cudaGetErrorString(err));
    return {safe_eviction, conflict_penalty};
}

torch::Tensor eviction_scores(
    torch::Tensor recency,
    torch::Tensor frequency,
    torch::Tensor attn_weights,
    torch::Tensor valid_mask,
    float w_recency,
    float w_frequency,
    float w_attention
) {
    CHECK_INPUT(recency);
    CHECK_INPUT(frequency);
    CHECK_INPUT(attn_weights);
    CHECK_INPUT(valid_mask);

    int num_blocks = recency.size(0);
    int seq_len = attn_weights.size(1);

    auto scores = torch::empty({num_blocks},
                               torch::TensorOptions().dtype(torch::kFloat32).device(recency.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_eviction_scores(
        recency.data_ptr<float>(),
        frequency.data_ptr<float>(),
        attn_weights.data_ptr<float>(),
        valid_mask.data_ptr<bool>(),
        num_blocks, seq_len,
        w_recency, w_frequency, w_attention,
        scores.data_ptr<float>(),
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "eviction_scores failed: ", cudaGetErrorString(err));
    return scores;
}

void async_restore_kv(
    torch::Tensor pinned_kv,
    torch::Tensor gpu_kv,
    int num_tokens,
    int head_dim,
    int offset
) {
    CHECK_CONTIGUOUS(pinned_kv);
    CHECK_INPUT(gpu_kv);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_async_restore_kv(
        pinned_kv.data_ptr<float>(),
        gpu_kv.data_ptr<float>(),
        num_tokens, head_dim, offset,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "async_restore_kv failed: ", cudaGetErrorString(err));
}

void soft_lock(
    torch::Tensor locked_blocks,
    torch::Tensor priority_blocks
) {
    CHECK_INPUT(locked_blocks);
    CHECK_INPUT(priority_blocks);

    int num_priority = priority_blocks.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_soft_lock(
        (uint32_t*)locked_blocks.data_ptr<int32_t>(),
        priority_blocks.data_ptr<int>(),
        num_priority,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "soft_lock failed: ", cudaGetErrorString(err));
}

void soft_unlock(
    torch::Tensor locked_blocks,
    torch::Tensor priority_blocks
) {
    CHECK_INPUT(locked_blocks);
    CHECK_INPUT(priority_blocks);

    int num_priority = priority_blocks.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_soft_unlock(
        (uint32_t*)locked_blocks.data_ptr<int32_t>(),
        priority_blocks.data_ptr<int>(),
        num_priority,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "soft_unlock failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rank_drafts", &rank_drafts, "Rank draft tokens by confidence (CUDA)");
    m.def("detect_conflict", &detect_conflict, "Detect KV block conflicts (CUDA)");
    m.def("eviction_scores", &eviction_scores, "Compute eviction scores (CUDA)");
    m.def("async_restore_kv", &async_restore_kv, "Async KV restore (CUDA)");
    m.def("soft_lock", &soft_lock, "Apply soft locks (CUDA)");
    m.def("soft_unlock", &soft_unlock, "Release soft locks (CUDA)");
}
