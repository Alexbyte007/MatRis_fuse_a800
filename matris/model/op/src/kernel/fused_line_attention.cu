#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cfloat>
#include <limits>

namespace {

constexpr int kDim = 128;
constexpr int kThreads = 256;

__device__ float atomic_max_float(float* addr, float value) {
  int* addr_as_i = reinterpret_cast<int*>(addr);
  int old = *addr_as_i;
  while (__int_as_float(old) < value) {
    int assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(value));
    if (old == assumed) {
      break;
    }
  }
  return __int_as_float(old);
}

__global__ void fused_line_attention_init_kernel(
    float* __restrict__ source_max,
    float* __restrict__ target_max,
    float* __restrict__ source_sum,
    float* __restrict__ target_sum,
    float* __restrict__ source_out,
    float* __restrict__ target_out,
    int64_t total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  source_max[idx] = -FLT_MAX;
  target_max[idx] = -FLT_MAX;
  source_sum[idx] = 0.0f;
  target_sum[idx] = 0.0f;
  source_out[idx] = 0.0f;
  target_out[idx] = 0.0f;
}

__global__ void fused_line_attention_max_init_kernel(
    float* __restrict__ source_max,
    float* __restrict__ target_max,
    int64_t total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  source_max[idx] = -FLT_MAX;
  target_max[idx] = -FLT_MAX;
}

__global__ void fused_line_attention_single_max_init_kernel(
    float* __restrict__ out,
    int64_t total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  out[idx] = -FLT_MAX;
}

__global__ void fused_line_attention_zero4_kernel(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    float* __restrict__ d,
    int64_t total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  a[idx] = 0.0f;
  b[idx] = 0.0f;
  c[idx] = 0.0f;
  d[idx] = 0.0f;
}

__global__ void fused_line_attention_source_init_kernel(
    float* __restrict__ source_max,
    float* __restrict__ source_sum,
    float* __restrict__ source_out,
    int64_t total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  source_max[idx] = -FLT_MAX;
  source_sum[idx] = 0.0f;
  source_out[idx] = 0.0f;
}

__global__ void fused_line_attention_max_kernel(
    const float* __restrict__ source_logits,
    const float* __restrict__ target_logits,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ source_max,
    float* __restrict__ target_max,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  atomic_max_float(source_max + s * kDim + dim, source_logits[idx]);
  atomic_max_float(target_max + t * kDim + dim, target_logits[idx]);
}

__global__ void fused_line_attention_single_max_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ index,
    float* __restrict__ out,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t segment = index[row];
  atomic_max_float(out + segment * kDim + dim, logits[idx]);
}

__global__ void fused_line_attention_exp_sum_kernel(
    const float* __restrict__ source_logits,
    const float* __restrict__ target_logits,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ source_max,
    const float* __restrict__ target_max,
    float* __restrict__ source_alpha,
    float* __restrict__ target_alpha,
    float* __restrict__ source_sum,
    float* __restrict__ target_sum,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  float se = expf(source_logits[idx] - source_max[s * kDim + dim]);
  float te = expf(target_logits[idx] - target_max[t * kDim + dim]);
  source_alpha[idx] = se;
  target_alpha[idx] = te;
  atomicAdd(source_sum + s * kDim + dim, se);
  atomicAdd(target_sum + t * kDim + dim, te);
}

__global__ void fused_line_attention_norm_out_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ source_sum,
    const float* __restrict__ target_sum,
    float* __restrict__ source_alpha,
    float* __restrict__ target_alpha,
    float* __restrict__ source_out,
    float* __restrict__ target_out,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  float sa = source_alpha[idx] / source_sum[s * kDim + dim];
  float ta = target_alpha[idx] / target_sum[t * kDim + dim];
  source_alpha[idx] = sa;
  target_alpha[idx] = ta;
  float v = values[idx];
  atomicAdd(source_out + s * kDim + dim, sa * v);
  atomicAdd(target_out + t * kDim + dim, ta * v);
}

__global__ void fused_line_attention_source_exp_sum_kernel(
    const float* __restrict__ source_logits,
    const int64_t* __restrict__ source_index,
    const float* __restrict__ source_max,
    float* __restrict__ source_alpha,
    float* __restrict__ source_sum,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  float se = expf(source_logits[idx] - source_max[s * kDim + dim]);
  source_alpha[idx] = se;
  atomicAdd(source_sum + s * kDim + dim, se);
}

__global__ void fused_line_attention_source_norm_out_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ source_index,
    const float* __restrict__ source_sum,
    float* __restrict__ source_alpha,
    float* __restrict__ source_out,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  float sa = source_alpha[idx] / source_sum[s * kDim + dim];
  source_alpha[idx] = sa;
  atomicAdd(source_out + s * kDim + dim, sa * values[idx]);
}

__global__ void fused_line_attention_target_offsets_forward_kernel(
    const float* __restrict__ target_logits,
    const float* __restrict__ values,
    const int64_t* __restrict__ target_offsets,
    float* __restrict__ target_alpha,
    float* __restrict__ target_out,
    int64_t num_segments) {
  int64_t segment = blockIdx.x;
  int dim = threadIdx.x;
  if (segment >= num_segments || dim >= kDim) {
    return;
  }
  int64_t start = target_offsets[segment];
  int64_t end = target_offsets[segment + 1];
  float max_v = -FLT_MAX;
  for (int64_t row = start; row < end; ++row) {
    float v = target_logits[row * kDim + dim];
    max_v = fmaxf(max_v, v);
  }

  float sum_v = 0.0f;
  for (int64_t row = start; row < end; ++row) {
    int64_t idx = row * kDim + dim;
    float e = expf(target_logits[idx] - max_v);
    target_alpha[idx] = e;
    sum_v += e;
  }

  float out_v = 0.0f;
  float inv_sum = sum_v > 0.0f ? 1.0f / sum_v : 0.0f;
  for (int64_t row = start; row < end; ++row) {
    int64_t idx = row * kDim + dim;
    float alpha = target_alpha[idx] * inv_sum;
    target_alpha[idx] = alpha;
    out_v += alpha * values[idx];
  }
  target_out[segment * kDim + dim] = out_v;
}

__global__ void fused_line_attention_node_input_init_kernel(
    const float* __restrict__ node_feat,
    float* __restrict__ source_max,
    float* __restrict__ source_sum,
    float* __restrict__ fusion_node_feat,
    int64_t num_segments) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_segments * kDim;
  if (idx >= total) {
    return;
  }
  int64_t segment = idx / kDim;
  int dim = idx - segment * kDim;
  source_max[idx] = -FLT_MAX;
  source_sum[idx] = 0.0f;
  int64_t out_base = segment * (3 * kDim);
  fusion_node_feat[out_base + dim] = node_feat[idx];
  fusion_node_feat[out_base + kDim + dim] = 0.0f;
  fusion_node_feat[out_base + 2 * kDim + dim] = 0.0f;
}

__global__ void fused_line_attention_source_norm_to_node_input_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ source_index,
    const float* __restrict__ source_sum,
    float* __restrict__ source_alpha,
    float* __restrict__ fusion_node_feat,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  float sa = source_alpha[idx] / source_sum[s * kDim + dim];
  source_alpha[idx] = sa;
  atomicAdd(fusion_node_feat + s * (3 * kDim) + 2 * kDim + dim, sa * values[idx]);
}

__global__ void fused_line_attention_target_offsets_to_node_input_kernel(
    const float* __restrict__ target_logits,
    const float* __restrict__ values,
    const int64_t* __restrict__ target_offsets,
    float* __restrict__ target_alpha,
    float* __restrict__ fusion_node_feat,
    int64_t num_segments) {
  int64_t segment = blockIdx.x;
  int dim = threadIdx.x;
  if (segment >= num_segments || dim >= kDim) {
    return;
  }
  int64_t start = target_offsets[segment];
  int64_t end = target_offsets[segment + 1];
  float max_v = -FLT_MAX;
  for (int64_t row = start; row < end; ++row) {
    float v = target_logits[row * kDim + dim];
    max_v = fmaxf(max_v, v);
  }

  float sum_v = 0.0f;
  for (int64_t row = start; row < end; ++row) {
    int64_t idx = row * kDim + dim;
    float e = expf(target_logits[idx] - max_v);
    target_alpha[idx] = e;
    sum_v += e;
  }

  float out_v = 0.0f;
  float inv_sum = sum_v > 0.0f ? 1.0f / sum_v : 0.0f;
  for (int64_t row = start; row < end; ++row) {
    int64_t idx = row * kDim + dim;
    float alpha = target_alpha[idx] * inv_sum;
    target_alpha[idx] = alpha;
    out_v += alpha * values[idx];
  }
  fusion_node_feat[segment * (3 * kDim) + kDim + dim] = out_v;
}

__global__ void fused_line_attention_backward_kernel(
    const float* __restrict__ grad_source_out,
    const float* __restrict__ grad_target_out,
    const float* __restrict__ values,
    const float* __restrict__ source_out,
    const float* __restrict__ target_out,
    const float* __restrict__ source_alpha,
    const float* __restrict__ target_alpha,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_source_logits,
    float* __restrict__ grad_target_logits,
    float* __restrict__ grad_values,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  float v = values[idx];
  float sa = source_alpha[idx];
  float ta = target_alpha[idx];
  float gs = grad_source_out[s * kDim + dim];
  float gt = grad_target_out[t * kDim + dim];
  float os = source_out[s * kDim + dim];
  float ot = target_out[t * kDim + dim];
  grad_source_logits[idx] = sa * gs * (v - os);
  grad_target_logits[idx] = ta * gt * (v - ot);
  grad_values[idx] = sa * gs + ta * gt;
}

__global__ void fused_line_attention_backward_with_edge_direct_kernel(
    const float* __restrict__ grad_source_out,
    const float* __restrict__ grad_target_out,
    const float* __restrict__ grad_edge_direct,
    const float* __restrict__ values,
    const float* __restrict__ source_out,
    const float* __restrict__ target_out,
    const float* __restrict__ source_alpha,
    const float* __restrict__ target_alpha,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_source_logits,
    float* __restrict__ grad_target_logits,
    float* __restrict__ grad_values,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  float v = values[idx];
  float sa = source_alpha[idx];
  float ta = target_alpha[idx];
  float gs = grad_source_out[s * kDim + dim];
  float gt = grad_target_out[t * kDim + dim];
  float os = source_out[s * kDim + dim];
  float ot = target_out[t * kDim + dim];
  grad_source_logits[idx] = sa * gs * (v - os);
  grad_target_logits[idx] = ta * gt * (v - ot);
  grad_values[idx] = sa * gs + ta * gt + grad_edge_direct[idx];
}

__global__ void fused_line_attention_values_backward_with_edge_direct_kernel(
    const float* __restrict__ grad_source_out,
    const float* __restrict__ grad_target_out,
    const float* __restrict__ grad_edge_direct,
    const float* __restrict__ source_alpha,
    const float* __restrict__ target_alpha,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_values,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  float sa = source_alpha[idx];
  float ta = target_alpha[idx];
  float gs = grad_source_out[s * kDim + dim];
  float gt = grad_target_out[t * kDim + dim];
  grad_values[idx] = sa * gs + ta * gt + grad_edge_direct[idx];
}

}  // namespace

std::vector<torch::Tensor> fused_line_attention_forward(const torch::Tensor &source_logits,
                                                        const torch::Tensor &target_logits,
                                                        const torch::Tensor &values,
                                                        const torch::Tensor &source_index,
                                                        const torch::Tensor &target_index,
                                                        int64_t num_segments) {
  TORCH_CHECK(source_logits.is_cuda() && target_logits.is_cuda() && values.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "fused_line_attention_forward: tensors must be CUDA");
  TORCH_CHECK(source_logits.scalar_type() == torch::kFloat32 &&
                  target_logits.scalar_type() == torch::kFloat32 &&
                  values.scalar_type() == torch::kFloat32,
              "fused_line_attention_forward: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "fused_line_attention_forward: indices must be int64");
  TORCH_CHECK(source_logits.dim() == 2 && source_logits.size(1) == kDim,
              "fused_line_attention_forward: source_logits must be [E, 128]");
  TORCH_CHECK(source_logits.sizes() == target_logits.sizes() && source_logits.sizes() == values.sizes(),
              "fused_line_attention_forward: tensor shape mismatch");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == source_logits.size(0) &&
                  target_index.size(0) == source_logits.size(0),
              "fused_line_attention_forward: index shape mismatch");

  auto source_logits_c = source_logits.contiguous();
  auto target_logits_c = target_logits.contiguous();
  auto values_c = values.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_index_c = target_index.contiguous();
  auto opts = source_logits.options();
  auto source_max = torch::full({num_segments, kDim}, -std::numeric_limits<float>::infinity(), opts);
  auto target_max = torch::full({num_segments, kDim}, -std::numeric_limits<float>::infinity(), opts);
  auto source_sum = torch::zeros({num_segments, kDim}, opts);
  auto target_sum = torch::zeros({num_segments, kDim}, opts);
  auto source_out = torch::zeros({num_segments, kDim}, opts);
  auto target_out = torch::zeros({num_segments, kDim}, opts);
  auto source_alpha = torch::empty_like(source_logits_c);
  auto target_alpha = torch::empty_like(target_logits_c);

  int64_t rows = source_logits_c.size(0);
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_max_kernel<<<blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      target_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      target_max.data_ptr<float>(),
      rows);
  fused_line_attention_exp_sum_kernel<<<blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      target_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      target_max.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      target_alpha.data_ptr<float>(),
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      rows);
  fused_line_attention_norm_out_kernel<<<blocks, kThreads>>>(
      values_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      target_alpha.data_ptr<float>(),
      source_out.data_ptr<float>(),
      target_out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {source_out, target_out, source_alpha, target_alpha};
}

std::vector<torch::Tensor> fused_line_attention_forward_v2(const torch::Tensor &source_logits,
                                                           const torch::Tensor &target_logits,
                                                           const torch::Tensor &values,
                                                           const torch::Tensor &source_index,
                                                           const torch::Tensor &target_index,
                                                           int64_t num_segments) {
  TORCH_CHECK(source_logits.is_cuda() && target_logits.is_cuda() && values.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "fused_line_attention_forward_v2: tensors must be CUDA");
  TORCH_CHECK(source_logits.scalar_type() == torch::kFloat32 &&
                  target_logits.scalar_type() == torch::kFloat32 &&
                  values.scalar_type() == torch::kFloat32,
              "fused_line_attention_forward_v2: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "fused_line_attention_forward_v2: indices must be int64");
  TORCH_CHECK(source_logits.dim() == 2 && source_logits.size(1) == kDim,
              "fused_line_attention_forward_v2: source_logits must be [E, 128]");
  TORCH_CHECK(source_logits.sizes() == target_logits.sizes() && source_logits.sizes() == values.sizes(),
              "fused_line_attention_forward_v2: tensor shape mismatch");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == source_logits.size(0) &&
                  target_index.size(0) == source_logits.size(0),
              "fused_line_attention_forward_v2: index shape mismatch");

  auto source_logits_c = source_logits.contiguous();
  auto target_logits_c = target_logits.contiguous();
  auto values_c = values.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_index_c = target_index.contiguous();
  auto opts = source_logits.options();
  auto source_max = torch::empty({num_segments, kDim}, opts);
  auto target_max = torch::empty({num_segments, kDim}, opts);
  auto source_sum = torch::empty({num_segments, kDim}, opts);
  auto target_sum = torch::empty({num_segments, kDim}, opts);
  auto source_out = torch::empty({num_segments, kDim}, opts);
  auto target_out = torch::empty({num_segments, kDim}, opts);
  auto source_alpha = torch::empty_like(source_logits_c);
  auto target_alpha = torch::empty_like(target_logits_c);

  int64_t rows = source_logits_c.size(0);
  int64_t init_total = num_segments * kDim;
  int init_blocks = static_cast<int>((init_total + kThreads - 1) / kThreads);
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_init_kernel<<<init_blocks, kThreads>>>(
      source_max.data_ptr<float>(),
      target_max.data_ptr<float>(),
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      source_out.data_ptr<float>(),
      target_out.data_ptr<float>(),
      init_total);
  fused_line_attention_max_kernel<<<blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      target_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      target_max.data_ptr<float>(),
      rows);
  fused_line_attention_exp_sum_kernel<<<blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      target_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      target_max.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      target_alpha.data_ptr<float>(),
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      rows);
  fused_line_attention_norm_out_kernel<<<blocks, kThreads>>>(
      values_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      target_alpha.data_ptr<float>(),
      source_out.data_ptr<float>(),
      target_out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {source_out, target_out, source_alpha, target_alpha};
}

std::vector<torch::Tensor> fused_line_attention_max_atomic(const torch::Tensor &source_logits,
                                                           const torch::Tensor &target_logits,
                                                           const torch::Tensor &source_index,
                                                           const torch::Tensor &target_index,
                                                           int64_t num_segments) {
  TORCH_CHECK(source_logits.is_cuda() && target_logits.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "fused_line_attention_max_atomic: tensors must be CUDA");
  TORCH_CHECK(source_logits.scalar_type() == torch::kFloat32 && target_logits.scalar_type() == torch::kFloat32,
              "fused_line_attention_max_atomic: logits must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "fused_line_attention_max_atomic: indices must be int64");
  TORCH_CHECK(source_logits.dim() == 2 && source_logits.size(1) == kDim,
              "fused_line_attention_max_atomic: source_logits must be [E, 128]");
  TORCH_CHECK(source_logits.sizes() == target_logits.sizes(),
              "fused_line_attention_max_atomic: tensor shape mismatch");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == source_logits.size(0) &&
                  target_index.size(0) == source_logits.size(0),
              "fused_line_attention_max_atomic: index shape mismatch");

  auto source_logits_c = source_logits.contiguous();
  auto target_logits_c = target_logits.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_index_c = target_index.contiguous();
  auto opts = source_logits.options();
  auto source_max = torch::empty({num_segments, kDim}, opts);
  auto target_max = torch::empty({num_segments, kDim}, opts);

  int64_t rows = source_logits_c.size(0);
  int64_t init_total = num_segments * kDim;
  int init_blocks = static_cast<int>((init_total + kThreads - 1) / kThreads);
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_max_init_kernel<<<init_blocks, kThreads>>>(
      source_max.data_ptr<float>(),
      target_max.data_ptr<float>(),
      init_total);
  fused_line_attention_max_kernel<<<blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      target_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      target_max.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {source_max, target_max};
}

torch::Tensor fused_line_attention_single_max_atomic(const torch::Tensor &logits,
                                                     const torch::Tensor &index,
                                                     int64_t num_segments) {
  TORCH_CHECK(logits.is_cuda() && index.is_cuda(),
              "fused_line_attention_single_max_atomic: tensors must be CUDA");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "fused_line_attention_single_max_atomic: logits must be float32");
  TORCH_CHECK(index.scalar_type() == torch::kInt64,
              "fused_line_attention_single_max_atomic: index must be int64");
  TORCH_CHECK(logits.dim() == 2 && logits.size(1) == kDim,
              "fused_line_attention_single_max_atomic: logits must be [E, 128]");
  TORCH_CHECK(index.dim() == 1 && index.size(0) == logits.size(0),
              "fused_line_attention_single_max_atomic: index shape mismatch");

  auto logits_c = logits.contiguous();
  auto index_c = index.contiguous();
  auto out = torch::empty({num_segments, kDim}, logits.options());
  int64_t rows = logits_c.size(0);
  int64_t init_total = num_segments * kDim;
  int init_blocks = static_cast<int>((init_total + kThreads - 1) / kThreads);
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_single_max_init_kernel<<<init_blocks, kThreads>>>(
      out.data_ptr<float>(),
      init_total);
  fused_line_attention_single_max_kernel<<<blocks, kThreads>>>(
      logits_c.data_ptr<float>(),
      index_c.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> fused_line_attention_forward_with_max(const torch::Tensor &source_logits,
                                                                 const torch::Tensor &target_logits,
                                                                 const torch::Tensor &values,
                                                                 const torch::Tensor &source_index,
                                                                 const torch::Tensor &target_index,
                                                                 const torch::Tensor &source_max,
                                                                 const torch::Tensor &target_max,
                                                                 int64_t num_segments) {
  TORCH_CHECK(source_logits.is_cuda() && target_logits.is_cuda() && values.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda() && source_max.is_cuda() && target_max.is_cuda(),
              "fused_line_attention_forward_with_max: tensors must be CUDA");
  TORCH_CHECK(source_logits.scalar_type() == torch::kFloat32 &&
                  target_logits.scalar_type() == torch::kFloat32 &&
                  values.scalar_type() == torch::kFloat32 &&
                  source_max.scalar_type() == torch::kFloat32 &&
                  target_max.scalar_type() == torch::kFloat32,
              "fused_line_attention_forward_with_max: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "fused_line_attention_forward_with_max: indices must be int64");
  TORCH_CHECK(source_logits.dim() == 2 && source_logits.size(1) == kDim,
              "fused_line_attention_forward_with_max: source_logits must be [E, 128]");
  TORCH_CHECK(source_logits.sizes() == target_logits.sizes() && source_logits.sizes() == values.sizes(),
              "fused_line_attention_forward_with_max: tensor shape mismatch");
  TORCH_CHECK(source_max.dim() == 2 && target_max.dim() == 2 &&
                  source_max.size(0) == num_segments && target_max.size(0) == num_segments &&
                  source_max.size(1) == kDim && target_max.size(1) == kDim,
              "fused_line_attention_forward_with_max: max shape mismatch");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == source_logits.size(0) &&
                  target_index.size(0) == source_logits.size(0),
              "fused_line_attention_forward_with_max: index shape mismatch");

  auto source_logits_c = source_logits.contiguous();
  auto target_logits_c = target_logits.contiguous();
  auto values_c = values.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_index_c = target_index.contiguous();
  auto source_max_c = source_max.contiguous();
  auto target_max_c = target_max.contiguous();
  auto opts = source_logits.options();
  auto source_sum = torch::empty({num_segments, kDim}, opts);
  auto target_sum = torch::empty({num_segments, kDim}, opts);
  auto source_out = torch::empty({num_segments, kDim}, opts);
  auto target_out = torch::empty({num_segments, kDim}, opts);
  auto source_alpha = torch::empty_like(source_logits_c);
  auto target_alpha = torch::empty_like(target_logits_c);

  int64_t rows = source_logits_c.size(0);
  int64_t init_total = num_segments * kDim;
  int init_blocks = static_cast<int>((init_total + kThreads - 1) / kThreads);
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_zero4_kernel<<<init_blocks, kThreads>>>(
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      source_out.data_ptr<float>(),
      target_out.data_ptr<float>(),
      init_total);
  fused_line_attention_exp_sum_kernel<<<blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      target_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_max_c.data_ptr<float>(),
      target_max_c.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      target_alpha.data_ptr<float>(),
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      rows);
  fused_line_attention_norm_out_kernel<<<blocks, kThreads>>>(
      values_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      target_alpha.data_ptr<float>(),
      source_out.data_ptr<float>(),
      target_out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {source_out, target_out, source_alpha, target_alpha};
}

std::vector<torch::Tensor> fused_line_attention_forward_target_offsets(const torch::Tensor &source_logits,
                                                                       const torch::Tensor &target_logits,
                                                                       const torch::Tensor &values,
                                                                       const torch::Tensor &source_index,
                                                                       const torch::Tensor &target_index,
                                                                       const torch::Tensor &target_offsets,
                                                                       int64_t num_segments) {
  TORCH_CHECK(source_logits.is_cuda() && target_logits.is_cuda() && values.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda() && target_offsets.is_cuda(),
              "fused_line_attention_forward_target_offsets: tensors must be CUDA");
  TORCH_CHECK(source_logits.scalar_type() == torch::kFloat32 &&
                  target_logits.scalar_type() == torch::kFloat32 &&
                  values.scalar_type() == torch::kFloat32,
              "fused_line_attention_forward_target_offsets: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64 &&
                  target_offsets.scalar_type() == torch::kInt64,
              "fused_line_attention_forward_target_offsets: indices must be int64");
  TORCH_CHECK(source_logits.dim() == 2 && source_logits.size(1) == kDim,
              "fused_line_attention_forward_target_offsets: source_logits must be [E, 128]");
  TORCH_CHECK(source_logits.sizes() == target_logits.sizes() && source_logits.sizes() == values.sizes(),
              "fused_line_attention_forward_target_offsets: tensor shape mismatch");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == source_logits.size(0) &&
                  target_index.size(0) == source_logits.size(0),
              "fused_line_attention_forward_target_offsets: index shape mismatch");
  TORCH_CHECK(target_offsets.dim() == 1 && target_offsets.size(0) == num_segments + 1,
              "fused_line_attention_forward_target_offsets: target_offsets shape mismatch");

  auto source_logits_c = source_logits.contiguous();
  auto target_logits_c = target_logits.contiguous();
  auto values_c = values.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_offsets_c = target_offsets.contiguous();
  auto opts = source_logits.options();
  auto source_max = torch::empty({num_segments, kDim}, opts);
  auto source_sum = torch::empty({num_segments, kDim}, opts);
  auto source_out = torch::empty({num_segments, kDim}, opts);
  auto target_out = torch::empty({num_segments, kDim}, opts);
  auto source_alpha = torch::empty_like(source_logits_c);
  auto target_alpha = torch::empty_like(target_logits_c);

  int64_t rows = source_logits_c.size(0);
  int64_t init_total = num_segments * kDim;
  int init_blocks = static_cast<int>((init_total + kThreads - 1) / kThreads);
  int row_blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_source_init_kernel<<<init_blocks, kThreads>>>(
      source_max.data_ptr<float>(),
      source_sum.data_ptr<float>(),
      source_out.data_ptr<float>(),
      init_total);
  fused_line_attention_single_max_kernel<<<row_blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      rows);
  fused_line_attention_source_exp_sum_kernel<<<row_blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      source_sum.data_ptr<float>(),
      rows);
  fused_line_attention_source_norm_out_kernel<<<row_blocks, kThreads>>>(
      values_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      source_sum.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      source_out.data_ptr<float>(),
      rows);
  fused_line_attention_target_offsets_forward_kernel<<<static_cast<int>(num_segments), kDim>>>(
      target_logits_c.data_ptr<float>(),
      values_c.data_ptr<float>(),
      target_offsets_c.data_ptr<int64_t>(),
      target_alpha.data_ptr<float>(),
      target_out.data_ptr<float>(),
      num_segments);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {source_out, target_out, source_alpha, target_alpha};
}

std::vector<torch::Tensor> fused_line_attention_node_input_forward_target_offsets(
    const torch::Tensor &source_logits,
    const torch::Tensor &target_logits,
    const torch::Tensor &values,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &target_offsets,
    const torch::Tensor &node_feat,
    int64_t num_segments) {
  TORCH_CHECK(source_logits.is_cuda() && target_logits.is_cuda() && values.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda() && target_offsets.is_cuda() &&
                  node_feat.is_cuda(),
              "fused_line_attention_node_input_forward_target_offsets: tensors must be CUDA");
  TORCH_CHECK(source_logits.scalar_type() == torch::kFloat32 &&
                  target_logits.scalar_type() == torch::kFloat32 &&
                  values.scalar_type() == torch::kFloat32 &&
                  node_feat.scalar_type() == torch::kFloat32,
              "fused_line_attention_node_input_forward_target_offsets: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64 &&
                  target_offsets.scalar_type() == torch::kInt64,
              "fused_line_attention_node_input_forward_target_offsets: indices must be int64");
  TORCH_CHECK(source_logits.dim() == 2 && source_logits.size(1) == kDim,
              "fused_line_attention_node_input_forward_target_offsets: source_logits must be [E, 128]");
  TORCH_CHECK(source_logits.sizes() == target_logits.sizes() && source_logits.sizes() == values.sizes(),
              "fused_line_attention_node_input_forward_target_offsets: tensor shape mismatch");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == source_logits.size(0) &&
                  target_index.size(0) == source_logits.size(0),
              "fused_line_attention_node_input_forward_target_offsets: index shape mismatch");
  TORCH_CHECK(target_offsets.dim() == 1 && target_offsets.size(0) == num_segments + 1,
              "fused_line_attention_node_input_forward_target_offsets: target_offsets shape mismatch");
  TORCH_CHECK(node_feat.dim() == 2 && node_feat.size(0) == num_segments && node_feat.size(1) == kDim,
              "fused_line_attention_node_input_forward_target_offsets: node_feat must be [N, 128]");

  auto source_logits_c = source_logits.contiguous();
  auto target_logits_c = target_logits.contiguous();
  auto values_c = values.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_offsets_c = target_offsets.contiguous();
  auto node_feat_c = node_feat.contiguous();
  auto opts = source_logits.options();
  auto source_max = torch::empty({num_segments, kDim}, opts);
  auto source_sum = torch::empty({num_segments, kDim}, opts);
  auto fusion_node_feat = torch::empty({num_segments, 3 * kDim}, opts);
  auto source_alpha = torch::empty_like(source_logits_c);
  auto target_alpha = torch::empty_like(target_logits_c);

  int64_t rows = source_logits_c.size(0);
  int64_t init_total = num_segments * kDim;
  int init_blocks = static_cast<int>((init_total + kThreads - 1) / kThreads);
  int row_blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_node_input_init_kernel<<<init_blocks, kThreads>>>(
      node_feat_c.data_ptr<float>(),
      source_max.data_ptr<float>(),
      source_sum.data_ptr<float>(),
      fusion_node_feat.data_ptr<float>(),
      num_segments);
  fused_line_attention_single_max_kernel<<<row_blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      rows);
  fused_line_attention_source_exp_sum_kernel<<<row_blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      source_sum.data_ptr<float>(),
      rows);
  fused_line_attention_source_norm_to_node_input_kernel<<<row_blocks, kThreads>>>(
      values_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      source_sum.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      fusion_node_feat.data_ptr<float>(),
      rows);
  fused_line_attention_target_offsets_to_node_input_kernel<<<static_cast<int>(num_segments), kDim>>>(
      target_logits_c.data_ptr<float>(),
      values_c.data_ptr<float>(),
      target_offsets_c.data_ptr<int64_t>(),
      target_alpha.data_ptr<float>(),
      fusion_node_feat.data_ptr<float>(),
      num_segments);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {fusion_node_feat, source_alpha, target_alpha};
}

std::vector<torch::Tensor> fused_line_attention_backward(const torch::Tensor &grad_source_out,
                                                         const torch::Tensor &grad_target_out,
                                                         const torch::Tensor &values,
                                                         const torch::Tensor &source_out,
                                                         const torch::Tensor &target_out,
                                                         const torch::Tensor &source_alpha,
                                                         const torch::Tensor &target_alpha,
                                                         const torch::Tensor &source_index,
                                                         const torch::Tensor &target_index) {
  TORCH_CHECK(grad_source_out.is_cuda() && grad_target_out.is_cuda() && values.is_cuda() &&
                  source_out.is_cuda() && target_out.is_cuda() && source_alpha.is_cuda() &&
                  target_alpha.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "fused_line_attention_backward: tensors must be CUDA");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32 && grad_source_out.scalar_type() == torch::kFloat32 &&
                  grad_target_out.scalar_type() == torch::kFloat32,
              "fused_line_attention_backward: float tensors must be float32");
  auto grad_source_logits = torch::empty_like(values);
  auto grad_target_logits = torch::empty_like(values);
  auto grad_values = torch::empty_like(values);
  auto grad_source_out_c = grad_source_out.contiguous();
  auto grad_target_out_c = grad_target_out.contiguous();
  auto values_c = values.contiguous();
  auto source_out_c = source_out.contiguous();
  auto target_out_c = target_out.contiguous();
  auto source_alpha_c = source_alpha.contiguous();
  auto target_alpha_c = target_alpha.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_index_c = target_index.contiguous();
  int64_t rows = values_c.size(0);
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_backward_kernel<<<blocks, kThreads>>>(
      grad_source_out_c.data_ptr<float>(),
      grad_target_out_c.data_ptr<float>(),
      values_c.data_ptr<float>(),
      source_out_c.data_ptr<float>(),
      target_out_c.data_ptr<float>(),
      source_alpha_c.data_ptr<float>(),
      target_alpha_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      grad_source_logits.data_ptr<float>(),
      grad_target_logits.data_ptr<float>(),
      grad_values.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_source_logits, grad_target_logits, grad_values};
}

std::vector<torch::Tensor> fused_line_attention_backward_with_edge_direct(
    const torch::Tensor &grad_source_out,
    const torch::Tensor &grad_target_out,
    const torch::Tensor &grad_edge_direct,
    const torch::Tensor &values,
    const torch::Tensor &source_out,
    const torch::Tensor &target_out,
    const torch::Tensor &source_alpha,
    const torch::Tensor &target_alpha,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index) {
  TORCH_CHECK(grad_source_out.is_cuda() && grad_target_out.is_cuda() && grad_edge_direct.is_cuda() &&
                  values.is_cuda() && source_out.is_cuda() && target_out.is_cuda() &&
                  source_alpha.is_cuda() && target_alpha.is_cuda() && source_index.is_cuda() &&
                  target_index.is_cuda(),
              "fused_line_attention_backward_with_edge_direct: tensors must be CUDA");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32 && grad_source_out.scalar_type() == torch::kFloat32 &&
                  grad_target_out.scalar_type() == torch::kFloat32 &&
                  grad_edge_direct.scalar_type() == torch::kFloat32,
              "fused_line_attention_backward_with_edge_direct: float tensors must be float32");
  TORCH_CHECK(values.sizes() == grad_edge_direct.sizes(),
              "fused_line_attention_backward_with_edge_direct: values/grad_edge_direct shape mismatch");
  auto grad_source_logits = torch::empty_like(values);
  auto grad_target_logits = torch::empty_like(values);
  auto grad_values = torch::empty_like(values);
  auto grad_source_out_c = grad_source_out.contiguous();
  auto grad_target_out_c = grad_target_out.contiguous();
  auto grad_edge_direct_c = grad_edge_direct.contiguous();
  auto values_c = values.contiguous();
  auto source_out_c = source_out.contiguous();
  auto target_out_c = target_out.contiguous();
  auto source_alpha_c = source_alpha.contiguous();
  auto target_alpha_c = target_alpha.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_index_c = target_index.contiguous();
  int64_t rows = values_c.size(0);
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_backward_with_edge_direct_kernel<<<blocks, kThreads>>>(
      grad_source_out_c.data_ptr<float>(),
      grad_target_out_c.data_ptr<float>(),
      grad_edge_direct_c.data_ptr<float>(),
      values_c.data_ptr<float>(),
      source_out_c.data_ptr<float>(),
      target_out_c.data_ptr<float>(),
      source_alpha_c.data_ptr<float>(),
      target_alpha_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      grad_source_logits.data_ptr<float>(),
      grad_target_logits.data_ptr<float>(),
      grad_values.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_source_logits, grad_target_logits, grad_values};
}

torch::Tensor fused_line_attention_values_backward_with_edge_direct(
    const torch::Tensor &grad_source_out,
    const torch::Tensor &grad_target_out,
    const torch::Tensor &grad_edge_direct,
    const torch::Tensor &source_alpha,
    const torch::Tensor &target_alpha,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index) {
  TORCH_CHECK(grad_source_out.is_cuda() && grad_target_out.is_cuda() && grad_edge_direct.is_cuda() &&
                  source_alpha.is_cuda() && target_alpha.is_cuda() && source_index.is_cuda() &&
                  target_index.is_cuda(),
              "fused_line_attention_values_backward_with_edge_direct: tensors must be CUDA");
  TORCH_CHECK(grad_source_out.scalar_type() == torch::kFloat32 &&
                  grad_target_out.scalar_type() == torch::kFloat32 &&
                  grad_edge_direct.scalar_type() == torch::kFloat32 &&
                  source_alpha.scalar_type() == torch::kFloat32 &&
                  target_alpha.scalar_type() == torch::kFloat32,
              "fused_line_attention_values_backward_with_edge_direct: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "fused_line_attention_values_backward_with_edge_direct: indices must be int64");
  TORCH_CHECK(grad_edge_direct.sizes() == source_alpha.sizes() &&
                  grad_edge_direct.sizes() == target_alpha.sizes(),
              "fused_line_attention_values_backward_with_edge_direct: edge tensor shape mismatch");
  auto grad_source_out_c = grad_source_out.contiguous();
  auto grad_target_out_c = grad_target_out.contiguous();
  auto grad_edge_direct_c = grad_edge_direct.contiguous();
  auto source_alpha_c = source_alpha.contiguous();
  auto target_alpha_c = target_alpha.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_index_c = target_index.contiguous();
  auto grad_values = torch::empty_like(grad_edge_direct_c);
  int64_t rows = grad_edge_direct_c.size(0);
  if (rows == 0) {
    return grad_values;
  }
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_values_backward_with_edge_direct_kernel<<<blocks, kThreads>>>(
      grad_source_out_c.data_ptr<float>(),
      grad_target_out_c.data_ptr<float>(),
      grad_edge_direct_c.data_ptr<float>(),
      source_alpha_c.data_ptr<float>(),
      target_alpha_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      grad_values.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_values;
}
