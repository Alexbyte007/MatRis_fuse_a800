#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kThreadsN128 = 128;
constexpr int kDimN128 = 128;

__device__ __forceinline__ float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

void cublas_row_major_matmul(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    int64_t rows,
    int64_t k,
    int64_t cols,
    int64_t a_row_stride,
    int64_t b_row_stride,
    int64_t c_row_stride,
    const char* label) {
  TORCH_CHECK(rows <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  k <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  cols <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  a_row_stride <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  b_row_stride <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  c_row_stride <= static_cast<int64_t>(std::numeric_limits<int>::max()),
              label, ": dimensions exceed cuBLAS int range");
  const int m = static_cast<int>(cols);
  const int n = static_cast<int>(rows);
  const int kk = static_cast<int>(k);
  const int lda = static_cast<int>(b_row_stride);
  const int ldb = static_cast<int>(a_row_stride);
  const int ldc = static_cast<int>(c_row_stride);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasSgemm(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m,
      n,
      kk,
      &alpha,
      b,
      lda,
      a,
      ldb,
      &beta,
      c,
      ldc);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, label, ": cublasSgemm failed");
}

__global__ void silu_input_grad_epilogue_kernel(
    const float* __restrict__ grad_core_linear,
    const float* __restrict__ grad_gate_linear,
    const float* __restrict__ core_hidden,
    const float* __restrict__ gate_hidden,
    float* __restrict__ grad_core_hidden,
    float* __restrict__ grad_gate_hidden,
    int64_t numel) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  float core_x = core_hidden[idx];
  float gate_x = gate_hidden[idx];
  float core_sig = sigmoidf_fast(core_x);
  float gate_sig = sigmoidf_fast(gate_x);
  float core_silu_grad = core_sig * (1.0f + core_x * (1.0f - core_sig));
  float gate_silu_grad = gate_sig * (1.0f + gate_x * (1.0f - gate_sig));
  grad_core_hidden[idx] = grad_core_linear[idx] * core_silu_grad;
  grad_gate_hidden[idx] = grad_gate_linear[idx] * gate_silu_grad;
}

__global__ void residual_old_grad_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ res_weight,
    float* __restrict__ grad_old,
    int64_t numel,
    int64_t dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  grad_old[idx] = grad_out[idx] * res_weight[idx % dim];
}

__global__ void input_grad_only_gated_tail_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ core,
    const float* __restrict__ gate,
    const float* __restrict__ core_weight,
    const float* __restrict__ core_bias,
    const float* __restrict__ gate_weight,
    const float* __restrict__ gate_bias,
    float* __restrict__ grad_core,
    float* __restrict__ grad_gate,
    int rows,
    int dim,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core[kThreads];
  __shared__ float s_gate[kThreads];

  int base = row * dim;
  float core_v = 0.0f;
  float gate_v = 0.0f;
  if (tid < dim) {
    core_v = core[base + tid];
    gate_v = gate[base + tid];
  }

  s_core[tid] = tid < dim ? core_v : 0.0f;
  s_gate[tid] = tid < dim ? gate_v : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_mean = s_core[0] / static_cast<float>(dim);
  float gate_mean = s_gate[0] / static_cast<float>(dim);
  __syncthreads();

  float core_centered = tid < dim ? core_v - core_mean : 0.0f;
  float gate_centered = tid < dim ? gate_v - gate_mean : 0.0f;
  s_core[tid] = tid < dim ? core_centered * core_centered : 0.0f;
  s_gate[tid] = tid < dim ? gate_centered * gate_centered : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_rstd = rsqrtf(s_core[0] / static_cast<float>(dim) + eps);
  float gate_rstd = rsqrtf(s_gate[0] / static_cast<float>(dim) + eps);
  __syncthreads();

  float core_xhat = core_centered * core_rstd;
  float gate_xhat = gate_centered * gate_rstd;
  float core_ln = 0.0f;
  float gate_ln = 0.0f;
  float grad_core_norm = 0.0f;
  float grad_gate_norm = 0.0f;

  if (tid < dim) {
    core_ln = core_xhat * core_weight[tid] + core_bias[tid];
    gate_ln = gate_xhat * gate_weight[tid] + gate_bias[tid];
    float core_sig = sigmoidf_fast(core_ln);
    float core_act = core_ln * core_sig;
    float gate_act = sigmoidf_fast(gate_ln);
    float grad = grad_out[base + tid];
    float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
    float grad_core_ln = grad * gate_act * core_silu_grad;
    float grad_gate_ln = grad * core_act * gate_act * (1.0f - gate_act);
    grad_core_norm = grad_core_ln * core_weight[tid];
    grad_gate_norm = grad_gate_ln * gate_weight[tid];
  }

  s_core[tid] = tid < dim ? grad_core_norm : 0.0f;
  s_gate[tid] = tid < dim ? grad_gate_norm : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_sum_grad = s_core[0];
  float gate_sum_grad = s_gate[0];
  __syncthreads();

  s_core[tid] = tid < dim ? grad_core_norm * core_xhat : 0.0f;
  s_gate[tid] = tid < dim ? grad_gate_norm * gate_xhat : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_sum_grad_xhat = s_core[0];
  float gate_sum_grad_xhat = s_gate[0];

  if (tid < dim) {
    float inv_dim = 1.0f / static_cast<float>(dim);
    grad_core[base + tid] =
        (grad_core_norm * static_cast<float>(dim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
        core_rstd * inv_dim;
    grad_gate[base + tid] =
        (grad_gate_norm * static_cast<float>(dim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
        gate_rstd * inv_dim;
  }
}

__global__ void input_grad_only_gated_tail_backward_n128_v2_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ core,
    const float* __restrict__ gate,
    const float* __restrict__ core_weight,
    const float* __restrict__ core_bias,
    const float* __restrict__ gate_weight,
    const float* __restrict__ gate_bias,
    float* __restrict__ grad_core,
    float* __restrict__ grad_gate,
    int rows,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core_sum[kThreadsN128];
  __shared__ float s_gate_sum[kThreadsN128];
  __shared__ float s_core_aux[kThreadsN128];
  __shared__ float s_gate_aux[kThreadsN128];

  int base = row * kDimN128;
  float core_v = core[base + tid];
  float gate_v = gate[base + tid];

  s_core_sum[tid] = core_v;
  s_gate_sum[tid] = gate_v;
  __syncthreads();

  for (int stride = kThreadsN128 / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core_sum[tid] += s_core_sum[tid + stride];
      s_gate_sum[tid] += s_gate_sum[tid + stride];
    }
    __syncthreads();
  }

  constexpr float inv_dim = 1.0f / static_cast<float>(kDimN128);
  float core_mean = s_core_sum[0] * inv_dim;
  float gate_mean = s_gate_sum[0] * inv_dim;
  __syncthreads();

  float core_centered = core_v - core_mean;
  float gate_centered = gate_v - gate_mean;

  s_core_sum[tid] = core_centered * core_centered;
  s_gate_sum[tid] = gate_centered * gate_centered;
  __syncthreads();

  for (int stride = kThreadsN128 / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core_sum[tid] += s_core_sum[tid + stride];
      s_gate_sum[tid] += s_gate_sum[tid + stride];
    }
    __syncthreads();
  }

  float core_var = s_core_sum[0] * inv_dim;
  float gate_var = s_gate_sum[0] * inv_dim;
  float core_rstd = rsqrtf(core_var + eps);
  float gate_rstd = rsqrtf(gate_var + eps);
  __syncthreads();

  float core_xhat = core_centered * core_rstd;
  float gate_xhat = gate_centered * gate_rstd;
  float core_ln = core_xhat * core_weight[tid] + core_bias[tid];
  float gate_ln = gate_xhat * gate_weight[tid] + gate_bias[tid];
  float core_sig = sigmoidf_fast(core_ln);
  float core_act = core_ln * core_sig;
  float gate_act = sigmoidf_fast(gate_ln);
  float grad = grad_out[base + tid];
  float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
  float grad_core_ln = grad * gate_act * core_silu_grad;
  float grad_gate_ln = grad * core_act * gate_act * (1.0f - gate_act);
  float grad_core_norm = grad_core_ln * core_weight[tid];
  float grad_gate_norm = grad_gate_ln * gate_weight[tid];

  s_core_sum[tid] = grad_core_norm;
  s_gate_sum[tid] = grad_gate_norm;
  s_core_aux[tid] = grad_core_norm * core_xhat;
  s_gate_aux[tid] = grad_gate_norm * gate_xhat;
  __syncthreads();

  for (int stride = kThreadsN128 / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core_sum[tid] += s_core_sum[tid + stride];
      s_gate_sum[tid] += s_gate_sum[tid + stride];
      s_core_aux[tid] += s_core_aux[tid + stride];
      s_gate_aux[tid] += s_gate_aux[tid + stride];
    }
    __syncthreads();
  }

  float core_sum_grad = s_core_sum[0];
  float gate_sum_grad = s_gate_sum[0];
  float core_sum_grad_xhat = s_core_aux[0];
  float gate_sum_grad_xhat = s_gate_aux[0];

  grad_core[base + tid] =
      (grad_core_norm * static_cast<float>(kDimN128) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
      core_rstd * inv_dim;
  grad_gate[base + tid] =
      (grad_gate_norm * static_cast<float>(kDimN128) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
      gate_rstd * inv_dim;
}

__global__ void input_grad_only_gated_tail_backward_stack_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ core,
    const float* __restrict__ gate,
    const float* __restrict__ core_weight,
    const float* __restrict__ core_bias,
    const float* __restrict__ gate_weight,
    const float* __restrict__ gate_bias,
    float* __restrict__ grad_stack,
    int rows,
    int dim,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core[kThreads];
  __shared__ float s_gate[kThreads];

  int base = row * dim;
  float core_v = 0.0f;
  float gate_v = 0.0f;
  if (tid < dim) {
    core_v = core[base + tid];
    gate_v = gate[base + tid];
  }

  s_core[tid] = tid < dim ? core_v : 0.0f;
  s_gate[tid] = tid < dim ? gate_v : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_mean = s_core[0] / static_cast<float>(dim);
  float gate_mean = s_gate[0] / static_cast<float>(dim);
  __syncthreads();

  float core_centered = tid < dim ? core_v - core_mean : 0.0f;
  float gate_centered = tid < dim ? gate_v - gate_mean : 0.0f;
  s_core[tid] = tid < dim ? core_centered * core_centered : 0.0f;
  s_gate[tid] = tid < dim ? gate_centered * gate_centered : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_rstd = rsqrtf(s_core[0] / static_cast<float>(dim) + eps);
  float gate_rstd = rsqrtf(s_gate[0] / static_cast<float>(dim) + eps);
  __syncthreads();

  float core_xhat = core_centered * core_rstd;
  float gate_xhat = gate_centered * gate_rstd;
  float core_ln = 0.0f;
  float gate_ln = 0.0f;
  float grad_core_norm = 0.0f;
  float grad_gate_norm = 0.0f;

  if (tid < dim) {
    core_ln = core_xhat * core_weight[tid] + core_bias[tid];
    gate_ln = gate_xhat * gate_weight[tid] + gate_bias[tid];
    float core_sig = sigmoidf_fast(core_ln);
    float core_act = core_ln * core_sig;
    float gate_act = sigmoidf_fast(gate_ln);
    float grad = grad_out[base + tid];
    float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
    float grad_core_ln = grad * gate_act * core_silu_grad;
    float grad_gate_ln = grad * core_act * gate_act * (1.0f - gate_act);
    grad_core_norm = grad_core_ln * core_weight[tid];
    grad_gate_norm = grad_gate_ln * gate_weight[tid];
  }

  s_core[tid] = tid < dim ? grad_core_norm : 0.0f;
  s_gate[tid] = tid < dim ? grad_gate_norm : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_sum_grad = s_core[0];
  float gate_sum_grad = s_gate[0];
  __syncthreads();

  s_core[tid] = tid < dim ? grad_core_norm * core_xhat : 0.0f;
  s_gate[tid] = tid < dim ? grad_gate_norm * gate_xhat : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_sum_grad_xhat = s_core[0];
  float gate_sum_grad_xhat = s_gate[0];

  if (tid < dim) {
    float inv_dim = 1.0f / static_cast<float>(dim);
    grad_stack[base + tid] =
        (grad_core_norm * static_cast<float>(dim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
        core_rstd * inv_dim;
    grad_stack[static_cast<int64_t>(rows) * dim + base + tid] =
        (grad_gate_norm * static_cast<float>(dim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
        gate_rstd * inv_dim;
  }
}

__global__ void param_grad_gated_tail_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ core,
    const float* __restrict__ gate,
    const float* __restrict__ core_weight,
    const float* __restrict__ core_bias,
    const float* __restrict__ gate_weight,
    const float* __restrict__ gate_bias,
    float* __restrict__ grad_core,
    float* __restrict__ grad_gate,
    float* __restrict__ grad_core_weight,
    float* __restrict__ grad_core_bias,
    float* __restrict__ grad_gate_weight,
    float* __restrict__ grad_gate_bias,
    int rows,
    int dim,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core[kThreads];
  __shared__ float s_gate[kThreads];

  int base = row * dim;
  float core_v = 0.0f;
  float gate_v = 0.0f;
  if (tid < dim) {
    core_v = core[base + tid];
    gate_v = gate[base + tid];
  }

  s_core[tid] = tid < dim ? core_v : 0.0f;
  s_gate[tid] = tid < dim ? gate_v : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_mean = s_core[0] / static_cast<float>(dim);
  float gate_mean = s_gate[0] / static_cast<float>(dim);
  __syncthreads();

  float core_centered = tid < dim ? core_v - core_mean : 0.0f;
  float gate_centered = tid < dim ? gate_v - gate_mean : 0.0f;
  s_core[tid] = tid < dim ? core_centered * core_centered : 0.0f;
  s_gate[tid] = tid < dim ? gate_centered * gate_centered : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_rstd = rsqrtf(s_core[0] / static_cast<float>(dim) + eps);
  float gate_rstd = rsqrtf(s_gate[0] / static_cast<float>(dim) + eps);
  __syncthreads();

  float core_xhat = core_centered * core_rstd;
  float gate_xhat = gate_centered * gate_rstd;
  float grad_core_ln = 0.0f;
  float grad_gate_ln = 0.0f;
  float grad_core_norm = 0.0f;
  float grad_gate_norm = 0.0f;

  if (tid < dim) {
    float core_ln = core_xhat * core_weight[tid] + core_bias[tid];
    float gate_ln = gate_xhat * gate_weight[tid] + gate_bias[tid];
    float core_sig = sigmoidf_fast(core_ln);
    float core_act = core_ln * core_sig;
    float gate_act = sigmoidf_fast(gate_ln);
    float grad = grad_out[base + tid];
    float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
    grad_core_ln = grad * gate_act * core_silu_grad;
    grad_gate_ln = grad * core_act * gate_act * (1.0f - gate_act);
    grad_core_norm = grad_core_ln * core_weight[tid];
    grad_gate_norm = grad_gate_ln * gate_weight[tid];

    atomicAdd(grad_core_weight + tid, grad_core_ln * core_xhat);
    atomicAdd(grad_core_bias + tid, grad_core_ln);
    atomicAdd(grad_gate_weight + tid, grad_gate_ln * gate_xhat);
    atomicAdd(grad_gate_bias + tid, grad_gate_ln);
  }

  s_core[tid] = tid < dim ? grad_core_norm : 0.0f;
  s_gate[tid] = tid < dim ? grad_gate_norm : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_sum_grad = s_core[0];
  float gate_sum_grad = s_gate[0];
  __syncthreads();

  s_core[tid] = tid < dim ? grad_core_norm * core_xhat : 0.0f;
  s_gate[tid] = tid < dim ? grad_gate_norm * gate_xhat : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core[tid] += s_core[tid + stride];
      s_gate[tid] += s_gate[tid + stride];
    }
    __syncthreads();
  }
  float core_sum_grad_xhat = s_core[0];
  float gate_sum_grad_xhat = s_gate[0];

  if (tid < dim) {
    float inv_dim = 1.0f / static_cast<float>(dim);
    grad_core[base + tid] =
        (grad_core_norm * static_cast<float>(dim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
        core_rstd * inv_dim;
    grad_gate[base + tid] =
        (grad_gate_norm * static_cast<float>(dim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
        gate_rstd * inv_dim;
  }
}

}  // namespace

std::vector<torch::Tensor> input_grad_only_gated_tail_backward(
    const torch::Tensor &grad_out,
    const torch::Tensor &core,
    const torch::Tensor &gate,
    const torch::Tensor &core_weight,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_weight,
    const torch::Tensor &gate_bias,
    double eps) {
  TORCH_CHECK(grad_out.is_cuda() && core.is_cuda() && gate.is_cuda(), "gated_tail_backward: tensors must be CUDA");
  TORCH_CHECK(core_weight.is_cuda() && core_bias.is_cuda() && gate_weight.is_cuda() && gate_bias.is_cuda(),
              "gated_tail_backward: norm tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 && core.scalar_type() == torch::kFloat32 &&
                  gate.scalar_type() == torch::kFloat32 && core_weight.scalar_type() == torch::kFloat32 &&
                  core_bias.scalar_type() == torch::kFloat32 && gate_weight.scalar_type() == torch::kFloat32 &&
                  gate_bias.scalar_type() == torch::kFloat32,
              "gated_tail_backward: all tensors must be float32");
  TORCH_CHECK(core.dim() == 2 && gate.sizes() == core.sizes() && grad_out.sizes() == core.sizes(),
              "gated_tail_backward: core/gate/grad_out must have same 2D shape");
  int64_t dim64 = core.size(1);
  TORCH_CHECK(dim64 == 128 || dim64 == 256, "gated_tail_backward: dim must be 128 or 256");
  TORCH_CHECK(core_weight.numel() == dim64 && core_bias.numel() == dim64 &&
                  gate_weight.numel() == dim64 && gate_bias.numel() == dim64,
              "gated_tail_backward: norm tensors must match feature dim");

  auto grad_out_c = grad_out.contiguous();
  auto core_c = core.contiguous();
  auto gate_c = gate.contiguous();
  auto core_weight_c = core_weight.contiguous();
  auto core_bias_c = core_bias.contiguous();
  auto gate_weight_c = gate_weight.contiguous();
  auto gate_bias_c = gate_bias.contiguous();
  auto grad_core = torch::empty_like(core_c);
  auto grad_gate = torch::empty_like(gate_c);

  int rows = static_cast<int>(core_c.size(0));
  int dim = static_cast<int>(dim64);
  input_grad_only_gated_tail_backward_kernel<<<rows, kThreads>>>(
      grad_out_c.data_ptr<float>(),
      core_c.data_ptr<float>(),
      gate_c.data_ptr<float>(),
      core_weight_c.data_ptr<float>(),
      core_bias_c.data_ptr<float>(),
      gate_weight_c.data_ptr<float>(),
      gate_bias_c.data_ptr<float>(),
      grad_core.data_ptr<float>(),
      grad_gate.data_ptr<float>(),
      rows,
      dim,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core, grad_gate};
}

std::vector<torch::Tensor> input_grad_only_gated_tail_backward_n128_v2(
    const torch::Tensor &grad_out,
    const torch::Tensor &core,
    const torch::Tensor &gate,
    const torch::Tensor &core_weight,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_weight,
    const torch::Tensor &gate_bias,
    double eps) {
  TORCH_CHECK(grad_out.is_cuda() && core.is_cuda() && gate.is_cuda(),
              "gated_tail_backward_n128_v2: tensors must be CUDA");
  TORCH_CHECK(core_weight.is_cuda() && core_bias.is_cuda() && gate_weight.is_cuda() && gate_bias.is_cuda(),
              "gated_tail_backward_n128_v2: norm tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 && core.scalar_type() == torch::kFloat32 &&
                  gate.scalar_type() == torch::kFloat32 && core_weight.scalar_type() == torch::kFloat32 &&
                  core_bias.scalar_type() == torch::kFloat32 && gate_weight.scalar_type() == torch::kFloat32 &&
                  gate_bias.scalar_type() == torch::kFloat32,
              "gated_tail_backward_n128_v2: all tensors must be float32");
  TORCH_CHECK(core.dim() == 2 && gate.sizes() == core.sizes() && grad_out.sizes() == core.sizes(),
              "gated_tail_backward_n128_v2: core/gate/grad_out must have same 2D shape");
  TORCH_CHECK(core.size(1) == kDimN128, "gated_tail_backward_n128_v2: dim must be 128");
  TORCH_CHECK(core_weight.numel() == kDimN128 && core_bias.numel() == kDimN128 &&
                  gate_weight.numel() == kDimN128 && gate_bias.numel() == kDimN128,
              "gated_tail_backward_n128_v2: norm tensors must match feature dim");

  auto grad_out_c = grad_out.contiguous();
  auto core_c = core.contiguous();
  auto gate_c = gate.contiguous();
  auto core_weight_c = core_weight.contiguous();
  auto core_bias_c = core_bias.contiguous();
  auto gate_weight_c = gate_weight.contiguous();
  auto gate_bias_c = gate_bias.contiguous();
  auto grad_core = torch::empty_like(core_c);
  auto grad_gate = torch::empty_like(gate_c);

  int rows = static_cast<int>(core_c.size(0));
  input_grad_only_gated_tail_backward_n128_v2_kernel<<<rows, kThreadsN128>>>(
      grad_out_c.data_ptr<float>(),
      core_c.data_ptr<float>(),
      gate_c.data_ptr<float>(),
      core_weight_c.data_ptr<float>(),
      core_bias_c.data_ptr<float>(),
      gate_weight_c.data_ptr<float>(),
      gate_bias_c.data_ptr<float>(),
      grad_core.data_ptr<float>(),
      grad_gate.data_ptr<float>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core, grad_gate};
}

std::vector<torch::Tensor> gated_tail_second_silu_input_grad_macro(
    const torch::Tensor &grad_out,
    const torch::Tensor &core,
    const torch::Tensor &gate,
    const torch::Tensor &core_weight,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_weight,
    const torch::Tensor &gate_bias,
    double eps,
    const torch::Tensor &core_second_weight,
    const torch::Tensor &gate_second_weight,
    const torch::Tensor &core_first_hidden,
    const torch::Tensor &gate_first_hidden,
    bool use_tail_bwd_v2) {
  TORCH_CHECK(grad_out.is_cuda() && core.is_cuda() && gate.is_cuda(),
              "gated_tail_second_silu_input_grad_macro: tail tensors must be CUDA");
  TORCH_CHECK(core_weight.is_cuda() && core_bias.is_cuda() && gate_weight.is_cuda() && gate_bias.is_cuda(),
              "gated_tail_second_silu_input_grad_macro: norm tensors must be CUDA");
  TORCH_CHECK(core_second_weight.is_cuda() && gate_second_weight.is_cuda() &&
                  core_first_hidden.is_cuda() && gate_first_hidden.is_cuda(),
              "gated_tail_second_silu_input_grad_macro: second weights and hidden tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 && core.scalar_type() == torch::kFloat32 &&
                  gate.scalar_type() == torch::kFloat32 && core_weight.scalar_type() == torch::kFloat32 &&
                  core_bias.scalar_type() == torch::kFloat32 && gate_weight.scalar_type() == torch::kFloat32 &&
                  gate_bias.scalar_type() == torch::kFloat32 &&
                  core_second_weight.scalar_type() == torch::kFloat32 &&
                  gate_second_weight.scalar_type() == torch::kFloat32 &&
                  core_first_hidden.scalar_type() == torch::kFloat32 &&
                  gate_first_hidden.scalar_type() == torch::kFloat32,
              "gated_tail_second_silu_input_grad_macro: all tensors must be float32");
  TORCH_CHECK(core.dim() == 2 && gate.sizes() == core.sizes() && grad_out.sizes() == core.sizes(),
              "gated_tail_second_silu_input_grad_macro: core/gate/grad_out must have same 2D shape");
  TORCH_CHECK(core_first_hidden.sizes() == core.sizes() && gate_first_hidden.sizes() == core.sizes(),
              "gated_tail_second_silu_input_grad_macro: hidden tensors must match tail shape");
  const int64_t rows = core.size(0);
  const int64_t dim64 = core.size(1);
  TORCH_CHECK(dim64 == 128 || dim64 == 256,
              "gated_tail_second_silu_input_grad_macro: dim must be 128 or 256");
  TORCH_CHECK(!use_tail_bwd_v2 || dim64 == kDimN128,
              "gated_tail_second_silu_input_grad_macro: v2 tail path only supports dim 128");
  TORCH_CHECK(core_second_weight.sizes() == torch::IntArrayRef({dim64, dim64}) &&
                  gate_second_weight.sizes() == torch::IntArrayRef({dim64, dim64}),
              "gated_tail_second_silu_input_grad_macro: second weights must be [dim, dim]");

  std::vector<torch::Tensor> grad_tail;
  if (use_tail_bwd_v2) {
    grad_tail = input_grad_only_gated_tail_backward_n128_v2(
        grad_out,
        core,
        gate,
        core_weight,
        core_bias,
        gate_weight,
        gate_bias,
        eps);
  } else {
    grad_tail = input_grad_only_gated_tail_backward(
        grad_out,
        core,
        gate,
        core_weight,
        core_bias,
        gate_weight,
        gate_bias,
        eps);
  }
  TORCH_CHECK(grad_tail.size() == 2,
              "gated_tail_second_silu_input_grad_macro: tail backward returned wrong arity");

  auto grad_core = grad_tail[0].contiguous();
  auto grad_gate = grad_tail[1].contiguous();
  auto core_second_weight_c = core_second_weight.contiguous();
  auto gate_second_weight_c = gate_second_weight.contiguous();
  auto core_first_hidden_c = core_first_hidden.contiguous();
  auto gate_first_hidden_c = gate_first_hidden.contiguous();

  auto grad_core_linear = torch::empty_like(grad_core);
  auto grad_gate_linear = torch::empty_like(grad_gate);
  auto grad_core_hidden = torch::empty_like(grad_core);
  auto grad_gate_hidden = torch::empty_like(grad_gate);
  if (rows == 0) {
    return {grad_core_hidden, grad_gate_hidden};
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublas_row_major_matmul(
      handle,
      grad_core.data_ptr<float>(),
      core_second_weight_c.data_ptr<float>(),
      grad_core_linear.data_ptr<float>(),
      rows,
      dim64,
      dim64,
      dim64,
      dim64,
      dim64,
      "gated_tail_second_silu_input_grad_macro core second input-grad");
  cublas_row_major_matmul(
      handle,
      grad_gate.data_ptr<float>(),
      gate_second_weight_c.data_ptr<float>(),
      grad_gate_linear.data_ptr<float>(),
      rows,
      dim64,
      dim64,
      dim64,
      dim64,
      dim64,
      "gated_tail_second_silu_input_grad_macro gate second input-grad");

  const int threads = 256;
  const int64_t numel = rows * dim64;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  silu_input_grad_epilogue_kernel<<<blocks, threads>>>(
      grad_core_linear.data_ptr<float>(),
      grad_gate_linear.data_ptr<float>(),
      core_first_hidden_c.data_ptr<float>(),
      gate_first_hidden_c.data_ptr<float>(),
      grad_core_hidden.data_ptr<float>(),
      grad_gate_hidden.data_ptr<float>(),
      numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core_hidden, grad_gate_hidden};
}

std::vector<torch::Tensor> gated_tail_second_silu_residual_input_grad_macro(
    const torch::Tensor &grad_out,
    const torch::Tensor &core,
    const torch::Tensor &gate,
    const torch::Tensor &core_weight,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_weight,
    const torch::Tensor &gate_bias,
    double eps,
    const torch::Tensor &core_second_weight,
    const torch::Tensor &gate_second_weight,
    const torch::Tensor &core_first_hidden,
    const torch::Tensor &gate_first_hidden,
    bool use_tail_bwd_v2,
    const torch::Tensor &old_feat,
    const torch::Tensor &res_weight) {
  TORCH_CHECK(old_feat.is_cuda() && res_weight.is_cuda(),
              "gated_tail_second_silu_residual_input_grad_macro: residual tensors must be CUDA");
  TORCH_CHECK(old_feat.scalar_type() == torch::kFloat32 && res_weight.scalar_type() == torch::kFloat32,
              "gated_tail_second_silu_residual_input_grad_macro: residual tensors must be float32");
  TORCH_CHECK(old_feat.sizes() == grad_out.sizes(),
              "gated_tail_second_silu_residual_input_grad_macro: old_feat must match grad_out");
  TORCH_CHECK(res_weight.dim() == 2 && res_weight.size(0) == 1 && res_weight.size(1) == grad_out.size(1),
              "gated_tail_second_silu_residual_input_grad_macro: res_weight must be [1, dim]");

  auto grads = gated_tail_second_silu_input_grad_macro(
      grad_out,
      core,
      gate,
      core_weight,
      core_bias,
      gate_weight,
      gate_bias,
      eps,
      core_second_weight,
      gate_second_weight,
      core_first_hidden,
      gate_first_hidden,
      use_tail_bwd_v2);
  TORCH_CHECK(grads.size() == 2,
              "gated_tail_second_silu_residual_input_grad_macro: base macro returned wrong arity");

  auto grad_out_c = grad_out.contiguous();
  auto res_weight_c = res_weight.contiguous();
  auto grad_old = torch::empty_like(old_feat);
  const int64_t rows = grad_out_c.size(0);
  const int64_t dim64 = grad_out_c.size(1);
  const int64_t numel = rows * dim64;
  if (numel > 0) {
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    residual_old_grad_kernel<<<blocks, threads>>>(
        grad_out_c.data_ptr<float>(),
        res_weight_c.data_ptr<float>(),
        grad_old.data_ptr<float>(),
        numel,
        dim64);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grads[0], grads[1], grad_old, torch::Tensor()};
}

torch::Tensor input_grad_only_gated_tail_backward_stack(
    const torch::Tensor &grad_out,
    const torch::Tensor &core,
    const torch::Tensor &gate,
    const torch::Tensor &core_weight,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_weight,
    const torch::Tensor &gate_bias,
    double eps) {
  TORCH_CHECK(grad_out.is_cuda() && core.is_cuda() && gate.is_cuda(), "gated_tail_backward_stack: tensors must be CUDA");
  TORCH_CHECK(core_weight.is_cuda() && core_bias.is_cuda() && gate_weight.is_cuda() && gate_bias.is_cuda(),
              "gated_tail_backward_stack: norm tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 && core.scalar_type() == torch::kFloat32 &&
                  gate.scalar_type() == torch::kFloat32 && core_weight.scalar_type() == torch::kFloat32 &&
                  core_bias.scalar_type() == torch::kFloat32 && gate_weight.scalar_type() == torch::kFloat32 &&
                  gate_bias.scalar_type() == torch::kFloat32,
              "gated_tail_backward_stack: all tensors must be float32");
  TORCH_CHECK(core.dim() == 2 && gate.sizes() == core.sizes() && grad_out.sizes() == core.sizes(),
              "gated_tail_backward_stack: core/gate/grad_out must have same 2D shape");
  int64_t dim64 = core.size(1);
  TORCH_CHECK(dim64 == 128 || dim64 == 256, "gated_tail_backward_stack: dim must be 128 or 256");
  TORCH_CHECK(core_weight.numel() == dim64 && core_bias.numel() == dim64 &&
                  gate_weight.numel() == dim64 && gate_bias.numel() == dim64,
              "gated_tail_backward_stack: norm tensors must match feature dim");

  auto grad_out_c = grad_out.contiguous();
  auto core_c = core.contiguous();
  auto gate_c = gate.contiguous();
  auto core_weight_c = core_weight.contiguous();
  auto core_bias_c = core_bias.contiguous();
  auto gate_weight_c = gate_weight.contiguous();
  auto gate_bias_c = gate_bias.contiguous();

  int rows = static_cast<int>(core_c.size(0));
  int dim = static_cast<int>(dim64);
  auto grad_stack = torch::empty({2, rows, dim}, core_c.options());
  input_grad_only_gated_tail_backward_stack_kernel<<<rows, kThreads>>>(
      grad_out_c.data_ptr<float>(),
      core_c.data_ptr<float>(),
      gate_c.data_ptr<float>(),
      core_weight_c.data_ptr<float>(),
      core_bias_c.data_ptr<float>(),
      gate_weight_c.data_ptr<float>(),
      gate_bias_c.data_ptr<float>(),
      grad_stack.data_ptr<float>(),
      rows,
      dim,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_stack;
}

std::vector<torch::Tensor> param_grad_gated_tail_backward(
    const torch::Tensor &grad_out,
    const torch::Tensor &core,
    const torch::Tensor &gate,
    const torch::Tensor &core_weight,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_weight,
    const torch::Tensor &gate_bias,
    double eps) {
  TORCH_CHECK(grad_out.is_cuda() && core.is_cuda() && gate.is_cuda(), "gated_tail_param_backward: tensors must be CUDA");
  TORCH_CHECK(core_weight.is_cuda() && core_bias.is_cuda() && gate_weight.is_cuda() && gate_bias.is_cuda(),
              "gated_tail_param_backward: norm tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 && core.scalar_type() == torch::kFloat32 &&
                  gate.scalar_type() == torch::kFloat32 && core_weight.scalar_type() == torch::kFloat32 &&
                  core_bias.scalar_type() == torch::kFloat32 && gate_weight.scalar_type() == torch::kFloat32 &&
                  gate_bias.scalar_type() == torch::kFloat32,
              "gated_tail_param_backward: all tensors must be float32");
  TORCH_CHECK(core.dim() == 2 && gate.sizes() == core.sizes() && grad_out.sizes() == core.sizes(),
              "gated_tail_param_backward: core/gate/grad_out must have same 2D shape");
  int64_t dim64 = core.size(1);
  TORCH_CHECK(dim64 == 128 || dim64 == 256, "gated_tail_param_backward: dim must be 128 or 256");
  TORCH_CHECK(core_weight.numel() == dim64 && core_bias.numel() == dim64 &&
                  gate_weight.numel() == dim64 && gate_bias.numel() == dim64,
              "gated_tail_param_backward: norm tensors must match feature dim");

  auto grad_out_c = grad_out.contiguous();
  auto core_c = core.contiguous();
  auto gate_c = gate.contiguous();
  auto core_weight_c = core_weight.contiguous();
  auto core_bias_c = core_bias.contiguous();
  auto gate_weight_c = gate_weight.contiguous();
  auto gate_bias_c = gate_bias.contiguous();
  auto grad_core = torch::empty_like(core_c);
  auto grad_gate = torch::empty_like(gate_c);
  auto grad_core_weight = torch::zeros_like(core_weight_c);
  auto grad_core_bias = torch::zeros_like(core_bias_c);
  auto grad_gate_weight = torch::zeros_like(gate_weight_c);
  auto grad_gate_bias = torch::zeros_like(gate_bias_c);

  int rows = static_cast<int>(core_c.size(0));
  int dim = static_cast<int>(dim64);
  param_grad_gated_tail_backward_kernel<<<rows, kThreads>>>(
      grad_out_c.data_ptr<float>(),
      core_c.data_ptr<float>(),
      gate_c.data_ptr<float>(),
      core_weight_c.data_ptr<float>(),
      core_bias_c.data_ptr<float>(),
      gate_weight_c.data_ptr<float>(),
      gate_bias_c.data_ptr<float>(),
      grad_core.data_ptr<float>(),
      grad_gate.data_ptr<float>(),
      grad_core_weight.data_ptr<float>(),
      grad_core_bias.data_ptr<float>(),
      grad_gate_weight.data_ptr<float>(),
      grad_gate_bias.data_ptr<float>(),
      rows,
      dim,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core, grad_gate, grad_core_weight, grad_core_bias, grad_gate_weight, grad_gate_bias};
}
