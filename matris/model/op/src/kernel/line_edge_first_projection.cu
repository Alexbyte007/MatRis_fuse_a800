#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <array>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kDim = 128;
constexpr int kP44Threads = 256;
constexpr int kProjectTileM = 16;
constexpr int kProjectTileN = 16;
constexpr int kProjectTileK = 16;
constexpr int kProjectTile32M = 32;
constexpr int kProjectTile32N = 32;
constexpr int kProjectTile32K = 32;

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
    float beta,
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

bool use_p108_grouped_alpha_gemm() {
  const char* value = std::getenv("MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_GROUPED_ALPHA");
  return value != nullptr && value[0] == '1';
}

bool use_p108_split_edge_gemm() {
  const char* value = std::getenv("MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SPLIT_EDGE");
  return value != nullptr && value[0] == '1';
}

bool use_p108_inplace_hidden_gemm() {
  const char* value = std::getenv("MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_INPLACE_HIDDEN");
  return value != nullptr && value[0] == '1';
}

void cublas_row_major_grouped_pair_matmul(
    cublasHandle_t handle,
    const float* a0,
    const float* b0,
    float* c0,
    const float* a1,
    const float* b1,
    float* c1,
    int64_t rows,
    int64_t k,
    int64_t cols,
    int64_t a_row_stride,
    int64_t b_row_stride,
    int64_t c_row_stride,
    const torch::Tensor& anchor,
    const char* label) {
  TORCH_CHECK(rows <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  k <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  cols <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  a_row_stride <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  b_row_stride <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                  c_row_stride <= static_cast<int64_t>(std::numeric_limits<int>::max()),
              label, ": dimensions exceed cuBLAS int range");

  const cublasOperation_t transa_array[1] = {CUBLAS_OP_N};
  const cublasOperation_t transb_array[1] = {CUBLAS_OP_N};
  const int m_array[1] = {static_cast<int>(cols)};
  const int n_array[1] = {static_cast<int>(rows)};
  const int k_array[1] = {static_cast<int>(k)};
  const float alpha_array[1] = {1.0f};
  const float beta_array[1] = {0.0f};
  const int lda_array[1] = {static_cast<int>(b_row_stride)};
  const int ldb_array[1] = {static_cast<int>(a_row_stride)};
  const int ldc_array[1] = {static_cast<int>(c_row_stride)};
  const int group_size[1] = {2};
  const std::array<const float*, 2> h_Aarray = {b0, b1};
  const std::array<const float*, 2> h_Barray = {a0, a1};
  const std::array<float*, 2> h_Carray = {c0, c1};

  auto ptr_options = torch::TensorOptions().dtype(torch::kInt64).device(anchor.device());
  auto d_Aarray_storage = torch::empty({2}, ptr_options);
  auto d_Barray_storage = torch::empty({2}, ptr_options);
  auto d_Carray_storage = torch::empty({2}, ptr_options);
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_Aarray_storage.data_ptr<int64_t>(),
      h_Aarray.data(),
      sizeof(const float*) * h_Aarray.size(),
      cudaMemcpyHostToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_Barray_storage.data_ptr<int64_t>(),
      h_Barray.data(),
      sizeof(const float*) * h_Barray.size(),
      cudaMemcpyHostToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_Carray_storage.data_ptr<int64_t>(),
      h_Carray.data(),
      sizeof(float*) * h_Carray.size(),
      cudaMemcpyHostToDevice,
      stream.stream()));

  const float* const* Aarray = reinterpret_cast<const float* const*>(d_Aarray_storage.data_ptr<int64_t>());
  const float* const* Barray = reinterpret_cast<const float* const*>(d_Barray_storage.data_ptr<int64_t>());
  float* const* Carray = reinterpret_cast<float* const*>(d_Carray_storage.data_ptr<int64_t>());
  cublasStatus_t status = cublasSgemmGroupedBatched(
      handle,
      transa_array,
      transb_array,
      m_array,
      n_array,
      k_array,
      alpha_array,
      Aarray,
      lda_array,
      Barray,
      ldb_array,
      beta_array,
      Carray,
      ldc_array,
      1,
      group_size);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, label, ": cublasSgemmGroupedBatched failed");
}

__device__ __forceinline__ float sigmoidf_stable_line_edge(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void line_edge_fill_silu_grad_hidden_kernel(
    const float* __restrict__ grad_core,
    const float* __restrict__ grad_gate,
    const float* __restrict__ core_projected,
    const float* __restrict__ gate_projected,
    float* __restrict__ grad_hidden,
    int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int dim = static_cast<int>(threadIdx.x);
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t base = row * kDim;
  const float core_x = core_projected[base + dim];
  const float core_sig = sigmoidf_stable_line_edge(core_x);
  const float core_silu_grad = core_sig * (1.0f + core_x * (1.0f - core_sig));
  const float gate_x = gate_projected[base + dim];
  const float gate_sig = sigmoidf_stable_line_edge(gate_x);
  const float gate_silu_grad = gate_sig * (1.0f + gate_x * (1.0f - gate_sig));
  const int64_t out_base = row * 2 * kDim;
  grad_hidden[out_base + dim] = grad_core[base + dim] * core_silu_grad;
  grad_hidden[out_base + kDim + dim] = grad_gate[base + dim] * gate_silu_grad;
}

__global__ void line_edge_apply_silu_grad_inplace_kernel(
    float* __restrict__ grad_core,
    float* __restrict__ grad_gate,
    const float* __restrict__ core_projected,
    const float* __restrict__ gate_projected,
    int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int dim = static_cast<int>(threadIdx.x);
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t base = row * kDim;
  const float core_x = core_projected[base + dim];
  const float core_sig = sigmoidf_stable_line_edge(core_x);
  const float core_silu_grad = core_sig * (1.0f + core_x * (1.0f - core_sig));
  const float gate_x = gate_projected[base + dim];
  const float gate_sig = sigmoidf_stable_line_edge(gate_x);
  const float gate_silu_grad = gate_sig * (1.0f + gate_x * (1.0f - gate_sig));
  grad_core[base + dim] *= core_silu_grad;
  grad_gate[base + dim] *= gate_silu_grad;
}

__global__ void line_edge_add_alpha_to_grad_cat_kernel(
    const float* __restrict__ grad_source_alpha,
    const float* __restrict__ grad_target_alpha,
    float* __restrict__ grad_cat,
    int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int dim = static_cast<int>(threadIdx.x);
  if (row >= rows || dim >= kDim) {
    return;
  }
  grad_cat[row * 3 * kDim + dim] += grad_source_alpha[row * kDim + dim] + grad_target_alpha[row * kDim + dim];
}

__global__ void line_edge_gather_cat_forward_kernel(
    const float* __restrict__ node_feat,
    const float* __restrict__ edge_feat,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ out,
    int64_t rows) {
  int64_t row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  int64_t source = source_index[row];
  int64_t target = target_index[row];
  out[row * 3 * kDim + dim] = edge_feat[row * kDim + dim];
  out[row * 3 * kDim + kDim + dim] = node_feat[target * kDim + dim];
  out[row * 3 * kDim + 2 * kDim + dim] = node_feat[source * kDim + dim];
}

__global__ void line_edge_cat_grad_scatter_backward_kernel(
    const float* __restrict__ grad_cat,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows) {
  int64_t row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  int64_t source = source_index[row];
  int64_t target = target_index[row];
  const int64_t base = row * 3 * kDim;
  grad_edge[row * kDim + dim] = grad_cat[base + dim];
  atomicAdd(&grad_node[target * kDim + dim], grad_cat[base + kDim + dim]);
  atomicAdd(&grad_node[source * kDim + dim], grad_cat[base + 2 * kDim + dim]);
}

__global__ void line_edge_node_pair_grad_scatter_backward_kernel(
    const float* __restrict__ grad_node_pair,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    int64_t rows) {
  int64_t row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  int64_t source = source_index[row];
  int64_t target = target_index[row];
  const int64_t base = row * 2 * kDim;
  atomicAdd(&grad_node[target * kDim + dim], grad_node_pair[base + dim]);
  atomicAdd(&grad_node[source * kDim + dim], grad_node_pair[base + kDim + dim]);
}

__global__ void line_node_triple_cat_forward_kernel(
    const float* __restrict__ node_feat,
    const float* __restrict__ target_feat,
    const float* __restrict__ source_feat,
    float* __restrict__ out,
    int64_t rows) {
  int64_t row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t out_base = row * 3 * kDim;
  out[out_base + dim] = node_feat[row * kDim + dim];
  out[out_base + kDim + dim] = target_feat[row * kDim + dim];
  out[out_base + 2 * kDim + dim] = source_feat[row * kDim + dim];
}

__global__ void line_node_triple_cat_backward_kernel(
    const float* __restrict__ grad_cat,
    float* __restrict__ grad_node,
    float* __restrict__ grad_target,
    float* __restrict__ grad_source,
    int64_t rows) {
  int64_t row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t base = row * 3 * kDim;
  grad_node[row * kDim + dim] = grad_cat[base + dim];
  grad_target[row * kDim + dim] = grad_cat[base + kDim + dim];
  grad_source[row * kDim + dim] = grad_cat[base + 2 * kDim + dim];
}

__global__ void directed_edge_gather_cat_forward_kernel(
    const float* __restrict__ node_feat,
    const float* __restrict__ edge_feat,
    const int64_t* __restrict__ edge_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ out,
    int64_t rows) {
  int64_t row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t edge = edge_index[row];
  const int64_t source = source_index[row];
  const int64_t target = target_index[row];
  out[row * 3 * kDim + dim] = edge_feat[edge * kDim + dim];
  out[row * 3 * kDim + kDim + dim] = node_feat[target * kDim + dim];
  out[row * 3 * kDim + 2 * kDim + dim] = node_feat[source * kDim + dim];
}

__global__ void directed_edge_cat_grad_scatter_backward_kernel(
    const float* __restrict__ grad_cat,
    const int64_t* __restrict__ edge_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows) {
  int64_t row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t edge = edge_index[row];
  const int64_t source = source_index[row];
  const int64_t target = target_index[row];
  const int64_t base = row * 3 * kDim;
  atomicAdd(&grad_edge[edge * kDim + dim], grad_cat[base + dim]);
  atomicAdd(&grad_node[target * kDim + dim], grad_cat[base + kDim + dim]);
  atomicAdd(&grad_node[source * kDim + dim], grad_cat[base + 2 * kDim + dim]);
}

__global__ void refine_line_edge_gather_cat_forward_kernel(
    const float* __restrict__ node_feat,
    const float* __restrict__ edge_feat,
    const float* __restrict__ atom_feat,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ out,
    int64_t rows) {
  int64_t row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t atom = atom_index[row];
  const int64_t source = source_index[row];
  const int64_t target = target_index[row];
  out[row * 4 * kDim + dim] = edge_feat[row * kDim + dim];
  out[row * 4 * kDim + kDim + dim] = atom_feat[atom * kDim + dim];
  out[row * 4 * kDim + 2 * kDim + dim] = node_feat[target * kDim + dim];
  out[row * 4 * kDim + 3 * kDim + dim] = node_feat[source * kDim + dim];
}

__global__ void refine_line_edge_cat_grad_scatter_backward_kernel(
    const float* __restrict__ grad_cat,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom,
    int64_t rows) {
  int64_t row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t atom = atom_index[row];
  const int64_t source = source_index[row];
  const int64_t target = target_index[row];
  const int64_t base = row * 4 * kDim;
  grad_edge[row * kDim + dim] = grad_cat[base + dim];
  atomicAdd(&grad_atom[atom * kDim + dim], grad_cat[base + kDim + dim]);
  atomicAdd(&grad_node[target * kDim + dim], grad_cat[base + 2 * kDim + dim]);
  atomicAdd(&grad_node[source * kDim + dim], grad_cat[base + 3 * kDim + dim]);
}

__device__ __forceinline__ void store_refine_line_project_scatter_value(
    float value,
    int64_t row,
    int col,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom) {
  if (col < kDim) {
    grad_edge[row * kDim + col] = value;
  } else if (col < 2 * kDim) {
    atomicAdd(&grad_atom[atom_index[row] * kDim + (col - kDim)], value);
  } else if (col < 3 * kDim) {
    atomicAdd(&grad_node[target_index[row] * kDim + (col - 2 * kDim)], value);
  } else {
    atomicAdd(&grad_node[source_index[row] * kDim + (col - 3 * kDim)], value);
  }
}

__global__ void refine_line_project_grad_scatter_backward_tile32_kernel(
    const float* __restrict__ grad_projected,
    const float* __restrict__ weight,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom,
    int64_t rows,
    int64_t hidden) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  for (int64_t k0 = 0; k0 < hidden; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int64_t global_k = k0 + k;
      s_grad[r][k] = (global_row < rows && global_k < hidden)
                         ? grad_projected[global_row * hidden + global_k]
                         : 0.0f;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int64_t global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < hidden && global_col < 4 * kDim)
                           ? weight[global_k * 4 * kDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }

  if (row0 < rows && col0 < 4 * kDim) {
    store_refine_line_project_scatter_value(
        acc00, row0, col0, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
  if (row0 < rows && col0 + 1 < 4 * kDim) {
    store_refine_line_project_scatter_value(
        acc01, row0, col0 + 1, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
  if (row0 + 1 < rows && col0 < 4 * kDim) {
    store_refine_line_project_scatter_value(
        acc10, row0 + 1, col0, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
  if (row0 + 1 < rows && col0 + 1 < 4 * kDim) {
    store_refine_line_project_scatter_value(
        acc11, row0 + 1, col0 + 1, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
}

__device__ __forceinline__ void store_refine_line_project_scatter_add_value(
    float value,
    int64_t row,
    int col,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ grad_edge_in,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom,
    int64_t rows) {
  if (row >= rows || col >= 4 * kDim) {
    return;
  }
  if (col < kDim) {
    const int64_t offset = row * kDim + col;
    grad_edge[offset] = grad_edge_in[offset] + value;
  } else if (col < 2 * kDim) {
    atomicAdd(&grad_atom[atom_index[row] * kDim + (col - kDim)], value);
  } else if (col < 3 * kDim) {
    atomicAdd(&grad_node[target_index[row] * kDim + (col - 2 * kDim)], value);
  } else {
    atomicAdd(&grad_node[source_index[row] * kDim + (col - 3 * kDim)], value);
  }
}

__global__ void refine_line_project_dual_grad_scatter_add_tile32_kernel(
    const float* __restrict__ grad_core,
    const float* __restrict__ grad_gate,
    const float* __restrict__ weight,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ grad_node_in,
    const float* __restrict__ grad_edge_in,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom,
    int64_t rows) {
  __shared__ float s_core[kProjectTile32M][kDim];
  __shared__ float s_gate[kProjectTile32M][kDim];
  __shared__ float s_weight_core[kDim][kProjectTile32N + 1];
  __shared__ float s_weight_gate[kDim][kProjectTile32N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  for (int idx = tid; idx < kProjectTile32M * kDim; idx += blockDim.x * blockDim.y) {
    const int r = idx / kDim;
    const int k = idx - r * kDim;
    const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
    s_core[r][k] = global_row < rows ? grad_core[global_row * kDim + k] : 0.0f;
    s_gate[r][k] = global_row < rows ? grad_gate[global_row * kDim + k] : 0.0f;
  }
  for (int idx = tid; idx < kDim * kProjectTile32N; idx += blockDim.x * blockDim.y) {
    const int k = idx / kProjectTile32N;
    const int c = idx - k * kProjectTile32N;
    const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
    s_weight_core[k][c] = global_col < 4 * kDim ? weight[k * 4 * kDim + global_col] : 0.0f;
    s_weight_gate[k][c] = global_col < 4 * kDim ? weight[(kDim + k) * 4 * kDim + global_col] : 0.0f;
  }
  __syncthreads();

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;
  #pragma unroll
  for (int kk = 0; kk < kDim; ++kk) {
    const float c0 = s_core[ty * 2][kk];
    const float g0 = s_gate[ty * 2][kk];
    const float c1 = s_core[ty * 2 + 1][kk];
    const float g1 = s_gate[ty * 2 + 1][kk];
    const float wc0 = s_weight_core[kk][tx * 2];
    const float wc1 = s_weight_core[kk][tx * 2 + 1];
    const float wg0 = s_weight_gate[kk][tx * 2];
    const float wg1 = s_weight_gate[kk][tx * 2 + 1];
    acc00 += c0 * wc0 + g0 * wg0;
    acc01 += c0 * wc1 + g0 * wg1;
    acc10 += c1 * wc0 + g1 * wg0;
    acc11 += c1 * wc1 + g1 * wg1;
  }

  store_refine_line_project_scatter_add_value(
      acc00, row0, col0, atom_index, source_index, target_index, grad_edge_in,
      grad_node, grad_edge, grad_atom, rows);
  store_refine_line_project_scatter_add_value(
      acc01, row0, col0 + 1, atom_index, source_index, target_index, grad_edge_in,
      grad_node, grad_edge, grad_atom, rows);
  store_refine_line_project_scatter_add_value(
      acc10, row0 + 1, col0, atom_index, source_index, target_index, grad_edge_in,
      grad_node, grad_edge, grad_atom, rows);
  store_refine_line_project_scatter_add_value(
      acc11, row0 + 1, col0 + 1, atom_index, source_index, target_index, grad_edge_in,
      grad_node, grad_edge, grad_atom, rows);
}

__global__ void line_edge_project_grad_scatter_backward_kernel(
    const float* __restrict__ grad_projected,
    const float* __restrict__ weight,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows,
    int64_t hidden) {
  const int64_t row = blockIdx.x;
  const int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }

  float grad_edge_value = 0.0f;
  float grad_target_value = 0.0f;
  float grad_source_value = 0.0f;
  const int64_t grad_base = row * hidden;
  for (int64_t j = 0; j < hidden; ++j) {
    const float g = grad_projected[grad_base + j];
    const int64_t weight_base = j * 3 * kDim;
    grad_edge_value += g * weight[weight_base + dim];
    grad_target_value += g * weight[weight_base + kDim + dim];
    grad_source_value += g * weight[weight_base + 2 * kDim + dim];
  }

  grad_edge[row * kDim + dim] = grad_edge_value;
  atomicAdd(&grad_node[target_index[row] * kDim + dim], grad_target_value);
  atomicAdd(&grad_node[source_index[row] * kDim + dim], grad_source_value);
}

__global__ void line_edge_project_grad_scatter_backward_tiled_kernel(
    const float* __restrict__ grad_projected,
    const float* __restrict__ weight,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows,
    int64_t hidden) {
  __shared__ float s_grad[kProjectTileM][kProjectTileK];
  __shared__ float s_weight[kProjectTileK][kProjectTileN + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kProjectTileM + ty;
  const int col = static_cast<int>(blockIdx.x) * kProjectTileN + tx;

  float acc = 0.0f;
  for (int64_t k0 = 0; k0 < hidden; k0 += kProjectTileK) {
    const int64_t grad_k = k0 + tx;
    const int64_t weight_k = k0 + ty;
    s_grad[ty][tx] = (row < rows && grad_k < hidden)
                         ? grad_projected[row * hidden + grad_k]
                         : 0.0f;
    s_weight[ty][tx] = (col < 3 * kDim && weight_k < hidden)
                           ? weight[weight_k * 3 * kDim + col]
                           : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < kProjectTileK; ++kk) {
      acc += s_grad[ty][kk] * s_weight[kk][tx];
    }
    __syncthreads();
  }

  if (row >= rows || col >= 3 * kDim) {
    return;
  }
  if (col < kDim) {
    grad_edge[row * kDim + col] = acc;
  } else if (col < 2 * kDim) {
    atomicAdd(&grad_node[target_index[row] * kDim + (col - kDim)], acc);
  } else {
    atomicAdd(&grad_node[source_index[row] * kDim + (col - 2 * kDim)], acc);
  }
}

__device__ __forceinline__ void store_line_edge_project_scatter_value(
    float value,
    int64_t row,
    int col,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge) {
  if (col < kDim) {
    grad_edge[row * kDim + col] = value;
  } else if (col < 2 * kDim) {
    atomicAdd(&grad_node[target_index[row] * kDim + (col - kDim)], value);
  } else {
    atomicAdd(&grad_node[source_index[row] * kDim + (col - 2 * kDim)], value);
  }
}

__device__ __forceinline__ void store_directed_edge_project_scatter_value(
    float value,
    int64_t row,
    int col,
    const int64_t* __restrict__ edge_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge) {
  if (col < kDim) {
    atomicAdd(&grad_edge[edge_index[row] * kDim + col], value);
  } else if (col < 2 * kDim) {
    atomicAdd(&grad_node[target_index[row] * kDim + (col - kDim)], value);
  } else {
    atomicAdd(&grad_node[source_index[row] * kDim + (col - 2 * kDim)], value);
  }
}

__global__ void line_edge_project_grad_scatter_backward_tile32_kernel(
    const float* __restrict__ grad_projected,
    const float* __restrict__ weight,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows,
    int64_t hidden) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];

  const int tx = threadIdx.x;  // 0..15
  const int ty = threadIdx.y;  // 0..15
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  for (int64_t k0 = 0; k0 < hidden; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int64_t global_k = k0 + k;
      s_grad[r][k] = (global_row < rows && global_k < hidden)
                         ? grad_projected[global_row * hidden + global_k]
                         : 0.0f;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int64_t global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < hidden && global_col < 3 * kDim)
                           ? weight[global_k * 3 * kDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }

  if (row0 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc00, row0, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc01, row0, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc10, row0 + 1, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc11, row0 + 1, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
}

__global__ void line_edge_silu_project_grad_scatter_backward_tile32_kernel(
    const float* __restrict__ grad_core,
    const float* __restrict__ grad_gate,
    const float* __restrict__ core_projected,
    const float* __restrict__ gate_projected,
    const float* __restrict__ weight,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  constexpr int kHidden = 2 * kDim;
  for (int k0 = 0; k0 < kHidden; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int global_k = k0 + k;
      float value = 0.0f;
      if (global_row < rows && global_k < kHidden) {
        const int64_t base = global_row * kDim;
        if (global_k < kDim) {
          const float x = core_projected[base + global_k];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_core[base + global_k] * silu_grad;
        } else {
          const int kk = global_k - kDim;
          const float x = gate_projected[base + kk];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_gate[base + kk] * silu_grad;
        }
      }
      s_grad[r][k] = value;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < kHidden && global_col < 3 * kDim)
                           ? weight[global_k * 3 * kDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }

  if (row0 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc00, row0, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc01, row0, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc10, row0 + 1, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc11, row0 + 1, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
}

__global__ void directed_edge_silu_project_grad_scatter_backward_tile32_kernel(
    const float* __restrict__ grad_core,
    const float* __restrict__ grad_gate,
    const float* __restrict__ core_projected,
    const float* __restrict__ gate_projected,
    const float* __restrict__ weight,
    const int64_t* __restrict__ edge_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  constexpr int kHidden = 2 * kDim;
  for (int k0 = 0; k0 < kHidden; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int global_k = k0 + k;
      float value = 0.0f;
      if (global_row < rows && global_k < kHidden) {
        const int64_t base = global_row * kDim;
        if (global_k < kDim) {
          const float x = core_projected[base + global_k];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_core[base + global_k] * silu_grad;
        } else {
          const int kk = global_k - kDim;
          const float x = gate_projected[base + kk];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_gate[base + kk] * silu_grad;
        }
      }
      s_grad[r][k] = value;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < kHidden && global_col < 3 * kDim)
                           ? weight[global_k * 3 * kDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }

  if (row0 < rows && col0 < 3 * kDim) {
    store_directed_edge_project_scatter_value(
        acc00, row0, col0, edge_index, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 < rows && col0 + 1 < 3 * kDim) {
    store_directed_edge_project_scatter_value(
        acc01, row0, col0 + 1, edge_index, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 < 3 * kDim) {
    store_directed_edge_project_scatter_value(
        acc10, row0 + 1, col0, edge_index, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 + 1 < 3 * kDim) {
    store_directed_edge_project_scatter_value(
        acc11, row0 + 1, col0 + 1, edge_index, source_index, target_index, grad_node, grad_edge);
  }
}

__device__ __forceinline__ float alpha_input_grad_acc_line_edge(
    const float* __restrict__ grad_source_logits,
    const float* __restrict__ grad_target_logits,
    const float* __restrict__ source_weight,
    const float* __restrict__ target_weight,
    int64_t row,
    int col,
    int64_t rows) {
  if (row >= rows || col >= kDim) {
    return 0.0f;
  }
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kDim; ++k) {
    acc += grad_source_logits[row * kDim + k] * source_weight[k * kDim + col];
    acc += grad_target_logits[row * kDim + k] * target_weight[k * kDim + col];
  }
  return acc;
}

__global__ void line_edge_silu_project_alpha_grad_scatter_backward_tile32_kernel(
    const float* __restrict__ grad_core,
    const float* __restrict__ grad_gate,
    const float* __restrict__ core_projected,
    const float* __restrict__ gate_projected,
    const float* __restrict__ first_weight,
    const float* __restrict__ grad_source_logits,
    const float* __restrict__ grad_target_logits,
    const float* __restrict__ source_alpha_weight,
    const float* __restrict__ target_alpha_weight,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  constexpr int kHidden = 2 * kDim;
  for (int k0 = 0; k0 < kHidden; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int global_k = k0 + k;
      float value = 0.0f;
      if (global_row < rows && global_k < kHidden) {
        const int64_t base = global_row * kDim;
        if (global_k < kDim) {
          const float x = core_projected[base + global_k];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_core[base + global_k] * silu_grad;
        } else {
          const int kk = global_k - kDim;
          const float x = gate_projected[base + kk];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_gate[base + kk] * silu_grad;
        }
      }
      s_grad[r][k] = value;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < kHidden && global_col < 3 * kDim)
                           ? first_weight[global_k * 3 * kDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }

  if (col0 < kDim) {
    acc00 += alpha_input_grad_acc_line_edge(
        grad_source_logits, grad_target_logits, source_alpha_weight, target_alpha_weight, row0, col0, rows);
    acc10 += alpha_input_grad_acc_line_edge(
        grad_source_logits, grad_target_logits, source_alpha_weight, target_alpha_weight, row0 + 1, col0, rows);
  }
  if (col0 + 1 < kDim) {
    acc01 += alpha_input_grad_acc_line_edge(
        grad_source_logits, grad_target_logits, source_alpha_weight, target_alpha_weight, row0, col0 + 1, rows);
    acc11 += alpha_input_grad_acc_line_edge(
        grad_source_logits, grad_target_logits, source_alpha_weight, target_alpha_weight, row0 + 1, col0 + 1, rows);
  }

  if (row0 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc00, row0, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc01, row0, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc10, row0 + 1, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc11, row0 + 1, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
}

__global__ void line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32_kernel(
    const float* __restrict__ grad_core,
    const float* __restrict__ grad_gate,
    const float* __restrict__ core_projected,
    const float* __restrict__ gate_projected,
    const float* __restrict__ first_weight,
    const float* __restrict__ grad_source_logits,
    const float* __restrict__ grad_target_logits,
    const float* __restrict__ source_alpha_weight,
    const float* __restrict__ target_alpha_weight,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  constexpr int kHidden = 2 * kDim;
  for (int k0 = 0; k0 < kHidden; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int global_k = k0 + k;
      float value = 0.0f;
      if (global_row < rows && global_k < kHidden) {
        const int64_t base = global_row * kDim;
        if (global_k < kDim) {
          const float x = core_projected[base + global_k];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_core[base + global_k] * silu_grad;
        } else {
          const int kk = global_k - kDim;
          const float x = gate_projected[base + kk];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_gate[base + kk] * silu_grad;
        }
      }
      s_grad[r][k] = value;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < kHidden && global_col < 3 * kDim)
                           ? first_weight[global_k * 3 * kDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }

  if (static_cast<int>(blockIdx.x) * kProjectTile32N < kDim) {
    float alpha00 = 0.0f;
    float alpha01 = 0.0f;
    float alpha10 = 0.0f;
    float alpha11 = 0.0f;

    for (int k0 = 0; k0 < kDim; k0 += kProjectTile32K) {
      for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
        const int r = idx / kProjectTile32K;
        const int k = idx - r * kProjectTile32K;
        const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
        const int global_k = k0 + k;
        s_grad[r][k] = (global_row < rows && global_k < kDim)
                           ? grad_source_logits[global_row * kDim + global_k]
                           : 0.0f;
      }
      for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
        const int k = idx / kProjectTile32N;
        const int c = idx - k * kProjectTile32N;
        const int global_k = k0 + k;
        const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
        s_weight[k][c] = (global_k < kDim && global_col < kDim)
                             ? source_alpha_weight[global_k * kDim + global_col]
                             : 0.0f;
      }
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < kProjectTile32K; ++kk) {
        const float a0 = s_grad[ty * 2][kk];
        const float a1 = s_grad[ty * 2 + 1][kk];
        const float b0 = s_weight[kk][tx * 2];
        const float b1 = s_weight[kk][tx * 2 + 1];
        alpha00 += a0 * b0;
        alpha01 += a0 * b1;
        alpha10 += a1 * b0;
        alpha11 += a1 * b1;
      }
      __syncthreads();

      for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
        const int r = idx / kProjectTile32K;
        const int k = idx - r * kProjectTile32K;
        const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
        const int global_k = k0 + k;
        s_grad[r][k] = (global_row < rows && global_k < kDim)
                           ? grad_target_logits[global_row * kDim + global_k]
                           : 0.0f;
      }
      for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
        const int k = idx / kProjectTile32N;
        const int c = idx - k * kProjectTile32N;
        const int global_k = k0 + k;
        const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
        s_weight[k][c] = (global_k < kDim && global_col < kDim)
                             ? target_alpha_weight[global_k * kDim + global_col]
                             : 0.0f;
      }
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < kProjectTile32K; ++kk) {
        const float a0 = s_grad[ty * 2][kk];
        const float a1 = s_grad[ty * 2 + 1][kk];
        const float b0 = s_weight[kk][tx * 2];
        const float b1 = s_weight[kk][tx * 2 + 1];
        alpha00 += a0 * b0;
        alpha01 += a0 * b1;
        alpha10 += a1 * b0;
        alpha11 += a1 * b1;
      }
      __syncthreads();
    }

    acc00 += alpha00;
    acc01 += alpha01;
    acc10 += alpha10;
    acc11 += alpha11;
  }

  if (row0 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc00, row0, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc01, row0, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc10, row0 + 1, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc11, row0 + 1, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
}

__device__ __forceinline__ void store_line_edge_project_target_tmp_value(
    float value,
    int64_t row,
    int col,
    const int64_t* __restrict__ source_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_target_tmp) {
  if (col < kDim) {
    grad_edge[row * kDim + col] = value;
  } else if (col < 2 * kDim) {
    grad_target_tmp[row * kDim + (col - kDim)] = value;
  } else {
    atomicAdd(&grad_node[source_index[row] * kDim + (col - 2 * kDim)], value);
  }
}

__global__ void line_edge_silu_project_alpha_grad_scatter_backward_target_tmp_tile32_kernel(
    const float* __restrict__ grad_core,
    const float* __restrict__ grad_gate,
    const float* __restrict__ core_projected,
    const float* __restrict__ gate_projected,
    const float* __restrict__ first_weight,
    const float* __restrict__ grad_source_logits,
    const float* __restrict__ grad_target_logits,
    const float* __restrict__ source_alpha_weight,
    const float* __restrict__ target_alpha_weight,
    const int64_t* __restrict__ source_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_target_tmp,
    int64_t rows) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  constexpr int kHidden = 2 * kDim;
  for (int k0 = 0; k0 < kHidden; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int global_k = k0 + k;
      float value = 0.0f;
      if (global_row < rows && global_k < kHidden) {
        const int64_t base = global_row * kDim;
        if (global_k < kDim) {
          const float x = core_projected[base + global_k];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_core[base + global_k] * silu_grad;
        } else {
          const int kk = global_k - kDim;
          const float x = gate_projected[base + kk];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_gate[base + kk] * silu_grad;
        }
      }
      s_grad[r][k] = value;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < kHidden && global_col < 3 * kDim)
                           ? first_weight[global_k * 3 * kDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }

  if (col0 < kDim) {
    acc00 += alpha_input_grad_acc_line_edge(
        grad_source_logits, grad_target_logits, source_alpha_weight, target_alpha_weight, row0, col0, rows);
    acc10 += alpha_input_grad_acc_line_edge(
        grad_source_logits, grad_target_logits, source_alpha_weight, target_alpha_weight, row0 + 1, col0, rows);
  }
  if (col0 + 1 < kDim) {
    acc01 += alpha_input_grad_acc_line_edge(
        grad_source_logits, grad_target_logits, source_alpha_weight, target_alpha_weight, row0, col0 + 1, rows);
    acc11 += alpha_input_grad_acc_line_edge(
        grad_source_logits, grad_target_logits, source_alpha_weight, target_alpha_weight, row0 + 1, col0 + 1, rows);
  }

  if (row0 < rows && col0 < 3 * kDim) {
    store_line_edge_project_target_tmp_value(
        acc00, row0, col0, source_index, grad_node, grad_edge, grad_target_tmp);
  }
  if (row0 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_target_tmp_value(
        acc01, row0, col0 + 1, source_index, grad_node, grad_edge, grad_target_tmp);
  }
  if (row0 + 1 < rows && col0 < 3 * kDim) {
    store_line_edge_project_target_tmp_value(
        acc10, row0 + 1, col0, source_index, grad_node, grad_edge, grad_target_tmp);
  }
  if (row0 + 1 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_target_tmp_value(
        acc11, row0 + 1, col0 + 1, source_index, grad_node, grad_edge, grad_target_tmp);
  }
}

__global__ void line_edge_target_tmp_reduce_kernel(
    const float* __restrict__ grad_target_tmp,
    const int64_t* __restrict__ target_offsets,
    float* __restrict__ grad_node,
    int64_t node_rows) {
  const int dim = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t node = static_cast<int64_t>(blockIdx.y);
  if (node >= node_rows || dim >= kDim) {
    return;
  }
  const int64_t start = target_offsets[node];
  const int64_t end = target_offsets[node + 1];
  float acc = 0.0f;
  for (int64_t row = start; row < end; ++row) {
    acc += grad_target_tmp[row * kDim + dim];
  }
  grad_node[node * kDim + dim] += acc;
}

__global__ void line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32_kernel(
    const float* __restrict__ grad_core,
    const float* __restrict__ grad_gate,
    const float* __restrict__ core_projected,
    const float* __restrict__ gate_projected,
    const float* __restrict__ first_weight,
    const float* __restrict__ grad_source_out,
    const float* __restrict__ grad_target_out,
    const float* __restrict__ values,
    const float* __restrict__ source_out,
    const float* __restrict__ target_out,
    const float* __restrict__ source_alpha,
    const float* __restrict__ target_alpha,
    const float* __restrict__ source_alpha_weight,
    const float* __restrict__ target_alpha_weight,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];
  __shared__ float s_source_logit_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_target_logit_grad[kProjectTile32M][kProjectTile32K];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  constexpr int kHidden = 2 * kDim;
  for (int k0 = 0; k0 < kHidden; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int global_k = k0 + k;
      float value = 0.0f;
      if (global_row < rows && global_k < kHidden) {
        const int64_t base = global_row * kDim;
        if (global_k < kDim) {
          const float x = core_projected[base + global_k];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_core[base + global_k] * silu_grad;
        } else {
          const int kk = global_k - kDim;
          const float x = gate_projected[base + kk];
          const float sig = sigmoidf_stable_line_edge(x);
          const float silu_grad = sig * (1.0f + x * (1.0f - sig));
          value = grad_gate[base + kk] * silu_grad;
        }
      }
      s_grad[r][k] = value;
    }
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int global_k = k0 + k;
      float source_value = 0.0f;
      float target_value = 0.0f;
      if (global_row < rows && global_k < kDim) {
        const int64_t s = source_index[global_row];
        const int64_t t = target_index[global_row];
        const int64_t edge_idx = global_row * kDim + global_k;
        const float v = values[edge_idx];
        const float gs = grad_source_out[s * kDim + global_k];
        const float gt = grad_target_out[t * kDim + global_k];
        source_value = source_alpha[edge_idx] * gs * (v - source_out[s * kDim + global_k]);
        target_value = target_alpha[edge_idx] * gt * (v - target_out[t * kDim + global_k]);
      }
      s_source_logit_grad[r][k] = source_value;
      s_target_logit_grad[r][k] = target_value;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < kHidden && global_col < 3 * kDim)
                           ? first_weight[global_k * 3 * kDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
      const int global_k = k0 + kk;
      if (global_k < kDim && col0 < kDim) {
        const float source_w0 = source_alpha_weight[global_k * kDim + col0];
        const float target_w0 = target_alpha_weight[global_k * kDim + col0];
        acc00 += s_source_logit_grad[ty * 2][kk] * source_w0 +
                 s_target_logit_grad[ty * 2][kk] * target_w0;
        acc10 += s_source_logit_grad[ty * 2 + 1][kk] * source_w0 +
                 s_target_logit_grad[ty * 2 + 1][kk] * target_w0;
      }
      if (global_k < kDim && col0 + 1 < kDim) {
        const float source_w1 = source_alpha_weight[global_k * kDim + col0 + 1];
        const float target_w1 = target_alpha_weight[global_k * kDim + col0 + 1];
        acc01 += s_source_logit_grad[ty * 2][kk] * source_w1 +
                 s_target_logit_grad[ty * 2][kk] * target_w1;
        acc11 += s_source_logit_grad[ty * 2 + 1][kk] * source_w1 +
                 s_target_logit_grad[ty * 2 + 1][kk] * target_w1;
      }
    }
    __syncthreads();
  }

  if (row0 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc00, row0, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc01, row0, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc10, row0 + 1, col0, source_index, target_index, grad_node, grad_edge);
  }
  if (row0 + 1 < rows && col0 + 1 < 3 * kDim) {
    store_line_edge_project_scatter_value(
        acc11, row0 + 1, col0 + 1, source_index, target_index, grad_node, grad_edge);
  }
}

__global__ void line_edge_w8a8_tail_project_scatter_backward_n128_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ core_pre,
    const float* __restrict__ gate_pre,
    const float* __restrict__ core_projected,
    const float* __restrict__ gate_projected,
    const int8_t* __restrict__ core_q_weight,
    const int8_t* __restrict__ gate_q_weight,
    const float* __restrict__ core_weight_scale,
    const float* __restrict__ gate_weight_scale,
    const float* __restrict__ core_norm_weight,
    const float* __restrict__ core_norm_bias,
    const float* __restrict__ gate_norm_weight,
    const float* __restrict__ gate_norm_bias,
    const float* __restrict__ first_weight,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    int64_t rows,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core_grad[kDim];
  __shared__ float s_gate_grad[kDim];
  __shared__ float s_reduce_core[kP44Threads];
  __shared__ float s_reduce_gate[kP44Threads];

  const int64_t base = static_cast<int64_t>(row) * kDim;
  float core_v = 0.0f;
  float gate_v = 0.0f;
  if (tid < kDim) {
    core_v = core_pre[base + tid];
    gate_v = gate_pre[base + tid];
  }
  __syncthreads();

  s_reduce_core[tid] = tid < kDim ? core_v : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? gate_v : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_mean = s_reduce_core[0] * (1.0f / static_cast<float>(kDim));
  const float gate_mean = s_reduce_gate[0] * (1.0f / static_cast<float>(kDim));
  __syncthreads();

  const float core_centered = tid < kDim ? core_v - core_mean : 0.0f;
  const float gate_centered = tid < kDim ? gate_v - gate_mean : 0.0f;
  s_reduce_core[tid] = tid < kDim ? core_centered * core_centered : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? gate_centered * gate_centered : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_rstd = rsqrtf(s_reduce_core[0] * (1.0f / static_cast<float>(kDim)) + eps);
  const float gate_rstd = rsqrtf(s_reduce_gate[0] * (1.0f / static_cast<float>(kDim)) + eps);
  __syncthreads();

  const float core_xhat = core_centered * core_rstd;
  const float gate_xhat = gate_centered * gate_rstd;
  float grad_core_norm = 0.0f;
  float grad_gate_norm = 0.0f;
  if (tid < kDim) {
    const float core_ln = core_xhat * core_norm_weight[tid] + core_norm_bias[tid];
    const float gate_ln = gate_xhat * gate_norm_weight[tid] + gate_norm_bias[tid];
    const float core_sig = sigmoidf_stable_line_edge(core_ln);
    const float core_act = core_ln * core_sig;
    const float gate_act = sigmoidf_stable_line_edge(gate_ln);
    const float grad = grad_out[base + tid];
    const float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
    const float grad_core_ln = grad * gate_act * core_silu_grad;
    const float grad_gate_ln = grad * core_act * gate_act * (1.0f - gate_act);
    grad_core_norm = grad_core_ln * core_norm_weight[tid];
    grad_gate_norm = grad_gate_ln * gate_norm_weight[tid];
  }

  s_reduce_core[tid] = tid < kDim ? grad_core_norm : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? grad_gate_norm : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_sum_grad = s_reduce_core[0];
  const float gate_sum_grad = s_reduce_gate[0];
  __syncthreads();

  s_reduce_core[tid] = tid < kDim ? grad_core_norm * core_xhat : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? grad_gate_norm * gate_xhat : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_sum_grad_xhat = s_reduce_core[0];
  const float gate_sum_grad_xhat = s_reduce_gate[0];
  __syncthreads();

  if (tid < kDim) {
    const float inv_dim = 1.0f / static_cast<float>(kDim);
    const float grad_core_pre =
        (grad_core_norm * static_cast<float>(kDim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
        core_rstd * inv_dim;
    const float grad_gate_pre =
        (grad_gate_norm * static_cast<float>(kDim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
        gate_rstd * inv_dim;
    s_core_grad[tid] = grad_core_pre;
    s_gate_grad[tid] = grad_gate_pre;
  }
  __syncthreads();

  if (tid < kDim) {
    float grad_core_input = 0.0f;
    float grad_gate_input = 0.0f;
    for (int j = 0; j < kDim; ++j) {
      grad_core_input += s_core_grad[j] *
                         static_cast<float>(core_q_weight[j * kDim + tid]) *
                         core_weight_scale[j];
      grad_gate_input += s_gate_grad[j] *
                         static_cast<float>(gate_q_weight[j * kDim + tid]) *
                         gate_weight_scale[j];
    }

    float core_proj = core_projected[base + tid];
    float gate_proj = gate_projected[base + tid];
    float core_sig = sigmoidf_stable_line_edge(core_proj);
    float gate_sig = sigmoidf_stable_line_edge(gate_proj);
    float core_silu_grad = core_sig * (1.0f + core_proj * (1.0f - core_sig));
    float gate_silu_grad = gate_sig * (1.0f + gate_proj * (1.0f - gate_sig));
    s_core_grad[tid] = grad_core_input * core_silu_grad;
    s_gate_grad[tid] = grad_gate_input * gate_silu_grad;
  }
  __syncthreads();

  if (tid < kDim) {
    float grad_edge_value = 0.0f;
    float grad_target_value = 0.0f;
    float grad_source_value = 0.0f;
    for (int j = 0; j < kDim; ++j) {
      const float core_g = s_core_grad[j];
      const float gate_g = s_gate_grad[j];
      const int64_t core_weight_base = static_cast<int64_t>(j) * 3 * kDim;
      const int64_t gate_weight_base = static_cast<int64_t>(kDim + j) * 3 * kDim;
      grad_edge_value += core_g * first_weight[core_weight_base + tid] +
                         gate_g * first_weight[gate_weight_base + tid];
      grad_target_value += core_g * first_weight[core_weight_base + kDim + tid] +
                           gate_g * first_weight[gate_weight_base + kDim + tid];
      grad_source_value += core_g * first_weight[core_weight_base + 2 * kDim + tid] +
                           gate_g * first_weight[gate_weight_base + 2 * kDim + tid];
    }
    grad_edge[base + tid] = grad_edge_value;
    atomicAdd(&grad_node[target_index[row] * kDim + tid], grad_target_value);
    atomicAdd(&grad_node[source_index[row] * kDim + tid], grad_source_value);
  }
}

}  // namespace

torch::Tensor line_edge_gather_cat_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index) {
  TORCH_CHECK(node_feat.is_cuda() && edge_feat.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_gather_cat_forward: all tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 && edge_feat.scalar_type() == torch::kFloat32,
              "line_edge_gather_cat_forward: features must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_gather_cat_forward: indices must be int64");
  TORCH_CHECK(node_feat.dim() == 2 && edge_feat.dim() == 2 && node_feat.size(1) == kDim && edge_feat.size(1) == kDim,
              "line_edge_gather_cat_forward: feature tensors must be [rows, 128]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 && source_index.size(0) == edge_feat.size(0) &&
                  target_index.size(0) == edge_feat.size(0),
              "line_edge_gather_cat_forward: index sizes must match edge rows");

  auto node_c = node_feat.contiguous();
  auto edge_c = edge_feat.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto out = edge_c.new_empty({edge_c.size(0), 3 * kDim});
  int64_t rows = edge_c.size(0);
  line_edge_gather_cat_forward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      node_c.data_ptr<float>(),
      edge_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> line_edge_cat_grad_scatter_backward(
    const torch::Tensor &grad_cat,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows) {
  TORCH_CHECK(grad_cat.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_cat_grad_scatter_backward: all tensors must be CUDA");
  TORCH_CHECK(grad_cat.scalar_type() == torch::kFloat32, "line_edge_cat_grad_scatter_backward: grad_cat must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_cat_grad_scatter_backward: indices must be int64");
  TORCH_CHECK(grad_cat.dim() == 2 && grad_cat.size(1) == 3 * kDim,
              "line_edge_cat_grad_scatter_backward: grad_cat must be [rows, 384]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 && source_index.size(0) == grad_cat.size(0) &&
                  target_index.size(0) == grad_cat.size(0),
              "line_edge_cat_grad_scatter_backward: index sizes must match grad rows");
  TORCH_CHECK(node_rows >= 0, "line_edge_cat_grad_scatter_backward: node_rows must be non-negative");

  auto grad_cat_c = grad_cat.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_cat_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_cat_c.new_empty({grad_cat_c.size(0), kDim});
  int64_t rows = grad_cat_c.size(0);
  line_edge_cat_grad_scatter_backward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      grad_cat_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

torch::Tensor line_node_triple_cat_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &target_feat,
    const torch::Tensor &source_feat) {
  TORCH_CHECK(node_feat.is_cuda() && target_feat.is_cuda() && source_feat.is_cuda(),
              "line_node_triple_cat_forward: all tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 &&
                  target_feat.scalar_type() == torch::kFloat32 &&
                  source_feat.scalar_type() == torch::kFloat32,
              "line_node_triple_cat_forward: feature tensors must be float32");
  TORCH_CHECK(node_feat.dim() == 2 && target_feat.dim() == 2 && source_feat.dim() == 2 &&
                  node_feat.size(1) == kDim && target_feat.size(1) == kDim && source_feat.size(1) == kDim,
              "line_node_triple_cat_forward: feature tensors must be [rows, 128]");
  TORCH_CHECK(node_feat.size(0) == target_feat.size(0) && node_feat.size(0) == source_feat.size(0),
              "line_node_triple_cat_forward: row counts must match");
  const int64_t rows = node_feat.size(0);
  auto node_c = node_feat.contiguous();
  auto target_c = target_feat.contiguous();
  auto source_c = source_feat.contiguous();
  auto out = node_c.new_empty({rows, 3 * kDim});
  if (rows == 0) {
    return out;
  }
  line_node_triple_cat_forward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      node_c.data_ptr<float>(),
      target_c.data_ptr<float>(),
      source_c.data_ptr<float>(),
      out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> line_node_triple_cat_backward(const torch::Tensor &grad_cat) {
  TORCH_CHECK(grad_cat.is_cuda(), "line_node_triple_cat_backward: grad_cat must be CUDA");
  TORCH_CHECK(grad_cat.scalar_type() == torch::kFloat32,
              "line_node_triple_cat_backward: grad_cat must be float32");
  TORCH_CHECK(grad_cat.dim() == 2 && grad_cat.size(1) == 3 * kDim,
              "line_node_triple_cat_backward: grad_cat must be [rows, 384]");
  const int64_t rows = grad_cat.size(0);
  auto grad_cat_c = grad_cat.contiguous();
  auto grad_node = grad_cat_c.new_empty({rows, kDim});
  auto grad_target = grad_cat_c.new_empty({rows, kDim});
  auto grad_source = grad_cat_c.new_empty({rows, kDim});
  if (rows == 0) {
    return {grad_node, grad_target, grad_source};
  }
  line_node_triple_cat_backward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      grad_cat_c.data_ptr<float>(),
      grad_node.data_ptr<float>(),
      grad_target.data_ptr<float>(),
      grad_source.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_target, grad_source};
}

torch::Tensor directed_edge_gather_cat_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &edge_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index) {
  TORCH_CHECK(node_feat.is_cuda() && edge_feat.is_cuda() && edge_index.is_cuda() && source_index.is_cuda() &&
                  target_index.is_cuda(),
              "directed_edge_gather_cat_forward: all tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 && edge_feat.scalar_type() == torch::kFloat32,
              "directed_edge_gather_cat_forward: features must be float32");
  TORCH_CHECK(edge_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "directed_edge_gather_cat_forward: indices must be int64");
  TORCH_CHECK(node_feat.dim() == 2 && edge_feat.dim() == 2 && node_feat.size(1) == kDim && edge_feat.size(1) == kDim,
              "directed_edge_gather_cat_forward: feature tensors must be [rows, 128]");
  TORCH_CHECK(edge_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  edge_index.size(0) == source_index.size(0) && target_index.size(0) == source_index.size(0),
              "directed_edge_gather_cat_forward: index sizes must match");

  auto node_c = node_feat.contiguous();
  auto edge_c = edge_feat.contiguous();
  auto edge_index_c = edge_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  const int64_t rows = source_c.size(0);
  auto out = edge_c.new_empty({rows, 3 * kDim});
  if (rows == 0) {
    return out;
  }
  directed_edge_gather_cat_forward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      node_c.data_ptr<float>(),
      edge_c.data_ptr<float>(),
      edge_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> directed_edge_cat_grad_scatter_backward(
    const torch::Tensor &grad_cat,
    const torch::Tensor &edge_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows) {
  TORCH_CHECK(grad_cat.is_cuda() && edge_index.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "directed_edge_cat_grad_scatter_backward: all tensors must be CUDA");
  TORCH_CHECK(grad_cat.scalar_type() == torch::kFloat32,
              "directed_edge_cat_grad_scatter_backward: grad_cat must be float32");
  TORCH_CHECK(edge_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "directed_edge_cat_grad_scatter_backward: indices must be int64");
  TORCH_CHECK(grad_cat.dim() == 2 && grad_cat.size(1) == 3 * kDim,
              "directed_edge_cat_grad_scatter_backward: grad_cat must be [rows, 384]");
  TORCH_CHECK(edge_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  edge_index.size(0) == grad_cat.size(0) && source_index.size(0) == grad_cat.size(0) &&
                  target_index.size(0) == grad_cat.size(0),
              "directed_edge_cat_grad_scatter_backward: index sizes must match grad rows");
  TORCH_CHECK(node_rows >= 0 && edge_rows >= 0,
              "directed_edge_cat_grad_scatter_backward: row counts must be non-negative");

  auto grad_cat_c = grad_cat.contiguous();
  auto edge_index_c = edge_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_cat_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_cat_c.new_zeros({edge_rows, kDim});
  const int64_t rows = grad_cat_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  directed_edge_cat_grad_scatter_backward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      grad_cat_c.data_ptr<float>(),
      edge_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

torch::Tensor refine_line_edge_gather_cat_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index) {
  TORCH_CHECK(node_feat.is_cuda() && edge_feat.is_cuda() && atom_feat.is_cuda() && atom_index.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "refine_line_edge_gather_cat_forward: all tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 && edge_feat.scalar_type() == torch::kFloat32 &&
                  atom_feat.scalar_type() == torch::kFloat32,
              "refine_line_edge_gather_cat_forward: features must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_edge_gather_cat_forward: indices must be int64");
  TORCH_CHECK(node_feat.dim() == 2 && edge_feat.dim() == 2 && atom_feat.dim() == 2 &&
                  node_feat.size(1) == kDim && edge_feat.size(1) == kDim && atom_feat.size(1) == kDim,
              "refine_line_edge_gather_cat_forward: feature tensors must be [rows, 128]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == edge_feat.size(0) && source_index.size(0) == edge_feat.size(0) &&
                  target_index.size(0) == edge_feat.size(0),
              "refine_line_edge_gather_cat_forward: index sizes must match edge rows");

  auto node_c = node_feat.contiguous();
  auto edge_c = edge_feat.contiguous();
  auto atom_c = atom_feat.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  const int64_t rows = edge_c.size(0);
  auto out = edge_c.new_empty({rows, 4 * kDim});
  if (rows == 0) {
    return out;
  }
  refine_line_edge_gather_cat_forward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      node_c.data_ptr<float>(),
      edge_c.data_ptr<float>(),
      atom_c.data_ptr<float>(),
      atom_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> refine_line_edge_cat_grad_scatter_backward(
    const torch::Tensor &grad_cat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows) {
  TORCH_CHECK(grad_cat.is_cuda() && atom_index.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "refine_line_edge_cat_grad_scatter_backward: all tensors must be CUDA");
  TORCH_CHECK(grad_cat.scalar_type() == torch::kFloat32,
              "refine_line_edge_cat_grad_scatter_backward: grad_cat must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_edge_cat_grad_scatter_backward: indices must be int64");
  TORCH_CHECK(grad_cat.dim() == 2 && grad_cat.size(1) == 4 * kDim,
              "refine_line_edge_cat_grad_scatter_backward: grad_cat must be [rows, 512]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == grad_cat.size(0) && source_index.size(0) == grad_cat.size(0) &&
                  target_index.size(0) == grad_cat.size(0),
              "refine_line_edge_cat_grad_scatter_backward: index sizes must match grad rows");
  TORCH_CHECK(node_rows >= 0 && edge_rows >= 0 && atom_rows >= 0,
              "refine_line_edge_cat_grad_scatter_backward: row counts must be non-negative");

  auto grad_cat_c = grad_cat.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_cat_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_cat_c.new_empty({edge_rows, kDim});
  auto grad_atom = grad_cat_c.new_zeros({atom_rows, kDim});
  const int64_t rows = grad_cat_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge, grad_atom};
  }
  refine_line_edge_cat_grad_scatter_backward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      grad_cat_c.data_ptr<float>(),
      atom_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      grad_atom.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge, grad_atom};
}

std::vector<torch::Tensor> refine_line_project_grad_scatter_backward_tile32(
    const torch::Tensor &grad_projected,
    const torch::Tensor &weight,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows) {
  TORCH_CHECK(grad_projected.is_cuda() && weight.is_cuda() && atom_index.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "refine_line_project_grad_scatter_backward_tile32: all tensors must be CUDA");
  TORCH_CHECK(grad_projected.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
              "refine_line_project_grad_scatter_backward_tile32: tensors must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_project_grad_scatter_backward_tile32: indices must be int64");
  TORCH_CHECK(grad_projected.dim() == 2 && weight.dim() == 2,
              "refine_line_project_grad_scatter_backward_tile32: grad_projected and weight must be 2D");
  TORCH_CHECK(weight.size(1) == 4 * kDim,
              "refine_line_project_grad_scatter_backward_tile32: weight must have 512 input columns");
  TORCH_CHECK(grad_projected.size(1) == weight.size(0),
              "refine_line_project_grad_scatter_backward_tile32: grad_projected hidden dim must match weight rows");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == grad_projected.size(0) &&
                  source_index.size(0) == grad_projected.size(0) &&
                  target_index.size(0) == grad_projected.size(0),
              "refine_line_project_grad_scatter_backward_tile32: index sizes must match grad rows");
  TORCH_CHECK(node_rows >= 0 && edge_rows >= 0 && atom_rows >= 0,
              "refine_line_project_grad_scatter_backward_tile32: row counts must be non-negative");

  auto grad_projected_c = grad_projected.contiguous();
  auto weight_c = weight.contiguous();
  auto atom_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_projected_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_projected_c.new_empty({edge_rows, kDim});
  auto grad_atom = grad_projected_c.new_zeros({atom_rows, kDim});
  const int64_t rows = grad_projected_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge, grad_atom};
  }

  dim3 block(16, 16);
  dim3 grid((4 * kDim + kProjectTile32N - 1) / kProjectTile32N,
            static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  refine_line_project_grad_scatter_backward_tile32_kernel<<<grid, block>>>(
      grad_projected_c.data_ptr<float>(),
      weight_c.data_ptr<float>(),
      atom_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      grad_atom.data_ptr<float>(),
      rows,
      grad_projected_c.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge, grad_atom};
}

std::vector<torch::Tensor> refine_line_project_dual_grad_scatter_add_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &weight,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &grad_node_in,
    const torch::Tensor &grad_edge_in,
    int64_t atom_rows) {
  TORCH_CHECK(grad_core.is_cuda() && grad_gate.is_cuda() && weight.is_cuda() &&
                  atom_index.is_cuda() && source_index.is_cuda() && target_index.is_cuda() &&
                  grad_node_in.is_cuda() && grad_edge_in.is_cuda(),
              "refine_line_project_dual_grad_scatter_add_tile32: all tensors must be CUDA");
  TORCH_CHECK(grad_core.scalar_type() == torch::kFloat32 &&
                  grad_gate.scalar_type() == torch::kFloat32 &&
                  weight.scalar_type() == torch::kFloat32 &&
                  grad_node_in.scalar_type() == torch::kFloat32 &&
                  grad_edge_in.scalar_type() == torch::kFloat32,
              "refine_line_project_dual_grad_scatter_add_tile32: float tensors must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_project_dual_grad_scatter_add_tile32: indices must be int64");
  TORCH_CHECK(grad_core.dim() == 2 && grad_gate.sizes() == grad_core.sizes() &&
                  grad_core.size(1) == kDim,
              "refine_line_project_dual_grad_scatter_add_tile32: grad_core/gate must be [rows,128]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == 2 * kDim && weight.size(1) == 4 * kDim,
              "refine_line_project_dual_grad_scatter_add_tile32: weight must be [256,512]");
  TORCH_CHECK(grad_node_in.dim() == 2 && grad_node_in.size(1) == kDim &&
                  grad_edge_in.sizes() == grad_core.sizes(),
              "refine_line_project_dual_grad_scatter_add_tile32: input grads have wrong shape");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == grad_core.size(0) &&
                  source_index.size(0) == grad_core.size(0) &&
                  target_index.size(0) == grad_core.size(0),
              "refine_line_project_dual_grad_scatter_add_tile32: index sizes must match rows");
  TORCH_CHECK(atom_rows >= 0, "refine_line_project_dual_grad_scatter_add_tile32: atom_rows must be non-negative");

  auto grad_core_c = grad_core.contiguous();
  auto grad_gate_c = grad_gate.contiguous();
  auto weight_c = weight.contiguous();
  auto atom_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_node_in.contiguous().clone();
  auto grad_edge_in_c = grad_edge_in.contiguous();
  auto grad_edge = grad_edge_in_c.new_empty(grad_edge_in_c.sizes());
  auto grad_atom = grad_core_c.new_zeros({atom_rows, kDim});
  const int64_t rows = grad_core_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge.zero_(), grad_atom};
  }

  dim3 block(16, 16);
  dim3 grid((4 * kDim + kProjectTile32N - 1) / kProjectTile32N,
            static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  refine_line_project_dual_grad_scatter_add_tile32_kernel<<<grid, block>>>(
      grad_core_c.data_ptr<float>(),
      grad_gate_c.data_ptr<float>(),
      weight_c.data_ptr<float>(),
      atom_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge_in_c.data_ptr<float>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      grad_atom.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge, grad_atom};
}

std::vector<torch::Tensor> line_edge_project_grad_scatter_backward(
    const torch::Tensor &grad_projected,
    const torch::Tensor &weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows) {
  TORCH_CHECK(grad_projected.is_cuda() && weight.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_project_grad_scatter_backward: all tensors must be CUDA");
  TORCH_CHECK(grad_projected.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
              "line_edge_project_grad_scatter_backward: tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_project_grad_scatter_backward: indices must be int64");
  TORCH_CHECK(grad_projected.dim() == 2 && weight.dim() == 2,
              "line_edge_project_grad_scatter_backward: grad_projected and weight must be 2D");
  TORCH_CHECK(weight.size(1) == 3 * kDim,
              "line_edge_project_grad_scatter_backward: weight must have 384 input columns");
  TORCH_CHECK(grad_projected.size(1) == weight.size(0),
              "line_edge_project_grad_scatter_backward: grad_projected hidden dim must match weight rows");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_projected.size(0) &&
                  target_index.size(0) == grad_projected.size(0),
              "line_edge_project_grad_scatter_backward: index sizes must match grad rows");
  TORCH_CHECK(node_rows >= 0, "line_edge_project_grad_scatter_backward: node_rows must be non-negative");

  auto grad_projected_c = grad_projected.contiguous();
  auto weight_c = weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_projected_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_projected_c.new_empty({grad_projected_c.size(0), kDim});
  const int64_t rows = grad_projected_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  line_edge_project_grad_scatter_backward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      grad_projected_c.data_ptr<float>(),
      weight_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows,
      grad_projected_c.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> line_edge_project_grad_scatter_backward_tiled(
    const torch::Tensor &grad_projected,
    const torch::Tensor &weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows) {
  TORCH_CHECK(grad_projected.is_cuda() && weight.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_project_grad_scatter_backward_tiled: all tensors must be CUDA");
  TORCH_CHECK(grad_projected.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
              "line_edge_project_grad_scatter_backward_tiled: tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_project_grad_scatter_backward_tiled: indices must be int64");
  TORCH_CHECK(grad_projected.dim() == 2 && weight.dim() == 2,
              "line_edge_project_grad_scatter_backward_tiled: grad_projected and weight must be 2D");
  TORCH_CHECK(weight.size(1) == 3 * kDim,
              "line_edge_project_grad_scatter_backward_tiled: weight must have 384 input columns");
  TORCH_CHECK(grad_projected.size(1) == weight.size(0),
              "line_edge_project_grad_scatter_backward_tiled: grad_projected hidden dim must match weight rows");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_projected.size(0) &&
                  target_index.size(0) == grad_projected.size(0),
              "line_edge_project_grad_scatter_backward_tiled: index sizes must match grad rows");
  TORCH_CHECK(node_rows >= 0, "line_edge_project_grad_scatter_backward_tiled: node_rows must be non-negative");

  auto grad_projected_c = grad_projected.contiguous();
  auto weight_c = weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_projected_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_projected_c.new_empty({grad_projected_c.size(0), kDim});
  const int64_t rows = grad_projected_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  dim3 block(kProjectTileN, kProjectTileM);
  dim3 grid((3 * kDim + kProjectTileN - 1) / kProjectTileN,
            static_cast<unsigned int>((rows + kProjectTileM - 1) / kProjectTileM));
  line_edge_project_grad_scatter_backward_tiled_kernel<<<grid, block>>>(
      grad_projected_c.data_ptr<float>(),
      weight_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows,
      grad_projected_c.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> line_edge_project_grad_scatter_backward_tile32(
    const torch::Tensor &grad_projected,
    const torch::Tensor &weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows) {
  TORCH_CHECK(grad_projected.is_cuda() && weight.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_project_grad_scatter_backward_tile32: all tensors must be CUDA");
  TORCH_CHECK(grad_projected.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
              "line_edge_project_grad_scatter_backward_tile32: tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_project_grad_scatter_backward_tile32: indices must be int64");
  TORCH_CHECK(grad_projected.dim() == 2 && weight.dim() == 2,
              "line_edge_project_grad_scatter_backward_tile32: grad_projected and weight must be 2D");
  TORCH_CHECK(weight.size(1) == 3 * kDim,
              "line_edge_project_grad_scatter_backward_tile32: weight must have 384 input columns");
  TORCH_CHECK(grad_projected.size(1) == weight.size(0),
              "line_edge_project_grad_scatter_backward_tile32: grad_projected hidden dim must match weight rows");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_projected.size(0) &&
                  target_index.size(0) == grad_projected.size(0),
              "line_edge_project_grad_scatter_backward_tile32: index sizes must match grad rows");
  TORCH_CHECK(node_rows >= 0, "line_edge_project_grad_scatter_backward_tile32: node_rows must be non-negative");

  auto grad_projected_c = grad_projected.contiguous();
  auto weight_c = weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_projected_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_projected_c.new_empty({grad_projected_c.size(0), kDim});
  const int64_t rows = grad_projected_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  dim3 block(16, 16);
  dim3 grid((3 * kDim + kProjectTile32N - 1) / kProjectTile32N,
            static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  line_edge_project_grad_scatter_backward_tile32_kernel<<<grid, block>>>(
      grad_projected_c.data_ptr<float>(),
      weight_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows,
      grad_projected_c.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> line_edge_silu_project_grad_scatter_backward_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows) {
  TORCH_CHECK(grad_core.is_cuda() && grad_gate.is_cuda() && core_projected.is_cuda() &&
                  gate_projected.is_cuda() && weight.is_cuda() && source_index.is_cuda() &&
                  target_index.is_cuda(),
              "line_edge_silu_project_grad_scatter_backward_tile32: all tensors must be CUDA");
  TORCH_CHECK(grad_core.scalar_type() == torch::kFloat32 && grad_gate.scalar_type() == torch::kFloat32 &&
                  core_projected.scalar_type() == torch::kFloat32 &&
                  gate_projected.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
              "line_edge_silu_project_grad_scatter_backward_tile32: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_silu_project_grad_scatter_backward_tile32: indices must be int64");
  TORCH_CHECK(grad_core.dim() == 2 && grad_gate.sizes() == grad_core.sizes() &&
                  core_projected.sizes() == grad_core.sizes() && gate_projected.sizes() == grad_core.sizes(),
              "line_edge_silu_project_grad_scatter_backward_tile32: core/gate tensors must be matching 2D");
  TORCH_CHECK(grad_core.size(1) == kDim,
              "line_edge_silu_project_grad_scatter_backward_tile32: hidden dim must be 128");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({2 * kDim, 3 * kDim}),
              "line_edge_silu_project_grad_scatter_backward_tile32: weight must be [256, 384]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_core.size(0) &&
                  target_index.size(0) == grad_core.size(0),
              "line_edge_silu_project_grad_scatter_backward_tile32: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0, "line_edge_silu_project_grad_scatter_backward_tile32: node_rows must be non-negative");

  auto grad_core_c = grad_core.contiguous();
  auto grad_gate_c = grad_gate.contiguous();
  auto core_projected_c = core_projected.contiguous();
  auto gate_projected_c = gate_projected.contiguous();
  auto weight_c = weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_core_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_core_c.new_empty({grad_core_c.size(0), kDim});
  const int64_t rows = grad_core_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  dim3 block(16, 16);
  dim3 grid((3 * kDim + kProjectTile32N - 1) / kProjectTile32N,
            static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  line_edge_silu_project_grad_scatter_backward_tile32_kernel<<<grid, block>>>(
      grad_core_c.data_ptr<float>(),
      grad_gate_c.data_ptr<float>(),
      core_projected_c.data_ptr<float>(),
      gate_projected_c.data_ptr<float>(),
      weight_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> directed_edge_silu_project_grad_scatter_backward_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &weight,
    const torch::Tensor &edge_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows) {
  TORCH_CHECK(grad_core.is_cuda() && grad_gate.is_cuda() && core_projected.is_cuda() &&
                  gate_projected.is_cuda() && weight.is_cuda() && edge_index.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "directed_edge_silu_project_grad_scatter_backward_tile32: all tensors must be CUDA");
  TORCH_CHECK(grad_core.scalar_type() == torch::kFloat32 && grad_gate.scalar_type() == torch::kFloat32 &&
                  core_projected.scalar_type() == torch::kFloat32 &&
                  gate_projected.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
              "directed_edge_silu_project_grad_scatter_backward_tile32: float tensors must be float32");
  TORCH_CHECK(edge_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "directed_edge_silu_project_grad_scatter_backward_tile32: indices must be int64");
  TORCH_CHECK(grad_core.dim() == 2 && grad_gate.sizes() == grad_core.sizes() &&
                  core_projected.sizes() == grad_core.sizes() && gate_projected.sizes() == grad_core.sizes(),
              "directed_edge_silu_project_grad_scatter_backward_tile32: core/gate tensors must be matching 2D");
  TORCH_CHECK(grad_core.size(1) == kDim,
              "directed_edge_silu_project_grad_scatter_backward_tile32: hidden dim must be 128");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({2 * kDim, 3 * kDim}),
              "directed_edge_silu_project_grad_scatter_backward_tile32: weight must be [256, 384]");
  TORCH_CHECK(edge_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  edge_index.size(0) == grad_core.size(0) &&
                  source_index.size(0) == grad_core.size(0) &&
                  target_index.size(0) == grad_core.size(0),
              "directed_edge_silu_project_grad_scatter_backward_tile32: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0 && edge_rows >= 0,
              "directed_edge_silu_project_grad_scatter_backward_tile32: row counts must be non-negative");

  auto grad_core_c = grad_core.contiguous();
  auto grad_gate_c = grad_gate.contiguous();
  auto core_projected_c = core_projected.contiguous();
  auto gate_projected_c = gate_projected.contiguous();
  auto weight_c = weight.contiguous();
  auto edge_index_c = edge_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_core_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_core_c.new_zeros({edge_rows, kDim});
  const int64_t rows = grad_core_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  dim3 block(16, 16);
  dim3 grid((3 * kDim + kProjectTile32N - 1) / kProjectTile32N,
            static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  directed_edge_silu_project_grad_scatter_backward_tile32_kernel<<<grid, block>>>(
      grad_core_c.data_ptr<float>(),
      grad_gate_c.data_ptr<float>(),
      core_projected_c.data_ptr<float>(),
      gate_projected_c.data_ptr<float>(),
      weight_c.data_ptr<float>(),
      edge_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> line_edge_silu_project_alpha_grad_scatter_backward_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_logits,
    const torch::Tensor &grad_target_logits,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows) {
  TORCH_CHECK(grad_core.is_cuda() && grad_gate.is_cuda() && core_projected.is_cuda() &&
                  gate_projected.is_cuda() && first_weight.is_cuda() &&
                  grad_source_logits.is_cuda() && grad_target_logits.is_cuda() &&
                  source_alpha_weight.is_cuda() && target_alpha_weight.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: all tensors must be CUDA");
  TORCH_CHECK(grad_core.scalar_type() == torch::kFloat32 && grad_gate.scalar_type() == torch::kFloat32 &&
                  core_projected.scalar_type() == torch::kFloat32 &&
                  gate_projected.scalar_type() == torch::kFloat32 &&
                  first_weight.scalar_type() == torch::kFloat32 &&
                  grad_source_logits.scalar_type() == torch::kFloat32 &&
                  grad_target_logits.scalar_type() == torch::kFloat32 &&
                  source_alpha_weight.scalar_type() == torch::kFloat32 &&
                  target_alpha_weight.scalar_type() == torch::kFloat32,
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: indices must be int64");
  TORCH_CHECK(grad_core.dim() == 2 && grad_gate.sizes() == grad_core.sizes() &&
                  core_projected.sizes() == grad_core.sizes() && gate_projected.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: core/gate tensors must be matching 2D");
  TORCH_CHECK(grad_core.size(1) == kDim,
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: hidden dim must be 128");
  TORCH_CHECK(first_weight.sizes() == torch::IntArrayRef({2 * kDim, 3 * kDim}),
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: first_weight must be [256, 384]");
  TORCH_CHECK(grad_source_logits.sizes() == grad_core.sizes() &&
                  grad_target_logits.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: alpha grad tensors must be [rows, 128]");
  TORCH_CHECK(source_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  target_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}),
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: alpha weights must be [128, 128]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_core.size(0) &&
                  target_index.size(0) == grad_core.size(0),
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0,
              "line_edge_silu_project_alpha_grad_scatter_backward_tile32: node_rows must be non-negative");

  auto grad_core_c = grad_core.contiguous();
  auto grad_gate_c = grad_gate.contiguous();
  auto core_projected_c = core_projected.contiguous();
  auto gate_projected_c = gate_projected.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto grad_source_logits_c = grad_source_logits.contiguous();
  auto grad_target_logits_c = grad_target_logits.contiguous();
  auto source_alpha_weight_c = source_alpha_weight.contiguous();
  auto target_alpha_weight_c = target_alpha_weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_core_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_core_c.new_empty({grad_core_c.size(0), kDim});
  const int64_t rows = grad_core_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  dim3 block(16, 16);
  dim3 grid((3 * kDim + kProjectTile32N - 1) / kProjectTile32N,
            static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  line_edge_silu_project_alpha_grad_scatter_backward_tile32_kernel<<<grid, block>>>(
      grad_core_c.data_ptr<float>(),
      grad_gate_c.data_ptr<float>(),
      core_projected_c.data_ptr<float>(),
      gate_projected_c.data_ptr<float>(),
      first_weight_c.data_ptr<float>(),
      grad_source_logits_c.data_ptr<float>(),
      grad_target_logits_c.data_ptr<float>(),
      source_alpha_weight_c.data_ptr<float>(),
      target_alpha_weight_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_logits,
    const torch::Tensor &grad_target_logits,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows) {
  TORCH_CHECK(grad_core.is_cuda() && grad_gate.is_cuda() && core_projected.is_cuda() &&
                  gate_projected.is_cuda() && first_weight.is_cuda() &&
                  grad_source_logits.is_cuda() && grad_target_logits.is_cuda() &&
                  source_alpha_weight.is_cuda() && target_alpha_weight.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: all tensors must be CUDA");
  TORCH_CHECK(grad_core.scalar_type() == torch::kFloat32 && grad_gate.scalar_type() == torch::kFloat32 &&
                  core_projected.scalar_type() == torch::kFloat32 &&
                  gate_projected.scalar_type() == torch::kFloat32 &&
                  first_weight.scalar_type() == torch::kFloat32 &&
                  grad_source_logits.scalar_type() == torch::kFloat32 &&
                  grad_target_logits.scalar_type() == torch::kFloat32 &&
                  source_alpha_weight.scalar_type() == torch::kFloat32 &&
                  target_alpha_weight.scalar_type() == torch::kFloat32,
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: indices must be int64");
  TORCH_CHECK(grad_core.dim() == 2 && grad_gate.sizes() == grad_core.sizes() &&
                  core_projected.sizes() == grad_core.sizes() && gate_projected.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: core/gate tensors must be matching 2D");
  TORCH_CHECK(grad_core.size(1) == kDim,
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: hidden dim must be 128");
  TORCH_CHECK(first_weight.sizes() == torch::IntArrayRef({2 * kDim, 3 * kDim}),
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: first_weight must be [256, 384]");
  TORCH_CHECK(grad_source_logits.sizes() == grad_core.sizes() &&
                  grad_target_logits.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: alpha grad tensors must be [rows, 128]");
  TORCH_CHECK(source_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  target_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}),
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: alpha weights must be [128, 128]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_core.size(0) &&
                  target_index.size(0) == grad_core.size(0),
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0,
              "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32: node_rows must be non-negative");

  auto grad_core_c = grad_core.contiguous();
  auto grad_gate_c = grad_gate.contiguous();
  auto core_projected_c = core_projected.contiguous();
  auto gate_projected_c = gate_projected.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto grad_source_logits_c = grad_source_logits.contiguous();
  auto grad_target_logits_c = grad_target_logits.contiguous();
  auto source_alpha_weight_c = source_alpha_weight.contiguous();
  auto target_alpha_weight_c = target_alpha_weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_core_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_core_c.new_empty({grad_core_c.size(0), kDim});
  const int64_t rows = grad_core_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  dim3 block(16, 16);
  dim3 grid((3 * kDim + kProjectTile32N - 1) / kProjectTile32N,
            static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32_kernel<<<grid, block>>>(
      grad_core_c.data_ptr<float>(),
      grad_gate_c.data_ptr<float>(),
      core_projected_c.data_ptr<float>(),
      gate_projected_c.data_ptr<float>(),
      first_weight_c.data_ptr<float>(),
      grad_source_logits_c.data_ptr<float>(),
      grad_target_logits_c.data_ptr<float>(),
      source_alpha_weight_c.data_ptr<float>(),
      target_alpha_weight_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_logits,
    const torch::Tensor &grad_target_logits,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows) {
  TORCH_CHECK(grad_core.is_cuda() && grad_gate.is_cuda() && core_projected.is_cuda() &&
                  gate_projected.is_cuda() && first_weight.is_cuda() &&
                  grad_source_logits.is_cuda() && grad_target_logits.is_cuda() &&
                  source_alpha_weight.is_cuda() && target_alpha_weight.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: all tensors must be CUDA");
  TORCH_CHECK(grad_core.scalar_type() == torch::kFloat32 && grad_gate.scalar_type() == torch::kFloat32 &&
                  core_projected.scalar_type() == torch::kFloat32 &&
                  gate_projected.scalar_type() == torch::kFloat32 &&
                  first_weight.scalar_type() == torch::kFloat32 &&
                  grad_source_logits.scalar_type() == torch::kFloat32 &&
                  grad_target_logits.scalar_type() == torch::kFloat32 &&
                  source_alpha_weight.scalar_type() == torch::kFloat32 &&
                  target_alpha_weight.scalar_type() == torch::kFloat32,
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: indices must be int64");
  TORCH_CHECK(grad_core.dim() == 2 && grad_gate.sizes() == grad_core.sizes() &&
                  core_projected.sizes() == grad_core.sizes() && gate_projected.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: core/gate tensors must be matching 2D");
  TORCH_CHECK(grad_core.size(1) == kDim,
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: hidden dim must be 128");
  TORCH_CHECK(first_weight.sizes() == torch::IntArrayRef({2 * kDim, 3 * kDim}),
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: first_weight must be [256, 384]");
  TORCH_CHECK(grad_source_logits.sizes() == grad_core.sizes() &&
                  grad_target_logits.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: alpha grad tensors must be [rows, 128]");
  TORCH_CHECK(source_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  target_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}),
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: alpha weights must be [128, 128]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_core.size(0) &&
                  target_index.size(0) == grad_core.size(0),
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0,
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: node_rows must be non-negative");

  const c10::cuda::CUDAGuard device_guard(grad_core.device());
  auto grad_core_c = grad_core.contiguous();
  auto grad_gate_c = grad_gate.contiguous();
  auto core_projected_c = core_projected.contiguous();
  auto gate_projected_c = gate_projected.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto grad_source_logits_c = grad_source_logits.contiguous();
  auto grad_target_logits_c = grad_target_logits.contiguous();
  auto source_alpha_weight_c = source_alpha_weight.contiguous();
  auto target_alpha_weight_c = target_alpha_weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  const int64_t rows = grad_core_c.size(0);
  auto grad_node = grad_core_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_core_c.new_empty({rows, kDim});
  if (rows == 0) {
    return {grad_node, grad_edge};
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const bool split_edge = use_p108_split_edge_gemm();
  const bool grouped_alpha = use_p108_grouped_alpha_gemm();
  const bool inplace_hidden = use_p108_inplace_hidden_gemm();
  TORCH_CHECK(!(split_edge && grouped_alpha),
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: split edge and grouped alpha are mutually exclusive experiments");
  TORCH_CHECK(!(split_edge && inplace_hidden),
              "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm: split edge and inplace hidden are mutually exclusive experiments");
  if (inplace_hidden) {
    line_edge_apply_silu_grad_inplace_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
        grad_core_c.data_ptr<float>(),
        grad_gate_c.data_ptr<float>(),
        core_projected_c.data_ptr<float>(),
        gate_projected_c.data_ptr<float>(),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto grad_cat = split_edge ? grad_core_c.new_empty({rows, 2 * kDim}) : grad_core_c.new_empty({rows, 3 * kDim});
  if (inplace_hidden) {
    cublas_row_major_matmul(
        handle,
        grad_core_c.data_ptr<float>(),
        first_weight_c.data_ptr<float>(),
        grad_cat.data_ptr<float>(),
        rows,
        kDim,
        3 * kDim,
        kDim,
        3 * kDim,
        3 * kDim,
        0.0f,
        "line_edge dense_gemm first projection core inplace");
    cublas_row_major_matmul(
        handle,
        grad_gate_c.data_ptr<float>(),
        first_weight_c.data_ptr<float>() + kDim * 3 * kDim,
        grad_cat.data_ptr<float>(),
        rows,
        kDim,
        3 * kDim,
        kDim,
        3 * kDim,
        3 * kDim,
        1.0f,
        "line_edge dense_gemm first projection gate inplace");
  } else if (split_edge) {
    auto grad_hidden = grad_core_c.new_empty({rows, 2 * kDim});
    line_edge_fill_silu_grad_hidden_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
        grad_core_c.data_ptr<float>(),
        grad_gate_c.data_ptr<float>(),
        core_projected_c.data_ptr<float>(),
        gate_projected_c.data_ptr<float>(),
        grad_hidden.data_ptr<float>(),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cublas_row_major_matmul(
        handle,
        grad_hidden.data_ptr<float>(),
        first_weight_c.data_ptr<float>(),
        grad_edge.data_ptr<float>(),
        rows,
        2 * kDim,
        kDim,
        2 * kDim,
        3 * kDim,
        kDim,
        0.0f,
        "line_edge dense_gemm first projection edge");
    cublas_row_major_matmul(
        handle,
        grad_hidden.data_ptr<float>(),
        first_weight_c.data_ptr<float>() + kDim,
        grad_cat.data_ptr<float>(),
        rows,
        2 * kDim,
        2 * kDim,
        2 * kDim,
        3 * kDim,
        2 * kDim,
        0.0f,
        "line_edge dense_gemm first projection node pair");
  } else {
    auto grad_hidden = grad_core_c.new_empty({rows, 2 * kDim});
    line_edge_fill_silu_grad_hidden_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
        grad_core_c.data_ptr<float>(),
        grad_gate_c.data_ptr<float>(),
        core_projected_c.data_ptr<float>(),
        gate_projected_c.data_ptr<float>(),
        grad_hidden.data_ptr<float>(),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cublas_row_major_matmul(
        handle,
        grad_hidden.data_ptr<float>(),
        first_weight_c.data_ptr<float>(),
        grad_cat.data_ptr<float>(),
        rows,
        2 * kDim,
        3 * kDim,
        2 * kDim,
        3 * kDim,
        3 * kDim,
        0.0f,
        "line_edge dense_gemm first projection");
  }
  if (grouped_alpha) {
    auto grad_source_alpha = grad_core_c.new_empty({rows, kDim});
    auto grad_target_alpha = grad_core_c.new_empty({rows, kDim});
    cublas_row_major_grouped_pair_matmul(
        handle,
        grad_source_logits_c.data_ptr<float>(),
        source_alpha_weight_c.data_ptr<float>(),
        grad_source_alpha.data_ptr<float>(),
        grad_target_logits_c.data_ptr<float>(),
        target_alpha_weight_c.data_ptr<float>(),
        grad_target_alpha.data_ptr<float>(),
        rows,
        kDim,
        kDim,
        kDim,
        kDim,
        kDim,
        grad_core_c,
        "line_edge dense_gemm grouped alpha");
    line_edge_add_alpha_to_grad_cat_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
        grad_source_alpha.data_ptr<float>(),
        grad_target_alpha.data_ptr<float>(),
        grad_cat.data_ptr<float>(),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    float* alpha_grad_out = split_edge ? grad_edge.data_ptr<float>() : grad_cat.data_ptr<float>();
    const int64_t alpha_grad_stride = split_edge ? kDim : 3 * kDim;
    cublas_row_major_matmul(
        handle,
        grad_source_logits_c.data_ptr<float>(),
        source_alpha_weight_c.data_ptr<float>(),
        alpha_grad_out,
        rows,
        kDim,
        kDim,
        kDim,
        kDim,
        alpha_grad_stride,
        1.0f,
        "line_edge dense_gemm source alpha");
    cublas_row_major_matmul(
        handle,
        grad_target_logits_c.data_ptr<float>(),
        target_alpha_weight_c.data_ptr<float>(),
        alpha_grad_out,
        rows,
        kDim,
        kDim,
        kDim,
        kDim,
        alpha_grad_stride,
        1.0f,
        "line_edge dense_gemm target alpha");
  }

  if (split_edge) {
    line_edge_node_pair_grad_scatter_backward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
        grad_cat.data_ptr<float>(),
        source_c.data_ptr<int64_t>(),
        target_c.data_ptr<int64_t>(),
        grad_node.data_ptr<float>(),
        rows);
  } else {
    line_edge_cat_grad_scatter_backward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
        grad_cat.data_ptr<float>(),
        source_c.data_ptr<int64_t>(),
        target_c.data_ptr<int64_t>(),
        grad_node.data_ptr<float>(),
        grad_edge.data_ptr<float>(),
        rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_logits,
    const torch::Tensor &grad_target_logits,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &target_offsets,
    int64_t node_rows) {
  TORCH_CHECK(grad_core.is_cuda() && grad_gate.is_cuda() && core_projected.is_cuda() &&
                  gate_projected.is_cuda() && first_weight.is_cuda() &&
                  grad_source_logits.is_cuda() && grad_target_logits.is_cuda() &&
                  source_alpha_weight.is_cuda() && target_alpha_weight.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda() && target_offsets.is_cuda(),
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: all tensors must be CUDA");
  TORCH_CHECK(grad_core.scalar_type() == torch::kFloat32 && grad_gate.scalar_type() == torch::kFloat32 &&
                  core_projected.scalar_type() == torch::kFloat32 &&
                  gate_projected.scalar_type() == torch::kFloat32 &&
                  first_weight.scalar_type() == torch::kFloat32 &&
                  grad_source_logits.scalar_type() == torch::kFloat32 &&
                  grad_target_logits.scalar_type() == torch::kFloat32 &&
                  source_alpha_weight.scalar_type() == torch::kFloat32 &&
                  target_alpha_weight.scalar_type() == torch::kFloat32,
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64 &&
                  target_offsets.scalar_type() == torch::kInt64,
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: indices must be int64");
  TORCH_CHECK(grad_core.dim() == 2 && grad_gate.sizes() == grad_core.sizes() &&
                  core_projected.sizes() == grad_core.sizes() && gate_projected.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: core/gate tensors must be matching 2D");
  TORCH_CHECK(grad_core.size(1) == kDim,
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: hidden dim must be 128");
  TORCH_CHECK(first_weight.sizes() == torch::IntArrayRef({2 * kDim, 3 * kDim}),
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: first_weight must be [256, 384]");
  TORCH_CHECK(grad_source_logits.sizes() == grad_core.sizes() &&
                  grad_target_logits.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: alpha grad tensors must be [rows, 128]");
  TORCH_CHECK(source_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  target_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}),
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: alpha weights must be [128, 128]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_core.size(0) &&
                  target_index.size(0) == grad_core.size(0),
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: index sizes must match rows");
  TORCH_CHECK(target_offsets.dim() == 1 && target_offsets.size(0) == node_rows + 1,
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: target_offsets must be [node_rows + 1]");
  TORCH_CHECK(node_rows >= 0,
              "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32: node_rows must be non-negative");

  auto grad_core_c = grad_core.contiguous();
  auto grad_gate_c = grad_gate.contiguous();
  auto core_projected_c = core_projected.contiguous();
  auto gate_projected_c = gate_projected.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto grad_source_logits_c = grad_source_logits.contiguous();
  auto grad_target_logits_c = grad_target_logits.contiguous();
  auto source_alpha_weight_c = source_alpha_weight.contiguous();
  auto target_alpha_weight_c = target_alpha_weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_offsets_c = target_offsets.contiguous();
  auto grad_node = grad_core_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_core_c.new_empty({grad_core_c.size(0), kDim});
  const int64_t rows = grad_core_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  auto grad_target_tmp = grad_core_c.new_empty({rows, kDim});

  dim3 block(16, 16);
  dim3 grid((3 * kDim + kProjectTile32N - 1) / kProjectTile32N,
            static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  line_edge_silu_project_alpha_grad_scatter_backward_target_tmp_tile32_kernel<<<grid, block>>>(
      grad_core_c.data_ptr<float>(),
      grad_gate_c.data_ptr<float>(),
      core_projected_c.data_ptr<float>(),
      gate_projected_c.data_ptr<float>(),
      first_weight_c.data_ptr<float>(),
      grad_source_logits_c.data_ptr<float>(),
      grad_target_logits_c.data_ptr<float>(),
      source_alpha_weight_c.data_ptr<float>(),
      target_alpha_weight_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      grad_target_tmp.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 reduce_block(kDim);
  dim3 reduce_grid(1, static_cast<unsigned int>(node_rows));
  line_edge_target_tmp_reduce_kernel<<<reduce_grid, reduce_block>>>(
      grad_target_tmp.data_ptr<float>(),
      target_offsets_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      node_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_out,
    const torch::Tensor &grad_target_out,
    const torch::Tensor &values,
    const torch::Tensor &source_out,
    const torch::Tensor &target_out,
    const torch::Tensor &source_alpha,
    const torch::Tensor &target_alpha,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows) {
  TORCH_CHECK(grad_core.is_cuda() && grad_gate.is_cuda() && core_projected.is_cuda() &&
                  gate_projected.is_cuda() && first_weight.is_cuda() && grad_source_out.is_cuda() &&
                  grad_target_out.is_cuda() && values.is_cuda() && source_out.is_cuda() &&
                  target_out.is_cuda() && source_alpha.is_cuda() && target_alpha.is_cuda() &&
                  source_alpha_weight.is_cuda() && target_alpha_weight.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: all tensors must be CUDA");
  TORCH_CHECK(grad_core.scalar_type() == torch::kFloat32 && grad_gate.scalar_type() == torch::kFloat32 &&
                  core_projected.scalar_type() == torch::kFloat32 &&
                  gate_projected.scalar_type() == torch::kFloat32 &&
                  first_weight.scalar_type() == torch::kFloat32 &&
                  grad_source_out.scalar_type() == torch::kFloat32 &&
                  grad_target_out.scalar_type() == torch::kFloat32 &&
                  values.scalar_type() == torch::kFloat32 &&
                  source_out.scalar_type() == torch::kFloat32 &&
                  target_out.scalar_type() == torch::kFloat32 &&
                  source_alpha.scalar_type() == torch::kFloat32 &&
                  target_alpha.scalar_type() == torch::kFloat32 &&
                  source_alpha_weight.scalar_type() == torch::kFloat32 &&
                  target_alpha_weight.scalar_type() == torch::kFloat32,
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: indices must be int64");
  TORCH_CHECK(grad_core.dim() == 2 && grad_gate.sizes() == grad_core.sizes() &&
                  core_projected.sizes() == grad_core.sizes() && gate_projected.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: core/gate tensors must be matching 2D");
  TORCH_CHECK(grad_core.size(1) == kDim,
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: hidden dim must be 128");
  TORCH_CHECK(first_weight.sizes() == torch::IntArrayRef({2 * kDim, 3 * kDim}),
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: first_weight must be [256, 384]");
  TORCH_CHECK(values.sizes() == grad_core.sizes() && source_alpha.sizes() == grad_core.sizes() &&
                  target_alpha.sizes() == grad_core.sizes(),
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: edge tensors must be [rows, 128]");
  TORCH_CHECK(grad_source_out.dim() == 2 && grad_target_out.sizes() == grad_source_out.sizes() &&
                  source_out.sizes() == grad_source_out.sizes() && target_out.sizes() == grad_source_out.sizes() &&
                  grad_source_out.size(1) == kDim,
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: node tensors must be [node_rows, 128]");
  TORCH_CHECK(source_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  target_alpha_weight.sizes() == torch::IntArrayRef({kDim, kDim}),
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: alpha weights must be [128, 128]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_core.size(0) &&
                  target_index.size(0) == grad_core.size(0),
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0,
              "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32: node_rows must be non-negative");

  auto grad_core_c = grad_core.contiguous();
  auto grad_gate_c = grad_gate.contiguous();
  auto core_projected_c = core_projected.contiguous();
  auto gate_projected_c = gate_projected.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto grad_source_out_c = grad_source_out.contiguous();
  auto grad_target_out_c = grad_target_out.contiguous();
  auto values_c = values.contiguous();
  auto source_out_c = source_out.contiguous();
  auto target_out_c = target_out.contiguous();
  auto source_alpha_c = source_alpha.contiguous();
  auto target_alpha_c = target_alpha.contiguous();
  auto source_alpha_weight_c = source_alpha_weight.contiguous();
  auto target_alpha_weight_c = target_alpha_weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_node = grad_core_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_core_c.new_empty({grad_core_c.size(0), kDim});
  const int64_t rows = grad_core_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  dim3 block(16, 16);
  dim3 grid((3 * kDim + kProjectTile32N - 1) / kProjectTile32N,
            static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32_kernel<<<grid, block>>>(
      grad_core_c.data_ptr<float>(),
      grad_gate_c.data_ptr<float>(),
      core_projected_c.data_ptr<float>(),
      gate_projected_c.data_ptr<float>(),
      first_weight_c.data_ptr<float>(),
      grad_source_out_c.data_ptr<float>(),
      grad_target_out_c.data_ptr<float>(),
      values_c.data_ptr<float>(),
      source_out_c.data_ptr<float>(),
      target_out_c.data_ptr<float>(),
      source_alpha_c.data_ptr<float>(),
      target_alpha_c.data_ptr<float>(),
      source_alpha_weight_c.data_ptr<float>(),
      target_alpha_weight_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}

std::vector<torch::Tensor> line_edge_w8a8_tail_project_scatter_backward_n128(
    const torch::Tensor &grad_out,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    const torch::Tensor &first_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    double eps) {
  TORCH_CHECK(grad_out.is_cuda() && core_pre.is_cuda() && gate_pre.is_cuda() && core_projected.is_cuda() &&
                  gate_projected.is_cuda() && core_q_weight.is_cuda() && gate_q_weight.is_cuda() &&
                  core_weight_scale.is_cuda() && gate_weight_scale.is_cuda() && core_norm_weight.is_cuda() &&
                  core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda() &&
                  first_weight.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "line_edge_w8a8_tail_project_scatter_backward_n128: all tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 && core_pre.scalar_type() == torch::kFloat32 &&
                  gate_pre.scalar_type() == torch::kFloat32 && core_projected.scalar_type() == torch::kFloat32 &&
                  gate_projected.scalar_type() == torch::kFloat32 && core_weight_scale.scalar_type() == torch::kFloat32 &&
                  gate_weight_scale.scalar_type() == torch::kFloat32 && core_norm_weight.scalar_type() == torch::kFloat32 &&
                  core_norm_bias.scalar_type() == torch::kFloat32 && gate_norm_weight.scalar_type() == torch::kFloat32 &&
                  gate_norm_bias.scalar_type() == torch::kFloat32 && first_weight.scalar_type() == torch::kFloat32,
              "line_edge_w8a8_tail_project_scatter_backward_n128: float tensors must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "line_edge_w8a8_tail_project_scatter_backward_n128: qweight tensors must be int8");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "line_edge_w8a8_tail_project_scatter_backward_n128: indices must be int64");
  TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(1) == kDim,
              "line_edge_w8a8_tail_project_scatter_backward_n128: grad_out must be [rows, 128]");
  TORCH_CHECK(core_pre.sizes() == grad_out.sizes() && gate_pre.sizes() == grad_out.sizes() &&
                  core_projected.sizes() == grad_out.sizes() && gate_projected.sizes() == grad_out.sizes(),
              "line_edge_w8a8_tail_project_scatter_backward_n128: saved row tensors must match grad_out");
  TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2 &&
                  core_q_weight.size(0) == kDim && core_q_weight.size(1) == kDim &&
                  gate_q_weight.size(0) == kDim && gate_q_weight.size(1) == kDim,
              "line_edge_w8a8_tail_project_scatter_backward_n128: qweight must be [128, 128]");
  TORCH_CHECK(first_weight.dim() == 2 && first_weight.size(0) == 2 * kDim && first_weight.size(1) == 3 * kDim,
              "line_edge_w8a8_tail_project_scatter_backward_n128: first_weight must be [256, 384]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == grad_out.size(0) && target_index.size(0) == grad_out.size(0),
              "line_edge_w8a8_tail_project_scatter_backward_n128: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0, "line_edge_w8a8_tail_project_scatter_backward_n128: node_rows must be non-negative");

  auto grad_out_c = grad_out.contiguous();
  auto core_pre_c = core_pre.contiguous();
  auto gate_pre_c = gate_pre.contiguous();
  auto core_projected_c = core_projected.contiguous();
  auto gate_projected_c = gate_projected.contiguous();
  auto core_q_weight_c = core_q_weight.contiguous();
  auto gate_q_weight_c = gate_q_weight.contiguous();
  auto core_weight_scale_c = core_weight_scale.contiguous();
  auto gate_weight_scale_c = gate_weight_scale.contiguous();
  auto core_norm_weight_c = core_norm_weight.contiguous();
  auto core_norm_bias_c = core_norm_bias.contiguous();
  auto gate_norm_weight_c = gate_norm_weight.contiguous();
  auto gate_norm_bias_c = gate_norm_bias.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();

  auto grad_node = grad_out_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_out_c.new_empty({grad_out_c.size(0), kDim});
  const int64_t rows = grad_out_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge};
  }
  line_edge_w8a8_tail_project_scatter_backward_n128_kernel<<<static_cast<unsigned int>(rows), kP44Threads>>>(
      grad_out_c.data_ptr<float>(),
      core_pre_c.data_ptr<float>(),
      gate_pre_c.data_ptr<float>(),
      core_projected_c.data_ptr<float>(),
      gate_projected_c.data_ptr<float>(),
      core_q_weight_c.data_ptr<int8_t>(),
      gate_q_weight_c.data_ptr<int8_t>(),
      core_weight_scale_c.data_ptr<float>(),
      gate_weight_scale_c.data_ptr<float>(),
      core_norm_weight_c.data_ptr<float>(),
      core_norm_bias_c.data_ptr<float>(),
      gate_norm_weight_c.data_ptr<float>(),
      gate_norm_bias_c.data_ptr<float>(),
      first_weight_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge};
}
