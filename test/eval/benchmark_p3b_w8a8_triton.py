import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from fairchem.core.datasets import AseDBDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications.base import MatRISCalculator
from quant.fusion import apply_gated_mlp_fusion


FUSION_MODE = "line_all_candidate_gated_mlp_second_fused_fp32"

CANDIDATE_BRANCHES = {
    "a8_line_nonlinear_core_gate": (
        "attn_block_line_graph.edge_nonlinear_update",
        "refine_block_line_graph.edge_nonlinear_update",
        "attn_block_line_graph.node_nonlinear_update",
    ),
    "a8_line_edge_core_gate": (
        "attn_block_line_graph.edge_nonlinear_update",
        "refine_block_line_graph.edge_nonlinear_update",
    ),
    "a8_line_node_core_gate": (
        "attn_block_line_graph.node_nonlinear_update",
    ),
}


@triton.jit
def _quantize_per_tensor_kernel(
    x_ptr,
    qx_ptr,
    scale_ptr,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    scaled = x / scale
    q_pos = tl.floor(scaled + 0.5)
    q_neg = tl.ceil(scaled - 0.5)
    q = tl.where(scaled >= 0.0, q_pos, q_neg)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(qx_ptr + offsets, q, mask=mask)


@triton.jit
def _block_activation_scale_kernel(
    x_ptr,
    scale_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    tile_abs_max = tl.full((), 0.0, dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        tile_abs_max = tl.maximum(tile_abs_max, tl.max(tl.abs(x), axis=None))
    tl.store(scale_ptr + pid_m, tl.maximum(tile_abs_max / 127.0, 1.0e-12))


@triton.jit
def _w8a8_matmul_dequant_kernel(
    a_ptr,
    b_ptr,
    weight_scale_ptr,
    activation_scale_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0,
        )
        b = tl.load(
            b_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc += tl.dot(a, b, out_dtype=tl.int32)

    activation_scale = tl.load(activation_scale_ptr)
    weight_scale = tl.load(weight_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * (activation_scale * weight_scale[None, :])
    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        out += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _w8a8_block_scale_quant_matmul_dequant_kernel(
    x_ptr,
    b_ptr,
    weight_scale_ptr,
    block_scale_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    activation_scale = tl.load(block_scale_ptr + pid_m)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        scaled = x / activation_scale
        q_pos = tl.floor(scaled + 0.5)
        q_neg = tl.ceil(scaled - 0.5)
        a = tl.where(scaled >= 0.0, q_pos, q_neg)
        a = tl.minimum(tl.maximum(a, -127.0), 127.0).to(tl.int8)
        b = tl.load(
            b_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc += tl.dot(a, b, out_dtype=tl.int32)

    weight_scale = tl.load(weight_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * (activation_scale * weight_scale[None, :])
    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        out += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _w8a8_fused_quant_matmul_dequant_kernel(
    x_ptr,
    b_ptr,
    weight_scale_ptr,
    activation_scale_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    activation_scale = tl.load(activation_scale_ptr)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        scaled = x / activation_scale
        q_pos = tl.floor(scaled + 0.5)
        q_neg = tl.ceil(scaled - 0.5)
        a = tl.where(scaled >= 0.0, q_pos, q_neg)
        a = tl.minimum(tl.maximum(a, -127.0), 127.0).to(tl.int8)
        b = tl.load(
            b_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc += tl.dot(a, b, out_dtype=tl.int32)

    weight_scale = tl.load(weight_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * (activation_scale * weight_scale[None, :])
    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        out += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _w8a8_tile_dynamic_quant_matmul_dequant_kernel(
    x_ptr,
    b_ptr,
    weight_scale_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tile_abs_max = tl.full((), 0.0, dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        tile_abs_max = tl.maximum(tile_abs_max, tl.max(tl.abs(x), axis=None))
    activation_scale = tl.maximum(tile_abs_max / 127.0, 1.0e-12)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        scaled = x / activation_scale
        q_pos = tl.floor(scaled + 0.5)
        q_neg = tl.ceil(scaled - 0.5)
        a = tl.where(scaled >= 0.0, q_pos, q_neg)
        a = tl.minimum(tl.maximum(a, -127.0), 127.0).to(tl.int8)
        b = tl.load(
            b_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc += tl.dot(a, b, out_dtype=tl.int32)

    weight_scale = tl.load(weight_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * (activation_scale * weight_scale[None, :])
    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        out += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P3b-3 Triton W8A8 microbenchmark for MatRIS fused line GatedMLP shapes."
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--output-dir", default="results/p3b_w8a8_triton_microbench")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="e", choices=("e", "ef", "efs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--speedup-threshold", type=float, default=1.0)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=("a8_line_edge_core_gate",),
        choices=tuple(CANDIDATE_BRANCHES),
        help="Candidates to benchmark. Default keeps the first pass focused on large edge shapes.",
    )
    parser.add_argument(
        "--points",
        nargs="+",
        default=("fused_first", "fused_second"),
        choices=("fused_first", "fused_second"),
    )
    return parser.parse_args()


def select_indices(dataset_len: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= dataset_len:
        return list(range(dataset_len))
    rng = random.Random(seed)
    indices = rng.sample(range(dataset_len), limit)
    indices.sort()
    return indices


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def branch_name(module_name: str) -> str | None:
    for branches in CANDIDATE_BRANCHES.values():
        for branch in branches:
            if branch in module_name:
                return branch
    return None


def point_name(module_name: str) -> str | None:
    if module_name.endswith(".fused_first"):
        return "fused_first"
    if module_name.endswith(".fused_second"):
        return "fused_second"
    return None


def collect_shapes(calc: MatRISCalculator, dataset: AseDBDataset, indices: list[int], device: str) -> list[dict]:
    captured: dict[tuple[str, str, str, int, int, int], dict] = {}
    hooks = []

    def make_hook(module_name: str):
        def hook(module, inputs):
            point = point_name(module_name)
            branch = branch_name(module_name)
            if point is None or branch is None:
                return
            x = inputs[0]
            if x.ndim < 2:
                return
            m = int(x.reshape(-1, x.shape[-1]).shape[0])
            k = int(x.shape[-1])
            n = int(module.weight.shape[0])
            for candidate, branches in CANDIDATE_BRANCHES.items():
                if branch not in branches:
                    continue
                key = (candidate, branch, point, m, k, n)
                captured[key] = {
                    "candidate": candidate,
                    "branch": branch,
                    "fused_point": point,
                    "M": m,
                    "K": k,
                    "N": n,
                    "source_module": module_name,
                }

        return hook

    for name, module in calc.model.named_modules():
        if point_name(name) is not None and branch_name(name) is not None:
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        for sample_index in indices:
            atoms = dataset.get_atoms(sample_index)
            atoms.calc = calc
            _ = atoms.get_potential_energy()
            if "f" in calc.task:
                _ = atoms.get_forces()
            if "s" in calc.task:
                _ = atoms.get_stress()
            sync(device)
    finally:
        for hook in hooks:
            hook.remove()

    return sorted(
        captured.values(),
        key=lambda x: (x["candidate"], x["branch"], x["fused_point"], x["M"], x["K"], x["N"]),
    )


def measure_ms(fn, device: str, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    sync(device)

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def quantize_weight_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 127
    weight_fp32 = weight.float().contiguous()
    scale = weight_fp32.abs().amax(dim=1).clamp_min(1e-12) / qmax
    q_weight = torch.round(weight_fp32 / scale.reshape(-1, 1)).clamp(-qmax, qmax).to(torch.int8)
    return q_weight.t().contiguous(), scale.contiguous()


def quantize_activation_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 127
    scale = x.float().abs().amax().clamp_min(1e-12) / qmax
    q_x = torch.round(x.float() / scale).clamp(-qmax, qmax).to(torch.int8).contiguous()
    return q_x, scale.reshape(())


def quantize_activation_triton(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q_x = torch.empty_like(x, dtype=torch.int8)
    total = x.numel()
    block_size = 1024
    grid = (triton.cdiv(total, block_size),)
    _quantize_per_tensor_kernel[grid](x, q_x, scale, total, BLOCK_SIZE=block_size)
    return q_x


def triton_w8a8_linear(
    q_x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    m, k = q_x.shape
    n = q_weight_t.shape[1]
    out = torch.empty((m, n), device=q_x.device, dtype=torch.float32)
    bias_arg = bias if bias is not None else out
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _w8a8_matmul_dequant_kernel[grid](
        q_x,
        q_weight_t,
        weight_scale,
        activation_scale,
        bias_arg,
        out,
        M=m,
        K=k,
        N=n,
        has_bias=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def triton_w8a8_static_full_linear(
    x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
    **config,
) -> torch.Tensor:
    q_x = quantize_activation_triton(x, activation_scale)
    return triton_w8a8_linear(q_x, q_weight_t, weight_scale, activation_scale, bias, **config)


def triton_w8a8_fused_quant_linear(
    x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    m, k = x.shape
    n = q_weight_t.shape[1]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    bias_arg = bias if bias is not None else out
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _w8a8_fused_quant_matmul_dequant_kernel[grid](
        x,
        q_weight_t,
        weight_scale,
        activation_scale,
        bias_arg,
        out,
        M=m,
        K=k,
        N=n,
        has_bias=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def triton_w8a8_dynamic_fused_quant_linear(
    x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    **config,
) -> torch.Tensor:
    activation_scale = x.float().abs().amax().clamp_min(1e-12) / 127.0
    return triton_w8a8_fused_quant_linear(x, q_weight_t, weight_scale, activation_scale, bias, **config)


def triton_w8a8_tile_dynamic_fused_quant_linear(
    x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    m, k = x.shape
    n = q_weight_t.shape[1]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    bias_arg = bias if bias is not None else out
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _w8a8_tile_dynamic_quant_matmul_dequant_kernel[grid](
        x,
        q_weight_t,
        weight_scale,
        bias_arg,
        out,
        M=m,
        K=k,
        N=n,
        has_bias=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def triton_w8a8_block_dynamic_fused_quant_linear(
    x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    m, k = x.shape
    n = q_weight_t.shape[1]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    block_scales = torch.empty((triton.cdiv(m, block_m),), device=x.device, dtype=torch.float32)
    scale_grid = (triton.cdiv(m, block_m),)
    _block_activation_scale_kernel[scale_grid](
        x,
        block_scales,
        M=m,
        K=k,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    bias_arg = bias if bias is not None else out
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _w8a8_block_scale_quant_matmul_dequant_kernel[grid](
        x,
        q_weight_t,
        weight_scale,
        block_scales,
        bias_arg,
        out,
        M=m,
        K=k,
        N=n,
        has_bias=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def triton_w8a8_dynamic_full_linear(
    x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    **config,
) -> torch.Tensor:
    activation_scale = x.float().abs().amax().clamp_min(1e-12) / 127.0
    q_x = quantize_activation_triton(x, activation_scale)
    return triton_w8a8_linear(q_x, q_weight_t, weight_scale, activation_scale, bias, **config)


def torch_w8a8_full_linear(
    x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    q_x, activation_scale = quantize_activation_torch(x)
    acc = torch._int_mm(q_x, q_weight_t)
    out = acc.float() * (activation_scale * weight_scale.reshape(1, -1))
    if bias is not None:
        out = out + bias
    return out


def candidate_configs(k: int) -> list[dict]:
    block_k = 64 if k % 64 == 0 else 32
    return [
        {"block_m": 16, "block_n": 64, "block_k": block_k, "num_warps": 4, "num_stages": 4},
        {"block_m": 32, "block_n": 64, "block_k": block_k, "num_warps": 4, "num_stages": 4},
        {"block_m": 32, "block_n": 128, "block_k": block_k, "num_warps": 4, "num_stages": 4},
        {"block_m": 64, "block_n": 64, "block_k": block_k, "num_warps": 4, "num_stages": 4},
        {"block_m": 64, "block_n": 128, "block_k": block_k, "num_warps": 4, "num_stages": 4},
        {"block_m": 128, "block_n": 64, "block_k": block_k, "num_warps": 4, "num_stages": 4},
        {"block_m": 128, "block_n": 128, "block_k": block_k, "num_warps": 4, "num_stages": 4},
        {"block_m": 64, "block_n": 128, "block_k": block_k, "num_warps": 8, "num_stages": 4},
        {"block_m": 128, "block_n": 128, "block_k": block_k, "num_warps": 8, "num_stages": 4},
    ]


def reference_error(
    x: torch.Tensor,
    q_x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
    config: dict,
) -> tuple[float, float]:
    out = triton_w8a8_linear(q_x, q_weight_t, weight_scale, activation_scale, bias, **config)
    dq_x = q_x.float() * activation_scale
    dq_weight = q_weight_t.t().float() * weight_scale.reshape(-1, 1)
    ref_out = F.linear(dq_x, dq_weight, bias)
    diff = (out - ref_out).abs()
    return float(diff.max().detach().cpu()), float(diff.mean().detach().cpu())


def benchmark_shape(shape: dict, modules: dict[str, torch.nn.Module], args: argparse.Namespace) -> dict:
    if args.device != "cuda":
        raise RuntimeError("P3b-3 Triton benchmark requires CUDA.")

    module = modules[shape["source_module"]]
    m = int(shape["M"])
    k = int(shape["K"])
    weight = module.weight.detach().float().contiguous()
    bias = module.bias.detach().float().contiguous() if module.bias is not None else None
    q_weight_t, weight_scale = quantize_weight_per_channel(weight)
    x = torch.randn((m, k), device=args.device, dtype=torch.float32)
    q_x, dynamic_scale = quantize_activation_torch(x)
    static_scale = dynamic_scale.detach().clone()

    fp32_mean, fp32_std = measure_ms(
        lambda: F.linear(x, weight, bias),
        args.device,
        args.warmup,
        args.repeat,
    )
    torch_full_mean, torch_full_std = measure_ms(
        lambda: torch_w8a8_full_linear(x, q_weight_t, weight_scale, bias),
        args.device,
        args.warmup,
        args.repeat,
    )

    rows = []
    best_prequant = None
    best_static = None
    best_dynamic = None
    for config in candidate_configs(k):
        prequant_mean, prequant_std = measure_ms(
            lambda config=config: triton_w8a8_linear(
                q_x,
                q_weight_t,
                weight_scale,
                dynamic_scale,
                bias,
                **config,
            ),
            args.device,
            args.warmup,
            args.repeat,
        )
        static_mean, static_std = measure_ms(
            lambda config=config: triton_w8a8_static_full_linear(
                x,
                q_weight_t,
                weight_scale,
                static_scale,
                bias,
                **config,
            ),
            args.device,
            args.warmup,
            args.repeat,
        )
        dynamic_mean, dynamic_std = measure_ms(
            lambda config=config: triton_w8a8_dynamic_full_linear(
                x,
                q_weight_t,
                weight_scale,
                bias,
                **config,
            ),
            args.device,
            args.warmup,
            args.repeat,
        )
        fused_static_mean, fused_static_std = measure_ms(
            lambda config=config: triton_w8a8_fused_quant_linear(
                x,
                q_weight_t,
                weight_scale,
                static_scale,
                bias,
                **config,
            ),
            args.device,
            args.warmup,
            args.repeat,
        )
        fused_dynamic_mean, fused_dynamic_std = measure_ms(
            lambda config=config: triton_w8a8_dynamic_fused_quant_linear(
                x,
                q_weight_t,
                weight_scale,
                bias,
                **config,
            ),
            args.device,
            args.warmup,
            args.repeat,
        )
        tile_dynamic_mean, tile_dynamic_std = measure_ms(
            lambda config=config: triton_w8a8_tile_dynamic_fused_quant_linear(
                x,
                q_weight_t,
                weight_scale,
                bias,
                **config,
            ),
            args.device,
            args.warmup,
            args.repeat,
        )
        block_dynamic_mean, block_dynamic_std = measure_ms(
            lambda config=config: triton_w8a8_block_dynamic_fused_quant_linear(
                x,
                q_weight_t,
                weight_scale,
                bias,
                **config,
            ),
            args.device,
            args.warmup,
            args.repeat,
        )
        max_abs_diff, mean_abs_diff = reference_error(
            x,
            q_x,
            q_weight_t,
            weight_scale,
            dynamic_scale,
            bias,
            config,
        )
        row = {
            **config,
            "prequant_latency_ms_mean": prequant_mean,
            "prequant_latency_ms_std": prequant_std,
            "prequant_speedup_vs_fp32": fp32_mean / prequant_mean if prequant_mean > 0 else 0.0,
            "static_full_latency_ms_mean": static_mean,
            "static_full_latency_ms_std": static_std,
            "static_full_speedup_vs_fp32": fp32_mean / static_mean if static_mean > 0 else 0.0,
            "dynamic_full_latency_ms_mean": dynamic_mean,
            "dynamic_full_latency_ms_std": dynamic_std,
            "dynamic_full_speedup_vs_fp32": fp32_mean / dynamic_mean if dynamic_mean > 0 else 0.0,
            "fused_static_latency_ms_mean": fused_static_mean,
            "fused_static_latency_ms_std": fused_static_std,
            "fused_static_speedup_vs_fp32": fp32_mean / fused_static_mean if fused_static_mean > 0 else 0.0,
            "fused_dynamic_latency_ms_mean": fused_dynamic_mean,
            "fused_dynamic_latency_ms_std": fused_dynamic_std,
            "fused_dynamic_speedup_vs_fp32": fp32_mean / fused_dynamic_mean if fused_dynamic_mean > 0 else 0.0,
            "tile_dynamic_latency_ms_mean": tile_dynamic_mean,
            "tile_dynamic_latency_ms_std": tile_dynamic_std,
            "tile_dynamic_speedup_vs_fp32": fp32_mean / tile_dynamic_mean if tile_dynamic_mean > 0 else 0.0,
            "block_dynamic_latency_ms_mean": block_dynamic_mean,
            "block_dynamic_latency_ms_std": block_dynamic_std,
            "block_dynamic_speedup_vs_fp32": fp32_mean / block_dynamic_mean if block_dynamic_mean > 0 else 0.0,
            "max_abs_diff_vs_fake_quant_ref": max_abs_diff,
            "mean_abs_diff_vs_fake_quant_ref": mean_abs_diff,
        }
        rows.append(row)
        if best_prequant is None or row["prequant_latency_ms_mean"] < best_prequant["prequant_latency_ms_mean"]:
            best_prequant = row
        if best_static is None or row["static_full_latency_ms_mean"] < best_static["static_full_latency_ms_mean"]:
            best_static = row
        if best_dynamic is None or row["dynamic_full_latency_ms_mean"] < best_dynamic["dynamic_full_latency_ms_mean"]:
            best_dynamic = row
    best_fused_static = min(rows, key=lambda item: item["fused_static_latency_ms_mean"])
    best_fused_dynamic = min(rows, key=lambda item: item["fused_dynamic_latency_ms_mean"])
    best_tile_dynamic = min(rows, key=lambda item: item["tile_dynamic_latency_ms_mean"])
    best_block_dynamic = min(rows, key=lambda item: item["block_dynamic_latency_ms_mean"])

    return {
        **shape,
        "bias": bias is not None,
        "activation_quant_dynamic": "torch_amax + triton_elementwise_quantize",
        "activation_quant_static": "precomputed_per_tensor_scale + triton_elementwise_quantize",
        "weight_quant": "per_output_channel_symmetric_int8",
        "accumulation": "int32",
        "output_dtype": "fp32",
        "fp32_fused_latency_ms_mean": fp32_mean,
        "fp32_fused_latency_ms_std": fp32_std,
        "torch_full_latency_ms_mean": torch_full_mean,
        "torch_full_latency_ms_std": torch_full_std,
        "torch_full_speedup_vs_fp32": fp32_mean / torch_full_mean if torch_full_mean > 0 else 0.0,
        "triton_configs": rows,
        "best_prequant": best_prequant,
        "best_static_full": best_static,
        "best_dynamic_full": best_dynamic,
        "best_fused_static_full": best_fused_static,
        "best_fused_dynamic_full": best_fused_dynamic,
        "best_tile_dynamic_full": best_tile_dynamic,
        "best_block_dynamic_full": best_block_dynamic,
    }


def summarize(results: list[dict]) -> dict:
    best_prequant = max(
        results,
        key=lambda row: row["best_prequant"]["prequant_speedup_vs_fp32"],
        default=None,
    )
    best_static = max(
        results,
        key=lambda row: row["best_static_full"]["static_full_speedup_vs_fp32"],
        default=None,
    )
    best_dynamic = max(
        results,
        key=lambda row: row["best_dynamic_full"]["dynamic_full_speedup_vs_fp32"],
        default=None,
    )
    best_fused_static = max(
        results,
        key=lambda row: row["best_fused_static_full"]["fused_static_speedup_vs_fp32"],
        default=None,
    )
    best_fused_dynamic = max(
        results,
        key=lambda row: row["best_fused_dynamic_full"]["fused_dynamic_speedup_vs_fp32"],
        default=None,
    )
    best_tile_dynamic = max(
        results,
        key=lambda row: row["best_tile_dynamic_full"]["tile_dynamic_speedup_vs_fp32"],
        default=None,
    )
    best_block_dynamic = max(
        results,
        key=lambda row: row["best_block_dynamic_full"]["block_dynamic_speedup_vs_fp32"],
        default=None,
    )
    return {
        "total_shapes": len(results),
        "best_prequant_shape": _best_shape_payload(best_prequant, "best_prequant", "prequant_speedup_vs_fp32"),
        "best_static_full_shape": _best_shape_payload(best_static, "best_static_full", "static_full_speedup_vs_fp32"),
        "best_dynamic_full_shape": _best_shape_payload(best_dynamic, "best_dynamic_full", "dynamic_full_speedup_vs_fp32"),
        "best_fused_static_full_shape": _best_shape_payload(
            best_fused_static,
            "best_fused_static_full",
            "fused_static_speedup_vs_fp32",
        ),
        "best_fused_dynamic_full_shape": _best_shape_payload(
            best_fused_dynamic,
            "best_fused_dynamic_full",
            "fused_dynamic_speedup_vs_fp32",
        ),
        "best_tile_dynamic_full_shape": _best_shape_payload(
            best_tile_dynamic,
            "best_tile_dynamic_full",
            "tile_dynamic_speedup_vs_fp32",
        ),
        "best_block_dynamic_full_shape": _best_shape_payload(
            best_block_dynamic,
            "best_block_dynamic_full",
            "block_dynamic_speedup_vs_fp32",
        ),
        "prequant_speedup_count_gt_1": sum(
            row["best_prequant"]["prequant_speedup_vs_fp32"] > 1.0 for row in results
        ),
        "static_full_speedup_count_gt_1": sum(
            row["best_static_full"]["static_full_speedup_vs_fp32"] > 1.0 for row in results
        ),
        "dynamic_full_speedup_count_gt_1": sum(
            row["best_dynamic_full"]["dynamic_full_speedup_vs_fp32"] > 1.0 for row in results
        ),
        "fused_static_full_speedup_count_gt_1": sum(
            row["best_fused_static_full"]["fused_static_speedup_vs_fp32"] > 1.0 for row in results
        ),
        "fused_dynamic_full_speedup_count_gt_1": sum(
            row["best_fused_dynamic_full"]["fused_dynamic_speedup_vs_fp32"] > 1.0 for row in results
        ),
        "tile_dynamic_full_speedup_count_gt_1": sum(
            row["best_tile_dynamic_full"]["tile_dynamic_speedup_vs_fp32"] > 1.0 for row in results
        ),
        "block_dynamic_full_speedup_count_gt_1": sum(
            row["best_block_dynamic_full"]["block_dynamic_speedup_vs_fp32"] > 1.0 for row in results
        ),
    }


def _best_shape_payload(row: dict | None, key: str, speedup_key: str) -> dict | None:
    if row is None:
        return None
    best = row[key]
    return {
        "candidate": row["candidate"],
        "branch": row["branch"],
        "fused_point": row["fused_point"],
        "M": row["M"],
        "K": row["K"],
        "N": row["N"],
        "fp32_ms": row["fp32_fused_latency_ms_mean"],
        "latency_ms": best[speedup_key.replace("speedup_vs_fp32", "latency_ms_mean")],
        "speedup": best[speedup_key],
        "config": {
            "block_m": best["block_m"],
            "block_n": best["block_n"],
            "block_k": best["block_k"],
            "num_warps": best["num_warps"],
            "num_stages": best["num_stages"],
        },
    }


def build_analysis(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# P3b-3 Triton W8A8 Analysis",
        "",
        "## Headline",
        "",
    ]
    best_dynamic = summary["best_dynamic_full_shape"]
    best_static = summary["best_static_full_shape"]
    best_prequant = summary["best_prequant_shape"]
    best_fused_dynamic = summary["best_fused_dynamic_full_shape"]
    best_fused_static = summary["best_fused_static_full_shape"]
    best_tile_dynamic = summary["best_tile_dynamic_full_shape"]
    best_block_dynamic = summary["best_block_dynamic_full_shape"]
    if best_dynamic is not None:
        lines.append(
            "- Best dynamic full-path speedup: "
            f"`{best_dynamic['speedup']:.3f}x` at "
            f"`M={best_dynamic['M']} K={best_dynamic['K']} N={best_dynamic['N']}`."
        )
    if best_static is not None:
        lines.append(
            "- Best static-scale full-path speedup: "
            f"`{best_static['speedup']:.3f}x` at "
            f"`M={best_static['M']} K={best_static['K']} N={best_static['N']}`."
        )
    if best_fused_dynamic is not None:
        lines.append(
            "- Best fused-quant dynamic full-path speedup: "
            f"`{best_fused_dynamic['speedup']:.3f}x` at "
            f"`M={best_fused_dynamic['M']} K={best_fused_dynamic['K']} N={best_fused_dynamic['N']}`."
        )
    if best_fused_static is not None:
        lines.append(
            "- Best fused-quant static full-path speedup: "
            f"`{best_fused_static['speedup']:.3f}x` at "
            f"`M={best_fused_static['M']} K={best_fused_static['K']} N={best_fused_static['N']}`."
        )
    if best_tile_dynamic is not None:
        lines.append(
            "- Best tile-dynamic fused full-path speedup: "
            f"`{best_tile_dynamic['speedup']:.3f}x` at "
            f"`M={best_tile_dynamic['M']} K={best_tile_dynamic['K']} N={best_tile_dynamic['N']}`."
        )
    if best_block_dynamic is not None:
        lines.append(
            "- Best block-dynamic fused full-path speedup: "
            f"`{best_block_dynamic['speedup']:.3f}x` at "
            f"`M={best_block_dynamic['M']} K={best_block_dynamic['K']} N={best_block_dynamic['N']}`."
        )
    if best_prequant is not None:
        lines.append(
            "- Best prequantized-input speedup: "
            f"`{best_prequant['speedup']:.3f}x` at "
            f"`M={best_prequant['M']} K={best_prequant['K']} N={best_prequant['N']}`."
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            (
                "- `prequantized-input` isolates Triton tensor-core matmul plus dequant/bias. "
                "If this is not faster than FP32, the current Triton matmul kernel itself needs work."
            ),
            (
                "- `static-scale full` adds only elementwise quantization. "
                "The gap from prequantized-input estimates quantize-kernel cost without runtime amax."
            ),
            (
                "- `dynamic full` adds runtime activation amax. "
                "The gap from static-scale full is the practical cost of dynamic activation scale."
            ),
            (
                "- `fused-quant full` quantizes FP32 activation inside the Triton matmul tile, "
                "removing the standalone quantize kernel and global `q_x` write."
            ),
            (
                "- `tile-dynamic full` computes activation scale per matmul tile. "
                "It is a speed probe with different quantization semantics and needs a P3a-style precision screen."
            ),
            (
                "- `block-dynamic full` computes one scale per M block, stores a small scale vector, "
                "then reuses it across N tiles. It trades one extra launch for less duplicated activation scanning."
            ),
            "",
            "## Next Optimization Hooks",
            "",
            "- Tune `BLOCK_M/BLOCK_N/BLOCK_K`, `num_warps`, and `num_stages` around the winning edge shapes.",
            "- If dynamic full is the blocker, evaluate calibration/static activation scale on P3a precision gates.",
            "- If prequantized-input is the blocker, inspect generated PTX/Nsight Compute to confirm INT8 MMA usage.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_summary(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as fp:
        json.dump(payload, fp, indent=2)

    lines = [
        "# P3b-3 Triton W8A8 Microbenchmark",
        "",
        f"- device: `{payload['device']}`",
        f"- torch: `{payload['torch_version']}`",
        f"- triton: `{payload['triton_version']}`",
        f"- fusion_mode: `{payload['fusion_mode']}`",
        f"- candidates: `{', '.join(payload['candidates'])}`",
        f"- points: `{', '.join(payload['points'])}`",
        "",
        "## Shape Results",
        "",
        "| candidate | branch | point | M | K | N | fp32 ms | torch full ms | torch speedup | best prequant ms | prequant speedup | split static ms | split static speedup | split dynamic ms | split dynamic speedup | fused static ms | fused static speedup | fused dynamic ms | fused dynamic speedup | tile dynamic ms | tile dynamic speedup | block dynamic ms | block dynamic speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        pre = row["best_prequant"]
        static = row["best_static_full"]
        dyn = row["best_dynamic_full"]
        fused_static = row["best_fused_static_full"]
        fused_dyn = row["best_fused_dynamic_full"]
        tile_dyn = row["best_tile_dynamic_full"]
        block_dyn = row["best_block_dynamic_full"]
        lines.append(
            "| {candidate} | {branch} | {fused_point} | {M} | {K} | {N} | "
            "{fp32:.6f} | {torch_full:.6f} | {torch_speedup:.3f} | "
            "{pre_ms:.6f} | {pre_speedup:.3f} | "
            "{static_ms:.6f} | {static_speedup:.3f} | "
            "{dyn_ms:.6f} | {dyn_speedup:.3f} | "
            "{fused_static_ms:.6f} | {fused_static_speedup:.3f} | "
            "{fused_dyn_ms:.6f} | {fused_dyn_speedup:.3f} | "
            "{tile_dyn_ms:.6f} | {tile_dyn_speedup:.3f} | "
            "{block_dyn_ms:.6f} | {block_dyn_speedup:.3f} |".format(
                candidate=row["candidate"],
                branch=row["branch"],
                fused_point=row["fused_point"],
                M=row["M"],
                K=row["K"],
                N=row["N"],
                fp32=row["fp32_fused_latency_ms_mean"],
                torch_full=row["torch_full_latency_ms_mean"],
                torch_speedup=row["torch_full_speedup_vs_fp32"],
                pre_ms=pre["prequant_latency_ms_mean"],
                pre_speedup=pre["prequant_speedup_vs_fp32"],
                static_ms=static["static_full_latency_ms_mean"],
                static_speedup=static["static_full_speedup_vs_fp32"],
                dyn_ms=dyn["dynamic_full_latency_ms_mean"],
                dyn_speedup=dyn["dynamic_full_speedup_vs_fp32"],
                fused_static_ms=fused_static["fused_static_latency_ms_mean"],
                fused_static_speedup=fused_static["fused_static_speedup_vs_fp32"],
                fused_dyn_ms=fused_dyn["fused_dynamic_latency_ms_mean"],
                fused_dyn_speedup=fused_dyn["fused_dynamic_speedup_vs_fp32"],
                tile_dyn_ms=tile_dyn["tile_dynamic_latency_ms_mean"],
                tile_dyn_speedup=tile_dyn["tile_dynamic_speedup_vs_fp32"],
                block_dyn_ms=block_dyn["block_dynamic_latency_ms_mean"],
                block_dyn_speedup=block_dyn["block_dynamic_speedup_vs_fp32"],
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (output_dir / "analysis.md").write_text(build_analysis(payload))


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset = AseDBDataset(config={"src": args.dataset_src})
    indices = select_indices(len(dataset), args.limit, args.sample_seed)
    calc = MatRISCalculator(model=args.model, task=args.task, device=args.device)
    fused_modules = apply_gated_mlp_fusion(calc.model, FUSION_MODE)
    modules = dict(calc.model.named_modules())

    shapes = [
        shape
        for shape in collect_shapes(calc, dataset, indices, args.device)
        if shape["candidate"] in set(args.candidates) and shape["fused_point"] in set(args.points)
    ]
    results = [benchmark_shape(shape, modules, args) for shape in shapes]
    payload = {
        "phase": "P3b-3",
        "note": (
            "Triton W8A8 prototype. Reports prequantized input, static activation scale, "
            "and dynamic activation scale paths. Weight quantization is precomputed."
        ),
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "sample_indices": indices,
        "model": args.model,
        "task": args.task,
        "fusion_mode": FUSION_MODE,
        "fused_module_count": len(fused_modules),
        "candidates": list(args.candidates),
        "points": list(args.points),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "speedup_threshold": args.speedup_threshold,
        "results": results,
        "summary": summarize(results),
    }
    write_summary(Path(args.output_dir), payload)
    print(f"Wrote {Path(args.output_dir) / 'summary.json'}")
    print(f"Wrote {Path(args.output_dir) / 'summary.md'}")
    print(f"Wrote {Path(args.output_dir) / 'analysis.md'}")


if __name__ == "__main__":
    main()
