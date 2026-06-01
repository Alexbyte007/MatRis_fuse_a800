import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant.layers import PackedW8A32Linear


SHAPES = [
    ("attn_edge_first", 2086, 384, 256),
    ("attn_edge_first", 2760, 384, 256),
    ("attn_edge_first", 3672, 384, 256),
    ("attn_edge_first", 5088, 384, 256),
    ("attn_edge_first", 7064, 384, 256),
    ("attn_edge_second", 2086, 256, 256),
    ("attn_edge_second", 2760, 256, 256),
    ("attn_edge_second", 3672, 256, 256),
    ("attn_edge_second", 5088, 256, 256),
    ("attn_edge_second", 7064, 256, 256),
    ("refine_edge_first", 2086, 512, 256),
    ("refine_edge_first", 2760, 512, 256),
    ("refine_edge_first", 3672, 512, 256),
    ("refine_edge_first", 5088, 512, 256),
    ("refine_edge_first", 7064, 512, 256),
    ("refine_edge_second", 2086, 256, 256),
    ("refine_edge_second", 2760, 256, 256),
    ("refine_edge_second", 3672, 256, 256),
    ("refine_edge_second", 5088, 256, 256),
    ("refine_edge_second", 7064, 256, 256),
]


@triton.jit
def _w8a32_dequant_matmul_kernel(
    x_ptr,
    q_weight_ptr,
    scale_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        qw = tl.load(
            q_weight_ptr + offs_n[None, :] * K + k_idxs[:, None],
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
        w = qw * scale[None, :]
        acc += tl.dot(x, w, input_precision="tf32")

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _triton_w8a32(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    m, k = x.shape
    n = q_weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    bias_arg = bias if bias is not None else x.new_empty(0)
    _w8a32_dequant_matmul_kernel[grid](
        x,
        q_weight,
        scale,
        bias_arg,
        out,
        m,
        k,
        n,
        bias is not None,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _bench(fn, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def _pack_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q_weight = torch.round(weight.float() / scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    return q_weight, scale.contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton W8A32 operator microbenchmark.")
    parser.add_argument("--output-dir", default="results/p3c_w8a32_triton_microbench_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260503)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        (16, 64, 32, 4, 3),
        (16, 64, 64, 4, 3),
        (16, 128, 64, 4, 3),
        (32, 64, 32, 4, 3),
        (32, 64, 64, 4, 3),
        (32, 64, 128, 4, 3),
        (32, 128, 32, 4, 3),
        (32, 128, 64, 4, 3),
        (32, 128, 128, 4, 3),
        (32, 256, 64, 4, 3),
        (64, 64, 32, 4, 3),
        (64, 64, 64, 4, 3),
        (64, 128, 32, 4, 3),
        (64, 128, 64, 4, 3),
        (64, 128, 128, 4, 3),
        (64, 256, 64, 4, 3),
        (128, 64, 64, 4, 3),
        (128, 128, 64, 4, 3),
        (128, 128, 128, 4, 3),
        (32, 128, 64, 8, 3),
        (64, 128, 64, 8, 3),
        (64, 256, 64, 8, 3),
        (128, 128, 64, 8, 3),
    ]

    results = []
    for branch, m, k, n in SHAPES:
        x = torch.randn((m, k), device=args.device, dtype=torch.float32)
        weight = torch.randn((n, k), device=args.device, dtype=torch.float32) / (k**0.5)
        bias = torch.randn((n,), device=args.device, dtype=torch.float32)
        q_weight, scale = _pack_weight(weight)
        dq_weight = q_weight.float() * scale[:, None]

        linear = torch.nn.Linear(k, n, bias=True, device=args.device, dtype=torch.float32)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        packed = PackedW8A32Linear(linear, f"bench.{branch}", use_cuda_kernel=True).to(args.device)

        fp32_ms, fp32_std = _bench(lambda: F.linear(x, weight, bias), args.warmup, args.repeat)
        dq_cached_ms, dq_cached_std = _bench(lambda: F.linear(x, dq_weight, bias), args.warmup, args.repeat)
        cuda_ms, cuda_std = _bench(lambda: packed(x), args.warmup, args.repeat)

        best = None
        for block_m, block_n, block_k, num_warps, num_stages in configs:
            y = _triton_w8a32(x, q_weight, scale, bias, block_m, block_n, block_k, num_warps, num_stages)
            ref = F.linear(x, dq_weight, bias)
            max_diff = (y - ref).abs().max().item()
            mean_diff = (y - ref).abs().mean().item()
            ms, std = _bench(
                lambda bm=block_m, bn=block_n, bk=block_k, nw=num_warps, ns=num_stages: _triton_w8a32(
                    x, q_weight, scale, bias, bm, bn, bk, nw, ns
                ),
                args.warmup,
                args.repeat,
            )
            item = {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "latency_ms_mean": ms,
                "latency_ms_std": std,
                "speedup_vs_fp32": fp32_ms / ms,
                "speedup_vs_dq_cached": dq_cached_ms / ms,
                "max_abs_diff_vs_dq_ref": max_diff,
                "mean_abs_diff_vs_dq_ref": mean_diff,
            }
            if best is None or item["latency_ms_mean"] < best["latency_ms_mean"]:
                best = item

        assert best is not None
        results.append(
            {
                "branch": branch,
                "M": m,
                "K": k,
                "N": n,
                "fp32_latency_ms_mean": fp32_ms,
                "fp32_latency_ms_std": fp32_std,
                "dq_cached_latency_ms_mean": dq_cached_ms,
                "dq_cached_latency_ms_std": dq_cached_std,
                "cuda_w8a32_latency_ms_mean": cuda_ms,
                "cuda_w8a32_latency_ms_std": cuda_std,
                "cuda_w8a32_speedup_vs_fp32": fp32_ms / cuda_ms,
                "best_triton": best,
            }
        )
        print(
            f"{branch} M={m} K={k} N={n}: "
            f"fp32={fp32_ms:.4f}ms cuda={cuda_ms:.4f}ms "
            f"triton={best['latency_ms_mean']:.4f}ms "
            f"triton_speedup={best['speedup_vs_fp32']:.3f}x"
        )

    summary = {
        "phase": "p3c_w8a32_triton_microbench",
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else args.device,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "results": results,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# P3c W8A32 Triton Microbenchmark",
        "",
        f"- device: `{summary['device']}`",
        f"- torch: `{torch.__version__}`",
        f"- triton: `{triton.__version__}`",
        "",
        "| branch | M | K | N | fp32 ms | cuda W8A32 ms | cuda speedup | best Triton ms | Triton speedup | config |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        b = r["best_triton"]
        config = f"{b['block_m']}x{b['block_n']}x{b['block_k']}/w{b['num_warps']}/s{b['num_stages']}"
        lines.append(
            f"| {r['branch']} | {r['M']} | {r['K']} | {r['N']} | "
            f"{r['fp32_latency_ms_mean']:.6f} | {r['cuda_w8a32_latency_ms_mean']:.6f} | "
            f"{r['cuda_w8a32_speedup_vs_fp32']:.3f} | {b['latency_ms_mean']:.6f} | "
            f"{b['speedup_vs_fp32']:.3f} | `{config}` |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    triton_faster = [r for r in results if r["best_triton"]["speedup_vs_fp32"] > 1.0]
    best = max(results, key=lambda r: r["best_triton"]["speedup_vs_fp32"])
    analysis = [
        "# P3c W8A32 Triton Analysis",
        "",
        f"- Triton faster than FP32 on `{len(triton_faster)}/{len(results)}` shapes.",
        f"- Best Triton speedup: `{best['best_triton']['speedup_vs_fp32']:.3f}x` "
        f"at `{best['branch']}` M={best['M']} K={best['K']} N={best['N']}.",
        "- This path keeps activation FP32, so it is a mixed-precision/weight-bandwidth probe rather than INT8 tensor-core GEMM.",
    ]
    (out_dir / "analysis.md").write_text("\n".join(analysis) + "\n")


if __name__ == "__main__":
    main()
