from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OP_SRC = REPO_ROOT / "matris" / "model" / "op" / "src"
if str(OP_SRC) not in sys.path:
    sys.path.insert(0, str(OP_SRC))


def load_matris_op():
    import matris_op  # type: ignore

    return matris_op


def parse_rows(value: str) -> list[int]:
    rows = []
    for item in value.split(","):
        item = item.strip()
        if item:
            rows.append(int(item))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P89A microbench for input_grad_only_gated_tail_backward.")
    parser.add_argument("--rows", default="1,2,4,8,16,32,64,128,256,512,728,1024,2048,2086,3672,4096,5088,8192")
    parser.add_argument("--dim", type=int, default=128, choices=[128, 256])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--phase-rows", default="728,2086,3672,5088")
    parser.add_argument("--output-dir", default="results/p89_tail_backward_micro")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def cuda_time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    mean = sum(samples) / len(samples)
    if len(samples) > 1:
        var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std


def make_inputs(rows: int, dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    grad_out = torch.randn(rows, dim, device=device, dtype=torch.float32)
    core = torch.randn(rows, dim, device=device, dtype=torch.float32)
    gate = torch.randn(rows, dim, device=device, dtype=torch.float32)
    core_weight = torch.randn(dim, device=device, dtype=torch.float32)
    core_bias = torch.randn(dim, device=device, dtype=torch.float32)
    gate_weight = torch.randn(dim, device=device, dtype=torch.float32)
    gate_bias = torch.randn(dim, device=device, dtype=torch.float32)
    return grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias


def torch_stats(
    core: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    core_centered = core - core.mean(dim=1, keepdim=True)
    gate_centered = gate - gate.mean(dim=1, keepdim=True)
    core_rstd = torch.rsqrt((core_centered * core_centered).mean(dim=1, keepdim=True) + eps)
    gate_rstd = torch.rsqrt((gate_centered * gate_centered).mean(dim=1, keepdim=True) + eps)
    return core_centered * core_rstd, gate_centered * gate_rstd, core_rstd, gate_rstd


def torch_activation_grad(
    grad_out: torch.Tensor,
    core_xhat: torch.Tensor,
    gate_xhat: torch.Tensor,
    core_weight: torch.Tensor,
    core_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_ln = core_xhat * core_weight.reshape(1, -1) + core_bias.reshape(1, -1)
    gate_ln = gate_xhat * gate_weight.reshape(1, -1) + gate_bias.reshape(1, -1)
    core_sig = torch.sigmoid(core_ln)
    core_act = core_ln * core_sig
    gate_act = torch.sigmoid(gate_ln)
    core_silu_grad = core_sig * (1.0 + core_ln * (1.0 - core_sig))
    grad_core_ln = grad_out * gate_act * core_silu_grad
    grad_gate_ln = grad_out * core_act * gate_act * (1.0 - gate_act)
    return grad_core_ln * core_weight.reshape(1, -1), grad_gate_ln * gate_weight.reshape(1, -1)


def torch_ln_backward_writeback(
    grad_core_norm: torch.Tensor,
    grad_gate_norm: torch.Tensor,
    core_xhat: torch.Tensor,
    gate_xhat: torch.Tensor,
    core_rstd: torch.Tensor,
    gate_rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = grad_core_norm.shape[1]
    core_sum_grad = grad_core_norm.sum(dim=1, keepdim=True)
    gate_sum_grad = grad_gate_norm.sum(dim=1, keepdim=True)
    core_sum_grad_xhat = (grad_core_norm * core_xhat).sum(dim=1, keepdim=True)
    gate_sum_grad_xhat = (grad_gate_norm * gate_xhat).sum(dim=1, keepdim=True)
    grad_core = (grad_core_norm * dim - core_sum_grad - core_xhat * core_sum_grad_xhat) * core_rstd / dim
    grad_gate = (grad_gate_norm * dim - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) * gate_rstd / dim
    return grad_core, grad_gate


def torch_full_proxy(
    grad_out: torch.Tensor,
    core: torch.Tensor,
    gate: torch.Tensor,
    core_weight: torch.Tensor,
    core_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_xhat, gate_xhat, core_rstd, gate_rstd = torch_stats(core, gate, eps)
    grad_core_norm, grad_gate_norm = torch_activation_grad(
        grad_out,
        core_xhat,
        gate_xhat,
        core_weight,
        core_bias,
        gate_weight,
        gate_bias,
    )
    return torch_ln_backward_writeback(
        grad_core_norm,
        grad_gate_norm,
        core_xhat,
        gate_xhat,
        core_rstd,
        gate_rstd,
    )


def bench_current_kernel(matris_op, rows: int, dim: int, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda")
    inputs = make_inputs(rows, dim, device)
    grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias = inputs
    eps = 1.0e-5
    mean, std = cuda_time_ms(
        lambda: matris_op.input_grad_only_gated_tail_backward(
            grad_out,
            core,
            gate,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
            eps,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    stack_mean, stack_std = cuda_time_ms(
        lambda: matris_op.input_grad_only_gated_tail_backward_stack(
            grad_out,
            core,
            gate,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
            eps,
        ),
        warmup=max(10, args.warmup // 2),
        iters=max(50, args.iters // 2),
    )
    v2_available = dim == 128 and hasattr(matris_op, "input_grad_only_gated_tail_backward_n128_v2")
    v2_mean = None
    v2_std = None
    v2_core_max_abs = None
    v2_gate_max_abs = None
    v2_core_mean_abs = None
    v2_gate_mean_abs = None
    if v2_available:
        with torch.no_grad():
            ref_core, ref_gate = matris_op.input_grad_only_gated_tail_backward(
                grad_out,
                core,
                gate,
                core_weight,
                core_bias,
                gate_weight,
                gate_bias,
                eps,
            )
            v2_core, v2_gate = matris_op.input_grad_only_gated_tail_backward_n128_v2(
                grad_out,
                core,
                gate,
                core_weight,
                core_bias,
                gate_weight,
                gate_bias,
                eps,
            )
            torch.cuda.synchronize()
            core_diff = (ref_core - v2_core).abs()
            gate_diff = (ref_gate - v2_gate).abs()
            v2_core_max_abs = float(core_diff.max().item())
            v2_gate_max_abs = float(gate_diff.max().item())
            v2_core_mean_abs = float(core_diff.mean().item())
            v2_gate_mean_abs = float(gate_diff.mean().item())
        v2_mean, v2_std = cuda_time_ms(
            lambda: matris_op.input_grad_only_gated_tail_backward_n128_v2(
                grad_out,
                core,
                gate,
                core_weight,
                core_bias,
                gate_weight,
                gate_bias,
                eps,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
    bytes_rw = rows * dim * 4 * 10
    return {
        "rows": rows,
        "dim": dim,
        "current_ms_mean": mean,
        "current_ms_std": std,
        "current_us_per_row": mean * 1000.0 / max(rows, 1),
        "stack_ms_mean": stack_mean,
        "stack_ms_std": stack_std,
        "stack_us_per_row": stack_mean * 1000.0 / max(rows, 1),
        "v2_available": v2_available,
        "v2_ms_mean": v2_mean,
        "v2_ms_std": v2_std,
        "v2_us_per_row": v2_mean * 1000.0 / max(rows, 1) if v2_mean is not None else None,
        "v2_speedup_vs_current": mean / v2_mean if v2_mean and v2_mean > 0 else None,
        "v2_core_max_abs": v2_core_max_abs,
        "v2_gate_max_abs": v2_gate_max_abs,
        "v2_core_mean_abs": v2_core_mean_abs,
        "v2_gate_mean_abs": v2_gate_mean_abs,
        "rough_gb_s_current": bytes_rw / (mean / 1000.0) / 1.0e9 if mean > 0 else 0.0,
    }


def bench_torch_phase_proxy(rows: int, dim: int, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda")
    grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias = make_inputs(rows, dim, device)
    eps = 1.0e-5
    core_xhat, gate_xhat, core_rstd, gate_rstd = torch_stats(core, gate, eps)
    grad_core_norm, grad_gate_norm = torch_activation_grad(
        grad_out,
        core_xhat,
        gate_xhat,
        core_weight,
        core_bias,
        gate_weight,
        gate_bias,
    )

    stats_mean, stats_std = cuda_time_ms(
        lambda: torch_stats(core, gate, eps),
        warmup=max(10, args.warmup // 2),
        iters=max(50, args.iters // 2),
    )
    act_mean, act_std = cuda_time_ms(
        lambda: torch_activation_grad(
            grad_out,
            core_xhat,
            gate_xhat,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
        ),
        warmup=max(10, args.warmup // 2),
        iters=max(50, args.iters // 2),
    )
    ln_bwd_mean, ln_bwd_std = cuda_time_ms(
        lambda: torch_ln_backward_writeback(
            grad_core_norm,
            grad_gate_norm,
            core_xhat,
            gate_xhat,
            core_rstd,
            gate_rstd,
        ),
        warmup=max(10, args.warmup // 2),
        iters=max(50, args.iters // 2),
    )
    full_mean, full_std = cuda_time_ms(
        lambda: torch_full_proxy(
            grad_out,
            core,
            gate,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
            eps,
        ),
        warmup=max(10, args.warmup // 2),
        iters=max(50, args.iters // 2),
    )
    return {
        "rows": rows,
        "dim": dim,
        "torch_stats_ms_mean": stats_mean,
        "torch_stats_ms_std": stats_std,
        "torch_activation_grad_ms_mean": act_mean,
        "torch_activation_grad_ms_std": act_std,
        "torch_ln_backward_writeback_ms_mean": ln_bwd_mean,
        "torch_ln_backward_writeback_ms_std": ln_bwd_std,
        "torch_full_proxy_ms_mean": full_mean,
        "torch_full_proxy_ms_std": full_std,
    }


def linear_fit(rows: list[int], ms: list[float]) -> dict[str, float]:
    n = len(rows)
    sx = sum(float(x) for x in rows)
    sy = sum(ms)
    sxx = sum(float(x) * float(x) for x in rows)
    sxy = sum(float(x) * y for x, y in zip(rows, ms))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return {"intercept_ms": 0.0, "slope_us_per_row": 0.0}
    slope_ms = (n * sxy - sx * sy) / denom
    intercept = (sy - slope_ms * sx) / n
    return {"intercept_ms": intercept, "slope_us_per_row": slope_ms * 1000.0}


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# P89A Tail Backward Micro Breakdown",
        "",
        "## Static Kernel Structure",
        "",
        "- op: `input_grad_only_gated_tail_backward_kernel`",
        "- launch shape: `grid=(rows)`, `block=(256)`",
        "- supported dim: `128` or `256`; current W8A8 second-tail uses `dim=128`",
        "- per row phases: mean reduction, variance reduction, activation/gate derivative, grad reduction, grad*xhat reduction, final writeback",
        "- reductions per row: `4` reductions for each core/gate pair",
        "- approximate sync count per row/block: `39` `__syncthreads()` calls",
        "- dim=128 uses only half of the 256 threads for element work; the reduction loops still run over 256 lanes",
        "",
        "## Current Kernel Row Sweep",
        "",
        "| rows | current ms | us/row | v2 ms | v2 speedup | v2 max abs | stack ms | rough GB/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["current_rows"]:
        v2_ms = row.get("v2_ms_mean")
        v2_speedup = row.get("v2_speedup_vs_current")
        v2_max_abs = None
        if row.get("v2_core_max_abs") is not None and row.get("v2_gate_max_abs") is not None:
            v2_max_abs = max(float(row["v2_core_max_abs"]), float(row["v2_gate_max_abs"]))
        lines.append(
            f"| {row['rows']} | {row['current_ms_mean']:.6f} | {row['current_us_per_row']:.4f} | "
            f"{v2_ms:.6f} | {v2_speedup:.3f}x | {v2_max_abs:.3e} | "
            f"{row['stack_ms_mean']:.6f} | {row['rough_gb_s_current']:.2f} |"
            if v2_ms is not None and v2_speedup is not None and v2_max_abs is not None
            else f"| {row['rows']} | {row['current_ms_mean']:.6f} | {row['current_us_per_row']:.4f} | "
            f"n/a | n/a | n/a | {row['stack_ms_mean']:.6f} | {row['rough_gb_s_current']:.2f} |"
        )

    fit = summary["fit_all"]
    small_fit = summary["fit_small"]
    lines.extend(
        [
            "",
            "## Scaling Fit",
            "",
            f"- all rows fit: intercept `{fit['intercept_ms']:.6f} ms`, slope `{fit['slope_us_per_row']:.6f} us/row`",
            f"- rows <= 512 fit: intercept `{small_fit['intercept_ms']:.6f} ms`, slope `{small_fit['slope_us_per_row']:.6f} us/row`",
            "",
            "## Torch Phase Proxy",
            "",
            "This is not the custom kernel's internal timing. It is a decomposed PyTorch proxy to show which math families are expensive when separated.",
            "",
            "| rows | stats ms | activation grad ms | ln backward/writeback ms | full proxy ms |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["torch_phase_proxy"]:
        lines.append(
            f"| {row['rows']} | {row['torch_stats_ms_mean']:.6f} | "
            f"{row['torch_activation_grad_ms_mean']:.6f} | "
            f"{row['torch_ln_backward_writeback_ms_mean']:.6f} | "
            f"{row['torch_full_proxy_ms_mean']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## P89A Readout",
            "",
            "- The current kernel is not launch-only for normal MatRIS rows; latency scales with rows after a small intercept.",
            "- The static structure points at row-local LayerNorm reductions/synchronization as the main optimization target.",
            "- The current dim=128 path wastes half the 256-thread block for elementwise work, while still paying 256-lane reductions and syncs.",
            "- The stack-output variant is a layout experiment only; it does not remove the dominant reduction/sync work.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P89A microbench requires CUDA.")
    torch.manual_seed(args.seed)
    matris_op = load_matris_op()
    rows = parse_rows(args.rows)
    phase_rows = parse_rows(args.phase_rows)

    current_rows = [bench_current_kernel(matris_op, row, args.dim, args) for row in rows]
    torch_phase_proxy = [bench_torch_phase_proxy(row, args.dim, args) for row in phase_rows]
    fit_all = linear_fit([row["rows"] for row in current_rows], [row["current_ms_mean"] for row in current_rows])
    small = [row for row in current_rows if row["rows"] <= 512]
    fit_small = linear_fit([row["rows"] for row in small], [row["current_ms_mean"] for row in small])

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dim": args.dim,
        "warmup": args.warmup,
        "iters": args.iters,
        "current_rows": current_rows,
        "torch_phase_proxy": torch_phase_proxy,
        "fit_all": fit_all,
        "fit_small": fit_small,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "summary.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
