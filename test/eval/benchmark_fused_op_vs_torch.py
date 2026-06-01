from __future__ import annotations

import argparse
import json
import math
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated torch-vs-MatRIS CUDA fused op microbench.")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="results/fused_op_vs_torch_microbench_20260525.json")
    return parser.parse_args()


def cuda_time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    mean = sum(samples) / len(samples)
    std = math.sqrt(sum((x - mean) ** 2 for x in samples) / max(1, len(samples) - 1))
    return mean, std


def rel_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def make_index(rows: int, num_segments: int, *, device: torch.device, grouped: bool) -> torch.Tensor:
    base = torch.arange(rows, device=device, dtype=torch.long) % num_segments
    if grouped:
        return torch.sort(base).values.contiguous()
    perm = torch.randperm(rows, device=device)
    return base[perm].contiguous()


def torch_directed_forward(x: torch.Tensor, segment: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = torch.zeros((num_segments, x.shape[1]), device=x.device, dtype=x.dtype)
    return out.index_add(0, segment, x) * 0.5


def torch_directed_backward(grad_out: torch.Tensor, segment: torch.Tensor) -> torch.Tensor:
    return grad_out.index_select(0, segment) * 0.5


def bench_directed_average(matris_op, rows: int, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda")
    num_segments = rows // 2
    segment = torch.arange(num_segments, device=device, dtype=torch.long).repeat_interleave(2).contiguous()
    x = torch.randn((rows, 128), device=device)
    grad_out = torch.randn((num_segments, 128), device=device)

    ref_f = torch_directed_forward(x, segment, num_segments)
    cuda_f = matris_op.directed2undirected_average_forward(x, segment, num_segments)
    ref_b = torch_directed_backward(grad_out, segment)
    cuda_b = matris_op.directed2undirected_average_backward(grad_out, segment, rows)

    torch_ms, torch_std = cuda_time_ms(
        lambda: (torch_directed_forward(x, segment, num_segments), torch_directed_backward(grad_out, segment)),
        warmup=args.warmup,
        iters=args.iters,
    )
    cuda_ms, cuda_std = cuda_time_ms(
        lambda: (
            matris_op.directed2undirected_average_forward(x, segment, num_segments),
            matris_op.directed2undirected_average_backward(grad_out, segment, rows),
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    return {
        "op": "directed2undirected_average_forward_backward",
        "rows": rows,
        "num_segments": num_segments,
        "torch_ms": torch_ms,
        "torch_std": torch_std,
        "cuda_ms": cuda_ms,
        "cuda_std": cuda_std,
        "speedup": torch_ms / cuda_ms,
        "forward_error": rel_error(ref_f, cuda_f),
        "backward_error": rel_error(ref_b, cuda_b),
    }


def silu_grad(x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def torch_mlp_input_grad(
    grad_out: torch.Tensor,
    weight2: torch.Tensor,
    hidden: torch.Tensor,
    weight1: torch.Tensor,
) -> torch.Tensor:
    grad_activated = grad_out.matmul(weight2)
    grad_hidden = grad_activated * silu_grad(hidden)
    return grad_hidden.contiguous().matmul(weight1)


def bench_mlp_input_grad(matris_op, rows: int, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda")
    grad_out = torch.randn((rows, 128), device=device)
    hidden = torch.randn((rows, 128), device=device)
    weight1 = torch.randn((128, 128), device=device)
    weight2 = torch.randn((128, 128), device=device)
    ref = torch_mlp_input_grad(grad_out, weight2, hidden, weight1)
    out = matris_op.two_linear_silu_input_grad_backward_n128(grad_out, weight2, hidden, weight1)
    torch_ms, torch_std = cuda_time_ms(
        lambda: torch_mlp_input_grad(grad_out, weight2, hidden, weight1),
        warmup=args.warmup,
        iters=args.iters,
    )
    cuda_ms, cuda_std = cuda_time_ms(
        lambda: matris_op.two_linear_silu_input_grad_backward_n128(grad_out, weight2, hidden, weight1),
        warmup=args.warmup,
        iters=args.iters,
    )
    return {
        "op": "two_linear_silu_input_grad_backward_n128",
        "rows": rows,
        "torch_ms": torch_ms,
        "torch_std": torch_std,
        "cuda_ms": cuda_ms,
        "cuda_std": cuda_std,
        "speedup": torch_ms / cuda_ms,
        "error": rel_error(ref, out),
    }


def torch_gated_tail_backward(
    grad_out: torch.Tensor,
    core: torch.Tensor,
    gate: torch.Tensor,
    core_weight: torch.Tensor,
    core_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_centered = core - core.mean(dim=1, keepdim=True)
    gate_centered = gate - gate.mean(dim=1, keepdim=True)
    core_rstd = torch.rsqrt((core_centered * core_centered).mean(dim=1, keepdim=True) + eps)
    gate_rstd = torch.rsqrt((gate_centered * gate_centered).mean(dim=1, keepdim=True) + eps)
    core_xhat = core_centered * core_rstd
    gate_xhat = gate_centered * gate_rstd
    core_ln = core_xhat * core_weight.reshape(1, -1) + core_bias.reshape(1, -1)
    gate_ln = gate_xhat * gate_weight.reshape(1, -1) + gate_bias.reshape(1, -1)
    core_sig = torch.sigmoid(core_ln)
    core_act = core_ln * core_sig
    gate_act = torch.sigmoid(gate_ln)
    core_silu_grad = core_sig * (1.0 + core_ln * (1.0 - core_sig))
    grad_core_norm = grad_out * gate_act * core_silu_grad * core_weight.reshape(1, -1)
    grad_gate_norm = grad_out * core_act * gate_act * (1.0 - gate_act) * gate_weight.reshape(1, -1)
    dim = grad_out.shape[1]
    grad_core = (
        (grad_core_norm * dim - grad_core_norm.sum(dim=1, keepdim=True)
         - core_xhat * (grad_core_norm * core_xhat).sum(dim=1, keepdim=True))
        * core_rstd
        / dim
    )
    grad_gate = (
        (grad_gate_norm * dim - grad_gate_norm.sum(dim=1, keepdim=True)
         - gate_xhat * (grad_gate_norm * gate_xhat).sum(dim=1, keepdim=True))
        * gate_rstd
        / dim
    )
    return grad_core, grad_gate


def bench_gated_tail(matris_op, rows: int, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda")
    eps = 1.0e-5
    grad_out = torch.randn((rows, 128), device=device)
    core = torch.randn((rows, 128), device=device)
    gate = torch.randn((rows, 128), device=device)
    core_weight = torch.randn((128,), device=device)
    core_bias = torch.randn((128,), device=device)
    gate_weight = torch.randn((128,), device=device)
    gate_bias = torch.randn((128,), device=device)
    ref_core, ref_gate = torch_gated_tail_backward(
        grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps
    )
    out_core, out_gate = matris_op.input_grad_only_gated_tail_backward(
        grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps
    )
    torch_ms, torch_std = cuda_time_ms(
        lambda: torch_gated_tail_backward(
            grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    cuda_ms, cuda_std = cuda_time_ms(
        lambda: matris_op.input_grad_only_gated_tail_backward(
            grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    return {
        "op": "input_grad_only_gated_tail_backward",
        "rows": rows,
        "torch_ms": torch_ms,
        "torch_std": torch_std,
        "cuda_ms": cuda_ms,
        "cuda_std": cuda_std,
        "speedup": torch_ms / cuda_ms,
        "core_error": rel_error(ref_core, out_core),
        "gate_error": rel_error(ref_gate, out_gate),
    }


def torch_attention_side(
    logits: torch.Tensor,
    values: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = index.reshape(-1, 1).expand(-1, logits.shape[1])
    max_out = torch.full((num_segments, logits.shape[1]), -torch.inf, device=logits.device, dtype=logits.dtype)
    max_out.scatter_reduce_(0, expanded, logits, reduce="amax", include_self=True)
    exp_logits = torch.exp(logits - max_out.index_select(0, index))
    sums = torch.zeros_like(max_out)
    sums.index_add_(0, index, exp_logits)
    alpha = exp_logits / sums.index_select(0, index).clamp_min(1.0e-20)
    out = torch.zeros_like(max_out)
    out.index_add_(0, index, alpha * values)
    return out, alpha


def torch_attention_forward(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    values: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_out, source_alpha = torch_attention_side(source_logits, values, source_index, num_segments)
    target_out, target_alpha = torch_attention_side(target_logits, values, target_index, num_segments)
    return source_out, target_out, source_alpha, target_alpha


def torch_attention_backward(
    grad_source_out: torch.Tensor,
    grad_target_out: torch.Tensor,
    values: torch.Tensor,
    source_out: torch.Tensor,
    target_out: torch.Tensor,
    source_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gs = grad_source_out.index_select(0, source_index)
    gt = grad_target_out.index_select(0, target_index)
    os = source_out.index_select(0, source_index)
    ot = target_out.index_select(0, target_index)
    grad_source_logits = source_alpha * gs * (values - os)
    grad_target_logits = target_alpha * gt * (values - ot)
    grad_values = source_alpha * gs + target_alpha * gt
    return grad_source_logits, grad_target_logits, grad_values


def bench_attention(matris_op, rows: int, num_segments: int, grouped: bool, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda")
    source_index = make_index(rows, num_segments, device=device, grouped=grouped)
    target_index = make_index(rows, num_segments, device=device, grouped=grouped)
    source_logits = torch.randn((rows, 128), device=device)
    target_logits = torch.randn((rows, 128), device=device)
    values = torch.randn((rows, 128), device=device)
    grad_source_out = torch.randn((num_segments, 128), device=device)
    grad_target_out = torch.randn((num_segments, 128), device=device)

    ref_f = torch_attention_forward(source_logits, target_logits, values, source_index, target_index, num_segments)
    cuda_f = matris_op.fused_line_attention_forward(
        source_logits, target_logits, values, source_index, target_index, num_segments
    )
    ref_b = torch_attention_backward(
        grad_source_out, grad_target_out, values, ref_f[0], ref_f[1], ref_f[2], ref_f[3], source_index, target_index
    )
    cuda_b = matris_op.fused_line_attention_backward(
        grad_source_out, grad_target_out, values, cuda_f[0], cuda_f[1], cuda_f[2], cuda_f[3], source_index, target_index
    )

    torch_f_ms, torch_f_std = cuda_time_ms(
        lambda: torch_attention_forward(source_logits, target_logits, values, source_index, target_index, num_segments),
        warmup=args.warmup,
        iters=args.iters,
    )
    cuda_f_ms, cuda_f_std = cuda_time_ms(
        lambda: matris_op.fused_line_attention_forward(
            source_logits, target_logits, values, source_index, target_index, num_segments
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    torch_b_ms, torch_b_std = cuda_time_ms(
        lambda: torch_attention_backward(
            grad_source_out, grad_target_out, values, ref_f[0], ref_f[1], ref_f[2], ref_f[3], source_index, target_index
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    cuda_b_ms, cuda_b_std = cuda_time_ms(
        lambda: matris_op.fused_line_attention_backward(
            grad_source_out, grad_target_out, values, cuda_f[0], cuda_f[1], cuda_f[2], cuda_f[3], source_index, target_index
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    return {
        "op": "fused_line_attention_forward_backward",
        "rows": rows,
        "num_segments": num_segments,
        "grouped_index": grouped,
        "forward": {
            "torch_ms": torch_f_ms,
            "torch_std": torch_f_std,
            "cuda_ms": cuda_f_ms,
            "cuda_std": cuda_f_std,
            "speedup": torch_f_ms / cuda_f_ms,
            "source_out_error": rel_error(ref_f[0], cuda_f[0]),
            "target_out_error": rel_error(ref_f[1], cuda_f[1]),
            "source_alpha_error": rel_error(ref_f[2], cuda_f[2]),
            "target_alpha_error": rel_error(ref_f[3], cuda_f[3]),
        },
        "backward": {
            "torch_ms": torch_b_ms,
            "torch_std": torch_b_std,
            "cuda_ms": cuda_b_ms,
            "cuda_std": cuda_b_std,
            "speedup": torch_b_ms / cuda_b_ms,
            "grad_source_logits_error": rel_error(ref_b[0], cuda_b[0]),
            "grad_target_logits_error": rel_error(ref_b[1], cuda_b[1]),
            "grad_values_error": rel_error(ref_b[2], cuda_b[2]),
        },
        "forward_backward_total": {
            "torch_ms": torch_f_ms + torch_b_ms,
            "cuda_ms": cuda_f_ms + cuda_b_ms,
            "speedup": (torch_f_ms + torch_b_ms) / (cuda_f_ms + cuda_b_ms),
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    matris_op = load_matris_op()

    results: list[dict[str, Any]] = []
    for rows in (512, 2048, 8192):
        results.append(bench_directed_average(matris_op, rows, args))
    for rows in (512, 2048, 8192):
        results.append(bench_mlp_input_grad(matris_op, rows, args))
    for rows in (512, 2048, 8192):
        results.append(bench_gated_tail(matris_op, rows, args))

    # The same C++/CUDA attention op is used by line and atom attention; these are shape proxies.
    results.append(bench_attention(matris_op, rows=2048, num_segments=256, grouped=False, args=args))
    results.append(bench_attention(matris_op, rows=8192, num_segments=1024, grouped=False, args=args))
    results.append(bench_attention(matris_op, rows=2048, num_segments=256, grouped=True, args=args))
    results.append(bench_attention(matris_op, rows=8192, num_segments=1024, grouped=True, args=args))

    payload = {
        "metadata": {
            "device": torch.cuda.get_device_name(0),
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "note": "Synthetic isolated equivalent-subgraph benchmark; endpoint ablation ratios are separate.",
        },
        "results": results,
    }
    output = REPO_ROOT / args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
