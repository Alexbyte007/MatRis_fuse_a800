from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
OP_SRC = REPO_ROOT / "matris" / "model" / "op" / "src"
if str(OP_SRC) not in sys.path:
    sys.path.insert(0, str(OP_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local torch.compile experiment for refine FFN glue: smooth gather/reduce "
            "+ node/edge FFN + residual and the matching manual input-grad path."
        )
    )
    parser.add_argument("--rows", default="512,1024,2048,4096,8192")
    parser.add_argument(
        "--cases",
        default="",
        help=(
            "Optional explicit rows:nodes cases, for example "
            "'4096:256,8192:512'. When set, this overrides --rows/--node-ratio."
        ),
    )
    parser.add_argument("--node-ratio", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--output-json", default="results/compile_local_refine_ffn_glue_shape_sweep_20260526.json")
    return parser.parse_args()


def load_matris_op():
    import matris_op  # type: ignore

    return matris_op


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_cases(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.cases.strip():
        cases = []
        for item in args.cases.split(","):
            if not item.strip():
                continue
            rows_text, nodes_text = item.split(":", 1)
            rows = int(rows_text.strip())
            nodes = int(nodes_text.strip())
            if rows <= 0 or nodes <= 0:
                raise ValueError(f"Invalid rows:nodes case: {item!r}")
            cases.append((rows, nodes))
        return cases
    return [(rows, max(1, int(round(rows * args.node_ratio)))) for rows in parse_int_list(args.rows)]


def cuda_time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, float]:
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
    return {"mean_ms": mean, "std_ms": std, "min_ms": min(samples), "max_ms": max(samples)}


def rel_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).detach().abs()
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def silu_grad(x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def ffn_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = F.linear(x, w1, b1)
    out = F.linear(F.silu(hidden), w2, b2)
    return out, hidden


def ffn_input_grad(
    grad_out: torch.Tensor,
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    grad_hidden = grad_out.matmul(w2) * silu_grad(hidden)
    return grad_hidden.matmul(w1)


def refine_ffn_glue_forward_template(
    nonlinear: torch.Tensor,
    base_envelope: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    node_feat: torch.Tensor,
    edge_feat: torch.Tensor,
    node_res: torch.Tensor,
    edge_res: torch.Tensor,
    node_w1: torch.Tensor,
    node_b1: torch.Tensor,
    node_w2: torch.Tensor,
    node_b2: torch.Tensor,
    edge_w1: torch.Tensor,
    edge_b1: torch.Tensor,
    edge_w2: torch.Tensor,
    edge_b2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_i = base_envelope.index_select(0, source_index)
    base_j = base_envelope.index_select(0, target_index)
    smoothed = nonlinear * base_i * base_j
    node_agg = nonlinear.new_zeros((node_feat.shape[0], nonlinear.shape[1])).index_add(0, target_index, smoothed)
    delta_node, node_hidden = ffn_forward(node_agg, node_w1, node_b1, node_w2, node_b2)
    delta_edge, edge_hidden = ffn_forward(nonlinear, edge_w1, edge_b1, edge_w2, edge_b2)
    node_out = delta_node + node_res * node_feat
    edge_out = delta_edge + edge_res * edge_feat
    return node_out, edge_out, node_agg, node_hidden, edge_hidden


def refine_ffn_glue_forward_backward_template(
    nonlinear: torch.Tensor,
    base_envelope: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    node_feat: torch.Tensor,
    edge_feat: torch.Tensor,
    node_res: torch.Tensor,
    edge_res: torch.Tensor,
    node_w1: torch.Tensor,
    node_b1: torch.Tensor,
    node_w2: torch.Tensor,
    node_b2: torch.Tensor,
    edge_w1: torch.Tensor,
    edge_b1: torch.Tensor,
    edge_w2: torch.Tensor,
    edge_b2: torch.Tensor,
    grad_node_out: torch.Tensor,
    grad_edge_out: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    node_out, edge_out, _node_agg, node_hidden, edge_hidden = refine_ffn_glue_forward_template(
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        node_feat,
        edge_feat,
        node_res,
        edge_res,
        node_w1,
        node_b1,
        node_w2,
        node_b2,
        edge_w1,
        edge_b1,
        edge_w2,
        edge_b2,
    )
    grad_node_feat = grad_node_out * node_res
    grad_edge_feat = grad_edge_out * edge_res
    grad_node_agg = ffn_input_grad(grad_node_out, node_hidden, node_w1, node_w2)
    grad_edge_from_ffn = ffn_input_grad(grad_edge_out, edge_hidden, edge_w1, edge_w2)
    grad_smoothed = grad_node_agg.index_select(0, target_index)
    base_i = base_envelope.index_select(0, source_index)
    base_j = base_envelope.index_select(0, target_index)
    grad_nonlinear = grad_edge_from_ffn + grad_smoothed * base_i * base_j
    grad_base_source = grad_smoothed * nonlinear * base_j
    grad_base_target = grad_smoothed * nonlinear * base_i
    grad_base = torch.zeros_like(base_envelope)
    grad_base = grad_base.index_add(0, source_index, grad_base_source)
    grad_base = grad_base.index_add(0, target_index, grad_base_target)
    return node_out, edge_out, grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat


def cuda_ffn_input_grad(matris_op, grad_out: torch.Tensor, hidden: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return matris_op.two_linear_silu_input_grad_backward_n128(
        grad_out.contiguous(),
        w2.contiguous(),
        hidden.contiguous(),
        w1.contiguous(),
    )


def cuda_ffn_input_grad_cublas(matris_op, grad_out: torch.Tensor, hidden: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return matris_op.two_linear_silu_input_grad_backward_n128_cublas(
        grad_out.contiguous(),
        w2.contiguous(),
        hidden.contiguous(),
        w1.contiguous(),
    )


def cuda_mixed_forward(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        node_feat,
        edge_feat,
        node_res,
        edge_res,
        node_w1,
        node_b1,
        node_w2,
        node_b2,
        edge_w1,
        edge_b1,
        edge_w2,
        edge_b2,
        _grad_node_out,
        _grad_edge_out,
    ) = tensors
    node_agg = matris_op.refine_line_smooth_reduce_forward(
        nonlinear.contiguous(),
        base_envelope.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        int(node_feat.shape[0]),
    )
    delta_node, node_hidden = ffn_forward(node_agg, node_w1, node_b1, node_w2, node_b2)
    delta_edge, edge_hidden = ffn_forward(nonlinear, edge_w1, edge_b1, edge_w2, edge_b2)
    node_out = delta_node + node_res * node_feat
    edge_out = delta_edge + edge_res * edge_feat
    return node_out, edge_out, node_agg, node_hidden, edge_hidden


def cuda_mixed_forward_backward(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        node_feat,
        edge_feat,
        node_res,
        edge_res,
        node_w1,
        node_b1,
        node_w2,
        node_b2,
        edge_w1,
        edge_b1,
        edge_w2,
        edge_b2,
        grad_node_out,
        grad_edge_out,
    ) = tensors
    node_out, edge_out, _node_agg, node_hidden, edge_hidden = cuda_mixed_forward(matris_op, tensors)
    grad_node_feat = grad_node_out * node_res
    grad_edge_feat = grad_edge_out * edge_res
    grad_node_agg = cuda_ffn_input_grad(matris_op, grad_node_out, node_hidden, node_w1, node_w2)
    grad_edge_from_ffn = cuda_ffn_input_grad(matris_op, grad_edge_out, edge_hidden, edge_w1, edge_w2)
    grad_nonlinear_smooth, grad_base = matris_op.refine_line_smooth_reduce_backward(
        grad_node_agg.contiguous(),
        nonlinear.contiguous(),
        base_envelope.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
    )
    grad_nonlinear = grad_nonlinear_smooth + grad_edge_from_ffn
    return node_out, edge_out, grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat


def cuda_cublas_ffn_forward_backward(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        node_feat,
        edge_feat,
        node_res,
        edge_res,
        node_w1,
        node_b1,
        node_w2,
        node_b2,
        edge_w1,
        edge_b1,
        edge_w2,
        edge_b2,
        grad_node_out,
        grad_edge_out,
    ) = tensors
    node_out, edge_out, _node_agg, node_hidden, edge_hidden = cuda_mixed_forward(matris_op, tensors)
    grad_node_feat = grad_node_out * node_res
    grad_edge_feat = grad_edge_out * edge_res
    grad_node_agg = cuda_ffn_input_grad_cublas(matris_op, grad_node_out, node_hidden, node_w1, node_w2)
    grad_edge_from_ffn = cuda_ffn_input_grad_cublas(matris_op, grad_edge_out, edge_hidden, edge_w1, edge_w2)
    grad_nonlinear_smooth, grad_base = matris_op.refine_line_smooth_reduce_backward(
        grad_node_agg.contiguous(),
        nonlinear.contiguous(),
        base_envelope.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
    )
    grad_nonlinear = grad_nonlinear_smooth + grad_edge_from_ffn
    return node_out, edge_out, grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat


def cuda_macro_r12_forward_backward(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        node_feat,
        edge_feat,
        node_res,
        edge_res,
        node_w1,
        node_b1,
        node_w2,
        node_b2,
        edge_w1,
        edge_b1,
        edge_w2,
        edge_b2,
        grad_node_out,
        grad_edge_out,
    ) = tensors
    node_out, edge_out, _node_agg, node_hidden, edge_hidden = cuda_mixed_forward(matris_op, tensors)
    grad_node_agg = cuda_ffn_input_grad(matris_op, grad_node_out, node_hidden, node_w1, node_w2)
    grad_edge_from_ffn = cuda_ffn_input_grad(matris_op, grad_edge_out, edge_hidden, edge_w1, edge_w2)
    grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat = (
        matris_op.refine_line_smooth_reduce_backward_direct_residual(
            grad_node_agg.contiguous(),
            grad_edge_from_ffn.contiguous(),
            nonlinear.contiguous(),
            base_envelope.contiguous(),
            source_index.contiguous(),
            target_index.contiguous(),
            grad_node_out.contiguous(),
            grad_edge_out.contiguous(),
            node_res.contiguous(),
            edge_res.contiguous(),
        )
    )
    return node_out, edge_out, grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat


def cuda_cublas_ffn_r12_forward_backward(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        node_feat,
        edge_feat,
        node_res,
        edge_res,
        node_w1,
        node_b1,
        node_w2,
        node_b2,
        edge_w1,
        edge_b1,
        edge_w2,
        edge_b2,
        grad_node_out,
        grad_edge_out,
    ) = tensors
    node_out, edge_out, _node_agg, node_hidden, edge_hidden = cuda_mixed_forward(matris_op, tensors)
    grad_node_agg = cuda_ffn_input_grad_cublas(matris_op, grad_node_out, node_hidden, node_w1, node_w2)
    grad_edge_from_ffn = cuda_ffn_input_grad_cublas(matris_op, grad_edge_out, edge_hidden, edge_w1, edge_w2)
    grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat = (
        matris_op.refine_line_smooth_reduce_backward_direct_residual(
            grad_node_agg.contiguous(),
            grad_edge_from_ffn.contiguous(),
            nonlinear.contiguous(),
            base_envelope.contiguous(),
            source_index.contiguous(),
            target_index.contiguous(),
            grad_node_out.contiguous(),
            grad_edge_out.contiguous(),
            node_res.contiguous(),
            edge_res.contiguous(),
        )
    )
    return node_out, edge_out, grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat


def make_case(rows: int, num_nodes: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    nonlinear = torch.randn((rows, 128), device=device)
    base_envelope = torch.randn((num_nodes, 128), device=device)
    source_index = torch.randint(0, num_nodes, (rows,), device=device, dtype=torch.long).contiguous()
    target_index = torch.randint(0, num_nodes, (rows,), device=device, dtype=torch.long).contiguous()
    node_feat = torch.randn((num_nodes, 128), device=device)
    edge_feat = torch.randn((rows, 128), device=device)
    node_res = torch.randn((1, 128), device=device)
    edge_res = torch.randn((1, 128), device=device)
    node_w1 = torch.randn((128, 128), device=device)
    node_b1 = torch.randn((128,), device=device)
    node_w2 = torch.randn((128, 128), device=device)
    node_b2 = torch.randn((128,), device=device)
    edge_w1 = torch.randn((128, 128), device=device)
    edge_b1 = torch.randn((128,), device=device)
    edge_w2 = torch.randn((128, 128), device=device)
    edge_b2 = torch.randn((128,), device=device)
    grad_node_out = torch.randn((num_nodes, 128), device=device)
    grad_edge_out = torch.randn((rows, 128), device=device)
    return (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        node_feat,
        edge_feat,
        node_res,
        edge_res,
        node_w1,
        node_b1,
        node_w2,
        node_b2,
        edge_w1,
        edge_b1,
        edge_w2,
        edge_b2,
        grad_node_out,
        grad_edge_out,
    )


def bench_case(
    matris_op,
    compiled_forward: Callable[..., tuple[torch.Tensor, ...]],
    compiled_forward_backward: Callable[..., tuple[torch.Tensor, ...]],
    rows: int,
    num_nodes: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda")
    tensors = make_case(rows, num_nodes, device)
    eager_f = refine_ffn_glue_forward_template(*tensors[:-2])
    compiled_f = compiled_forward(*tensors[:-2])
    cuda_f = cuda_mixed_forward(matris_op, tensors)
    eager_fb = refine_ffn_glue_forward_backward_template(*tensors)
    compiled_fb = compiled_forward_backward(*tensors)
    cuda_fb = cuda_mixed_forward_backward(matris_op, tensors)
    cuda_cublas_fb = (
        cuda_cublas_ffn_forward_backward(matris_op, tensors)
        if hasattr(matris_op, "two_linear_silu_input_grad_backward_n128_cublas")
        else None
    )
    cuda_r12_fb = (
        cuda_macro_r12_forward_backward(matris_op, tensors)
        if hasattr(matris_op, "refine_line_smooth_reduce_backward_direct_residual")
        else None
    )
    cuda_cublas_r12_fb = (
        cuda_cublas_ffn_r12_forward_backward(matris_op, tensors)
        if (
            hasattr(matris_op, "two_linear_silu_input_grad_backward_n128_cublas")
            and hasattr(matris_op, "refine_line_smooth_reduce_backward_direct_residual")
        )
        else None
    )

    forward_time = {
        "torch_eager": cuda_time_ms(lambda: refine_ffn_glue_forward_template(*tensors[:-2]), warmup=args.warmup, iters=args.iters),
        "torch_compile": cuda_time_ms(lambda: compiled_forward(*tensors[:-2]), warmup=args.warmup, iters=args.iters),
        "cuda_mixed": cuda_time_ms(lambda: cuda_mixed_forward(matris_op, tensors), warmup=args.warmup, iters=args.iters),
    }
    forward_backward_time = {
        "torch_eager": cuda_time_ms(lambda: refine_ffn_glue_forward_backward_template(*tensors), warmup=args.warmup, iters=args.iters),
        "torch_compile": cuda_time_ms(lambda: compiled_forward_backward(*tensors), warmup=args.warmup, iters=args.iters),
        "cuda_mixed": cuda_time_ms(lambda: cuda_mixed_forward_backward(matris_op, tensors), warmup=args.warmup, iters=args.iters),
    }
    if cuda_r12_fb is not None:
        forward_backward_time["cuda_macro_r12"] = cuda_time_ms(
            lambda: cuda_macro_r12_forward_backward(matris_op, tensors),
            warmup=args.warmup,
            iters=args.iters,
        )
    if cuda_cublas_fb is not None:
        forward_backward_time["cuda_cublas_ffn"] = cuda_time_ms(
            lambda: cuda_cublas_ffn_forward_backward(matris_op, tensors),
            warmup=args.warmup,
            iters=args.iters,
        )
    if cuda_cublas_r12_fb is not None:
        forward_backward_time["cuda_cublas_ffn_r12"] = cuda_time_ms(
            lambda: cuda_cublas_ffn_r12_forward_backward(matris_op, tensors),
            warmup=args.warmup,
            iters=args.iters,
        )

    fb_speedups = {
        "torch_compile": forward_backward_time["torch_eager"]["mean_ms"] / forward_backward_time["torch_compile"]["mean_ms"],
        "cuda_mixed": forward_backward_time["torch_eager"]["mean_ms"] / forward_backward_time["cuda_mixed"]["mean_ms"],
    }
    if "cuda_macro_r12" in forward_backward_time:
        fb_speedups["cuda_macro_r12"] = (
            forward_backward_time["torch_eager"]["mean_ms"] / forward_backward_time["cuda_macro_r12"]["mean_ms"]
        )
    if "cuda_cublas_ffn" in forward_backward_time:
        fb_speedups["cuda_cublas_ffn"] = (
            forward_backward_time["torch_eager"]["mean_ms"] / forward_backward_time["cuda_cublas_ffn"]["mean_ms"]
        )
    if "cuda_cublas_ffn_r12" in forward_backward_time:
        fb_speedups["cuda_cublas_ffn_r12"] = (
            forward_backward_time["torch_eager"]["mean_ms"] / forward_backward_time["cuda_cublas_ffn_r12"]["mean_ms"]
        )

    return {
        "op": "refine_ffn_glue_line_like",
        "rows": rows,
        "num_nodes": num_nodes,
        "node_ratio": float(num_nodes / rows),
        "dim": 128,
        "forward_time": forward_time,
        "forward_backward_time": forward_backward_time,
        "speedup_vs_eager": {
            "forward": {
                "torch_compile": forward_time["torch_eager"]["mean_ms"] / forward_time["torch_compile"]["mean_ms"],
                "cuda_mixed": forward_time["torch_eager"]["mean_ms"] / forward_time["cuda_mixed"]["mean_ms"],
            },
            "forward_backward": fb_speedups,
        },
        "errors_vs_torch_eager": {
            "forward_compile_node_out": rel_error(eager_f[0], compiled_f[0]),
            "forward_compile_edge_out": rel_error(eager_f[1], compiled_f[1]),
            "forward_cuda_node_out": rel_error(eager_f[0], cuda_f[0]),
            "forward_cuda_edge_out": rel_error(eager_f[1], cuda_f[1]),
            "fb_compile_grad_nonlinear": rel_error(eager_fb[2], compiled_fb[2]),
            "fb_compile_grad_base": rel_error(eager_fb[3], compiled_fb[3]),
            "fb_cuda_grad_nonlinear": rel_error(eager_fb[2], cuda_fb[2]),
            "fb_cuda_grad_base": rel_error(eager_fb[3], cuda_fb[3]),
            "fb_cuda_r12_grad_nonlinear": rel_error(eager_fb[2], cuda_r12_fb[2]) if cuda_r12_fb is not None else None,
            "fb_cuda_r12_grad_base": rel_error(eager_fb[3], cuda_r12_fb[3]) if cuda_r12_fb is not None else None,
            "fb_cuda_r12_grad_node_feat": rel_error(eager_fb[4], cuda_r12_fb[4]) if cuda_r12_fb is not None else None,
            "fb_cuda_r12_grad_edge_feat": rel_error(eager_fb[5], cuda_r12_fb[5]) if cuda_r12_fb is not None else None,
            "fb_cuda_cublas_grad_nonlinear": rel_error(eager_fb[2], cuda_cublas_fb[2]) if cuda_cublas_fb is not None else None,
            "fb_cuda_cublas_grad_base": rel_error(eager_fb[3], cuda_cublas_fb[3]) if cuda_cublas_fb is not None else None,
            "fb_cuda_cublas_r12_grad_nonlinear": rel_error(eager_fb[2], cuda_cublas_r12_fb[2]) if cuda_cublas_r12_fb is not None else None,
            "fb_cuda_cublas_r12_grad_base": rel_error(eager_fb[3], cuda_cublas_r12_fb[3]) if cuda_cublas_r12_fb is not None else None,
            "fb_compile_grad_node_feat": rel_error(eager_fb[4], compiled_fb[4]),
            "fb_cuda_grad_node_feat": rel_error(eager_fb[4], cuda_fb[4]),
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_speedup(section: str, mode: str) -> float:
        return sum(float(item["speedup_vs_eager"][section][mode]) for item in results) / len(results)

    return {
        "num_cases": len(results),
        "rows": [int(item["rows"]) for item in results],
        "num_nodes": [int(item["num_nodes"]) for item in results],
        "forward_compile_speedup_mean": mean_speedup("forward", "torch_compile"),
        "forward_cuda_mixed_speedup_mean": mean_speedup("forward", "cuda_mixed"),
        "forward_backward_compile_speedup_mean": mean_speedup("forward_backward", "torch_compile"),
        "forward_backward_cuda_mixed_speedup_mean": mean_speedup("forward_backward", "cuda_mixed"),
        "forward_backward_cuda_macro_r12_speedup_mean": (
            mean_speedup("forward_backward", "cuda_macro_r12")
            if all("cuda_macro_r12" in item["speedup_vs_eager"]["forward_backward"] for item in results)
            else None
        ),
        "forward_backward_cuda_cublas_ffn_speedup_mean": (
            mean_speedup("forward_backward", "cuda_cublas_ffn")
            if all("cuda_cublas_ffn" in item["speedup_vs_eager"]["forward_backward"] for item in results)
            else None
        ),
        "forward_backward_cuda_cublas_ffn_r12_speedup_mean": (
            mean_speedup("forward_backward", "cuda_cublas_ffn_r12")
            if all("cuda_cublas_ffn_r12" in item["speedup_vs_eager"]["forward_backward"] for item in results)
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    matris_op = load_matris_op()
    if not hasattr(matris_op, "refine_line_smooth_reduce_forward"):
        raise RuntimeError("matris_op.refine_line_smooth_reduce_forward is unavailable")
    if not hasattr(matris_op, "two_linear_silu_input_grad_backward_n128"):
        raise RuntimeError("matris_op.two_linear_silu_input_grad_backward_n128 is unavailable")
    compiled_forward = torch.compile(refine_ffn_glue_forward_template, dynamic=True, mode=args.torch_compile_mode)
    compiled_forward_backward = torch.compile(
        refine_ffn_glue_forward_backward_template,
        dynamic=True,
        mode=args.torch_compile_mode,
    )
    cases = parse_cases(args)
    results = [
        bench_case(matris_op, compiled_forward, compiled_forward_backward, rows, num_nodes, args)
        for rows, num_nodes in cases
    ]
    payload = {
        "metadata": {
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_compile": {
                "enabled": True,
                "dynamic": True,
                "mode": args.torch_compile_mode,
                "logs_hint": 'TORCH_LOGS="graph_breaks,recompiles"',
            },
            "boundary": (
                "refine_line-like smooth gather/reduce + node_FFN/edge_FFN + residual, "
                "with manual input-grad path"
            ),
            "modes": {
                "torch_eager": "pure PyTorch ops",
                "torch_compile": "same PyTorch boundary compiled with dynamic=True",
                "cuda_mixed": "matris_op refine_line_smooth_reduce forward/backward + two_linear_silu_input_grad + PyTorch residual/FFN forward glue",
                "cuda_macro_r12": "cuda_mixed plus fused smooth backward/direct-grad/residual input-grad macro when available",
                "cuda_cublas_ffn": "cuda_mixed with cuBLAS two-linear SiLU input-grad",
                "cuda_cublas_ffn_r12": "cuBLAS FFN input-grad plus R-MACRO1/2 fused smooth/direct/residual backward",
            },
            "cases": [{"rows": rows, "num_nodes": num_nodes} for rows, num_nodes in cases],
            "rows": [rows for rows, _num_nodes in cases],
            "node_ratio": args.node_ratio,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "summary": summarize(results),
        "results": results,
    }
    output = REPO_ROOT / args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
