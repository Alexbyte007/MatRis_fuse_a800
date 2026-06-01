from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local torch.compile experiment for residual glue: update + res_weight * old_feat."
    )
    parser.add_argument("--rows", default="16,32,64,128,256,512,1024,2048,4096,8192")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--output-json", default="results/compile_local_residual_glue_shape_sweep_20260526.json")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


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


def residual_forward(update: torch.Tensor, old_feat: torch.Tensor, res_weight: torch.Tensor) -> torch.Tensor:
    return update + res_weight * old_feat


def residual_backward(
    grad_out: torch.Tensor,
    old_feat: torch.Tensor,
    res_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_update = grad_out
    grad_old = grad_out * res_weight
    grad_res = (grad_out * old_feat).sum(dim=0, keepdim=True)
    return grad_update, grad_old, grad_res


def residual_forward_backward(
    update: torch.Tensor,
    old_feat: torch.Tensor,
    res_weight: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = residual_forward(update, old_feat, res_weight)
    grad_update, grad_old, grad_res = residual_backward(grad_out, old_feat, res_weight)
    return out, grad_update, grad_old, grad_res


def residual_pair_forward(
    node_update: torch.Tensor,
    node_old: torch.Tensor,
    node_res: torch.Tensor,
    edge_update: torch.Tensor,
    edge_old: torch.Tensor,
    edge_res: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        node_update + node_res * node_old,
        edge_update + edge_res * edge_old,
    )


def residual_pair_forward_backward(
    node_update: torch.Tensor,
    node_old: torch.Tensor,
    node_res: torch.Tensor,
    edge_update: torch.Tensor,
    edge_old: torch.Tensor,
    edge_res: torch.Tensor,
    grad_node: torch.Tensor,
    grad_edge: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    node_out, edge_out = residual_pair_forward(node_update, node_old, node_res, edge_update, edge_old, edge_res)
    node_grad_update, node_grad_old, node_grad_res = residual_backward(grad_node, node_old, node_res)
    edge_grad_update, edge_grad_old, edge_grad_res = residual_backward(grad_edge, edge_old, edge_res)
    return (
        node_out,
        edge_out,
        node_grad_update,
        node_grad_old,
        node_grad_res,
        edge_grad_update,
        edge_grad_old,
        edge_grad_res,
    )


def make_case(rows: int, dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    update = torch.randn((rows, dim), device=device)
    old_feat = torch.randn((rows, dim), device=device)
    res_weight = torch.randn((1, dim), device=device)
    grad_out = torch.randn((rows, dim), device=device)
    return update, old_feat, res_weight, grad_out


def bench_single(
    compiled_forward: Callable[..., torch.Tensor],
    compiled_forward_backward: Callable[..., tuple[torch.Tensor, ...]],
    rows: int,
    dim: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda")
    update, old_feat, res_weight, grad_out = make_case(rows, dim, device)
    eager_f = residual_forward(update, old_feat, res_weight)
    compiled_f = compiled_forward(update, old_feat, res_weight).clone()
    eager_fb = residual_forward_backward(update, old_feat, res_weight, grad_out)
    compiled_fb = tuple(item.clone() for item in compiled_forward_backward(update, old_feat, res_weight, grad_out))

    return {
        "rows": rows,
        "dim": dim,
        "forward_time": {
            "torch_eager": cuda_time_ms(
                lambda: residual_forward(update, old_feat, res_weight),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "torch_compile": cuda_time_ms(
                lambda: compiled_forward(update, old_feat, res_weight),
                warmup=args.warmup,
                iters=args.iters,
            ),
        },
        "forward_backward_time": {
            "torch_eager": cuda_time_ms(
                lambda: residual_forward_backward(update, old_feat, res_weight, grad_out),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "torch_compile": cuda_time_ms(
                lambda: compiled_forward_backward(update, old_feat, res_weight, grad_out),
                warmup=args.warmup,
                iters=args.iters,
            ),
        },
        "errors": {
            "forward": rel_error(eager_f, compiled_f),
            "grad_update": rel_error(eager_fb[1], compiled_fb[1]),
            "grad_old": rel_error(eager_fb[2], compiled_fb[2]),
            "grad_res": rel_error(eager_fb[3], compiled_fb[3]),
        },
    }


def bench_pair(
    compiled_pair: Callable[..., tuple[torch.Tensor, ...]],
    rows: int,
    dim: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda")
    node_rows = max(1, int(round(rows * 0.25)))
    node_update, node_old, node_res, grad_node = make_case(node_rows, dim, device)
    edge_update, edge_old, edge_res, grad_edge = make_case(rows, dim, device)
    eager = residual_pair_forward_backward(
        node_update,
        node_old,
        node_res,
        edge_update,
        edge_old,
        edge_res,
        grad_node,
        grad_edge,
    )
    compiled = tuple(
        item.clone()
        for item in compiled_pair(
            node_update,
            node_old,
            node_res,
            edge_update,
            edge_old,
            edge_res,
            grad_node,
            grad_edge,
        )
    )
    return {
        "edge_rows": rows,
        "node_rows": node_rows,
        "dim": dim,
        "forward_backward_time": {
            "torch_eager": cuda_time_ms(
                lambda: residual_pair_forward_backward(
                    node_update,
                    node_old,
                    node_res,
                    edge_update,
                    edge_old,
                    edge_res,
                    grad_node,
                    grad_edge,
                ),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "torch_compile": cuda_time_ms(
                lambda: compiled_pair(
                    node_update,
                    node_old,
                    node_res,
                    edge_update,
                    edge_old,
                    edge_res,
                    grad_node,
                    grad_edge,
                ),
                warmup=args.warmup,
                iters=args.iters,
            ),
        },
        "errors": {
            f"out_{idx}": rel_error(a, b)
            for idx, (a, b) in enumerate(zip(eager, compiled))
        },
    }


def add_speedups(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for section in ("forward_time", "forward_backward_time"):
            if section not in row:
                continue
            eager = row[section]["torch_eager"]["mean_ms"]
            compiled = row[section]["torch_compile"]["mean_ms"]
            row[section]["compile_speedup_vs_eager"] = eager / compiled if compiled else float("inf")


def summarize(rows: list[dict[str, Any]], section: str) -> dict[str, float]:
    speedups = [row[section]["compile_speedup_vs_eager"] for row in rows if section in row]
    eager_ms = [row[section]["torch_eager"]["mean_ms"] for row in rows if section in row]
    compile_ms = [row[section]["torch_compile"]["mean_ms"] for row in rows if section in row]
    return {
        "avg_compile_speedup_vs_eager": sum(speedups) / len(speedups),
        "avg_eager_ms": sum(eager_ms) / len(eager_ms),
        "avg_compile_ms": sum(compile_ms) / len(compile_ms),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    rows_list = parse_int_list(args.rows)

    compiled_forward = torch.compile(residual_forward, dynamic=True, mode=args.torch_compile_mode)
    compiled_forward_backward = torch.compile(residual_forward_backward, dynamic=True, mode=args.torch_compile_mode)
    compiled_pair = torch.compile(residual_pair_forward_backward, dynamic=True, mode=args.torch_compile_mode)

    single_rows = [bench_single(compiled_forward, compiled_forward_backward, rows, args.dim, args) for rows in rows_list]
    pair_rows = [bench_pair(compiled_pair, rows, args.dim, args) for rows in rows_list]
    add_speedups(single_rows)
    add_speedups(pair_rows)

    payload = {
        "metadata": {
            "rows": rows_list,
            "dim": args.dim,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "torch_compile_mode": args.torch_compile_mode,
            "dynamic": True,
        },
        "single_residual": {
            "summary_forward": summarize(single_rows, "forward_time"),
            "summary_forward_backward": summarize(single_rows, "forward_backward_time"),
            "rows": single_rows,
        },
        "pair_residual": {
            "summary_forward_backward": summarize(pair_rows, "forward_backward_time"),
            "rows": pair_rows,
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
