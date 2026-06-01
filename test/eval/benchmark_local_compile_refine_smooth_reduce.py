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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local torch.compile experiment for refine_line smooth multiply + target reduce "
            "and its input-grad scatter boundary."
        )
    )
    parser.add_argument("--rows", default="512,1024,2048,4096,8192,16384")
    parser.add_argument("--node-ratio", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--output-json", default="results/compile_local_refine_smooth_reduce_shape_sweep_20260526.json")
    return parser.parse_args()


def load_matris_op():
    import matris_op  # type: ignore

    return matris_op


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


def make_index(rows: int, num_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.randint(0, num_nodes, (rows,), device=device, dtype=torch.long)
    target = torch.randint(0, num_nodes, (rows,), device=device, dtype=torch.long)
    return source.contiguous(), target.contiguous()


def torch_smooth_forward_template(
    nonlinear: torch.Tensor,
    base_envelope: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    base_i = base_envelope.index_select(0, source_index)
    base_j = base_envelope.index_select(0, target_index)
    weighted = nonlinear * base_i * base_j
    out = nonlinear.new_zeros((num_nodes, nonlinear.shape[1]))
    return out.index_add(0, target_index, weighted)


def torch_smooth_backward_template(
    grad_out: torch.Tensor,
    nonlinear: torch.Tensor,
    base_envelope: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad = grad_out.index_select(0, target_index)
    base_i = base_envelope.index_select(0, source_index)
    base_j = base_envelope.index_select(0, target_index)
    grad_nonlinear = grad * base_i * base_j
    grad_base_source = grad * nonlinear * base_j
    grad_base_target = grad * nonlinear * base_i
    grad_base = torch.zeros_like(base_envelope)
    grad_base = grad_base.index_add(0, source_index, grad_base_source)
    grad_base = grad_base.index_add(0, target_index, grad_base_target)
    return grad_nonlinear, grad_base


def torch_smooth_forward_backward_template(
    nonlinear: torch.Tensor,
    base_envelope: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    num_nodes: int,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch_smooth_forward_template(nonlinear, base_envelope, source_index, target_index, num_nodes)
    grad_nonlinear, grad_base = torch_smooth_backward_template(
        grad_out,
        nonlinear,
        base_envelope,
        source_index,
        target_index,
    )
    return out, grad_nonlinear, grad_base


def cuda_forward(matris_op, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    nonlinear, base_envelope, source_index, target_index, num_nodes, _grad_out = tensors
    return matris_op.refine_line_smooth_reduce_forward(
        nonlinear.contiguous(),
        base_envelope.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        int(num_nodes),
    )


def cuda_forward_backward(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nonlinear, base_envelope, source_index, target_index, _num_nodes, grad_out = tensors
    out = cuda_forward(matris_op, tensors)
    grad_nonlinear, grad_base = matris_op.refine_line_smooth_reduce_backward(
        grad_out.contiguous(),
        nonlinear.contiguous(),
        base_envelope.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
    )
    return out, grad_nonlinear, grad_base


def make_case(rows: int, node_ratio: float, device: torch.device) -> tuple[torch.Tensor, ...]:
    num_nodes = max(1, int(round(rows * node_ratio)))
    nonlinear = torch.randn((rows, 128), device=device)
    base_envelope = torch.randn((num_nodes, 128), device=device)
    source_index, target_index = make_index(rows, num_nodes, device)
    grad_out = torch.randn((num_nodes, 128), device=device)
    return nonlinear, base_envelope, source_index, target_index, num_nodes, grad_out


def bench_case(
    matris_op,
    compiled_forward: Callable[..., torch.Tensor],
    compiled_forward_backward: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    rows: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda")
    tensors = make_case(rows, args.node_ratio, device)
    nonlinear, base_envelope, source_index, target_index, num_nodes, grad_out = tensors
    eager_f = torch_smooth_forward_template(nonlinear, base_envelope, source_index, target_index, num_nodes)
    compiled_f = compiled_forward(nonlinear, base_envelope, source_index, target_index, num_nodes)
    cuda_f = cuda_forward(matris_op, tensors)

    eager_fb = torch_smooth_forward_backward_template(
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        num_nodes,
        grad_out,
    )
    compiled_fb = compiled_forward_backward(
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        num_nodes,
        grad_out,
    )
    cuda_fb = cuda_forward_backward(matris_op, tensors)

    forward_time = {
        "torch_eager": cuda_time_ms(
            lambda: torch_smooth_forward_template(nonlinear, base_envelope, source_index, target_index, num_nodes),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "torch_compile": cuda_time_ms(
            lambda: compiled_forward(nonlinear, base_envelope, source_index, target_index, num_nodes),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "cuda_fused": cuda_time_ms(lambda: cuda_forward(matris_op, tensors), warmup=args.warmup, iters=args.iters),
    }
    forward_backward_time = {
        "torch_eager": cuda_time_ms(
            lambda: torch_smooth_forward_backward_template(
                nonlinear,
                base_envelope,
                source_index,
                target_index,
                num_nodes,
                grad_out,
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "torch_compile": cuda_time_ms(
            lambda: compiled_forward_backward(
                nonlinear,
                base_envelope,
                source_index,
                target_index,
                num_nodes,
                grad_out,
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "cuda_fused": cuda_time_ms(lambda: cuda_forward_backward(matris_op, tensors), warmup=args.warmup, iters=args.iters),
    }

    return {
        "op": "refine_line_smooth_reduce",
        "rows": rows,
        "num_nodes": int(num_nodes),
        "dim": 128,
        "forward_time": forward_time,
        "forward_backward_time": forward_backward_time,
        "speedup_vs_eager": {
            "forward": {
                "torch_compile": forward_time["torch_eager"]["mean_ms"] / forward_time["torch_compile"]["mean_ms"],
                "cuda_fused": forward_time["torch_eager"]["mean_ms"] / forward_time["cuda_fused"]["mean_ms"],
            },
            "forward_backward": {
                "torch_compile": forward_backward_time["torch_eager"]["mean_ms"]
                / forward_backward_time["torch_compile"]["mean_ms"],
                "cuda_fused": forward_backward_time["torch_eager"]["mean_ms"]
                / forward_backward_time["cuda_fused"]["mean_ms"],
            },
        },
        "errors_vs_torch_eager": {
            "forward_compile": rel_error(eager_f, compiled_f),
            "forward_cuda": rel_error(eager_f, cuda_f),
            "fb_compile_out": rel_error(eager_fb[0], compiled_fb[0]),
            "fb_compile_grad_nonlinear": rel_error(eager_fb[1], compiled_fb[1]),
            "fb_compile_grad_base": rel_error(eager_fb[2], compiled_fb[2]),
            "fb_cuda_out": rel_error(eager_fb[0], cuda_fb[0]),
            "fb_cuda_grad_nonlinear": rel_error(eager_fb[1], cuda_fb[1]),
            "fb_cuda_grad_base": rel_error(eager_fb[2], cuda_fb[2]),
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_cases": len(results),
        "rows": [int(item["rows"]) for item in results],
        "num_nodes": [int(item["num_nodes"]) for item in results],
        "forward_compile_speedup_mean": sum(
            float(item["speedup_vs_eager"]["forward"]["torch_compile"]) for item in results
        )
        / len(results),
        "forward_cuda_speedup_mean": sum(float(item["speedup_vs_eager"]["forward"]["cuda_fused"]) for item in results)
        / len(results),
        "forward_backward_compile_speedup_mean": sum(
            float(item["speedup_vs_eager"]["forward_backward"]["torch_compile"]) for item in results
        )
        / len(results),
        "forward_backward_cuda_speedup_mean": sum(
            float(item["speedup_vs_eager"]["forward_backward"]["cuda_fused"]) for item in results
        )
        / len(results),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    matris_op = load_matris_op()
    compiled_forward = torch.compile(
        torch_smooth_forward_template,
        dynamic=True,
        mode=args.torch_compile_mode,
    )
    compiled_forward_backward = torch.compile(
        torch_smooth_forward_backward_template,
        dynamic=True,
        mode=args.torch_compile_mode,
    )
    results = [
        bench_case(matris_op, compiled_forward, compiled_forward_backward, rows, args)
        for rows in parse_int_list(args.rows)
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
            "boundary": "refine_line nonlinear * base[source] * base[target] + target index_add and manual input-grad scatter",
            "rows": parse_int_list(args.rows),
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
