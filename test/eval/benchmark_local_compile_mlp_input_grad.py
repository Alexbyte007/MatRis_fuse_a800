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
            "Local torch.compile experiment for the MLP/FFN input-grad boundary: "
            "SiLU backward elementwise chain and full two-linear SiLU input-grad."
        )
    )
    parser.add_argument("--rows", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--output-json", default="results/compile_local_mlp_input_grad_shape_sweep_20260526.json")
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


def silu_grad(x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def torch_mlp_silu_grad_template(grad_activated: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return grad_activated * silu_grad(hidden)


def torch_mlp_full_input_grad_template(
    grad_out: torch.Tensor,
    weight2: torch.Tensor,
    hidden: torch.Tensor,
    weight1: torch.Tensor,
) -> torch.Tensor:
    grad_activated = grad_out.matmul(weight2)
    grad_hidden = torch_mlp_silu_grad_template(grad_activated, hidden)
    return grad_hidden.contiguous().matmul(weight1)


def make_case(rows: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    grad_out = torch.randn((rows, 128), device=device)
    weight2 = torch.randn((128, 128), device=device)
    hidden = torch.randn((rows, 128), device=device)
    weight1 = torch.randn((128, 128), device=device)
    grad_activated = grad_out.matmul(weight2)
    return grad_out, weight2, hidden, weight1, grad_activated


def cuda_silu_grad(matris_op, grad_activated: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    if not hasattr(matris_op, "fuse_silu_bwd"):
        raise RuntimeError("matris_op.fuse_silu_bwd is required for the SiLU-grad CUDA comparison.")
    return matris_op.fuse_silu_bwd(grad_activated.contiguous(), hidden.contiguous())


def cuda_full_input_grad(matris_op, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    grad_out, weight2, hidden, weight1, _grad_activated = tensors
    if not hasattr(matris_op, "two_linear_silu_input_grad_backward_n128"):
        raise RuntimeError("matris_op.two_linear_silu_input_grad_backward_n128 is required.")
    return matris_op.two_linear_silu_input_grad_backward_n128(
        grad_out.contiguous(),
        weight2.contiguous(),
        hidden.contiguous(),
        weight1.contiguous(),
    )


def cuda_full_input_grad_cublas(matris_op, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    grad_out, weight2, hidden, weight1, _grad_activated = tensors
    return matris_op.two_linear_silu_input_grad_backward_n128_cublas(
        grad_out.contiguous(),
        weight2.contiguous(),
        hidden.contiguous(),
        weight1.contiguous(),
    )


def cuda_full_input_grad_cutlass_epilogue(matris_op, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    grad_out, weight2, hidden, weight1, _grad_activated = tensors
    return matris_op.two_linear_silu_input_grad_backward_n128_cutlass_epilogue(
        grad_out.contiguous(),
        weight2.contiguous(),
        hidden.contiguous(),
        weight1.contiguous(),
    )


def bench_case(
    matris_op,
    compiled_silu_grad: Callable[..., torch.Tensor],
    compiled_full: Callable[..., torch.Tensor],
    rows: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda")
    tensors = make_case(rows, device)
    grad_out, weight2, hidden, weight1, grad_activated = tensors

    silu_eager = torch_mlp_silu_grad_template(grad_activated, hidden)
    silu_compiled = compiled_silu_grad(grad_activated, hidden)
    silu_cuda = cuda_silu_grad(matris_op, grad_activated, hidden)

    full_eager = torch_mlp_full_input_grad_template(grad_out, weight2, hidden, weight1)
    full_compiled = compiled_full(grad_out, weight2, hidden, weight1)
    full_cuda = cuda_full_input_grad(matris_op, tensors)
    full_cuda_cublas = (
        cuda_full_input_grad_cublas(matris_op, tensors)
        if hasattr(matris_op, "two_linear_silu_input_grad_backward_n128_cublas")
        else None
    )
    full_cuda_cutlass = (
        cuda_full_input_grad_cutlass_epilogue(matris_op, tensors)
        if hasattr(matris_op, "two_linear_silu_input_grad_backward_n128_cutlass_epilogue")
        else None
    )

    silu_time = {
        "torch_eager": cuda_time_ms(
            lambda: torch_mlp_silu_grad_template(grad_activated, hidden),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "torch_compile": cuda_time_ms(
            lambda: compiled_silu_grad(grad_activated, hidden),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "cuda_fused": cuda_time_ms(
            lambda: cuda_silu_grad(matris_op, grad_activated, hidden),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }
    full_time = {
        "torch_eager": cuda_time_ms(
            lambda: torch_mlp_full_input_grad_template(grad_out, weight2, hidden, weight1),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "torch_compile": cuda_time_ms(
            lambda: compiled_full(grad_out, weight2, hidden, weight1),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "cuda_fused": cuda_time_ms(
            lambda: cuda_full_input_grad(matris_op, tensors),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }
    if full_cuda_cublas is not None:
        full_time["cuda_cublas"] = cuda_time_ms(
            lambda: cuda_full_input_grad_cublas(matris_op, tensors),
            warmup=args.warmup,
            iters=args.iters,
        )
    if full_cuda_cutlass is not None:
        full_time["cuda_cutlass_epilogue"] = cuda_time_ms(
            lambda: cuda_full_input_grad_cutlass_epilogue(matris_op, tensors),
            warmup=args.warmup,
            iters=args.iters,
        )

    return {
        "op": "mlp_two_linear_silu_input_grad",
        "rows": rows,
        "dim": 128,
        "silu_grad_boundary": {
            "time": silu_time,
            "speedup_vs_eager": {
                "torch_compile": silu_time["torch_eager"]["mean_ms"] / silu_time["torch_compile"]["mean_ms"],
                "cuda_fused": silu_time["torch_eager"]["mean_ms"] / silu_time["cuda_fused"]["mean_ms"],
            },
            "errors_vs_torch_eager": {
                "torch_compile": rel_error(silu_eager, silu_compiled),
                "cuda_fused": rel_error(silu_eager, silu_cuda),
            },
        },
        "full_input_grad_boundary": {
            "time": full_time,
            "speedup_vs_eager": {
                name: full_time["torch_eager"]["mean_ms"] / value["mean_ms"]
                for name, value in full_time.items()
                if name != "torch_eager"
            },
            "errors_vs_torch_eager": {
                "torch_compile": rel_error(full_eager, full_compiled),
                "cuda_fused": rel_error(full_eager, full_cuda),
                **(
                    {"cuda_cublas": rel_error(full_eager, full_cuda_cublas)}
                    if full_cuda_cublas is not None
                    else {}
                ),
                **(
                    {"cuda_cutlass_epilogue": rel_error(full_eager, full_cuda_cutlass)}
                    if full_cuda_cutlass is not None
                    else {}
                ),
            },
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_speedup(boundary: str, mode: str) -> float:
        return sum(float(item[boundary]["speedup_vs_eager"][mode]) for item in results) / len(results)

    return {
        "num_cases": len(results),
        "rows": [int(item["rows"]) for item in results],
        "silu_grad_boundary": {
            "compile_speedup_mean": mean_speedup("silu_grad_boundary", "torch_compile"),
            "cuda_speedup_mean": mean_speedup("silu_grad_boundary", "cuda_fused"),
        },
        "full_input_grad_boundary": {
            "speedup_mean": {
                mode: mean_speedup("full_input_grad_boundary", mode)
                for mode in sorted({mode for item in results for mode in item["full_input_grad_boundary"]["speedup_vs_eager"]})
            },
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    matris_op = load_matris_op()
    if hasattr(matris_op, "two_linear_silu_input_grad_backward_n128_cutlass_epilogue"):
        print("CUTLASS epilogue MLP input-grad op is available.")
    compiled_silu_grad = torch.compile(
        torch_mlp_silu_grad_template,
        dynamic=True,
        mode=args.torch_compile_mode,
    )
    compiled_full = torch.compile(
        torch_mlp_full_input_grad_template,
        dynamic=True,
        mode=args.torch_compile_mode,
    )
    results = [
        bench_case(matris_op, compiled_silu_grad, compiled_full, rows, args)
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
            "boundaries": {
                "silu_grad_boundary": "grad_activated * SiLU'(hidden), pure elementwise chain",
                "full_input_grad_boundary": "grad_out @ weight2 + SiLU grad + grad_hidden @ weight1",
            },
            "rows": parse_int_list(args.rows),
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
