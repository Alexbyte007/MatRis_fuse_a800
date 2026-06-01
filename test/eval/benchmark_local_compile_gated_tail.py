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
            "Local torch.compile experiment for GatedMLP tail input-grad: "
            "PyTorch layernorm/silu/sigmoid backward equivalent vs compile vs MatRIS CUDA op."
        )
    )
    parser.add_argument("--rows", default="128,256,512,1024,2048,8192")
    parser.add_argument("--dims", default="128,256")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--include-v2", action="store_true", help="Also benchmark input_grad_only_gated_tail_backward_n128_v2.")
    parser.add_argument("--output-json", default="results/compile_local_gated_tail_shape_sweep_20260526.json")
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


def torch_gated_tail_backward_template(
    grad_out: torch.Tensor,
    core: torch.Tensor,
    gate: torch.Tensor,
    core_weight: torch.Tensor,
    core_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    eps_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = eps_tensor.reshape(())
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
        (
            grad_core_norm * dim
            - grad_core_norm.sum(dim=1, keepdim=True)
            - core_xhat * (grad_core_norm * core_xhat).sum(dim=1, keepdim=True)
        )
        * core_rstd
        / dim
    )
    grad_gate = (
        (
            grad_gate_norm * dim
            - grad_gate_norm.sum(dim=1, keepdim=True)
            - gate_xhat * (grad_gate_norm * gate_xhat).sum(dim=1, keepdim=True)
        )
        * gate_rstd
        / dim
    )
    return grad_core, grad_gate


def torch_gated_tail_backward_eval_template(*args) -> tuple[torch.Tensor, torch.Tensor]:
    return torch_gated_tail_backward_template(*args)


def make_case(rows: int, dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    grad_out = torch.randn((rows, dim), device=device)
    core = torch.randn((rows, dim), device=device)
    gate = torch.randn((rows, dim), device=device)
    core_weight = torch.randn((dim,), device=device)
    core_bias = torch.randn((dim,), device=device)
    gate_weight = torch.randn((dim,), device=device)
    gate_bias = torch.randn((dim,), device=device)
    eps_tensor = torch.tensor(1.0e-5, device=device)
    return grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps_tensor


def cuda_tail(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps_tensor = tensors
    return matris_op.input_grad_only_gated_tail_backward(
        grad_out.contiguous(),
        core.contiguous(),
        gate.contiguous(),
        core_weight.contiguous(),
        core_bias.contiguous(),
        gate_weight.contiguous(),
        gate_bias.contiguous(),
        float(eps_tensor.item()),
    )


def cuda_tail_v2(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    grad_out, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps_tensor = tensors
    return matris_op.input_grad_only_gated_tail_backward_n128_v2(
        grad_out.contiguous(),
        core.contiguous(),
        gate.contiguous(),
        core_weight.contiguous(),
        core_bias.contiguous(),
        gate_weight.contiguous(),
        gate_bias.contiguous(),
        float(eps_tensor.item()),
    )


def bench_case(
    matris_op,
    compiled_eval: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    rows: int,
    dim: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda")
    tensors = make_case(rows, dim, device)
    eager = torch_gated_tail_backward_template(*tensors)
    compiled = compiled_eval(*tensors)
    cuda = cuda_tail(matris_op, tensors)

    result: dict[str, Any] = {
        "op": "input_grad_only_gated_tail_backward",
        "rows": rows,
        "dim": dim,
        "time": {
            "torch_eager": cuda_time_ms(lambda: torch_gated_tail_backward_template(*tensors), warmup=args.warmup, iters=args.iters),
            "torch_compile": cuda_time_ms(lambda: compiled_eval(*tensors), warmup=args.warmup, iters=args.iters),
            "cuda_fused": cuda_time_ms(lambda: cuda_tail(matris_op, tensors), warmup=args.warmup, iters=args.iters),
        },
        "errors_vs_torch_eager": {
            "torch_compile_grad_core": rel_error(eager[0], compiled[0]),
            "torch_compile_grad_gate": rel_error(eager[1], compiled[1]),
            "cuda_fused_grad_core": rel_error(eager[0], cuda[0]),
            "cuda_fused_grad_gate": rel_error(eager[1], cuda[1]),
        },
    }
    result["speedup_vs_eager"] = {
        "torch_compile": result["time"]["torch_eager"]["mean_ms"] / result["time"]["torch_compile"]["mean_ms"],
        "cuda_fused": result["time"]["torch_eager"]["mean_ms"] / result["time"]["cuda_fused"]["mean_ms"],
    }
    if args.include_v2 and dim == 128 and hasattr(matris_op, "input_grad_only_gated_tail_backward_n128_v2"):
        cuda_v2 = cuda_tail_v2(matris_op, tensors)
        result["time"]["cuda_fused_v2"] = cuda_time_ms(
            lambda: cuda_tail_v2(matris_op, tensors),
            warmup=args.warmup,
            iters=args.iters,
        )
        result["errors_vs_torch_eager"]["cuda_fused_v2_grad_core"] = rel_error(eager[0], cuda_v2[0])
        result["errors_vs_torch_eager"]["cuda_fused_v2_grad_gate"] = rel_error(eager[1], cuda_v2[1])
        result["speedup_vs_eager"]["cuda_fused_v2"] = (
            result["time"]["torch_eager"]["mean_ms"] / result["time"]["cuda_fused_v2"]["mean_ms"]
        )
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_dim: dict[str, Any] = {}
    for dim in sorted({int(item["dim"]) for item in results}):
        subset = [item for item in results if int(item["dim"]) == dim]
        by_dim[str(dim)] = {
            "rows": [int(item["rows"]) for item in subset],
            "compile_speedup_mean": sum(float(item["speedup_vs_eager"]["torch_compile"]) for item in subset) / len(subset),
            "cuda_speedup_mean": sum(float(item["speedup_vs_eager"]["cuda_fused"]) for item in subset) / len(subset),
        }
    return {"num_cases": len(results), "by_dim": by_dim}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    matris_op = load_matris_op()
    compiled_eval = torch.compile(
        torch_gated_tail_backward_eval_template,
        dynamic=True,
        mode=args.torch_compile_mode,
    )
    results = []
    for dim in parse_int_list(args.dims):
        if dim not in (128, 256):
            raise ValueError("The CUDA op only supports dim 128 or 256.")
        for rows in parse_int_list(args.rows):
            results.append(bench_case(matris_op, compiled_eval, rows, dim, args))
    payload = {
        "metadata": {
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_compile": {
                "enabled": True,
                "dynamic": True,
                "mode": args.torch_compile_mode,
                "scope": "local GatedMLP tail input-grad equivalent only",
                "logs_hint": 'TORCH_LOGS="graph_breaks,recompiles"',
            },
            "rows": parse_int_list(args.rows),
            "dims": parse_int_list(args.dims),
            "warmup": args.warmup,
            "iters": args.iters,
            "include_v2": bool(args.include_v2),
            "baseline_mapping": {
                "torch_eager": "PyTorch LayerNorm backward equivalent + SiLU/sigmoid gate input gradients",
                "torch_compile": "torch.compile(torch_eager_tail_backward, dynamic=True), local function only",
                "cuda_fused": "matris_op.input_grad_only_gated_tail_backward",
                "cuda_fused_v2": "matris_op.input_grad_only_gated_tail_backward_n128_v2 for dim=128 when requested",
            },
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
