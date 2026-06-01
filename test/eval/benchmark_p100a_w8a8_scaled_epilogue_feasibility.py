#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_quant_paths():
    from quant.layers import (
        _load_matris_op,
        cuda_w8a8_static_cutlass_dual_gated_tail,
        cuda_w8a8_static_wmma_dual_gated_tail_n128,
    )

    return (
        _load_matris_op(),
        cuda_w8a8_static_wmma_dual_gated_tail_n128,
        cuda_w8a8_static_cutlass_dual_gated_tail,
    )


def _cuda_time_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    out = fn()
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters), out


def _make_case(rows: int, *, seed: int, device: torch.device) -> dict[str, torch.Tensor | float]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rows * 17)
    core = (torch.randn((rows, 128), device=device, generator=gen, dtype=torch.float32) * 0.35).contiguous()
    gate = (torch.randn((rows, 128), device=device, generator=gen, dtype=torch.float32) * 0.35).contiguous()
    core_q_weight = torch.randint(-64, 65, (128, 128), device=device, generator=gen, dtype=torch.int8).contiguous()
    gate_q_weight = torch.randint(-64, 65, (128, 128), device=device, generator=gen, dtype=torch.int8).contiguous()
    core_weight_scale = (torch.rand((128,), device=device, generator=gen, dtype=torch.float32) * 0.006 + 0.001).contiguous()
    gate_weight_scale = (torch.rand((128,), device=device, generator=gen, dtype=torch.float32) * 0.006 + 0.001).contiguous()
    core_activation_scale = torch.tensor(0.0125, device=device, dtype=torch.float32)
    gate_activation_scale = torch.tensor(0.0125, device=device, dtype=torch.float32)
    core_bias = (torch.randn((128,), device=device, generator=gen, dtype=torch.float32) * 0.01).contiguous()
    gate_bias = (torch.randn((128,), device=device, generator=gen, dtype=torch.float32) * 0.01).contiguous()
    core_norm_weight = torch.ones((128,), device=device, dtype=torch.float32).contiguous()
    gate_norm_weight = torch.ones((128,), device=device, dtype=torch.float32).contiguous()
    core_norm_bias = torch.zeros((128,), device=device, dtype=torch.float32).contiguous()
    gate_norm_bias = torch.zeros((128,), device=device, dtype=torch.float32).contiguous()
    return {
        "core": core,
        "gate": gate,
        "core_q_weight": core_q_weight,
        "gate_q_weight": gate_q_weight,
        "core_weight_scale": core_weight_scale,
        "gate_weight_scale": gate_weight_scale,
        "core_activation_scale": core_activation_scale,
        "gate_activation_scale": gate_activation_scale,
        "core_bias": core_bias,
        "gate_bias": gate_bias,
        "core_norm_weight": core_norm_weight,
        "core_norm_bias": core_norm_bias,
        "gate_norm_weight": gate_norm_weight,
        "gate_norm_bias": gate_norm_bias,
        "eps": 1.0e-5,
    }


def _route_wmma_fused(wmma_fn: Callable[..., torch.Tensor | None], case: dict[str, Any]) -> torch.Tensor:
    out = wmma_fn(
        case["core"],
        case["gate"],
        case["core_q_weight"],
        case["core_weight_scale"],
        case["core_activation_scale"],
        case["core_bias"],
        case["gate_q_weight"],
        case["gate_weight_scale"],
        case["gate_activation_scale"],
        case["gate_bias"],
        case["core_norm_weight"],
        case["core_norm_bias"],
        case["gate_norm_weight"],
        case["gate_norm_bias"],
        case["eps"],
    )
    if out is None:
        raise RuntimeError("current WMMA fused second-tail path returned None")
    return out


def _route_cutlass_separate(cutlass_fn: Callable[..., torch.Tensor | None], case: dict[str, Any]) -> torch.Tensor:
    out = cutlass_fn(
        case["core"],
        case["gate"],
        case["core_q_weight"],
        case["core_weight_scale"],
        case["core_activation_scale"],
        case["core_bias"],
        case["gate_q_weight"],
        case["gate_weight_scale"],
        case["gate_activation_scale"],
        case["gate_bias"],
        case["core_norm_weight"],
        case["core_norm_bias"],
        case["gate_norm_weight"],
        case["gate_norm_bias"],
        case["eps"],
    )
    if out is None:
        raise RuntimeError("current MatRIS CUTLASS separate path returned None")
    return out


def _quantize_static(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8).contiguous()


def _route_torch_scaled_mm_tail(matris_op: Any, case: dict[str, Any]) -> torch.Tensor:
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is unavailable")
    if matris_op is None or not hasattr(matris_op, "fp32_gated_tail_forward_n128"):
        raise RuntimeError("matris_op.fp32_gated_tail_forward_n128 is unavailable")
    core_q = _quantize_static(case["core"], case["core_activation_scale"])
    gate_q = _quantize_static(case["gate"], case["gate_activation_scale"])
    core_pre = torch._scaled_mm(
        core_q,
        case["core_q_weight"].t().contiguous(),
        scale_a=case["core_activation_scale"].reshape(1, 1),
        scale_b=case["core_weight_scale"].reshape(1, 128),
        out_dtype=torch.float32,
    )
    gate_pre = torch._scaled_mm(
        gate_q,
        case["gate_q_weight"].t().contiguous(),
        scale_a=case["gate_activation_scale"].reshape(1, 1),
        scale_b=case["gate_weight_scale"].reshape(1, 128),
        out_dtype=torch.float32,
    )
    core_pre = core_pre + case["core_bias"]
    gate_pre = gate_pre + case["gate_bias"]
    return matris_op.fp32_gated_tail_forward_n128(
        core_pre.contiguous(),
        gate_pre.contiguous(),
        case["core_norm_weight"],
        case["core_norm_bias"],
        case["gate_norm_weight"],
        case["gate_norm_bias"],
        float(case["eps"]),
    )


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max().item()), "mean_abs": float(d.mean().item())}


def main() -> None:
    parser = argparse.ArgumentParser(description="P100A W8A8 scaled epilogue feasibility benchmark.")
    parser.add_argument("--rows", nargs="+", type=int, default=[236, 728, 1008, 1224, 1520, 4096, 6176])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/p100a_w8a8_scaled_epilogue_feasibility.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    # Keep route 1 on the same fast wrapper used by the current forward-only WMMA path.
    os.environ.pop("MATRIS_W8A8_DISABLE_FAST_WRAPPER", None)

    matris_op, wmma_fn, cutlass_fn = _load_quant_paths()
    if matris_op is None:
        raise RuntimeError("matris_op failed to load")

    results: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "routes": {
            "current_wmma_fused_second_tail": "MatRIS WMMA path: int8 pack + W8A8 GEMM + dequant + gated tail in one forward op",
            "current_cutlass_separate_dequant_tail": "MatRIS CUTLASS path: quantize + CUTLASS int8 GEMM + separate dequant/tail kernel",
            "torch_scaled_mm_epilogue_plus_matris_tail": "Proxy for vLLM-style scaled_mm epilogue + MatRIS tail; skipped on A800 if torch._scaled_mm rejects sm80",
        },
        "rows": [],
    }

    for rows in args.rows:
        case = _make_case(rows, seed=args.seed, device=device)
        row_result: dict[str, Any] = {"rows": rows, "timings_ms": {}, "diff_vs_wmma": {}, "status": {}}

        wmma_ms, wmma_out = _cuda_time_ms(lambda: _route_wmma_fused(wmma_fn, case), warmup=args.warmup, iters=args.iters)
        row_result["timings_ms"]["current_wmma_fused_second_tail"] = wmma_ms
        row_result["status"]["current_wmma_fused_second_tail"] = "ok"

        cutlass_ms, cutlass_out = _cuda_time_ms(
            lambda: _route_cutlass_separate(cutlass_fn, case), warmup=args.warmup, iters=args.iters
        )
        row_result["timings_ms"]["current_cutlass_separate_dequant_tail"] = cutlass_ms
        row_result["diff_vs_wmma"]["current_cutlass_separate_dequant_tail"] = _diff(wmma_out, cutlass_out)
        row_result["status"]["current_cutlass_separate_dequant_tail"] = "ok"

        try:
            scaled_ms, scaled_out = _cuda_time_ms(
                lambda: _route_torch_scaled_mm_tail(matris_op, case), warmup=args.warmup, iters=args.iters
            )
            row_result["timings_ms"]["torch_scaled_mm_epilogue_plus_matris_tail"] = scaled_ms
            row_result["diff_vs_wmma"]["torch_scaled_mm_epilogue_plus_matris_tail"] = _diff(wmma_out, scaled_out)
            row_result["status"]["torch_scaled_mm_epilogue_plus_matris_tail"] = "ok"
        except Exception as exc:
            row_result["status"]["torch_scaled_mm_epilogue_plus_matris_tail"] = f"skipped: {type(exc).__name__}: {str(exc).splitlines()[0]}"

        base = row_result["timings_ms"].get("current_wmma_fused_second_tail")
        if base:
            row_result["relative_to_wmma"] = {
                name: (float(ms) / float(base)) for name, ms in row_result["timings_ms"].items()
            }
        results["rows"].append(row_result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
