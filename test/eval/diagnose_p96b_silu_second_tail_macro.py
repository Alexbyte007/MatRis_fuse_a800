from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant.layers import (
    cuda_w8a8_static_wmma_dual_gated_tail_n128_saved_pre_autograd,
    cuda_w8a8_static_wmma_dual_silu_gated_tail_n128_saved_pre_autograd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P96B SiLU-prefix + W8A8 second-tail autograd macro correctness.")
    parser.add_argument("--rows", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="results/p96b_silu_second_tail_macro_correctness")
    return parser.parse_args()


def diff_summary(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean((a - b).float() ** 2)).item()) if diff.numel() else 0.0,
    }


def make_inputs(rows: int, seed: int) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("P96B correctness requires CUDA.")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    device = torch.device("cuda")
    core_q_weight = torch.randint(-64, 64, (128, 128), device=device, dtype=torch.int8, generator=generator)
    gate_q_weight = torch.randint(-64, 64, (128, 128), device=device, dtype=torch.int8, generator=generator)
    return {
        "core_raw": torch.randn(rows, 128, device=device, generator=generator, requires_grad=True),
        "gate_raw": torch.randn(rows, 128, device=device, generator=generator, requires_grad=True),
        "core_q_weight": core_q_weight,
        "gate_q_weight": gate_q_weight,
        "core_weight_scale": torch.rand(128, device=device, generator=generator).mul(0.02).add(0.001),
        "gate_weight_scale": torch.rand(128, device=device, generator=generator).mul(0.02).add(0.001),
        "core_activation_scale": torch.tensor(0.03, device=device),
        "gate_activation_scale": torch.tensor(0.03, device=device),
        "core_bias": torch.randn(128, device=device, generator=generator).mul(0.01),
        "gate_bias": torch.randn(128, device=device, generator=generator).mul(0.01),
        "core_norm_weight": torch.randn(128, device=device, generator=generator).mul(0.01).add(1.0),
        "core_norm_bias": torch.randn(128, device=device, generator=generator).mul(0.01),
        "gate_norm_weight": torch.randn(128, device=device, generator=generator).mul(0.01).add(1.0),
        "gate_norm_bias": torch.randn(128, device=device, generator=generator).mul(0.01),
        "grad_out": torch.randn(rows, 128, device=device, generator=generator),
    }


def run_reference(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    core = inputs["core_raw"].detach().clone().requires_grad_(True)
    gate = inputs["gate_raw"].detach().clone().requires_grad_(True)
    out = cuda_w8a8_static_wmma_dual_gated_tail_n128_saved_pre_autograd(
        F.silu(core),
        F.silu(gate),
        inputs["core_q_weight"],
        inputs["core_weight_scale"],
        inputs["core_activation_scale"],
        inputs["core_bias"],
        inputs["gate_q_weight"],
        inputs["gate_weight_scale"],
        inputs["gate_activation_scale"],
        inputs["gate_bias"],
        inputs["core_norm_weight"],
        inputs["core_norm_bias"],
        inputs["gate_norm_weight"],
        inputs["gate_norm_bias"],
        1.0e-5,
        None,
        None,
        "p96b.reference",
    )
    out.backward(inputs["grad_out"])
    torch.cuda.synchronize()
    return out.detach(), core.grad.detach(), gate.grad.detach()


def run_macro(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    core = inputs["core_raw"].detach().clone().requires_grad_(True)
    gate = inputs["gate_raw"].detach().clone().requires_grad_(True)
    out = cuda_w8a8_static_wmma_dual_silu_gated_tail_n128_saved_pre_autograd(
        core,
        gate,
        inputs["core_q_weight"],
        inputs["core_weight_scale"],
        inputs["core_activation_scale"],
        inputs["core_bias"],
        inputs["gate_q_weight"],
        inputs["gate_weight_scale"],
        inputs["gate_activation_scale"],
        inputs["gate_bias"],
        inputs["core_norm_weight"],
        inputs["core_norm_bias"],
        inputs["gate_norm_weight"],
        inputs["gate_norm_bias"],
        1.0e-5,
        None,
        None,
        "p96b.macro",
    )
    out.backward(inputs["grad_out"])
    torch.cuda.synchronize()
    return out.detach(), core.grad.detach(), gate.grad.detach()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = make_inputs(args.rows, args.seed)
    ref_out, ref_grad_core, ref_grad_gate = run_reference(inputs)
    macro_out, macro_grad_core, macro_grad_gate = run_macro(inputs)
    result = {
        "rows": args.rows,
        "seed": args.seed,
        "forward": diff_summary(macro_out, ref_out),
        "grad_core_raw": diff_summary(macro_grad_core, ref_grad_core),
        "grad_gate_raw": diff_summary(macro_grad_gate, ref_grad_gate),
    }
    result["pass"] = (
        result["forward"]["max_abs"] <= 1.0e-7
        and result["grad_core_raw"]["max_abs"] <= 1.0e-5
        and result["grad_gate_raw"]["max_abs"] <= 1.0e-5
    )
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P96B SiLU Second-Tail Macro Correctness",
        "",
        f"- rows: `{args.rows}`",
        f"- seed: `{args.seed}`",
        f"- PASS: `{str(result['pass']).lower()}`",
        "",
        "| item | max_abs | mean_abs | rmse |",
        "|---|---:|---:|---:|",
    ]
    for key in ("forward", "grad_core_raw", "grad_gate_raw"):
        row = result[key]
        lines.append(f"| `{key}` | `{row['max_abs']:.6e}` | `{row['mean_abs']:.6e}` | `{row['rmse']:.6e}` |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
