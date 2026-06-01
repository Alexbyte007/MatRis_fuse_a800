from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from infer_salex_lmdb_quant import (  # noqa: E402
    autocast_context,
    build_calculator,
    configure_precision,
    run_activation_calibration,
    select_group_aligned_keys,
)
from quant.layers import (  # noqa: E402
    _load_matris_op,
    cuda_w8a8_static_wmma_dual_gated_tail_n128_saved_pre_autograd,
)


P8D_QUANT_MODE = "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8"
FUSION_MODE = "p28_p26_all_ffn_mlp_input_grad_only"

REFINE_CORE_ALL = "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"
REFINE_GATE_ALL = "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"
ATTN_LINE_89 = (
    "interaction_block.8.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3,"
    "interaction_block.8.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3,"
    "interaction_block.9.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3,"
    "interaction_block.9.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"
)

KNOWN_MATRIS_ENV_KEYS = (
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY",
    "MATRIS_P29_MLP_BWD_KERNEL",
    "MATRIS_P78_FP32_GATED_TAIL_FORWARD",
    "MATRIS_P79_ATTN_LINE_GATHER_CAT",
    "MATRIS_P83C_LINE_ATTENTION_NODE_INPUT",
    "MATRIS_W8A8_BACKEND",
    "MATRIS_W8A8_DISABLE_FAST_WRAPPER",
    "MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE",
    "MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE_MIN_BLOCK",
    "MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE_MAX_BLOCK",
    "MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE_MIN_ROWS",
    "MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE_MAX_ROWS",
    "MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE",
    "MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE_MIN_BLOCK",
    "MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE_MAX_BLOCK",
    "MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE_MIN_ROWS",
    "MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE_MAX_ROWS",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE",
    "MATRIS_QUANT_INCLUDE_GLOBS",
    "MATRIS_QUANT_EXCLUDE_GLOBS",
)

BASE_ENV_FLAGS = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_P78_FP32_GATED_TAIL_FORWARD": "1",
    "MATRIS_P79_ATTN_LINE_GATHER_CAT": "1",
    "MATRIS_P83C_LINE_ATTENTION_NODE_INPUT": "1",
    "MATRIS_W8A8_DISABLE_FAST_WRAPPER": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local numerical diff for one W8A8 GatedMLP second-tail target: "
            "default Triton backend vs cuda_wmma_tail_n128_parallel."
        )
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-mode", default="fp32")
    parser.add_argument("--quant-mode", default=P8D_QUANT_MODE)
    parser.add_argument("--fusion-mode", default=FUSION_MODE)
    parser.add_argument(
        "--include-globs",
        default=f"{REFINE_CORE_ALL},{REFINE_GATE_ALL}",
        help="Comma-separated quant include globs used before model construction.",
    )
    parser.add_argument(
        "--modules",
        default=(
            "interaction_block.0.refine_block_line_graph.edge_nonlinear_update,"
            "interaction_block.9.refine_block_line_graph.edge_nonlinear_update"
        ),
        help="Comma-separated FusedInputGatedMLP module names to diagnose.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--capture-limit", type=int, default=8)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "results" / "p87_w8a8_backend_numerical_diff.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "results" / "p87_w8a8_backend_numerical_diff_summary.md"),
    )
    parser.add_argument("--grad-seed", type=int, default=20260513)
    parser.add_argument(
        "--enable-p87-saved-pre",
        action="store_true",
        help="Enable MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE during the diagnostic run.",
    )
    parser.add_argument(
        "--enable-p87-attn-saved-pre",
        action="store_true",
        help="Enable MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE during the diagnostic run.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def install_base_env(
    include_globs: str,
    enable_p87_saved_pre: bool,
    enable_p87_attn_saved_pre: bool,
) -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in KNOWN_MATRIS_ENV_KEYS}
    for key in KNOWN_MATRIS_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(BASE_ENV_FLAGS)
    os.environ.pop("MATRIS_W8A8_BACKEND", None)
    if include_globs:
        os.environ["MATRIS_QUANT_INCLUDE_GLOBS"] = include_globs
    if enable_p87_saved_pre:
        os.environ["MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE"] = "1"
    if enable_p87_attn_saved_pre:
        os.environ["MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE"] = "1"
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    for key in KNOWN_MATRIS_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in previous.items():
        if value is not None:
            os.environ[key] = value


@contextlib.contextmanager
def w8a8_backend(value: str | None):
    old = os.environ.get("MATRIS_W8A8_BACKEND")
    if value is None:
        os.environ.pop("MATRIS_W8A8_BACKEND", None)
    else:
        os.environ["MATRIS_W8A8_BACKEND"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MATRIS_W8A8_BACKEND", None)
        else:
            os.environ["MATRIS_W8A8_BACKEND"] = old


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(t.min().item()) if t.numel() else 0.0,
        "max": float(t.max().item()) if t.numel() else 0.0,
        "mean": float(t.mean().item()) if t.numel() else 0.0,
        "std": float(t.std(unbiased=False).item()) if t.numel() else 0.0,
        "nan_count": int(torch.isnan(t).sum().item()),
        "inf_count": int(torch.isinf(t).sum().item()),
    }


def diff_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float()
    cand = candidate.detach().float()
    if ref.shape != cand.shape:
        return {
            "shape_mismatch": True,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
        }
    diff = cand - ref
    abs_diff = diff.abs()
    ref_norm = torch.linalg.vector_norm(ref)
    diff_norm = torch.linalg.vector_norm(diff)
    denom = ref.abs().clamp_min(1.0e-12)
    rel = abs_diff / denom
    return {
        "shape_mismatch": False,
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()) if diff.numel() else 0.0,
        "rel_l2": float((diff_norm / ref_norm.clamp_min(1.0e-12)).item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
        "reference_nan_count": int(torch.isnan(ref).sum().item()),
        "candidate_nan_count": int(torch.isnan(cand).sum().item()),
        "reference_inf_count": int(torch.isinf(ref).sum().item()),
        "candidate_inf_count": int(torch.isinf(cand).sum().item()),
    }


def unavailable_diff(reason: str) -> dict[str, Any]:
    return {
        "shape_mismatch": False,
        "unavailable": True,
        "reason": reason,
        "max_abs": math.inf,
        "mean_abs": math.inf,
        "rmse": math.inf,
        "rel_l2": math.inf,
        "max_rel": math.inf,
        "mean_rel": math.inf,
    }


def capture_module_input(
    structures: AseDBDataset,
    calculator,
    module_name: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int]:
    modules = dict(calculator.model.named_modules())
    if module_name not in modules:
        available = [name for name in modules if name.endswith("edge_nonlinear_update")]
        raise KeyError(f"Cannot find module={module_name}. Example edge_nonlinear_update modules: {available[:8]}")

    captured: list[torch.Tensor] = []

    def pre_hook(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]) and not captured:
            captured.append(inputs[0].detach().contiguous())

    handle = modules[module_name].register_forward_pre_hook(pre_hook)
    keys = select_group_aligned_keys(len(structures), args.capture_limit + args.sample_offset, args.sample_seed)
    try:
        for graph_id in keys[args.sample_offset :]:
            captured.clear()
            atom = structures.get_atoms(int(graph_id))
            atom.calc = calculator
            with torch.enable_grad(), autocast_context(args.device, args.precision_mode):
                _ = atom.get_potential_energy()
                if args.task in ("ef", "efs", "efsm") and not captured:
                    _ = atom.get_forces()
                if args.task in ("efs", "efsm") and not captured:
                    _ = atom.get_stress()
            if captured:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                return captured[0], int(graph_id)
    finally:
        handle.remove()

    raise RuntimeError(f"Did not capture input for module={module_name} in {args.capture_limit} samples")


def tail_stages(module, core_pre: torch.Tensor, gate_pre: torch.Tensor) -> dict[str, torch.Tensor]:
    tail = module.fused_tail
    if tail is None:
        raise RuntimeError(f"module={module.module_name} does not have fused_tail")
    core_norm = tail.core_norm(core_pre) if tail.core_norm is not None else core_pre
    gate_norm = tail.gate_norm(gate_pre) if tail.gate_norm is not None else gate_pre
    core_silu = tail.activation_func(core_norm)
    gate_sigmoid = tail.activation_gate(gate_norm)
    final = core_silu * gate_sigmoid
    return {
        "core_second_output": core_pre,
        "gate_second_output": gate_pre,
        "core_norm": core_norm,
        "gate_norm": gate_norm,
        "silu_core": core_silu,
        "sigmoid_gate": gate_sigmoid,
        "core_times_gate": final,
        "final_output": final,
    }


def second_inputs(module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with w8a8_backend(None):
        core, gate = module._fused_first_projection(x)
        if module.core_second_prefix is None or module.gate_second_prefix is None:
            raise RuntimeError(f"module={module.module_name} does not have second-prefix modules")
        return module.core_second_prefix(core), module.gate_second_prefix(gate)


def default_second_tail(module, core: torch.Tensor, gate: torch.Tensor) -> dict[str, torch.Tensor]:
    with w8a8_backend(None):
        core_out, gate_out = module._triton_w8a8_static_fused_second(core, gate)
        if core_out is None or gate_out is None:
            raise RuntimeError(f"default W8A8 fused second returned None for module={module.module_name}")
        return tail_stages(module, core_out, gate_out)


def wmma_tail_with_pre(
    module,
    core: torch.Tensor,
    gate: torch.Tensor,
    backend: str | None,
) -> dict[str, torch.Tensor]:
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre"):
        raise RuntimeError("matris_op.quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre is unavailable")
    tail = module.fused_tail
    core_scale = module.core_second._activation_scale_for(core)
    gate_scale = module.gate_second._activation_scale_for(gate)
    core_bias = module.core_second.bias.float() if module.core_second.bias is not None else None
    gate_bias = module.gate_second.bias.float() if module.gate_second.bias is not None else None
    core_bias_arg = core_bias.contiguous() if core_bias is not None else core.new_empty(0)
    gate_bias_arg = gate_bias.contiguous() if gate_bias is not None else gate.new_empty(0)
    with w8a8_backend(backend):
        out, core_pre, gate_pre = matris_op.quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre(
            core.contiguous(),
            gate.contiguous(),
            module.core_second.q_weight.contiguous(),
            module.gate_second.q_weight.contiguous(),
            module.core_second.scale.float().contiguous(),
            module.gate_second.scale.float().contiguous(),
            core_scale.float().reshape(()).contiguous(),
            gate_scale.float().reshape(()).contiguous(),
            core_bias_arg,
            gate_bias_arg,
            core_bias is not None,
            gate_bias is not None,
            tail.core_norm.weight.float().contiguous(),
            tail.core_norm.bias.float().contiguous(),
            tail.gate_norm.weight.float().contiguous(),
            tail.gate_norm.bias.float().contiguous(),
            float(tail.core_norm.eps),
        )
    stages = tail_stages(module, core_pre, gate_pre)
    stages["cuda_final_output"] = out
    return stages


def default_second_tail_grad(
    module,
    core: torch.Tensor,
    gate: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_req = core.detach().clone().requires_grad_(True)
    gate_req = gate.detach().clone().requires_grad_(True)
    stages = default_second_tail(module, core_req, gate_req)
    grad_core, grad_gate = torch.autograd.grad(
        stages["final_output"],
        (core_req, gate_req),
        grad_out,
        retain_graph=False,
        create_graph=False,
    )
    return grad_core.detach(), grad_gate.detach()


def parallel_second_tail_grad(
    module,
    core: torch.Tensor,
    gate: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tail = module.fused_tail
    core_req = core.detach().clone().requires_grad_(True)
    gate_req = gate.detach().clone().requires_grad_(True)
    core_scale = module.core_second._activation_scale_for(core_req)
    gate_scale = module.gate_second._activation_scale_for(gate_req)
    core_bias = module.core_second.bias.float() if module.core_second.bias is not None else None
    gate_bias = module.gate_second.bias.float() if module.gate_second.bias is not None else None
    with w8a8_backend("cuda_wmma_tail_n128_parallel"):
        out = cuda_w8a8_static_wmma_dual_gated_tail_n128_saved_pre_autograd(
            core_req,
            gate_req,
            module.core_second.q_weight,
            module.core_second.scale,
            core_scale,
            core_bias,
            module.gate_second.q_weight,
            module.gate_second.scale,
            gate_scale,
            gate_bias,
            tail.core_norm.weight,
            tail.core_norm.bias,
            tail.gate_norm.weight,
            tail.gate_norm.bias,
            float(tail.core_norm.eps),
        )
    if out is None:
        raise RuntimeError("parallel saved_pre autograd wrapper returned None")
    grad_core, grad_gate = torch.autograd.grad(
        out,
        (core_req, gate_req),
        grad_out,
        retain_graph=False,
        create_graph=False,
    )
    return out.detach(), grad_core.detach(), grad_gate.detach()


def full_module_grad(
    module,
    x: torch.Tensor,
    backend: str | None,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, str | None]:
    x_req = x.detach().clone().requires_grad_(True)
    with w8a8_backend(backend):
        out = module(x_req)
    if not out.requires_grad:
        return out.detach(), None, "module output does not require grad"
    try:
        grad_x = torch.autograd.grad(
            out,
            x_req,
            grad_out,
            retain_graph=False,
            create_graph=False,
        )[0]
    except RuntimeError as exc:
        return out.detach(), None, str(exc)
    return out.detach(), grad_x.detach(), None


def run_one_module(module, x: torch.Tensor, module_name: str, grad_seed: int) -> dict[str, Any]:
    if not (hasattr(module, "core_second") and hasattr(module, "gate_second") and module.fused_tail is not None):
        raise RuntimeError(f"module={module_name} is not a fused second-tail GatedMLP")
    if not (
        hasattr(module.core_second, "q_weight")
        and hasattr(module.gate_second, "q_weight")
        and module.core_second.q_weight.shape == (128, 128)
        and module.gate_second.q_weight.shape == (128, 128)
    ):
        raise RuntimeError(f"module={module_name} is not a W8A8 n128 second-tail target")

    torch.manual_seed(grad_seed)
    if x.is_cuda:
        torch.cuda.manual_seed_all(grad_seed)

    with torch.no_grad():
        core_second_input, gate_second_input = second_inputs(module, x)
        default_stages = default_second_tail(module, core_second_input, gate_second_input)
        wmma_serial_stages = wmma_tail_with_pre(module, core_second_input, gate_second_input, backend=None)
        wmma_parallel_stages = wmma_tail_with_pre(
            module,
            core_second_input,
            gate_second_input,
            backend="cuda_wmma_tail_n128_parallel",
        )

    grad_out = torch.randn_like(default_stages["final_output"])
    default_grad_core, default_grad_gate = default_second_tail_grad(
        module,
        core_second_input,
        gate_second_input,
        grad_out,
    )
    parallel_out_for_grad, parallel_grad_core, parallel_grad_gate = parallel_second_tail_grad(
        module,
        core_second_input,
        gate_second_input,
        grad_out,
    )

    default_full_out, default_full_grad, default_full_grad_error = full_module_grad(module, x, None, grad_out)
    parallel_full_out, parallel_full_grad, parallel_full_grad_error = full_module_grad(
        module,
        x,
        "cuda_wmma_tail_n128_parallel",
        grad_out,
    )

    stage_names = (
        "core_second_output",
        "gate_second_output",
        "core_norm",
        "gate_norm",
        "silu_core",
        "sigmoid_gate",
        "core_times_gate",
        "final_output",
    )
    diffs: dict[str, Any] = {}
    for name in stage_names:
        diffs[f"default_vs_wmma_serial.{name}"] = diff_summary(default_stages[name], wmma_serial_stages[name])
        diffs[f"default_vs_wmma_parallel.{name}"] = diff_summary(default_stages[name], wmma_parallel_stages[name])
        diffs[f"wmma_serial_vs_parallel.{name}"] = diff_summary(wmma_serial_stages[name], wmma_parallel_stages[name])

    diffs["default_vs_wmma_parallel.cuda_final_output"] = diff_summary(
        default_stages["final_output"],
        wmma_parallel_stages["cuda_final_output"],
    )
    diffs["wmma_parallel_python_tail_vs_cuda_tail.final_output"] = diff_summary(
        wmma_parallel_stages["final_output"],
        wmma_parallel_stages["cuda_final_output"],
    )
    diffs["default_grad_core_second_input_vs_parallel"] = diff_summary(default_grad_core, parallel_grad_core)
    diffs["default_grad_gate_second_input_vs_parallel"] = diff_summary(default_grad_gate, parallel_grad_gate)
    diffs["parallel_grad_path_output_vs_parallel_with_pre"] = diff_summary(
        parallel_out_for_grad,
        wmma_parallel_stages["cuda_final_output"],
    )
    diffs["full_module_default_vs_parallel.output"] = diff_summary(default_full_out, parallel_full_out)
    if default_full_grad is None:
        diffs["full_module_default_vs_parallel.grad_input"] = unavailable_diff(
            f"default backend grad unavailable: {default_full_grad_error}"
        )
    elif parallel_full_grad is None:
        diffs["full_module_default_vs_parallel.grad_input"] = unavailable_diff(
            f"parallel backend grad unavailable: {parallel_full_grad_error}"
        )
    else:
        diffs["full_module_default_vs_parallel.grad_input"] = diff_summary(default_full_grad, parallel_full_grad)

    return {
        "module_name": module_name,
        "captured_input": tensor_summary(x),
        "core_second_input": tensor_summary(core_second_input),
        "gate_second_input": tensor_summary(gate_second_input),
        "core_activation_scale": float(module.core_second._activation_scale_for(core_second_input).detach().float().item()),
        "gate_activation_scale": float(module.gate_second._activation_scale_for(gate_second_input).detach().float().item()),
        "core_weight_scale": {
            "min": float(module.core_second.scale.detach().float().min().item()),
            "max": float(module.core_second.scale.detach().float().max().item()),
            "mean": float(module.core_second.scale.detach().float().mean().item()),
        },
        "gate_weight_scale": {
            "min": float(module.gate_second.scale.detach().float().min().item()),
            "max": float(module.gate_second.scale.detach().float().max().item()),
            "mean": float(module.gate_second.scale.detach().float().mean().item()),
        },
        "full_module_grad_errors": {
            "default": default_full_grad_error,
            "cuda_wmma_tail_n128_parallel": parallel_full_grad_error,
        },
        "diffs": diffs,
    }


def status_from_diff(diff: dict[str, Any], max_abs_tol: float = 1.0e-3, rel_l2_tol: float = 1.0e-4) -> str:
    if diff.get("unavailable"):
        return "FAIL"
    if diff.get("shape_mismatch"):
        return "FAIL"
    if diff.get("candidate_nan_count", 0) or diff.get("candidate_inf_count", 0):
        return "FAIL"
    max_abs = float(diff.get("max_abs", math.inf))
    rel_l2 = float(diff.get("rel_l2", math.inf))
    if max_abs <= max_abs_tol and rel_l2 <= rel_l2_tol:
        return "PASS"
    if max_abs <= 1.0e-2 and rel_l2 <= 1.0e-3:
        return "BORDERLINE"
    return "FAIL"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# P87 W8A8 backend numerical diff")
    lines.append("")
    lines.append(f"- quant_mode: `{payload['quant_mode']}`")
    lines.append(f"- fusion_mode: `{payload['fusion_mode']}`")
    lines.append(f"- include_globs: `{payload['include_globs']}`")
    lines.append(f"- calibration_limit: `{payload['activation_calibration'].get('num_samples', 0)}`")
    lines.append(f"- sample_seed: `{payload['sample_seed']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| module | graph_id | key diff | status | max_abs | rel_l2 |")
    lines.append("|---|---:|---|---|---:|---:|")
    key_names = (
        "default_vs_wmma_parallel.core_second_output",
        "default_vs_wmma_parallel.gate_second_output",
        "default_vs_wmma_parallel.cuda_final_output",
        "default_grad_core_second_input_vs_parallel",
        "default_grad_gate_second_input_vs_parallel",
        "full_module_default_vs_parallel.grad_input",
        "wmma_parallel_python_tail_vs_cuda_tail.final_output",
    )
    for result in payload["results"]:
        for key in key_names:
            diff = result["diffs"][key]
            lines.append(
                "| {module} | {graph_id} | `{key}` | {status} | {max_abs:.6g} | {rel_l2:.6g} |".format(
                    module=result["module_name"],
                    graph_id=result["graph_id"],
                    key=key,
                    status=status_from_diff(diff),
                    max_abs=float(diff.get("max_abs", math.nan)),
                    rel_l2=float(diff.get("rel_l2", math.nan)),
                )
            )
    lines.append("")
    forward_keys = (
        "default_vs_wmma_parallel.core_second_output",
        "default_vs_wmma_parallel.gate_second_output",
        "default_vs_wmma_parallel.cuda_final_output",
        "wmma_parallel_python_tail_vs_cuda_tail.final_output",
    )
    direct_grad_keys = (
        "default_grad_core_second_input_vs_parallel",
        "default_grad_gate_second_input_vs_parallel",
    )
    forward_ok = all(
        status_from_diff(result["diffs"][key]) in ("PASS", "BORDERLINE")
        for result in payload["results"]
        for key in forward_keys
    )
    direct_grad_ok = all(
        status_from_diff(result["diffs"][key]) in ("PASS", "BORDERLINE")
        for result in payload["results"]
        for key in direct_grad_keys
    )
    full_grad_errors = [
        (result["module_name"], result.get("full_module_grad_errors", {}).get("cuda_wmma_tail_n128_parallel"))
        for result in payload["results"]
    ]
    lines.append("## Finding")
    lines.append("")
    if forward_ok and direct_grad_ok and any(reason for _, reason in full_grad_errors):
        lines.append(
            "- The local CUDA math is numerically aligned with the default backend for the checked modules: "
            "second linear, dequant/tail output, and the explicit saved-pre input-grad path all pass near fp32 tolerance."
        )
        lines.append(
            "- The full module path still fails because `cuda_wmma_tail_n128_parallel` returns an output without `grad_fn`; "
            "so the main problem is the dispatch/autograd wrapper path, not the WMMA forward arithmetic itself."
        )
    elif forward_ok and direct_grad_ok:
        lines.append(
            "- The local CUDA math and full-module grad-input path are both aligned with the default backend "
            "for the checked modules. `cuda_wmma_tail_n128_parallel` now returns a differentiable output under this config."
        )
    else:
        lines.append("- See the table above; at least one local forward or explicit grad path differs.")
    for module_name, reason in full_grad_errors:
        if reason:
            lines.append(f"- `{module_name}` full-module parallel grad error: {reason}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If `core_second_output` or `gate_second_output` fails, the issue is before tail activation: "
        "activation scale, weight scale, bias, layout, rounding, clamp, or core/gate scale wiring."
    )
    lines.append(
        "- If second outputs pass but `cuda_final_output` fails, the issue is in the fused tail: "
        "LayerNorm, SiLU/sigmoid, multiply, or CUDA tail postprocess."
    )
    lines.append(
        "- If forward passes but `grad_*` fails, the issue is in saved-pre/input-grad backward: "
        "tail derivative, branch accumulation, or W8A8 grad-input matmul."
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    previous_env = install_base_env(
        args.include_globs,
        args.enable_p87_saved_pre,
        args.enable_p87_attn_saved_pre,
    )
    try:
        precision_info = configure_precision(args.device, args.precision_mode)
        structures = AseDBDataset(config={"src": args.dataset_src})
        calculator_args = argparse.Namespace(
            model=args.model,
            model_path=args.model_path,
            task=args.task,
            device=args.device,
            precision_mode=args.precision_mode,
            quant_mode=args.quant_mode,
            fusion_mode=args.fusion_mode,
            activation_calibration_limit=args.activation_calibration_limit,
            activation_calibration_seed=args.activation_calibration_seed,
        )
        calculator = build_calculator(calculator_args)
        calculator.model.eval()
        activation_calibration = run_activation_calibration(structures, calculator, calculator_args)
        calculator.model.eval()

        modules = dict(calculator.model.named_modules())
        results = []
        for offset, module_name in enumerate(split_csv(args.modules)):
            x, graph_id = capture_module_input(structures, calculator, module_name, args)
            module = modules[module_name]
            module.eval()
            with torch.enable_grad():
                result = run_one_module(module, x, module_name, args.grad_seed + offset)
            result["graph_id"] = graph_id
            results.append(result)

        payload = {
            "quant_mode": args.quant_mode,
            "fusion_mode": args.fusion_mode,
            "include_globs": args.include_globs,
            "modules": split_csv(args.modules),
            "sample_seed": args.sample_seed,
            "sample_offset": args.sample_offset,
            "activation_calibration": activation_calibration,
            "precision_info": precision_info,
            "env": {key: os.environ.get(key) for key in KNOWN_MATRIS_ENV_KEYS if os.environ.get(key) is not None},
            "results": results,
        }

        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_markdown(payload, Path(args.output_md))
        print(f"wrote {output_json}")
        print(f"wrote {args.output_md}")
    finally:
        restore_env(previous_env)


if __name__ == "__main__":
    main()
