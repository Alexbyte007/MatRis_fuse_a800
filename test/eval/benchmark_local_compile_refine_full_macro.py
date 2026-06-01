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
            "Local torch.compile experiment for a larger refine_line backward macro: "
            "smooth/reduce + FFN input-grad + residual grad + GatedMLP second-tail/first-projection scatter."
        )
    )
    parser.add_argument(
        "--cases",
        default="4096:320,6136:382,8192:528,12268:483,16388:658,24596:514",
        help="Explicit rows:nodes cases, for example '4096:320,8192:528'.",
    )
    parser.add_argument("--atom-ratio", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument(
        "--output-json",
        default="results/compile_local_refine_full_macro_realshape_20260526.json",
    )
    return parser.parse_args()


def load_matris_op():
    import matris_op  # type: ignore

    return matris_op


def parse_cases(text: str) -> list[tuple[int, int]]:
    cases = []
    for item in text.split(","):
        if not item.strip():
            continue
        rows_text, nodes_text = item.split(":", 1)
        rows = int(rows_text.strip())
        nodes = int(nodes_text.strip())
        if rows <= 0 or nodes <= 0:
            raise ValueError(f"Invalid rows:nodes case: {item!r}")
        cases.append((rows, nodes))
    return cases


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


def gated_tail_second_first_grad_projected(
    grad_out: torch.Tensor,
    core_second_out: torch.Tensor,
    gate_second_out: torch.Tensor,
    core_norm_weight: torch.Tensor,
    core_norm_bias: torch.Tensor,
    gate_norm_weight: torch.Tensor,
    gate_norm_bias: torch.Tensor,
    eps_tensor: torch.Tensor,
    core_second_weight: torch.Tensor,
    gate_second_weight: torch.Tensor,
    core_first_hidden: torch.Tensor,
    gate_first_hidden: torch.Tensor,
) -> torch.Tensor:
    eps = eps_tensor.reshape(())
    core_centered = core_second_out - core_second_out.mean(dim=1, keepdim=True)
    gate_centered = gate_second_out - gate_second_out.mean(dim=1, keepdim=True)
    core_rstd = torch.rsqrt((core_centered * core_centered).mean(dim=1, keepdim=True) + eps)
    gate_rstd = torch.rsqrt((gate_centered * gate_centered).mean(dim=1, keepdim=True) + eps)
    core_xhat = core_centered * core_rstd
    gate_xhat = gate_centered * gate_rstd
    core_ln = core_xhat * core_norm_weight.reshape(1, -1) + core_norm_bias.reshape(1, -1)
    gate_ln = gate_xhat * gate_norm_weight.reshape(1, -1) + gate_norm_bias.reshape(1, -1)
    core_sig = torch.sigmoid(core_ln)
    core_act = core_ln * core_sig
    gate_act = torch.sigmoid(gate_ln)
    core_silu_grad = core_sig * (1.0 + core_ln * (1.0 - core_sig))
    grad_core_norm = grad_out * gate_act * core_silu_grad * core_norm_weight.reshape(1, -1)
    grad_gate_norm = grad_out * core_act * gate_act * (1.0 - gate_act) * gate_norm_weight.reshape(1, -1)
    dim = grad_out.shape[1]
    grad_core_second = (
        (
            grad_core_norm * dim
            - grad_core_norm.sum(dim=1, keepdim=True)
            - core_xhat * (grad_core_norm * core_xhat).sum(dim=1, keepdim=True)
        )
        * core_rstd
        / dim
    )
    grad_gate_second = (
        (
            grad_gate_norm * dim
            - grad_gate_norm.sum(dim=1, keepdim=True)
            - gate_xhat * (grad_gate_norm * gate_xhat).sum(dim=1, keepdim=True)
        )
        * gate_rstd
        / dim
    )
    grad_core_first = grad_core_second.matmul(core_second_weight) * silu_grad(core_first_hidden)
    grad_gate_first = grad_gate_second.matmul(gate_second_weight) * silu_grad(gate_first_hidden)
    return torch.cat([grad_core_first, grad_gate_first], dim=1)


def refine_full_macro_backward_template(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        atom_index,
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
        core_second_out,
        gate_second_out,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps_tensor,
        core_second_weight,
        gate_second_weight,
        core_first_hidden,
        gate_first_hidden,
        first_weight,
        atom_template,
    ) = tensors

    base_i = base_envelope.index_select(0, source_index)
    base_j = base_envelope.index_select(0, target_index)
    smoothed = nonlinear * base_i * base_j
    node_agg = nonlinear.new_zeros((node_feat.shape[0], nonlinear.shape[1])).index_add(0, target_index, smoothed)
    delta_node, node_hidden = ffn_forward(node_agg, node_w1, node_b1, node_w2, node_b2)
    delta_edge, edge_hidden = ffn_forward(nonlinear, edge_w1, edge_b1, edge_w2, edge_b2)
    node_out = delta_node + node_res * node_feat
    edge_out = delta_edge + edge_res * edge_feat

    grad_node_feat = grad_node_out * node_res
    grad_edge_feat = grad_edge_out * edge_res
    grad_node_agg = ffn_input_grad(grad_node_out, node_hidden, node_w1, node_w2)
    grad_edge_from_ffn = ffn_input_grad(grad_edge_out, edge_hidden, edge_w1, edge_w2)
    grad_smoothed = grad_node_agg.index_select(0, target_index)
    grad_nonlinear = grad_edge_from_ffn + grad_smoothed * base_i * base_j
    grad_base_source = grad_smoothed * nonlinear * base_j
    grad_base_target = grad_smoothed * nonlinear * base_i
    grad_base = torch.zeros_like(base_envelope)
    grad_base = grad_base.index_add(0, source_index, grad_base_source)
    grad_base = grad_base.index_add(0, target_index, grad_base_target)

    grad_projected = gated_tail_second_first_grad_projected(
        grad_nonlinear,
        core_second_out,
        gate_second_out,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps_tensor,
        core_second_weight,
        gate_second_weight,
        core_first_hidden,
        gate_first_hidden,
    )
    grad_cat = grad_projected.matmul(first_weight)
    grad_edge_feat = grad_edge_feat + grad_cat[:, :128]
    grad_atom = torch.zeros_like(atom_template)
    grad_atom = grad_atom.index_add(0, atom_index, grad_cat[:, 128:256])
    grad_node_feat = grad_node_feat.index_add(0, target_index, grad_cat[:, 256:384])
    grad_node_feat = grad_node_feat.index_add(0, source_index, grad_cat[:, 384:512])
    return node_out, edge_out, grad_node_feat, grad_edge_feat, grad_atom, grad_base


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


def cuda_ffn_input_grad_cutlass_epilogue(
    matris_op,
    grad_out: torch.Tensor,
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    return matris_op.two_linear_silu_input_grad_backward_n128_cutlass_epilogue(
        grad_out.contiguous(),
        w2.contiguous(),
        hidden.contiguous(),
        w1.contiguous(),
    )


def cuda_full_macro_mixed(matris_op, tensors: tuple[torch.Tensor, ...], *, use_cublas_ffn: bool) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        atom_index,
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
        core_second_out,
        gate_second_out,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps_tensor,
        core_second_weight,
        gate_second_weight,
        core_first_hidden,
        gate_first_hidden,
        first_weight,
        atom_template,
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
    ffn_grad = cuda_ffn_input_grad_cublas if use_cublas_ffn else cuda_ffn_input_grad
    grad_node_agg = ffn_grad(matris_op, grad_node_out, node_hidden, node_w1, node_w2)
    grad_edge_from_ffn = ffn_grad(matris_op, grad_edge_out, edge_hidden, edge_w1, edge_w2)
    grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat = matris_op.refine_line_smooth_reduce_backward_direct_residual(
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
    grad_core_first, grad_gate_first = matris_op.gated_tail_second_silu_input_grad_macro(
        grad_nonlinear.contiguous(),
        core_second_out.contiguous(),
        gate_second_out.contiguous(),
        core_norm_weight.contiguous(),
        core_norm_bias.contiguous(),
        gate_norm_weight.contiguous(),
        gate_norm_bias.contiguous(),
        float(eps_tensor.item()),
        core_second_weight.contiguous(),
        gate_second_weight.contiguous(),
        core_first_hidden.contiguous(),
        gate_first_hidden.contiguous(),
        False,
    )
    grad_projected = torch.cat([grad_core_first, grad_gate_first], dim=1)
    grad_node_gate, grad_edge_gate, grad_atom = matris_op.refine_line_project_grad_scatter_backward_tile32(
        grad_projected.contiguous(),
        first_weight.contiguous(),
        atom_index.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        int(node_feat.shape[0]),
        int(edge_feat.shape[0]),
        int(atom_template.shape[0]),
    )
    return (
        node_out,
        edge_out,
        grad_node_feat + grad_node_gate,
        grad_edge_feat + grad_edge_gate,
        grad_atom,
        grad_base,
    )


def cuda_full_macro_mixed_cutlass_epilogue(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        atom_index,
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
        core_second_out,
        gate_second_out,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps_tensor,
        core_second_weight,
        gate_second_weight,
        core_first_hidden,
        gate_first_hidden,
        first_weight,
        atom_template,
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
    grad_node_agg = cuda_ffn_input_grad_cutlass_epilogue(matris_op, grad_node_out, node_hidden, node_w1, node_w2)
    grad_edge_from_ffn = cuda_ffn_input_grad_cutlass_epilogue(matris_op, grad_edge_out, edge_hidden, edge_w1, edge_w2)
    grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat = matris_op.refine_line_smooth_reduce_backward_direct_residual(
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
    grad_core_first, grad_gate_first = matris_op.gated_tail_second_silu_input_grad_macro(
        grad_nonlinear.contiguous(),
        core_second_out.contiguous(),
        gate_second_out.contiguous(),
        core_norm_weight.contiguous(),
        core_norm_bias.contiguous(),
        gate_norm_weight.contiguous(),
        gate_norm_bias.contiguous(),
        float(eps_tensor.item()),
        core_second_weight.contiguous(),
        gate_second_weight.contiguous(),
        core_first_hidden.contiguous(),
        gate_first_hidden.contiguous(),
        False,
    )
    grad_node_gate, grad_edge_gate, grad_atom = matris_op.refine_line_project_dual_grad_scatter_add_tile32(
        grad_core_first.contiguous(),
        grad_gate_first.contiguous(),
        first_weight.contiguous(),
        atom_index.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        grad_node_feat.contiguous(),
        grad_edge_feat.contiguous(),
        int(atom_template.shape[0]),
    )
    return (
        node_out,
        edge_out,
        grad_node_gate,
        grad_edge_gate,
        grad_atom,
        grad_base,
    )


def cuda_full_macro_grouped_ffn(matris_op, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        atom_index,
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
        core_second_out,
        gate_second_out,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps_tensor,
        core_second_weight,
        gate_second_weight,
        core_first_hidden,
        gate_first_hidden,
        first_weight,
        atom_template,
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
    grad_node_agg = None
    grad_edge_from_ffn = None
    if hasattr(matris_op, "two_linear_silu_input_grad_backward_n128_cublas_grouped_pair"):
        grad_node_agg, grad_edge_from_ffn = matris_op.two_linear_silu_input_grad_backward_n128_cublas_grouped_pair(
            grad_node_out.contiguous(),
            grad_edge_out.contiguous(),
            node_w2.contiguous(),
            edge_w2.contiguous(),
            node_hidden.contiguous(),
            edge_hidden.contiguous(),
            node_w1.contiguous(),
            edge_w1.contiguous(),
        )
    else:
        grad_node_agg = cuda_ffn_input_grad_cublas(matris_op, grad_node_out, node_hidden, node_w1, node_w2)
        grad_edge_from_ffn = cuda_ffn_input_grad_cublas(matris_op, grad_edge_out, edge_hidden, edge_w1, edge_w2)
    grad_nonlinear, grad_base, grad_node_feat, grad_edge_feat = matris_op.refine_line_smooth_reduce_backward_direct_residual(
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
    grad_core_first, grad_gate_first = matris_op.gated_tail_second_silu_input_grad_macro(
        grad_nonlinear.contiguous(),
        core_second_out.contiguous(),
        gate_second_out.contiguous(),
        core_norm_weight.contiguous(),
        core_norm_bias.contiguous(),
        gate_norm_weight.contiguous(),
        gate_norm_bias.contiguous(),
        float(eps_tensor.item()),
        core_second_weight.contiguous(),
        gate_second_weight.contiguous(),
        core_first_hidden.contiguous(),
        gate_first_hidden.contiguous(),
        False,
    )
    grad_node_gate, grad_edge_gate, grad_atom = matris_op.refine_line_project_dual_grad_scatter_add_tile32(
        grad_core_first.contiguous(),
        grad_gate_first.contiguous(),
        first_weight.contiguous(),
        atom_index.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        int(atom_template.shape[0]),
    )
    return (
        node_out,
        edge_out,
        grad_node_feat + grad_node_gate,
        grad_edge_feat + grad_edge_gate,
        grad_atom,
        grad_base,
    )


def cuda_full_macro_wrapper(matris_op, tensors: tuple[torch.Tensor, ...], *, use_cublas_ffn: bool) -> tuple[torch.Tensor, ...]:
    (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        atom_index,
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
        core_second_out,
        gate_second_out,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps_tensor,
        core_second_weight,
        gate_second_weight,
        core_first_hidden,
        gate_first_hidden,
        first_weight,
        atom_template,
    ) = tensors
    return tuple(
        matris_op.refine_line_full_backward_macro(
            nonlinear.contiguous(),
            base_envelope.contiguous(),
            source_index.contiguous(),
            target_index.contiguous(),
            atom_index.contiguous(),
            node_feat.contiguous(),
            edge_feat.contiguous(),
            node_res.contiguous(),
            edge_res.contiguous(),
            node_w1.contiguous(),
            node_b1.contiguous(),
            node_w2.contiguous(),
            node_b2.contiguous(),
            edge_w1.contiguous(),
            edge_b1.contiguous(),
            edge_w2.contiguous(),
            edge_b2.contiguous(),
            grad_node_out.contiguous(),
            grad_edge_out.contiguous(),
            core_second_out.contiguous(),
            gate_second_out.contiguous(),
            core_norm_weight.contiguous(),
            core_norm_bias.contiguous(),
            gate_norm_weight.contiguous(),
            gate_norm_bias.contiguous(),
            float(eps_tensor.item()),
            core_second_weight.contiguous(),
            gate_second_weight.contiguous(),
            core_first_hidden.contiguous(),
            gate_first_hidden.contiguous(),
            first_weight.contiguous(),
            int(atom_template.shape[0]),
            bool(use_cublas_ffn),
        )
    )


def cuda_full_macro_wrapper_legacy(matris_op, tensors: tuple[torch.Tensor, ...], *, use_cublas_ffn: bool) -> tuple[torch.Tensor, ...]:
    return cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=use_cublas_ffn)


def make_case(rows: int, num_nodes: int, atom_rows: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    nonlinear = torch.randn((rows, 128), device=device)
    base_envelope = torch.randn((num_nodes, 128), device=device)
    source_index = torch.randint(0, num_nodes, (rows,), device=device, dtype=torch.long).contiguous()
    target_index = torch.randint(0, num_nodes, (rows,), device=device, dtype=torch.long).contiguous()
    atom_index = torch.randint(0, atom_rows, (rows,), device=device, dtype=torch.long).contiguous()
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
    core_second_out = torch.randn((rows, 128), device=device)
    gate_second_out = torch.randn((rows, 128), device=device)
    core_norm_weight = torch.randn((128,), device=device)
    core_norm_bias = torch.randn((128,), device=device)
    gate_norm_weight = torch.randn((128,), device=device)
    gate_norm_bias = torch.randn((128,), device=device)
    eps_tensor = torch.tensor(1.0e-5, device=device)
    core_second_weight = torch.randn((128, 128), device=device)
    gate_second_weight = torch.randn((128, 128), device=device)
    core_first_hidden = torch.randn((rows, 128), device=device)
    gate_first_hidden = torch.randn((rows, 128), device=device)
    first_weight = torch.randn((256, 512), device=device)
    atom_template = torch.empty((atom_rows, 128), device=device)
    return (
        nonlinear,
        base_envelope,
        source_index,
        target_index,
        atom_index,
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
        core_second_out,
        gate_second_out,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps_tensor,
        core_second_weight,
        gate_second_weight,
        core_first_hidden,
        gate_first_hidden,
        first_weight,
        atom_template,
    )


def bench_case(
    matris_op,
    compiled_eval: Callable[..., tuple[torch.Tensor, ...]],
    rows: int,
    num_nodes: int,
    atom_rows: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda")
    tensors = make_case(rows, num_nodes, atom_rows, device)
    eager = refine_full_macro_backward_template(*tensors)
    compiled = compiled_eval(*tensors)

    has_cublas_ffn = hasattr(matris_op, "two_linear_silu_input_grad_backward_n128_cublas")
    has_cutlass_epilogue = hasattr(matris_op, "two_linear_silu_input_grad_backward_n128_cutlass_epilogue")
    has_macro = hasattr(matris_op, "refine_line_full_backward_macro")

    time = {
        "torch_eager": cuda_time_ms(
            lambda: refine_full_macro_backward_template(*tensors),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "torch_compile": cuda_time_ms(
            lambda: compiled_eval(*tensors),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "cuda_full_macro_wrapper_legacy": cuda_time_ms(
            lambda: cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=False),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }
    if has_cublas_ffn:
        time["cuda_full_macro_wrapper_legacy_cublas"] = cuda_time_ms(
            lambda: cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=True),
            warmup=args.warmup,
            iters=args.iters,
        )
    if has_cutlass_epilogue:
        time["cuda_full_macro_wrapper_cutlass_epilogue"] = cuda_time_ms(
            lambda: cuda_full_macro_mixed_cutlass_epilogue(matris_op, tensors),
            warmup=args.warmup,
            iters=args.iters,
        )
    if has_macro:
        time["cuda_full_macro_wrapper"] = cuda_time_ms(
            lambda: cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=False),
            warmup=args.warmup,
            iters=args.iters,
        )
    if has_macro and has_cublas_ffn:
        time["cuda_full_macro_wrapper_cublas_ffn"] = cuda_time_ms(
            lambda: cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=True),
            warmup=args.warmup,
            iters=args.iters,
        )
    result: dict[str, Any] = {
        "op": "refine_line_full_backward_macro_like",
        "rows": rows,
        "num_nodes": num_nodes,
        "atom_rows": atom_rows,
        "time": time,
        "speedup_vs_eager": {
            name: time["torch_eager"]["mean_ms"] / item["mean_ms"]
            for name, item in time.items()
            if name != "torch_eager"
        },
        "errors_vs_torch_eager": {
            "compile_grad_node": rel_error(eager[2], compiled[2]),
            "compile_grad_edge": rel_error(eager[3], compiled[3]),
            "compile_grad_atom": rel_error(eager[4], compiled[4]),
            "compile_grad_base": rel_error(eager[5], compiled[5]),
            "cuda_legacy_grad_node": rel_error(eager[2], cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=False)[2]),
            "cuda_legacy_grad_edge": rel_error(eager[3], cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=False)[3]),
            "cuda_legacy_grad_atom": rel_error(eager[4], cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=False)[4]),
            "cuda_legacy_grad_base": rel_error(eager[5], cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=False)[5]),
        },
    }
    if has_cublas_ffn:
        result["errors_vs_torch_eager"].update(
            {
                "cuda_legacy_cublas_grad_node": rel_error(eager[2], cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=True)[2]),
                "cuda_legacy_cublas_grad_edge": rel_error(eager[3], cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=True)[3]),
                "cuda_legacy_cublas_grad_atom": rel_error(eager[4], cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=True)[4]),
                "cuda_legacy_cublas_grad_base": rel_error(eager[5], cuda_full_macro_mixed(matris_op, tensors, use_cublas_ffn=True)[5]),
            }
        )
    if has_cutlass_epilogue:
        cuda_cutlass = cuda_full_macro_mixed_cutlass_epilogue(matris_op, tensors)
        result["errors_vs_torch_eager"].update(
            {
                "cuda_cutlass_epilogue_grad_node": rel_error(eager[2], cuda_cutlass[2]),
                "cuda_cutlass_epilogue_grad_edge": rel_error(eager[3], cuda_cutlass[3]),
                "cuda_cutlass_epilogue_grad_atom": rel_error(eager[4], cuda_cutlass[4]),
                "cuda_cutlass_epilogue_grad_base": rel_error(eager[5], cuda_cutlass[5]),
            }
        )
    if has_macro:
        result["errors_vs_torch_eager"].update(
            {
                "cuda_wrapper_grad_node": rel_error(eager[2], cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=False)[2]),
                "cuda_wrapper_grad_edge": rel_error(eager[3], cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=False)[3]),
                "cuda_wrapper_grad_atom": rel_error(eager[4], cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=False)[4]),
                "cuda_wrapper_grad_base": rel_error(eager[5], cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=False)[5]),
            }
        )
    if has_macro and has_cublas_ffn:
        result["errors_vs_torch_eager"].update(
            {
                "cuda_wrapper_cublas_grad_node": rel_error(eager[2], cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=True)[2]),
                "cuda_wrapper_cublas_grad_edge": rel_error(eager[3], cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=True)[3]),
                "cuda_wrapper_cublas_grad_atom": rel_error(eager[4], cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=True)[4]),
                "cuda_wrapper_cublas_grad_base": rel_error(eager[5], cuda_full_macro_wrapper(matris_op, tensors, use_cublas_ffn=True)[5]),
            }
        )
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({name for item in results for name in item["time"] if name != "torch_eager"})

    def mean_speedup(mode: str) -> float:
        return sum(float(item["speedup_vs_eager"][mode]) for item in results) / len(results)

    def weighted_mean_ms(mode: str) -> float:
        total_rows = sum(int(item["rows"]) for item in results)
        return sum(float(item["time"][mode]["mean_ms"]) * int(item["rows"]) for item in results) / total_rows

    return {
        "num_cases": len(results),
        "rows": [int(item["rows"]) for item in results],
        "num_nodes": [int(item["num_nodes"]) for item in results],
        "atom_rows": [int(item["atom_rows"]) for item in results],
        "mean_speedup_vs_eager": {mode: mean_speedup(mode) for mode in modes},
        "weighted_mean_ms": {mode: weighted_mean_ms(mode) for mode in ["torch_eager", *modes]},
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    matris_op = load_matris_op()
    required = [
        "refine_line_smooth_reduce_forward",
        "refine_line_smooth_reduce_backward_direct_residual",
        "refine_line_full_backward_macro",
        "two_linear_silu_input_grad_backward_n128",
        "gated_tail_second_silu_input_grad_macro",
        "refine_line_project_grad_scatter_backward_tile32",
    ]
    missing = [name for name in required if not hasattr(matris_op, name)]
    if missing:
        raise RuntimeError(f"Missing matris_op symbols: {missing}")
    compiled_eval = torch.compile(
        refine_full_macro_backward_template,
        dynamic=True,
        mode=args.torch_compile_mode,
    )
    results = []
    for rows, num_nodes in parse_cases(args.cases):
        atom_rows = max(1, int(round(num_nodes * args.atom_ratio)))
        results.append(bench_case(matris_op, compiled_eval, rows, num_nodes, atom_rows, args))
    payload = {
        "metadata": {
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_compile": {
                "enabled": True,
                "dynamic": True,
                "mode": args.torch_compile_mode,
                "scope": (
                    "refine_line backward-like big boundary: smooth/reduce + node/edge FFN input-grad "
                    "+ residual grad + GatedMLP tail/second/first input-grad + edge/atom/node scatter"
                ),
                "logs_hint": 'TORCH_LOGS="graph_breaks,recompiles"',
            },
            "cases": args.cases,
            "atom_ratio": args.atom_ratio,
            "warmup": args.warmup,
            "iters": args.iters,
            "baseline_mapping": {
                "torch_eager": "PyTorch equivalent full local boundary",
                "torch_compile": "torch.compile(torch_eager_full_boundary, dynamic=True), local function only",
                "cuda_full_macro_wrapper_legacy": "Old stitched CUDA path: smooth/reduce op + FFN input-grad op + GatedMLP second-tail macro + separate project/scatter op",
                "cuda_full_macro_wrapper_legacy_cublas": "Same as cuda_full_macro_wrapper_legacy but FFN input-grad uses the cuBLAS variant",
                "cuda_full_macro_wrapper_cutlass_epilogue": "Same local boundary but FFN input-grad uses experimental CUTLASS first-GEMM SiLU-grad epilogue",
                "cuda_full_macro_wrapper": "Current C++ macro path using direct dual scatter-add in the project/scatter step",
                "cuda_full_macro_wrapper_cublas_ffn": "Same as cuda_full_macro_wrapper but FFN input-grad uses the cuBLAS variant",
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
