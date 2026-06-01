from __future__ import annotations

import contextlib
import os
import time
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Any, Dict
from .functions import (
    MLP,
    GatedMLP,
    aggregate,
    get_normalization,
    Dimwise_softmax,
    directed2undirected_average_or_none,
    fused_line_attention_or_none,
    fused_line_attention_node_input_or_none,
    segment_softmax_weighted_sum_sorted_or_none,
    _use_line_attn_edge_bwd_fusion,
    aggressive_line_attn_eval_mode,
    aggressive_broad_bwd_mode,
    is_aggressive_line_attn_target,
    linear_input_grad_only,
    dual_linear_input_grad_only,
    use_p29_fused_alpha_input_grad_only,
    use_p35_all_line_attn_edge_first_dataflow,
    use_p35_atom_edge_first_dataflow,
    use_p35_refine_line_edge_first_dataflow,
    _load_matris_op,
    p68_grouped_ffn_pair_or_none,
    p79_attn_line_gather_cat_or_none,
    p80_attn_line_node_cat_or_none,
    use_p81_attn_reduce_detail_profile,
)
from torch.utils.checkpoint import checkpoint
from .p71_macro import p71_line_attention_eval_only_vjp, use_p71_line_attn_eval_only_vjp

THRESHOLD_VALUE = 60000 # Safe value for MatRIS-10M (A100-80GB)


def _attn_line_detail_profile_enabled(profile_prefix: str) -> bool:
    return (
        os.environ.get("MATRIS_ATTNLINE_DETAIL_PROFILE", "0") == "1"
        and profile_prefix.endswith(".attn_line")
    )


def _attn_line_record_function_enabled(profile_prefix: str) -> bool:
    return (
        os.environ.get("MATRIS_ATTNLINE_RECORD_FUNCTION", "0") == "1"
        and profile_prefix.endswith(".attn_line")
    )


def _sync_if_cuda_tensor(*values) -> None:
    if not torch.cuda.is_available():
        return
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            torch.cuda.synchronize()
            return
        if isinstance(value, (tuple, list)):
            _sync_if_cuda_tensor(*value)
            return


def _p37_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _p37_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _use_p37_interaction_ste(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P37_INTERACTION_STE", "0") != "1":
        return False
    if not profile_prefix.startswith("interaction_block."):
        return False
    try:
        block_idx = int(profile_prefix.split(".")[1])
    except (IndexError, ValueError):
        return False
    min_block = _p37_env_int("MATRIS_P37_INTERACTION_STE_MIN_BLOCK", 0)
    max_block = _p37_env_int("MATRIS_P37_INTERACTION_STE_MAX_BLOCK", 9)
    return min_block <= block_idx <= max_block


def _p37_straight_through(actual: Tensor | None, proxy: Tensor | None) -> Tensor | None:
    if actual is None:
        return None
    if proxy is None or actual.shape != proxy.shape or not torch.is_floating_point(actual):
        return actual.detach()
    scale = _p37_env_float("MATRIS_P37_INTERACTION_STE_SCALE", 1.0)
    return actual.detach() + (proxy - proxy.detach()) * scale


def _use_p53_one_block_custom_vjp(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P53_ONE_BLOCK_CUSTOM_VJP", "0") != "1":
        return False
    if not profile_prefix.startswith("interaction_block."):
        return False
    try:
        block_idx = int(profile_prefix.split(".")[1])
    except (IndexError, ValueError):
        return False
    target_idx = _p37_env_int("MATRIS_P53_ONE_BLOCK_INDEX", 9)
    return block_idx == target_idx


def _use_p53b_full_block_manual_vjp(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P53B_FULL_BLOCK_MANUAL_VJP", "0") != "1":
        return False
    if not profile_prefix.startswith("interaction_block."):
        return False
    try:
        block_idx = int(profile_prefix.split(".")[1])
    except (IndexError, ValueError):
        return False
    target_idx = _p37_env_int(
        "MATRIS_P53B_FULL_BLOCK_INDEX",
        _p37_env_int("MATRIS_P53_ONE_BLOCK_INDEX", 9),
    )
    return block_idx == target_idx


def _use_p99b_attn_line_module_vjp(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P99B_ATTN_LINE_MODULE_VJP", "0") != "1":
        return False
    if not profile_prefix.startswith("interaction_block.") or not profile_prefix.endswith(".attn_line"):
        return False
    try:
        block_idx = int(profile_prefix.split(".")[1])
    except (IndexError, ValueError):
        return False
    min_block = _p37_env_int("MATRIS_P99B_MIN_BLOCK", 0)
    max_block = _p37_env_int("MATRIS_P99B_MAX_BLOCK", 9)
    return min_block <= block_idx <= max_block


def _use_p101_a3_lite_attn_line_vjp(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P101_A3_LITE_ATTN_LINE_VJP", "0") != "1":
        return False
    if not profile_prefix.startswith("interaction_block.") or not profile_prefix.endswith(".attn_line"):
        return False
    try:
        block_idx = int(profile_prefix.split(".")[1])
    except (IndexError, ValueError):
        return False
    min_block = _p37_env_int("MATRIS_P101_MIN_BLOCK", _p37_env_int("MATRIS_P99B_MIN_BLOCK", 0))
    max_block = _p37_env_int("MATRIS_P101_MAX_BLOCK", _p37_env_int("MATRIS_P99B_MAX_BLOCK", 9))
    return min_block <= block_idx <= max_block


def _p99b_record_function(name: str):
    if os.environ.get("MATRIS_P99B_RECORD_FUNCTION", "0") == "1":
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


def _p101_use_cuda_gather_cat() -> bool:
    return os.environ.get("MATRIS_P101_USE_CUDA_GATHER_CAT", "1") == "1"


def _p101_use_node_input_attention() -> bool:
    return os.environ.get("MATRIS_P101_USE_NODE_INPUT_ATTENTION", "1") == "1"


def _p105_use_attn_line_edge_alpha_cuda_bwd() -> bool:
    return os.environ.get("MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD", "0") == "1"


def _p106_use_attention_backward_edge_direct() -> bool:
    return os.environ.get("MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT", "0") == "1"


def _p107_use_attention_alpha_project_fused() -> bool:
    return os.environ.get("MATRIS_P107_A_CUDA2_ATTN_ALPHA_PROJECT_FUSED", "0") == "1"


def _p108_use_attn_line_target_reduce_bwd() -> bool:
    return os.environ.get("MATRIS_P108_A_CUDA3_ATTN_LINE_TARGET_REDUCE_BWD", "0") == "1"


def _p108_use_attn_line_alpha_tiled_bwd() -> bool:
    return os.environ.get("MATRIS_P108_A_CUDA3_ATTN_LINE_ALPHA_TILED_BWD", "0") == "1"


def _p108_use_attn_line_dense_gemm_scatter_bwd() -> bool:
    return os.environ.get("MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_BWD", "0") == "1"


def _p108_use_attn_line_dense_gemm_op_bwd() -> bool:
    return os.environ.get("MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_OP_BWD", "0") == "1"


def _p108_attn_line_dense_gemm_scatter_threshold() -> int:
    return int(os.environ.get("MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_THRESHOLD", "4096"))


def _use_p58_refine_line_smooth_reduce(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P58_REFINE_LINE_SMOOTH_REDUCE", "0") != "1":
        return False
    return profile_prefix.endswith(".refine_line")


def _use_p58_refine_line_smooth_reduce_sorted(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P58_REFINE_LINE_SMOOTH_REDUCE_SORTED", "0") != "1":
        return False
    return profile_prefix.endswith(".refine_line")


def _refine_line_target_index_is_sorted(graph: dict[str, Tensor]) -> bool:
    target_offsets = graph.get("target_segment_offsets")
    return target_offsets is not None


def _use_p58_refine_line_edge_update(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P58_REFINE_LINE_EDGE_UPDATE", "0") != "1":
        return False
    return profile_prefix.endswith(".refine_line")


def _use_p102_refine_line_r1_ffn_pair_vjp(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P102_REFINE_LINE_R1_FFN_PAIR_VJP", "0") != "1":
        return False
    return profile_prefix.endswith(".refine_line")


def _use_p103_refine_line_r2_edge_update_edge_ffn_vjp(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P103_REFINE_LINE_R2_EDGE_UPDATE_EDGE_FFN_VJP", "0") != "1":
        return False
    return profile_prefix.endswith(".refine_line")


def _use_p104_refine_line_r3_block_vjp(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P104_REFINE_LINE_R3_BLOCK_VJP", "0") != "1":
        return False
    return profile_prefix.endswith(".refine_line")


class _P53OneBlockCustomVJP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        block: "Interaction_Block",
        batch_graph: Dict,
        node_feat: Tensor,
        edge_feat: Tensor,
        threebody_feat: Tensor | None,
        smooth_atom: Tensor | None,
        smooth_line: Tensor | None,
    ):
        smooth_weight = {
            "atom graph": smooth_atom.detach() if isinstance(smooth_atom, torch.Tensor) else None,
            "line graph": smooth_line.detach() if isinstance(smooth_line, torch.Tensor) else None,
        }
        ctx.block = block
        ctx.batch_graph = batch_graph
        ctx.has_threebody = isinstance(threebody_feat, torch.Tensor)
        ctx.has_smooth_atom = isinstance(smooth_atom, torch.Tensor)
        ctx.has_smooth_line = isinstance(smooth_line, torch.Tensor)
        ctx.save_for_backward(
            node_feat.detach(),
            edge_feat.detach(),
            threebody_feat.detach() if isinstance(threebody_feat, torch.Tensor) else node_feat.new_empty(0),
            smooth_atom.detach() if isinstance(smooth_atom, torch.Tensor) else node_feat.new_empty(0),
            smooth_line.detach() if isinstance(smooth_line, torch.Tensor) else node_feat.new_empty(0),
        )
        node_fwd = node_feat.detach()
        edge_fwd = edge_feat.detach()
        threebody_fwd = threebody_feat.detach() if isinstance(threebody_feat, torch.Tensor) else None
        original_requires_grad: list[bool] = []
        params = list(block.parameters())
        for param in params:
            original_requires_grad.append(bool(param.requires_grad))
            param.requires_grad_(False)

        block._p53_one_block_custom_vjp_active = True
        try:
            with torch.enable_grad():
                out_node, out_edge, out_threebody = block.forward(
                    batch_graph=batch_graph,
                    node_feat=node_fwd,
                    edge_feat=edge_fwd,
                    threebody_feat=threebody_fwd,
                    smooth_weight=smooth_weight,
                )
        finally:
            block._p53_one_block_custom_vjp_active = False
            for param, requires_grad in zip(params, original_requires_grad):
                param.requires_grad_(requires_grad)
        return (
            out_node.detach(),
            out_edge.detach(),
            out_threebody.detach() if isinstance(out_threebody, torch.Tensor) else None,
        )

    @staticmethod
    def backward(ctx, grad_node_out: Tensor | None, grad_edge_out: Tensor | None, grad_threebody_out: Tensor | None):
        node_saved, edge_saved, threebody_saved, smooth_atom_saved, smooth_line_saved = ctx.saved_tensors
        block = ctx.block
        smooth_atom = None
        smooth_line = None
        threebody = None
        tensor_inputs: list[Tensor] = []

        node = node_saved.detach().requires_grad_(True)
        edge = edge_saved.detach().requires_grad_(True)
        tensor_inputs.extend([node, edge])

        if ctx.has_threebody:
            threebody = threebody_saved.detach().requires_grad_(True)
            tensor_inputs.append(threebody)
        if ctx.has_smooth_atom:
            smooth_atom = smooth_atom_saved.detach().requires_grad_(True)
            tensor_inputs.append(smooth_atom)
        if ctx.has_smooth_line:
            smooth_line = smooth_line_saved.detach().requires_grad_(True)
            tensor_inputs.append(smooth_line)

        smooth_weight = {
            "atom graph": smooth_atom,
            "line graph": smooth_line,
        }
        original_requires_grad: list[bool] = []
        params = list(block.parameters())
        for param in params:
            original_requires_grad.append(bool(param.requires_grad))
            param.requires_grad_(False)

        block._p53_one_block_custom_vjp_active = True
        try:
            with torch.enable_grad():
                out_node, out_edge, out_threebody = block.forward(
                    batch_graph=ctx.batch_graph,
                    node_feat=node,
                    edge_feat=edge,
                    threebody_feat=threebody,
                    smooth_weight=smooth_weight,
                )
                outputs: list[Tensor] = []
                grad_outputs: list[Tensor] = []
                if grad_node_out is not None:
                    outputs.append(out_node)
                    grad_outputs.append(grad_node_out)
                if grad_edge_out is not None:
                    outputs.append(out_edge)
                    grad_outputs.append(grad_edge_out)
                if grad_threebody_out is not None and isinstance(out_threebody, torch.Tensor):
                    outputs.append(out_threebody)
                    grad_outputs.append(grad_threebody_out)
                if outputs:
                    grads = torch.autograd.grad(
                        outputs,
                        tensor_inputs,
                        grad_outputs=grad_outputs,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                else:
                    grads = tuple(None for _ in tensor_inputs)
        finally:
            block._p53_one_block_custom_vjp_active = False
            for param, requires_grad in zip(params, original_requires_grad):
                param.requires_grad_(requires_grad)

        grad_iter = iter(grads)
        grad_node = next(grad_iter)
        grad_edge = next(grad_iter)
        grad_threebody = next(grad_iter) if ctx.has_threebody else None
        grad_smooth_atom = next(grad_iter) if ctx.has_smooth_atom else None
        grad_smooth_line = next(grad_iter) if ctx.has_smooth_line else None
        return (
            None,
            None,
            grad_node,
            grad_edge,
            grad_threebody,
            grad_smooth_atom,
            grad_smooth_line,
        )


def _p53b_weight_bias(module: nn.Module) -> tuple[Tensor, Tensor | None]:
    if hasattr(module, "q_weight") and hasattr(module, "scale"):
        weight = module.q_weight.float() * module.scale.float().reshape(-1, 1)
        bias = getattr(module, "bias", None)
        return weight, bias.float() if isinstance(bias, Tensor) else None
    if hasattr(module, "weight_low"):
        weight = module.weight_low.float()
        bias = getattr(module, "bias_low", None)
        if bias is None:
            bias = getattr(module, "bias", None)
        return weight, bias.float() if isinstance(bias, Tensor) else None
    if hasattr(module, "weight"):
        weight = module.weight.float()
        bias = getattr(module, "bias", None)
        return weight, bias.float() if isinstance(bias, Tensor) else None
    raise TypeError(f"P53b unsupported linear-like module: {type(module).__name__}")


def _p53b_linear_forward(x: Tensor, module: nn.Module) -> tuple[Tensor, tuple[Tensor]]:
    weight, bias = _p53b_weight_bias(module)
    return F.linear(x.float(), weight, bias), (weight,)


def _p53b_linear_backward(grad_out: Tensor, cache: tuple[Tensor]) -> Tensor:
    (weight,) = cache
    return grad_out.float().matmul(weight)


def _p53b_silu_grad(x: Tensor) -> Tensor:
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def _p53b_layernorm_forward(
    x: Tensor,
    norm: nn.Module | None,
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, float] | None]:
    if norm is None:
        return x, None
    if not isinstance(norm, nn.LayerNorm):
        raise TypeError(f"P53b only supports LayerNorm tails, got {type(norm).__name__}")
    weight = norm.weight.float()
    bias = norm.bias.float()
    x_f = x.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    centered = x_f - mean
    var = centered.square().mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + float(norm.eps))
    x_hat = centered * rstd
    return x_hat * weight + bias, (x_hat, rstd, weight, float(norm.eps))


def _p53b_layernorm_backward(
    grad_out: Tensor,
    cache: tuple[Tensor, Tensor, Tensor, float] | None,
) -> Tensor:
    if cache is None:
        return grad_out
    x_hat, rstd, weight, _eps = cache
    grad_norm = grad_out.float() * weight
    hidden = grad_norm.shape[-1]
    sum_grad = grad_norm.sum(dim=-1, keepdim=True)
    sum_grad_xhat = (grad_norm * x_hat).sum(dim=-1, keepdim=True)
    return (grad_norm * hidden - sum_grad - x_hat * sum_grad_xhat) * (rstd / hidden)


def _p53b_mlp_forward(module: nn.Module, x: Tensor) -> tuple[Tensor, dict[str, Any]]:
    if hasattr(module, "first") and hasattr(module, "second"):
        h, first_cache = _p53b_linear_forward(x, module.first)
        a = F.silu(h)
        out, second_cache = _p53b_linear_forward(a, module.second)
        return out, {
            "kind": "fused_mlp",
            "h": h,
            "first": first_cache,
            "second": second_cache,
        }

    layers = list(getattr(module, "layers", []))
    if len(layers) == 4 and isinstance(layers[0], nn.Module) and isinstance(layers[3], nn.Module):
        h, first_cache = _p53b_linear_forward(x, layers[0])
        a = F.silu(h)
        out, second_cache = _p53b_linear_forward(a, layers[3])
        return out, {
            "kind": "fused_mlp",
            "h": h,
            "first": first_cache,
            "second": second_cache,
        }
    raise TypeError(f"P53b unsupported MLP module: {type(module).__name__}")


def _p53b_mlp_backward(grad_out: Tensor, cache: dict[str, Any]) -> Tensor:
    matris_op = _load_matris_op()
    weight1 = cache["first"][0]
    weight2 = cache["second"][0]
    hidden = cache["h"]
    if (
        matris_op is not None
        and hasattr(matris_op, "two_linear_silu_input_grad_backward_n128")
        and grad_out.is_cuda
        and weight1.is_cuda
        and weight2.is_cuda
        and hidden.is_cuda
        and grad_out.dtype == torch.float32
        and weight1.dtype == torch.float32
        and weight2.dtype == torch.float32
        and hidden.dtype == torch.float32
        and grad_out.ndim == 2
        and hidden.shape == grad_out.shape
        and grad_out.shape[-1] == 128
        and weight1.shape == (128, 128)
        and weight2.shape == (128, 128)
    ):
        return matris_op.two_linear_silu_input_grad_backward_n128(
            grad_out.contiguous(),
            weight2.contiguous(),
            hidden.contiguous(),
            weight1.contiguous(),
        )
    grad_a = _p53b_linear_backward(grad_out, cache["second"])
    grad_h = grad_a * _p53b_silu_grad(cache["h"])
    return _p53b_linear_backward(grad_h, cache["first"])


def _p53b_tail_forward(
    core: Tensor,
    gate: Tensor,
    core_norm: nn.Module | None,
    gate_norm: nn.Module | None,
) -> tuple[Tensor, dict[str, Any]]:
    core_ln, core_ln_cache = _p53b_layernorm_forward(core, core_norm)
    gate_ln, gate_ln_cache = _p53b_layernorm_forward(gate, gate_norm)
    core_act = F.silu(core_ln)
    gate_act = torch.sigmoid(gate_ln)
    core_weight = core_norm.weight.float() if isinstance(core_norm, nn.LayerNorm) else None
    core_bias = core_norm.bias.float() if isinstance(core_norm, nn.LayerNorm) else None
    gate_weight = gate_norm.weight.float() if isinstance(gate_norm, nn.LayerNorm) else None
    gate_bias = gate_norm.bias.float() if isinstance(gate_norm, nn.LayerNorm) else None
    return core_act * gate_act, {
        "core": core,
        "gate": gate,
        "core_ln": core_ln,
        "gate_ln": gate_ln,
        "gate_act": gate_act,
        "core_ln_cache": core_ln_cache,
        "gate_ln_cache": gate_ln_cache,
        "core_weight": core_weight,
        "core_bias": core_bias,
        "gate_weight": gate_weight,
        "gate_bias": gate_bias,
        "eps": float(core_norm.eps) if isinstance(core_norm, nn.LayerNorm) else 1.0e-5,
    }


def _p53b_tail_backward(grad_out: Tensor, cache: dict[str, Any]) -> tuple[Tensor, Tensor]:
    matris_op = _load_matris_op()
    if (
        matris_op is not None
        and hasattr(matris_op, "input_grad_only_gated_tail_backward")
        and isinstance(cache.get("core_weight"), Tensor)
        and isinstance(cache.get("core_bias"), Tensor)
        and isinstance(cache.get("gate_weight"), Tensor)
        and isinstance(cache.get("gate_bias"), Tensor)
        and grad_out.is_cuda
        and cache["core"].is_cuda
        and cache["gate"].is_cuda
        and grad_out.dtype == torch.float32
        and cache["core"].dtype == torch.float32
        and cache["gate"].dtype == torch.float32
        and grad_out.ndim == 2
        and grad_out.shape[-1] in (128, 256)
        and cache["core"].shape == grad_out.shape
        and cache["gate"].shape == grad_out.shape
    ):
        grad_core, grad_gate = matris_op.input_grad_only_gated_tail_backward(
            grad_out.contiguous(),
            cache["core"].contiguous(),
            cache["gate"].contiguous(),
            cache["core_weight"].contiguous(),
            cache["core_bias"].contiguous(),
            cache["gate_weight"].contiguous(),
            cache["gate_bias"].contiguous(),
            cache["eps"],
        )
        return grad_core, grad_gate
    grad_core_ln = grad_out.float() * cache["gate_act"] * _p53b_silu_grad(cache["core_ln"])
    grad_gate_ln = grad_out.float() * F.silu(cache["core_ln"]) * cache["gate_act"] * (1.0 - cache["gate_act"])
    return (
        _p53b_layernorm_backward(grad_core_ln, cache["core_ln_cache"]),
        _p53b_layernorm_backward(grad_gate_ln, cache["gate_ln_cache"]),
    )


def _p53b_silu_linear_tail_forward(module: nn.Module, x: Tensor) -> tuple[Tensor, dict[str, Any]]:
    layers = list(getattr(module, "children", lambda: [])())
    if (
        len(layers) == 3
        and isinstance(layers[1], nn.Dropout)
        and layers[1].p == 0.0
        and isinstance(layers[2], nn.Linear)
    ):
        h = F.silu(x.float())
        out, linear_cache = _p53b_linear_forward(h, layers[2])
        return out, {
            "kind": "silu_linear_tail",
            "input": x,
            "linear": linear_cache,
        }
    raise TypeError(f"P53b unsupported fused GatedMLP tail path: {type(module).__name__}")


def _p53b_silu_linear_tail_backward(grad_out: Tensor, cache: dict[str, Any]) -> Tensor:
    grad_h = _p53b_linear_backward(grad_out, cache["linear"])
    return grad_h * _p53b_silu_grad(cache["input"])


def _p53b_gated_forward(module: nn.Module, x: Tensor) -> tuple[Tensor, dict[str, Any]]:
    if hasattr(module, "fused_first"):
        core_hidden_dim = int(module.core_hidden_dim)
        gate_hidden_dim = int(module.gate_hidden_dim)
        if getattr(module, "fused_first", None) is not None:
            projected, first_cache = _p53b_linear_forward(x, module.fused_first)
            core_first, gate_first = projected.split([core_hidden_dim, gate_hidden_dim], dim=-1)
            split_first = True
        else:
            core_first, core_first_cache = _p53b_linear_forward(x, module.core_first)
            gate_first, gate_first_cache = _p53b_linear_forward(x, module.gate_first)
            first_cache = (core_first_cache, gate_first_cache)
            split_first = False

        if getattr(module, "core_second", None) is not None and getattr(module, "gate_second", None) is not None:
            core_second_in = F.silu(core_first)
            gate_second_in = F.silu(gate_first)
            core_second, core_second_cache = _p53b_linear_forward(core_second_in, module.core_second)
            gate_second, gate_second_cache = _p53b_linear_forward(gate_second_in, module.gate_second)
            tail_module = getattr(module, "fused_tail", None)
            core_norm = tail_module.core_norm if tail_module is not None else module.core_norm
            gate_norm = tail_module.gate_norm if tail_module is not None else module.gate_norm
            out, tail_cache = _p53b_tail_forward(core_second, gate_second, core_norm, gate_norm)
            return out, {
                "kind": "fused_gated",
                "two_linear": True,
                "split_first": split_first,
                "first": first_cache,
                "core_first": core_first,
                "gate_first": gate_first,
                "core_second": core_second_cache,
                "gate_second": gate_second_cache,
                "tail": tail_cache,
            }

        if getattr(module, "core_tail", None) is not None and getattr(module, "gate_tail", None) is not None:
            core_tail, core_tail_cache = _p53b_silu_linear_tail_forward(module.core_tail, core_first)
            gate_tail, gate_tail_cache = _p53b_silu_linear_tail_forward(module.gate_tail, gate_first)
            tail_module = getattr(module, "fused_tail", None)
            core_norm = tail_module.core_norm if tail_module is not None else module.core_norm
            gate_norm = tail_module.gate_norm if tail_module is not None else module.gate_norm
            out, tail_cache = _p53b_tail_forward(core_tail, gate_tail, core_norm, gate_norm)
            return out, {
                "kind": "fused_gated",
                "two_linear": False,
                "split_first": split_first,
                "first": first_cache,
                "core_tail": core_tail_cache,
                "gate_tail": gate_tail_cache,
                "tail": tail_cache,
            }

        out, tail_cache = _p53b_tail_forward(core_first, gate_first, module.core_norm, module.gate_norm)
        return out, {
            "kind": "fused_gated",
            "two_linear": False,
            "split_first": split_first,
            "first": first_cache,
            "tail": tail_cache,
        }

    if hasattr(module, "mlp_core") and hasattr(module, "mlp_gate"):
        core, core_cache = _p53b_mlp_forward(module.mlp_core, x)
        gate, gate_cache = _p53b_mlp_forward(module.mlp_gate, x)
        out, tail_cache = _p53b_tail_forward(core, gate, module.core_norm, module.gate_norm)
        return out, {
            "kind": "plain_gated",
            "core": core_cache,
            "gate": gate_cache,
            "tail": tail_cache,
        }
    raise TypeError(f"P53b unsupported gated module: {type(module).__name__}")


def _p53b_gated_backward(grad_out: Tensor, cache: dict[str, Any]) -> Tensor:
    if cache["kind"] == "plain_gated":
        grad_core, grad_gate = _p53b_tail_backward(grad_out, cache["tail"])
        return _p53b_mlp_backward(grad_core, cache["core"]) + _p53b_mlp_backward(grad_gate, cache["gate"])

    grad_core, grad_gate = _p53b_tail_backward(grad_out, cache["tail"])
    if cache["two_linear"]:
        grad_core_second_in = _p53b_linear_backward(grad_core, cache["core_second"])
        grad_gate_second_in = _p53b_linear_backward(grad_gate, cache["gate_second"])
        grad_core = grad_core_second_in * _p53b_silu_grad(cache["core_first"])
        grad_gate = grad_gate_second_in * _p53b_silu_grad(cache["gate_first"])
    elif "core_tail" in cache:
        grad_core = _p53b_silu_linear_tail_backward(grad_core, cache["core_tail"])
        grad_gate = _p53b_silu_linear_tail_backward(grad_gate, cache["gate_tail"])

    if cache["split_first"]:
        grad_first = torch.cat([grad_core, grad_gate], dim=-1)
        return _p53b_linear_backward(grad_first, cache["first"])

    core_first_cache, gate_first_cache = cache["first"]
    return _p53b_linear_backward(grad_core, core_first_cache) + _p53b_linear_backward(grad_gate, gate_first_cache)


def _p53b_gated_second_tail_projected_grad(
    grad_out: Tensor,
    core_second_out: Tensor,
    gate_second_out: Tensor,
    core_norm_weight: Tensor,
    core_norm_bias: Tensor,
    gate_norm_weight: Tensor,
    gate_norm_bias: Tensor,
    eps_tensor: Tensor,
    core_second_weight: Tensor,
    gate_second_weight: Tensor,
    core_first_hidden: Tensor,
    gate_first_hidden: Tensor,
) -> Tensor:
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
    gate_act = torch.sigmoid(gate_ln)
    core_act = core_ln * core_sig
    core_silu_grad = core_sig * (1.0 + core_ln * (1.0 - core_sig))
    grad_core_norm = grad_out.float() * gate_act * core_silu_grad * core_norm_weight.reshape(1, -1)
    grad_gate_norm = grad_out.float() * core_act * gate_act * (1.0 - gate_act) * gate_norm_weight.reshape(1, -1)
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
    grad_core_first = grad_core_second.matmul(core_second_weight) * _p53b_silu_grad(core_first_hidden)
    grad_gate_first = grad_gate_second.matmul(gate_second_weight) * _p53b_silu_grad(gate_first_hidden)
    return torch.cat([grad_core_first, grad_gate_first], dim=1)


def _p116_gated_second_tail_residual_forward_or_none(
    module: nn.Module,
    x: Tensor,
    old_feat: Tensor,
    res_weight: Tensor,
) -> tuple[Tensor, dict[str, Any]] | None:
    if os.environ.get("MATRIS_P116_GATED_TAIL_SECOND_RESIDUAL_MACRO", "0") != "1":
        return None
    if not (
        hasattr(module, "fused_first")
        and getattr(module, "fused_first", None) is not None
        and getattr(module, "core_second", None) is not None
        and getattr(module, "gate_second", None) is not None
        and getattr(module, "fused_tail", None) is not None
        and isinstance(module.core_second, nn.Linear)
        and isinstance(module.gate_second, nn.Linear)
        and isinstance(module.fused_tail.core_norm, nn.LayerNorm)
        and isinstance(module.fused_tail.gate_norm, nn.LayerNorm)
        and module.fused_tail.core_norm.weight is not None
        and module.fused_tail.core_norm.bias is not None
        and module.fused_tail.gate_norm.weight is not None
        and module.fused_tail.gate_norm.bias is not None
        and x.is_cuda
        and old_feat.is_cuda
        and res_weight.is_cuda
        and x.dtype == torch.float32
        and old_feat.dtype == torch.float32
        and res_weight.dtype == torch.float32
    ):
        return None

    core_hidden_dim = int(module.core_hidden_dim)
    gate_hidden_dim = int(module.gate_hidden_dim)
    projected, first_cache = _p53b_linear_forward(x, module.fused_first)
    core_first, gate_first = projected.split([core_hidden_dim, gate_hidden_dim], dim=-1)
    if not (
        core_first.shape == gate_first.shape == old_feat.shape
        and core_first.ndim == 2
        and core_first.shape[-1] in (128, 256)
        and res_weight.ndim == 2
        and res_weight.shape == (1, core_first.shape[-1])
        and module.core_second.weight.shape == (core_first.shape[-1], core_first.shape[-1])
        and module.gate_second.weight.shape == (core_first.shape[-1], core_first.shape[-1])
        and module.core_second.weight.dtype == torch.float32
        and module.gate_second.weight.dtype == torch.float32
        and module.core_second.weight.is_cuda
        and module.gate_second.weight.is_cuda
    ):
        return None

    core_second_in = F.silu(core_first)
    gate_second_in = F.silu(gate_first)
    core_second, core_second_cache = _p53b_linear_forward(core_second_in, module.core_second)
    gate_second, gate_second_cache = _p53b_linear_forward(gate_second_in, module.gate_second)
    core_update, tail_cache = _p53b_tail_forward(
        core_second,
        gate_second,
        module.fused_tail.core_norm,
        module.fused_tail.gate_norm,
    )
    return core_update + res_weight.float() * old_feat, {
        "kind": "p116_fused_gated_residual",
        "first": first_cache,
        "core_first": core_first,
        "gate_first": gate_first,
        "core_second": core_second_cache,
        "gate_second": gate_second_cache,
        "tail": tail_cache,
        "old_feat": old_feat,
        "res_weight": res_weight.float(),
    }


def _p116_gated_second_tail_residual_backward_or_none(
    grad_out: Tensor,
    cache: dict[str, Any],
) -> tuple[Tensor, Tensor] | None:
    if os.environ.get("MATRIS_P116_GATED_TAIL_SECOND_RESIDUAL_MACRO", "0") != "1":
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "gated_tail_second_silu_residual_input_grad_macro"):
        return None
    tail = cache["tail"]
    core_second_weight = cache["core_second"][0]
    gate_second_weight = cache["gate_second"][0]
    core_first = cache["core_first"]
    gate_first = cache["gate_first"]
    old_feat = cache["old_feat"]
    res_weight = cache["res_weight"]
    if not (
        isinstance(tail.get("core_weight"), Tensor)
        and isinstance(tail.get("core_bias"), Tensor)
        and isinstance(tail.get("gate_weight"), Tensor)
        and isinstance(tail.get("gate_bias"), Tensor)
        and grad_out.is_cuda
        and core_first.is_cuda
        and gate_first.is_cuda
        and old_feat.is_cuda
        and res_weight.is_cuda
        and core_second_weight.is_cuda
        and gate_second_weight.is_cuda
        and grad_out.dtype == torch.float32
        and core_first.dtype == torch.float32
        and gate_first.dtype == torch.float32
        and old_feat.dtype == torch.float32
        and res_weight.dtype == torch.float32
        and grad_out.shape == old_feat.shape == tail["core"].shape == tail["gate"].shape
        and core_first.shape == gate_first.shape == grad_out.shape
    ):
        return None
    grad_core, grad_gate, grad_old, _grad_res_weight = matris_op.gated_tail_second_silu_residual_input_grad_macro(
        grad_out.contiguous(),
        tail["core"].contiguous(),
        tail["gate"].contiguous(),
        tail["core_weight"].contiguous(),
        tail["core_bias"].contiguous(),
        tail["gate_weight"].contiguous(),
        tail["gate_bias"].contiguous(),
        tail["eps"],
        core_second_weight.contiguous(),
        gate_second_weight.contiguous(),
        core_first.contiguous(),
        gate_first.contiguous(),
        bool(core_first.shape[-1] == 128 and os.environ.get("MATRIS_P113_USE_TAIL_BWD_V2", "1") == "1"),
        old_feat.contiguous(),
        res_weight.contiguous(),
    )
    grad_first = torch.cat([grad_core, grad_gate], dim=-1)
    grad_x = _p53b_linear_backward(grad_first, cache["first"])
    return grad_x, grad_old


def _p53b_apply_update(module: nn.Module, x: Tensor) -> tuple[Tensor, dict[str, Any]]:
    if hasattr(module, "fused_first") or hasattr(module, "mlp_core"):
        return _p53b_gated_forward(module, x)
    return _p53b_mlp_forward(module, x)


def _p53b_apply_update_backward(grad_out: Tensor, cache: dict[str, Any]) -> Tensor:
    if cache.get("kind") in ("fused_gated", "plain_gated"):
        return _p53b_gated_backward(grad_out, cache)
    return _p53b_mlp_backward(grad_out, cache)


def _p53b_index_add(data: Tensor, index: Tensor, rows: int) -> Tensor:
    out = data.new_zeros((rows, data.shape[-1]))
    return out.index_add(0, index, data)


def _p53b_gather_backward(grad_gathered: Tensor, index: Tensor, rows: int) -> Tensor:
    out = grad_gathered.new_zeros((rows, grad_gathered.shape[-1]))
    return out.index_add(0, index, grad_gathered)


def _p53b_directed_average_forward(
    data: Tensor,
    directed2undirected: Tensor,
    rows: int,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    sums = _p53b_index_add(data, directed2undirected, rows)
    counts = torch.bincount(directed2undirected, minlength=rows).to(device=data.device, dtype=data.dtype).clamp_min_(1)
    return sums / counts.reshape(-1, 1), (directed2undirected, counts)


def _p53b_directed_average_backward(grad_out: Tensor, cache: tuple[Tensor, Tensor]) -> Tensor:
    directed2undirected, counts = cache
    return grad_out.index_select(0, directed2undirected) / counts.index_select(0, directed2undirected).reshape(-1, 1)


def _p53b_segment_softmax_sum_forward(
    logits: Tensor,
    values: Tensor,
    index: Tensor,
    rows: int,
) -> tuple[Tensor, Tensor]:
    expanded_index = index.reshape(-1, 1).expand(-1, logits.shape[-1])
    max_per = logits.new_full((rows, logits.shape[-1]), -float("inf"))
    max_per.scatter_reduce_(0, expanded_index, logits, reduce="amax", include_self=True)
    exp_logits = torch.exp(logits - max_per.index_select(0, index))
    denom = logits.new_zeros((rows, logits.shape[-1])).index_add(0, index, exp_logits)
    alpha = exp_logits / denom.index_select(0, index).clamp_min(1.0e-12)
    out = _p53b_index_add(alpha * values, index, rows)
    return out, alpha


def _p53b_attention_forward(
    source_logits: Tensor,
    target_logits: Tensor,
    values: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    rows: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    matris_op = _load_matris_op()
    if (
        matris_op is not None
        and hasattr(matris_op, "fused_line_attention_forward")
        and source_logits.is_cuda
        and target_logits.is_cuda
        and values.is_cuda
        and source_logits.dtype == torch.float32
        and target_logits.dtype == torch.float32
        and values.dtype == torch.float32
    ):
        source_out, target_out, source_alpha, target_alpha = matris_op.fused_line_attention_forward(
            source_logits.contiguous(),
            target_logits.contiguous(),
            values.contiguous(),
            source_index.contiguous(),
            target_index.contiguous(),
            int(rows),
        )
    else:
        source_out, source_alpha = _p53b_segment_softmax_sum_forward(source_logits, values, source_index, rows)
        target_out, target_alpha = _p53b_segment_softmax_sum_forward(target_logits, values, target_index, rows)
    return source_out, target_out, {
        "source_index": source_index,
        "target_index": target_index,
        "source_alpha": source_alpha,
        "target_alpha": target_alpha,
        "source_out": source_out,
        "target_out": target_out,
        "values": values,
    }


def _p53b_segment_softmax_sum_backward(
    grad_out: Tensor,
    alpha: Tensor,
    values: Tensor,
    out: Tensor,
    index: Tensor,
) -> tuple[Tensor, Tensor]:
    gathered_grad = grad_out.index_select(0, index)
    gathered_out = out.index_select(0, index)
    grad_values = alpha * gathered_grad
    grad_logits = alpha * gathered_grad * (values - gathered_out)
    return grad_logits, grad_values


def _p53b_attention_backward(
    grad_source_out: Tensor,
    grad_target_out: Tensor,
    cache: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    matris_op = _load_matris_op()
    if (
        matris_op is not None
        and hasattr(matris_op, "fused_line_attention_backward")
        and grad_source_out.is_cuda
        and grad_target_out.is_cuda
        and cache["values"].is_cuda
        and cache["source_alpha"].is_cuda
        and cache["target_alpha"].is_cuda
    ):
        grad_source_logits, grad_target_logits, grad_values = matris_op.fused_line_attention_backward(
            grad_source_out.contiguous(),
            grad_target_out.contiguous(),
            cache["values"].contiguous(),
            cache["source_out"].contiguous(),
            cache["target_out"].contiguous(),
            cache["source_alpha"].contiguous(),
            cache["target_alpha"].contiguous(),
            cache["source_index"].contiguous(),
            cache["target_index"].contiguous(),
        )
        return grad_source_logits, grad_target_logits, grad_values
    grad_source_logits, grad_values_source = _p53b_segment_softmax_sum_backward(
        grad_source_out,
        cache["source_alpha"],
        cache["values"],
        cache["source_out"],
        cache["source_index"],
    )
    grad_target_logits, grad_values_target = _p53b_segment_softmax_sum_backward(
        grad_target_out,
        cache["target_alpha"],
        cache["values"],
        cache["target_out"],
        cache["target_index"],
    )
    return grad_source_logits, grad_target_logits, grad_values_source + grad_values_target


def _p106_attention_backward_edge_direct_or_none(
    grad_source_out: Tensor,
    grad_target_out: Tensor,
    grad_edge_direct: Tensor,
    cache: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor] | None:
    if not _p106_use_attention_backward_edge_direct():
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "fused_line_attention_backward_with_edge_direct"):
        return None
    values = cache["values"]
    source_alpha = cache["source_alpha"]
    target_alpha = cache["target_alpha"]
    if not (
        grad_source_out.is_cuda
        and grad_target_out.is_cuda
        and grad_edge_direct.is_cuda
        and values.is_cuda
        and cache["source_out"].is_cuda
        and cache["target_out"].is_cuda
        and source_alpha.is_cuda
        and target_alpha.is_cuda
        and grad_source_out.dtype == torch.float32
        and grad_target_out.dtype == torch.float32
        and grad_edge_direct.dtype == torch.float32
        and values.dtype == torch.float32
        and source_alpha.dtype == torch.float32
        and target_alpha.dtype == torch.float32
        and grad_edge_direct.shape == values.shape
    ):
        return None
    return matris_op.fused_line_attention_backward_with_edge_direct(
        grad_source_out.contiguous(),
        grad_target_out.contiguous(),
        grad_edge_direct.contiguous(),
        values.contiguous(),
        cache["source_out"].contiguous(),
        cache["target_out"].contiguous(),
        source_alpha.contiguous(),
        target_alpha.contiguous(),
        cache["source_index"].contiguous(),
        cache["target_index"].contiguous(),
    )


def _p107_attention_values_backward_edge_direct_or_none(
    grad_source_out: Tensor,
    grad_target_out: Tensor,
    grad_edge_direct: Tensor,
    cache: dict[str, Any],
) -> Tensor | None:
    if not _p107_use_attention_alpha_project_fused():
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "fused_line_attention_values_backward_with_edge_direct"):
        return None
    values = cache["values"]
    source_alpha = cache["source_alpha"]
    target_alpha = cache["target_alpha"]
    if not (
        grad_source_out.is_cuda
        and grad_target_out.is_cuda
        and grad_edge_direct.is_cuda
        and values.is_cuda
        and source_alpha.is_cuda
        and target_alpha.is_cuda
        and grad_source_out.dtype == torch.float32
        and grad_target_out.dtype == torch.float32
        and grad_edge_direct.dtype == torch.float32
        and values.dtype == torch.float32
        and source_alpha.dtype == torch.float32
        and target_alpha.dtype == torch.float32
        and grad_edge_direct.shape == values.shape
    ):
        return None
    return matris_op.fused_line_attention_values_backward_with_edge_direct(
        grad_source_out.contiguous(),
        grad_target_out.contiguous(),
        grad_edge_direct.contiguous(),
        source_alpha.contiguous(),
        target_alpha.contiguous(),
        cache["source_index"].contiguous(),
        cache["target_index"].contiguous(),
    )


def _p53b_attention_layer_forward(
    layer: nn.Module,
    node_feat: Tensor,
    edge_feat: Tensor,
    graph: Dict,
    directed2undirected: Tensor | None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    source_index = graph["source_index"]
    target_index = graph["target_index"]
    is_atom = directed2undirected is not None
    if is_atom:
        edge_feat_0 = edge_feat.index_select(0, directed2undirected)
    else:
        edge_feat_0 = edge_feat
    source_node_feat = node_feat.index_select(0, source_index)
    target_node_feat = node_feat.index_select(0, target_index)
    edge_x = torch.cat([edge_feat_0, target_node_feat, source_node_feat], dim=1)
    edge_values, edge_cache = _p53b_apply_update(layer.edge_nonlinear_update, edge_x)
    source_logits, source_linear_cache = _p53b_linear_forward(edge_feat_0, layer.source_weight_linear)
    target_logits, target_linear_cache = _p53b_linear_forward(edge_feat_0, layer.target_weight_linear)
    source_out, target_out, attn_cache = _p53b_attention_forward(
        source_logits,
        target_logits,
        edge_values,
        source_index,
        target_index,
        int(node_feat.shape[0]),
    )
    if is_atom:
        edge_update, directed_cache = _p53b_directed_average_forward(
            edge_values,
            directed2undirected,
            int(edge_feat.shape[0]),
        )
    else:
        edge_update = edge_values
        directed_cache = None
    node_x = torch.cat([node_feat, target_out, source_out], dim=1)
    fused_node_residual = _p116_gated_second_tail_residual_forward_or_none(
        layer.node_nonlinear_update,
        node_x,
        node_feat,
        layer.node_res_weight,
    )
    if fused_node_residual is None:
        node_update, node_cache = _p53b_apply_update(layer.node_nonlinear_update, node_x)
        node_out = node_update + layer.node_res_weight.float() * node_feat
    else:
        node_out, node_cache = fused_node_residual
    edge_out = edge_update + layer.edge_res_weight.float() * edge_feat
    return node_out, edge_out, {
        "is_atom": is_atom,
        "source_index": source_index,
        "target_index": target_index,
        "directed2undirected": directed2undirected,
        "edge_rows": int(edge_feat.shape[0]),
        "node_rows": int(node_feat.shape[0]),
        "edge_cache": edge_cache,
        "node_cache": node_cache,
        "source_linear": source_linear_cache,
        "target_linear": target_linear_cache,
        "attn": attn_cache,
        "directed": directed_cache,
        "node_res": layer.node_res_weight.float(),
        "edge_res": layer.edge_res_weight.float(),
    }


def _p53b_attention_layer_backward(
    grad_node_out: Tensor,
    grad_edge_out: Tensor,
    cache: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    grad_node = grad_node_out.float() * cache["node_res"]
    grad_edge = grad_edge_out.float() * cache["edge_res"]

    grad_node_x = _p53b_apply_update_backward(grad_node_out, cache["node_cache"])
    dim = grad_node_out.shape[-1]
    grad_node = grad_node + grad_node_x[:, :dim]
    grad_target_out = grad_node_x[:, dim : 2 * dim]
    grad_source_out = grad_node_x[:, 2 * dim :]

    grad_source_logits, grad_target_logits, grad_edge_values = _p53b_attention_backward(
        grad_source_out,
        grad_target_out,
        cache["attn"],
    )
    if cache["is_atom"]:
        grad_edge_values = grad_edge_values + _p53b_directed_average_backward(grad_edge_out, cache["directed"])
    else:
        grad_edge_values = grad_edge_values + grad_edge_out

    grad_edge_feat_0 = _p53b_linear_backward(grad_source_logits, cache["source_linear"])
    grad_edge_feat_0 = grad_edge_feat_0 + _p53b_linear_backward(grad_target_logits, cache["target_linear"])
    grad_edge_x = _p53b_apply_update_backward(grad_edge_values, cache["edge_cache"])
    dim_edge = grad_edge_x.shape[-1] // 3
    grad_edge_feat_0 = grad_edge_feat_0 + grad_edge_x[:, :dim_edge]
    grad_target_node = grad_edge_x[:, dim_edge : 2 * dim_edge]
    grad_source_node = grad_edge_x[:, 2 * dim_edge :]
    grad_node = grad_node + _p53b_gather_backward(grad_target_node, cache["target_index"], cache["node_rows"])
    grad_node = grad_node + _p53b_gather_backward(grad_source_node, cache["source_index"], cache["node_rows"])
    if cache["is_atom"]:
        grad_edge = grad_edge + _p53b_gather_backward(
            grad_edge_feat_0,
            cache["directed2undirected"],
            cache["edge_rows"],
        )
    else:
        grad_edge = grad_edge + grad_edge_feat_0
    return grad_node, grad_edge


def _p101_line_gather_cat_forward(
    node_feat: Tensor,
    edge_feat: Tensor,
    source_index: Tensor,
    target_index: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    matris_op = _load_matris_op()
    if (
        _p101_use_cuda_gather_cat()
        and matris_op is not None
        and hasattr(matris_op, "line_edge_gather_cat_forward")
        and hasattr(matris_op, "line_edge_cat_grad_scatter_backward")
        and node_feat.is_cuda
        and edge_feat.is_cuda
        and source_index.is_cuda
        and target_index.is_cuda
        and node_feat.dtype == torch.float32
        and edge_feat.dtype == torch.float32
        and source_index.dtype == torch.int64
        and target_index.dtype == torch.int64
        and node_feat.ndim == 2
        and edge_feat.ndim == 2
        and node_feat.shape[-1] == 128
        and edge_feat.shape[-1] == 128
    ):
        return matris_op.line_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            source_index.contiguous(),
            target_index.contiguous(),
        ), {
            "kind": "cuda_line_edge_gather_cat",
            "source_index": source_index,
            "target_index": target_index,
            "node_rows": int(node_feat.shape[0]),
        }

    source_node_feat = node_feat.index_select(0, source_index)
    target_node_feat = node_feat.index_select(0, target_index)
    return torch.cat([edge_feat, target_node_feat, source_node_feat], dim=1), {
        "kind": "torch_line_edge_gather_cat",
        "source_index": source_index,
        "target_index": target_index,
        "node_rows": int(node_feat.shape[0]),
    }


def _p101_line_gather_cat_backward(grad_out: Tensor, cache: dict[str, Any]) -> tuple[Tensor, Tensor]:
    if cache["kind"] == "cuda_line_edge_gather_cat":
        matris_op = _load_matris_op()
        if matris_op is not None and hasattr(matris_op, "line_edge_cat_grad_scatter_backward"):
            grad_node, grad_edge = matris_op.line_edge_cat_grad_scatter_backward(
                grad_out.contiguous(),
                cache["source_index"].contiguous(),
                cache["target_index"].contiguous(),
                cache["node_rows"],
            )
            return grad_node, grad_edge

    dim = grad_out.shape[-1] // 3
    grad_edge = grad_out[:, :dim]
    grad_target_node = grad_out[:, dim : 2 * dim]
    grad_source_node = grad_out[:, 2 * dim :]
    grad_node = _p53b_gather_backward(grad_target_node, cache["target_index"], cache["node_rows"])
    grad_node = grad_node + _p53b_gather_backward(grad_source_node, cache["source_index"], cache["node_rows"])
    return grad_node, grad_edge


def _p101_attention_node_input_forward(
    source_logits: Tensor,
    target_logits: Tensor,
    values: Tensor,
    node_feat: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    target_offsets: Tensor | None,
) -> tuple[Tensor, dict[str, Any]]:
    rows = int(node_feat.shape[0])
    matris_op = _load_matris_op()
    if (
        _p101_use_node_input_attention()
        and matris_op is not None
        and hasattr(matris_op, "fused_line_attention_node_input_forward_target_offsets")
        and isinstance(target_offsets, Tensor)
        and source_logits.is_cuda
        and target_logits.is_cuda
        and values.is_cuda
        and node_feat.is_cuda
        and source_index.is_cuda
        and target_index.is_cuda
        and target_offsets.is_cuda
        and source_logits.dtype == torch.float32
        and target_logits.dtype == torch.float32
        and values.dtype == torch.float32
        and node_feat.dtype == torch.float32
        and source_logits.shape == target_logits.shape == values.shape
        and source_logits.ndim == 2
        and source_logits.shape[-1] == 128
        and node_feat.ndim == 2
        and node_feat.shape[-1] == 128
        and target_offsets.dtype == torch.int64
        and target_offsets.ndim == 1
        and target_offsets.numel() == rows + 1
    ):
        fusion_node_feat, source_alpha, target_alpha = matris_op.fused_line_attention_node_input_forward_target_offsets(
            source_logits.contiguous(),
            target_logits.contiguous(),
            values.contiguous(),
            source_index.contiguous(),
            target_index.contiguous(),
            target_offsets.contiguous(),
            node_feat.contiguous(),
            rows,
        )
        return fusion_node_feat, {
            "kind": "cuda_node_input",
            "values": values,
            "fusion_node_feat": fusion_node_feat,
            "source_alpha": source_alpha,
            "target_alpha": target_alpha,
            "source_index": source_index,
            "target_index": target_index,
        }

    source_out, target_out, attn_cache = _p53b_attention_forward(
        source_logits,
        target_logits,
        values,
        source_index,
        target_index,
        rows,
    )
    fusion_node_feat = torch.cat([node_feat, target_out, source_out], dim=1)
    attn_cache["kind"] = "torch_node_input"
    return fusion_node_feat, attn_cache


def _p101_attention_node_input_backward(
    grad_fusion_node_feat: Tensor,
    cache: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dim = grad_fusion_node_feat.shape[-1] // 3
    grad_node_direct = grad_fusion_node_feat[:, :dim].contiguous()
    grad_target_out = grad_fusion_node_feat[:, dim : 2 * dim].contiguous()
    grad_source_out = grad_fusion_node_feat[:, 2 * dim :].contiguous()
    if cache["kind"] == "cuda_node_input":
        target_out = cache["fusion_node_feat"][:, dim : 2 * dim].contiguous()
        source_out = cache["fusion_node_feat"][:, 2 * dim :].contiguous()
        attn_cache = {
            "source_index": cache["source_index"],
            "target_index": cache["target_index"],
            "source_alpha": cache["source_alpha"],
            "target_alpha": cache["target_alpha"],
            "source_out": source_out,
            "target_out": target_out,
            "values": cache["values"],
        }
    else:
        attn_cache = cache
    grad_source_logits, grad_target_logits, grad_values = _p53b_attention_backward(
        grad_source_out,
        grad_target_out,
        attn_cache,
    )
    return grad_node_direct, grad_source_logits, grad_target_logits, grad_values


def _p106_attention_node_input_backward_edge_direct_or_none(
    grad_fusion_node_feat: Tensor,
    cache: dict[str, Any],
    grad_edge_direct: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    dim = grad_fusion_node_feat.shape[-1] // 3
    grad_node_direct = grad_fusion_node_feat[:, :dim].contiguous()
    grad_target_out = grad_fusion_node_feat[:, dim : 2 * dim].contiguous()
    grad_source_out = grad_fusion_node_feat[:, 2 * dim :].contiguous()
    if cache["kind"] == "cuda_node_input":
        target_out = cache["fusion_node_feat"][:, dim : 2 * dim].contiguous()
        source_out = cache["fusion_node_feat"][:, 2 * dim :].contiguous()
        attn_cache = {
            "source_index": cache["source_index"],
            "target_index": cache["target_index"],
            "source_alpha": cache["source_alpha"],
            "target_alpha": cache["target_alpha"],
            "source_out": source_out,
            "target_out": target_out,
            "values": cache["values"],
        }
    else:
        attn_cache = cache
    fused = _p106_attention_backward_edge_direct_or_none(
        grad_source_out,
        grad_target_out,
        grad_edge_direct,
        attn_cache,
    )
    if fused is None:
        return None
    grad_source_logits, grad_target_logits, grad_values = fused
    return grad_node_direct, grad_source_logits, grad_target_logits, grad_values


def _p107_attention_node_input_values_backward_or_none(
    grad_fusion_node_feat: Tensor,
    cache: dict[str, Any],
    grad_edge_direct: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]] | None:
    dim = grad_fusion_node_feat.shape[-1] // 3
    grad_node_direct = grad_fusion_node_feat[:, :dim].contiguous()
    grad_target_out = grad_fusion_node_feat[:, dim : 2 * dim].contiguous()
    grad_source_out = grad_fusion_node_feat[:, 2 * dim :].contiguous()
    if cache["kind"] == "cuda_node_input":
        target_out = cache["fusion_node_feat"][:, dim : 2 * dim].contiguous()
        source_out = cache["fusion_node_feat"][:, 2 * dim :].contiguous()
        attn_cache = {
            "source_index": cache["source_index"],
            "target_index": cache["target_index"],
            "source_alpha": cache["source_alpha"],
            "target_alpha": cache["target_alpha"],
            "source_out": source_out,
            "target_out": target_out,
            "values": cache["values"],
        }
    else:
        attn_cache = cache
    grad_values = _p107_attention_values_backward_edge_direct_or_none(
        grad_source_out,
        grad_target_out,
        grad_edge_direct,
        attn_cache,
    )
    if grad_values is None:
        return None
    return grad_node_direct, grad_source_out, grad_target_out, grad_values, attn_cache


def _p105_edge_update_alpha_cuda_backward_or_none(
    grad_edge_values: Tensor,
    edge_cache: dict[str, Any],
    grad_source_logits: Tensor,
    grad_target_logits: Tensor,
    source_linear_cache: tuple[Tensor],
    target_linear_cache: tuple[Tensor],
    source_index: Tensor,
    target_index: Tensor,
    node_rows: int,
) -> tuple[Tensor, Tensor] | None:
    if not _p105_use_attn_line_edge_alpha_cuda_bwd():
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "line_edge_silu_project_alpha_grad_scatter_backward_tile32"):
        return None
    if not (
        edge_cache.get("kind") == "fused_gated"
        and edge_cache.get("two_linear") is True
        and edge_cache.get("split_first") is True
        and isinstance(edge_cache.get("first"), tuple)
    ):
        return None
    first_cache = edge_cache["first"]
    if len(first_cache) != 1:
        return None
    first_weight = first_cache[0]
    source_weight = source_linear_cache[0]
    target_weight = target_linear_cache[0]
    core_first = edge_cache.get("core_first")
    gate_first = edge_cache.get("gate_first")
    if not (
        isinstance(core_first, Tensor)
        and isinstance(gate_first, Tensor)
        and grad_edge_values.is_cuda
        and grad_source_logits.is_cuda
        and grad_target_logits.is_cuda
        and source_index.is_cuda
        and target_index.is_cuda
        and first_weight.is_cuda
        and source_weight.is_cuda
        and target_weight.is_cuda
        and grad_edge_values.dtype == torch.float32
        and grad_source_logits.dtype == torch.float32
        and grad_target_logits.dtype == torch.float32
        and core_first.dtype == torch.float32
        and gate_first.dtype == torch.float32
        and first_weight.dtype == torch.float32
        and source_weight.dtype == torch.float32
        and target_weight.dtype == torch.float32
        and grad_edge_values.ndim == 2
        and grad_source_logits.shape == grad_target_logits.shape == grad_edge_values.shape
        and core_first.shape == gate_first.shape == grad_edge_values.shape
        and grad_edge_values.shape[-1] == 128
        and first_weight.shape == (256, 384)
        and source_weight.shape == (128, 128)
        and target_weight.shape == (128, 128)
    ):
        return None

    grad_core, grad_gate = _p53b_tail_backward(grad_edge_values, edge_cache["tail"])
    grad_core_second_in = _p53b_linear_backward(grad_core, edge_cache["core_second"])
    grad_gate_second_in = _p53b_linear_backward(grad_gate, edge_cache["gate_second"])
    grad_node, grad_edge = matris_op.line_edge_silu_project_alpha_grad_scatter_backward_tile32(
        grad_core_second_in.contiguous(),
        grad_gate_second_in.contiguous(),
        core_first.contiguous(),
        gate_first.contiguous(),
        first_weight.contiguous(),
        grad_source_logits.contiguous(),
        grad_target_logits.contiguous(),
        source_weight.contiguous(),
        target_weight.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        int(node_rows),
    )
    return grad_node, grad_edge


def _p108_edge_update_alpha_tiled_cuda_backward_or_none(
    grad_edge_values: Tensor,
    edge_cache: dict[str, Any],
    grad_source_logits: Tensor,
    grad_target_logits: Tensor,
    source_linear_cache: tuple[Tensor],
    target_linear_cache: tuple[Tensor],
    source_index: Tensor,
    target_index: Tensor,
    node_rows: int,
) -> tuple[Tensor, Tensor] | None:
    if not _p108_use_attn_line_alpha_tiled_bwd():
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32"):
        return None
    if not (
        edge_cache.get("kind") == "fused_gated"
        and edge_cache.get("two_linear") is True
        and edge_cache.get("split_first") is True
        and isinstance(edge_cache.get("first"), tuple)
    ):
        return None
    first_cache = edge_cache["first"]
    if len(first_cache) != 1:
        return None
    first_weight = first_cache[0]
    source_weight = source_linear_cache[0]
    target_weight = target_linear_cache[0]
    core_first = edge_cache.get("core_first")
    gate_first = edge_cache.get("gate_first")
    if not (
        isinstance(core_first, Tensor)
        and isinstance(gate_first, Tensor)
        and grad_edge_values.is_cuda
        and grad_source_logits.is_cuda
        and grad_target_logits.is_cuda
        and source_index.is_cuda
        and target_index.is_cuda
        and first_weight.is_cuda
        and source_weight.is_cuda
        and target_weight.is_cuda
        and grad_edge_values.dtype == torch.float32
        and grad_source_logits.dtype == torch.float32
        and grad_target_logits.dtype == torch.float32
        and core_first.dtype == torch.float32
        and gate_first.dtype == torch.float32
        and first_weight.dtype == torch.float32
        and source_weight.dtype == torch.float32
        and target_weight.dtype == torch.float32
        and grad_edge_values.ndim == 2
        and grad_source_logits.shape == grad_target_logits.shape == grad_edge_values.shape
        and core_first.shape == gate_first.shape == grad_edge_values.shape
        and grad_edge_values.shape[-1] == 128
        and first_weight.shape == (256, 384)
        and source_weight.shape == (128, 128)
        and target_weight.shape == (128, 128)
    ):
        return None

    grad_core, grad_gate = _p53b_tail_backward(grad_edge_values, edge_cache["tail"])
    grad_core_second_in = _p53b_linear_backward(grad_core, edge_cache["core_second"])
    grad_gate_second_in = _p53b_linear_backward(grad_gate, edge_cache["gate_second"])
    grad_node, grad_edge = matris_op.line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32(
        grad_core_second_in.contiguous(),
        grad_gate_second_in.contiguous(),
        core_first.contiguous(),
        gate_first.contiguous(),
        first_weight.contiguous(),
        grad_source_logits.contiguous(),
        grad_target_logits.contiguous(),
        source_weight.contiguous(),
        target_weight.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        int(node_rows),
    )
    return grad_node, grad_edge


def _p108_edge_update_alpha_dense_gemm_scatter_backward_or_none(
    grad_edge_values: Tensor,
    edge_cache: dict[str, Any],
    grad_source_logits: Tensor,
    grad_target_logits: Tensor,
    source_linear_cache: tuple[Tensor],
    target_linear_cache: tuple[Tensor],
    source_index: Tensor,
    target_index: Tensor,
    node_rows: int,
) -> tuple[Tensor, Tensor] | None:
    if not _p108_use_attn_line_dense_gemm_scatter_bwd():
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "line_edge_cat_grad_scatter_backward"):
        return None
    if int(grad_edge_values.shape[0]) <= _p108_attn_line_dense_gemm_scatter_threshold():
        return None
    if not (
        edge_cache.get("kind") == "fused_gated"
        and edge_cache.get("two_linear") is True
        and edge_cache.get("split_first") is True
        and isinstance(edge_cache.get("first"), tuple)
    ):
        return None
    first_cache = edge_cache["first"]
    if len(first_cache) != 1:
        return None
    first_weight = first_cache[0]
    source_weight = source_linear_cache[0]
    target_weight = target_linear_cache[0]
    core_first = edge_cache.get("core_first")
    gate_first = edge_cache.get("gate_first")
    if not (
        isinstance(core_first, Tensor)
        and isinstance(gate_first, Tensor)
        and grad_edge_values.is_cuda
        and grad_source_logits.is_cuda
        and grad_target_logits.is_cuda
        and source_index.is_cuda
        and target_index.is_cuda
        and first_weight.is_cuda
        and source_weight.is_cuda
        and target_weight.is_cuda
        and grad_edge_values.dtype == torch.float32
        and grad_source_logits.dtype == torch.float32
        and grad_target_logits.dtype == torch.float32
        and core_first.dtype == torch.float32
        and gate_first.dtype == torch.float32
        and first_weight.dtype == torch.float32
        and source_weight.dtype == torch.float32
        and target_weight.dtype == torch.float32
        and grad_edge_values.ndim == 2
        and grad_source_logits.shape == grad_target_logits.shape == grad_edge_values.shape
        and core_first.shape == gate_first.shape == grad_edge_values.shape
        and grad_edge_values.shape[-1] == 128
        and first_weight.shape == (256, 384)
        and source_weight.shape == (128, 128)
        and target_weight.shape == (128, 128)
    ):
        return None

    grad_core, grad_gate = _p53b_tail_backward(grad_edge_values, edge_cache["tail"])
    grad_core_second_in = _p53b_linear_backward(grad_core, edge_cache["core_second"])
    grad_gate_second_in = _p53b_linear_backward(grad_gate, edge_cache["gate_second"])
    grad_core_first = grad_core_second_in * _p53b_silu_grad(core_first)
    grad_gate_first = grad_gate_second_in * _p53b_silu_grad(gate_first)
    grad_hidden = torch.cat([grad_core_first, grad_gate_first], dim=-1)
    grad_cat = grad_hidden.matmul(first_weight.contiguous())
    grad_alpha = grad_source_logits.contiguous().matmul(source_weight.contiguous())
    grad_alpha = grad_alpha + grad_target_logits.contiguous().matmul(target_weight.contiguous())
    grad_cat[:, :128] = grad_cat[:, :128] + grad_alpha
    grad_node, grad_edge = matris_op.line_edge_cat_grad_scatter_backward(
        grad_cat.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        int(node_rows),
    )
    return grad_node, grad_edge


def _p108_edge_update_alpha_dense_gemm_op_backward_or_none(
    grad_edge_values: Tensor,
    edge_cache: dict[str, Any],
    grad_source_logits: Tensor,
    grad_target_logits: Tensor,
    source_linear_cache: tuple[Tensor],
    target_linear_cache: tuple[Tensor],
    source_index: Tensor,
    target_index: Tensor,
    node_rows: int,
) -> tuple[Tensor, Tensor] | None:
    if not _p108_use_attn_line_dense_gemm_op_bwd():
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm"):
        return None
    if int(grad_edge_values.shape[0]) <= _p108_attn_line_dense_gemm_scatter_threshold():
        return None
    if not (
        edge_cache.get("kind") == "fused_gated"
        and edge_cache.get("two_linear") is True
        and edge_cache.get("split_first") is True
        and isinstance(edge_cache.get("first"), tuple)
    ):
        return None
    first_cache = edge_cache["first"]
    if len(first_cache) != 1:
        return None
    first_weight = first_cache[0]
    source_weight = source_linear_cache[0]
    target_weight = target_linear_cache[0]
    core_first = edge_cache.get("core_first")
    gate_first = edge_cache.get("gate_first")
    if not (
        isinstance(core_first, Tensor)
        and isinstance(gate_first, Tensor)
        and grad_edge_values.is_cuda
        and grad_source_logits.is_cuda
        and grad_target_logits.is_cuda
        and source_index.is_cuda
        and target_index.is_cuda
        and first_weight.is_cuda
        and source_weight.is_cuda
        and target_weight.is_cuda
        and grad_edge_values.dtype == torch.float32
        and grad_source_logits.dtype == torch.float32
        and grad_target_logits.dtype == torch.float32
        and core_first.dtype == torch.float32
        and gate_first.dtype == torch.float32
        and first_weight.dtype == torch.float32
        and source_weight.dtype == torch.float32
        and target_weight.dtype == torch.float32
        and grad_edge_values.ndim == 2
        and grad_source_logits.shape == grad_target_logits.shape == grad_edge_values.shape
        and core_first.shape == gate_first.shape == grad_edge_values.shape
        and grad_edge_values.shape[-1] == 128
        and first_weight.shape == (256, 384)
        and source_weight.shape == (128, 128)
        and target_weight.shape == (128, 128)
    ):
        return None

    grad_core, grad_gate = _p53b_tail_backward(grad_edge_values, edge_cache["tail"])
    grad_core_second_in = _p53b_linear_backward(grad_core, edge_cache["core_second"])
    grad_gate_second_in = _p53b_linear_backward(grad_gate, edge_cache["gate_second"])
    grad_node, grad_edge = matris_op.line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm(
        grad_core_second_in.contiguous(),
        grad_gate_second_in.contiguous(),
        core_first.contiguous(),
        gate_first.contiguous(),
        first_weight.contiguous(),
        grad_source_logits.contiguous(),
        grad_target_logits.contiguous(),
        source_weight.contiguous(),
        target_weight.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        int(node_rows),
    )
    return grad_node, grad_edge


def _p108_edge_update_alpha_target_reduce_cuda_backward_or_none(
    grad_edge_values: Tensor,
    edge_cache: dict[str, Any],
    grad_source_logits: Tensor,
    grad_target_logits: Tensor,
    source_linear_cache: tuple[Tensor],
    target_linear_cache: tuple[Tensor],
    source_index: Tensor,
    target_index: Tensor,
    target_offsets: Tensor | None,
    node_rows: int,
) -> tuple[Tensor, Tensor] | None:
    if not _p108_use_attn_line_target_reduce_bwd():
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32"):
        return None
    if target_offsets is None:
        return None
    if not (
        edge_cache.get("kind") == "fused_gated"
        and edge_cache.get("two_linear") is True
        and edge_cache.get("split_first") is True
        and isinstance(edge_cache.get("first"), tuple)
    ):
        return None
    first_cache = edge_cache["first"]
    if len(first_cache) != 1:
        return None
    first_weight = first_cache[0]
    source_weight = source_linear_cache[0]
    target_weight = target_linear_cache[0]
    core_first = edge_cache.get("core_first")
    gate_first = edge_cache.get("gate_first")
    if not (
        isinstance(core_first, Tensor)
        and isinstance(gate_first, Tensor)
        and grad_edge_values.is_cuda
        and grad_source_logits.is_cuda
        and grad_target_logits.is_cuda
        and source_index.is_cuda
        and target_index.is_cuda
        and target_offsets.is_cuda
        and first_weight.is_cuda
        and source_weight.is_cuda
        and target_weight.is_cuda
        and grad_edge_values.dtype == torch.float32
        and grad_source_logits.dtype == torch.float32
        and grad_target_logits.dtype == torch.float32
        and core_first.dtype == torch.float32
        and gate_first.dtype == torch.float32
        and first_weight.dtype == torch.float32
        and source_weight.dtype == torch.float32
        and target_weight.dtype == torch.float32
        and source_index.dtype == torch.int64
        and target_index.dtype == torch.int64
        and target_offsets.dtype == torch.int64
        and target_offsets.ndim == 1
        and target_offsets.numel() == int(node_rows) + 1
        and grad_edge_values.ndim == 2
        and grad_source_logits.shape == grad_target_logits.shape == grad_edge_values.shape
        and core_first.shape == gate_first.shape == grad_edge_values.shape
        and grad_edge_values.shape[-1] == 128
        and first_weight.shape == (256, 384)
        and source_weight.shape == (128, 128)
        and target_weight.shape == (128, 128)
    ):
        return None

    grad_core, grad_gate = _p53b_tail_backward(grad_edge_values, edge_cache["tail"])
    grad_core_second_in = _p53b_linear_backward(grad_core, edge_cache["core_second"])
    grad_gate_second_in = _p53b_linear_backward(grad_gate, edge_cache["gate_second"])
    grad_node, grad_edge = matris_op.line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32(
        grad_core_second_in.contiguous(),
        grad_gate_second_in.contiguous(),
        core_first.contiguous(),
        gate_first.contiguous(),
        first_weight.contiguous(),
        grad_source_logits.contiguous(),
        grad_target_logits.contiguous(),
        source_weight.contiguous(),
        target_weight.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        target_offsets.contiguous(),
        int(node_rows),
    )
    return grad_node, grad_edge


def _p107_edge_update_alpha_attention_cuda_backward_or_none(
    grad_edge_values: Tensor,
    edge_cache: dict[str, Any],
    grad_source_out: Tensor,
    grad_target_out: Tensor,
    attn_cache: dict[str, Any],
    source_linear_cache: tuple[Tensor],
    target_linear_cache: tuple[Tensor],
    source_index: Tensor,
    target_index: Tensor,
    node_rows: int,
) -> tuple[Tensor, Tensor] | None:
    if not _p107_use_attention_alpha_project_fused():
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32"):
        return None
    if not (
        edge_cache.get("kind") == "fused_gated"
        and edge_cache.get("two_linear") is True
        and edge_cache.get("split_first") is True
        and isinstance(edge_cache.get("first"), tuple)
    ):
        return None
    first_cache = edge_cache["first"]
    if len(first_cache) != 1:
        return None
    first_weight = first_cache[0]
    source_weight = source_linear_cache[0]
    target_weight = target_linear_cache[0]
    core_first = edge_cache.get("core_first")
    gate_first = edge_cache.get("gate_first")
    values = attn_cache["values"]
    source_out = attn_cache["source_out"]
    target_out = attn_cache["target_out"]
    source_alpha = attn_cache["source_alpha"]
    target_alpha = attn_cache["target_alpha"]
    if not (
        isinstance(core_first, Tensor)
        and isinstance(gate_first, Tensor)
        and grad_edge_values.is_cuda
        and grad_source_out.is_cuda
        and grad_target_out.is_cuda
        and values.is_cuda
        and source_out.is_cuda
        and target_out.is_cuda
        and source_alpha.is_cuda
        and target_alpha.is_cuda
        and source_index.is_cuda
        and target_index.is_cuda
        and first_weight.is_cuda
        and source_weight.is_cuda
        and target_weight.is_cuda
        and grad_edge_values.dtype == torch.float32
        and grad_source_out.dtype == torch.float32
        and grad_target_out.dtype == torch.float32
        and values.dtype == torch.float32
        and source_out.dtype == torch.float32
        and target_out.dtype == torch.float32
        and source_alpha.dtype == torch.float32
        and target_alpha.dtype == torch.float32
        and core_first.dtype == torch.float32
        and gate_first.dtype == torch.float32
        and first_weight.dtype == torch.float32
        and source_weight.dtype == torch.float32
        and target_weight.dtype == torch.float32
        and grad_edge_values.ndim == 2
        and core_first.shape == gate_first.shape == values.shape == source_alpha.shape == target_alpha.shape
        and grad_edge_values.shape == values.shape
        and grad_source_out.shape == grad_target_out.shape == source_out.shape == target_out.shape
        and grad_edge_values.shape[-1] == 128
        and first_weight.shape == (256, 384)
        and source_weight.shape == (128, 128)
        and target_weight.shape == (128, 128)
    ):
        return None

    grad_core, grad_gate = _p53b_tail_backward(grad_edge_values, edge_cache["tail"])
    grad_core_second_in = _p53b_linear_backward(grad_core, edge_cache["core_second"])
    grad_gate_second_in = _p53b_linear_backward(grad_gate, edge_cache["gate_second"])
    grad_node, grad_edge = matris_op.line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32(
        grad_core_second_in.contiguous(),
        grad_gate_second_in.contiguous(),
        core_first.contiguous(),
        gate_first.contiguous(),
        first_weight.contiguous(),
        grad_source_out.contiguous(),
        grad_target_out.contiguous(),
        values.contiguous(),
        source_out.contiguous(),
        target_out.contiguous(),
        source_alpha.contiguous(),
        target_alpha.contiguous(),
        source_weight.contiguous(),
        target_weight.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        int(node_rows),
    )
    return grad_node, grad_edge


def _p101_attention_layer_forward(
    layer: nn.Module,
    node_feat: Tensor,
    edge_feat: Tensor,
    graph: Dict,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    source_index = graph["source_index"]
    target_index = graph["target_index"]
    edge_x, gather_cache = _p101_line_gather_cat_forward(node_feat, edge_feat, source_index, target_index)
    edge_values, edge_cache = _p53b_apply_update(layer.edge_nonlinear_update, edge_x)
    source_logits, source_linear_cache = _p53b_linear_forward(edge_feat, layer.source_weight_linear)
    target_logits, target_linear_cache = _p53b_linear_forward(edge_feat, layer.target_weight_linear)
    node_x, attn_cache = _p101_attention_node_input_forward(
        source_logits,
        target_logits,
        edge_values,
        node_feat,
        source_index,
        target_index,
        graph.get("target_segment_offsets"),
    )
    node_update, node_cache = _p53b_apply_update(layer.node_nonlinear_update, node_x)
    node_out = node_update + layer.node_res_weight.float() * node_feat
    edge_out = edge_values + layer.edge_res_weight.float() * edge_feat
    return node_out, edge_out, {
        "source_index": source_index,
        "target_index": target_index,
        "target_segment_offsets": graph.get("target_segment_offsets"),
        "edge_rows": int(edge_feat.shape[0]),
        "node_rows": int(node_feat.shape[0]),
        "gather": gather_cache,
        "edge_cache": edge_cache,
        "node_cache": node_cache,
        "source_linear": source_linear_cache,
        "target_linear": target_linear_cache,
        "attn": attn_cache,
        "node_res": layer.node_res_weight.float(),
        "edge_res": layer.edge_res_weight.float(),
    }


def _p101_attention_layer_backward(
    grad_node_out: Tensor,
    grad_edge_out: Tensor,
    cache: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    if cache["node_cache"].get("kind") == "p116_fused_gated_residual":
        node_bwd = _p116_gated_second_tail_residual_backward_or_none(
            grad_node_out.float(),
            cache["node_cache"],
        )
    else:
        node_bwd = None
    if node_bwd is None:
        grad_node = grad_node_out.float() * cache["node_res"]
        grad_node_x = _p53b_apply_update_backward(grad_node_out, cache["node_cache"])
    else:
        grad_node_x, grad_node = node_bwd
    grad_edge = grad_edge_out.float() * cache["edge_res"]

    p107_node_input = _p107_attention_node_input_values_backward_or_none(
        grad_node_x,
        cache["attn"],
        grad_edge_out.float(),
    )
    if p107_node_input is None:
        fused_node_input = _p106_attention_node_input_backward_edge_direct_or_none(
            grad_node_x,
            cache["attn"],
            grad_edge_out.float(),
        )
        if fused_node_input is None:
            grad_node_direct, grad_source_logits, grad_target_logits, grad_edge_values = _p101_attention_node_input_backward(
                grad_node_x,
                cache["attn"],
            )
            grad_edge_values = grad_edge_values + grad_edge_out.float()
        else:
            grad_node_direct, grad_source_logits, grad_target_logits, grad_edge_values = fused_node_input
        p107_source_out_grad = None
        p107_target_out_grad = None
        p107_attn_cache = None
    else:
        (
            grad_node_direct,
            p107_source_out_grad,
            p107_target_out_grad,
            grad_edge_values,
            p107_attn_cache,
        ) = p107_node_input
        grad_source_logits = None
        grad_target_logits = None
    grad_node = grad_node + grad_node_direct

    fused_edge_alpha = None
    if p107_attn_cache is not None and p107_source_out_grad is not None and p107_target_out_grad is not None:
        fused_edge_alpha = _p107_edge_update_alpha_attention_cuda_backward_or_none(
            grad_edge_values,
            cache["edge_cache"],
            p107_source_out_grad,
            p107_target_out_grad,
            p107_attn_cache,
            cache["source_linear"],
            cache["target_linear"],
            cache["source_index"],
            cache["target_index"],
            cache["node_rows"],
        )
    if fused_edge_alpha is None and grad_source_logits is not None and grad_target_logits is not None:
        fused_edge_alpha = _p108_edge_update_alpha_dense_gemm_op_backward_or_none(
            grad_edge_values,
            cache["edge_cache"],
            grad_source_logits,
            grad_target_logits,
            cache["source_linear"],
            cache["target_linear"],
            cache["source_index"],
            cache["target_index"],
            cache["node_rows"],
        )
    if fused_edge_alpha is None and grad_source_logits is not None and grad_target_logits is not None:
        fused_edge_alpha = _p108_edge_update_alpha_dense_gemm_scatter_backward_or_none(
            grad_edge_values,
            cache["edge_cache"],
            grad_source_logits,
            grad_target_logits,
            cache["source_linear"],
            cache["target_linear"],
            cache["source_index"],
            cache["target_index"],
            cache["node_rows"],
        )
    if fused_edge_alpha is None and grad_source_logits is not None and grad_target_logits is not None:
        fused_edge_alpha = _p108_edge_update_alpha_tiled_cuda_backward_or_none(
            grad_edge_values,
            cache["edge_cache"],
            grad_source_logits,
            grad_target_logits,
            cache["source_linear"],
            cache["target_linear"],
            cache["source_index"],
            cache["target_index"],
            cache["node_rows"],
        )
    if fused_edge_alpha is None and grad_source_logits is not None and grad_target_logits is not None:
        fused_edge_alpha = _p108_edge_update_alpha_target_reduce_cuda_backward_or_none(
            grad_edge_values,
            cache["edge_cache"],
            grad_source_logits,
            grad_target_logits,
            cache["source_linear"],
            cache["target_linear"],
            cache["source_index"],
            cache["target_index"],
            cache.get("target_segment_offsets"),
            cache["node_rows"],
        )
    if fused_edge_alpha is None and grad_source_logits is not None and grad_target_logits is not None:
        fused_edge_alpha = _p105_edge_update_alpha_cuda_backward_or_none(
            grad_edge_values,
            cache["edge_cache"],
            grad_source_logits,
            grad_target_logits,
            cache["source_linear"],
            cache["target_linear"],
            cache["source_index"],
            cache["target_index"],
            cache["node_rows"],
        )
    if fused_edge_alpha is None:
        if grad_source_logits is None or grad_target_logits is None:
            grad_source_logits, grad_target_logits, grad_edge_values_without_direct = _p53b_attention_backward(
                p107_source_out_grad,
                p107_target_out_grad,
                p107_attn_cache,
            )
            grad_edge_values = grad_edge_values_without_direct + grad_edge_out.float()
        grad_edge_feat = _p53b_linear_backward(grad_source_logits, cache["source_linear"])
        grad_edge_feat = grad_edge_feat + _p53b_linear_backward(grad_target_logits, cache["target_linear"])
        grad_edge_x = _p53b_apply_update_backward(grad_edge_values, cache["edge_cache"])
        grad_node_from_gather, grad_edge_from_gather = _p101_line_gather_cat_backward(grad_edge_x, cache["gather"])
        grad_node = grad_node + grad_node_from_gather
        grad_edge = grad_edge + grad_edge_feat + grad_edge_from_gather
    else:
        grad_node_from_fused, grad_edge_from_fused = fused_edge_alpha
        grad_node = grad_node + grad_node_from_fused
        grad_edge = grad_edge + grad_edge_from_fused
    return grad_node, grad_edge


class _P99BAttnLineModuleManualVJP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        layer: nn.Module,
        graph: Dict,
        node_feat: Tensor,
        edge_feat: Tensor,
    ):
        with _p99b_record_function("P99B.attn_line_module.forward"):
            node_out, edge_out, cache = _p53b_attention_layer_forward(
                layer,
                node_feat.detach(),
                edge_feat.detach(),
                graph,
                None,
            )
        ctx.cache = cache
        return node_out, edge_out

    @staticmethod
    def backward(ctx, grad_node_out: Tensor | None, grad_edge_out: Tensor | None):
        cache = ctx.cache
        if grad_node_out is None and grad_edge_out is None:
            return None, None, None, None
        if grad_node_out is None:
            grad_node_out = grad_edge_out.new_zeros((cache["node_rows"], grad_edge_out.shape[-1]))
        if grad_edge_out is None:
            grad_edge_out = grad_node_out.new_zeros((cache["edge_rows"], grad_node_out.shape[-1]))
        with _p99b_record_function("P99B.attn_line_module.backward"):
            grad_node, grad_edge = _p53b_attention_layer_backward(
                grad_node_out,
                grad_edge_out,
                cache,
            )
        return None, None, grad_node, grad_edge


class _P101A3LiteAttnLineManualVJP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        layer: nn.Module,
        graph: Dict,
        node_feat: Tensor,
        edge_feat: Tensor,
    ):
        with _p99b_record_function("P101.a3_lite_attn_line.forward"):
            node_out, edge_out, cache = _p101_attention_layer_forward(
                layer,
                node_feat.detach(),
                edge_feat.detach(),
                graph,
            )
        ctx.cache = cache
        return node_out, edge_out

    @staticmethod
    def backward(ctx, grad_node_out: Tensor | None, grad_edge_out: Tensor | None):
        cache = ctx.cache
        if grad_node_out is None and grad_edge_out is None:
            return None, None, None, None
        if grad_node_out is None:
            grad_node_out = grad_edge_out.new_zeros((cache["node_rows"], grad_edge_out.shape[-1]))
        if grad_edge_out is None:
            grad_edge_out = grad_node_out.new_zeros((cache["edge_rows"], grad_node_out.shape[-1]))
        with _p99b_record_function("P101.a3_lite_attn_line.backward"):
            grad_node, grad_edge = _p101_attention_layer_backward(
                grad_node_out,
                grad_edge_out,
                cache,
            )
        return None, None, grad_node, grad_edge


def _p53b_refinement_forward(
    layer: nn.Module,
    node_feat: Tensor,
    edge_feat: Tensor,
    smooth_weight: Tensor,
    graph: Dict,
    directed2undirected: Tensor | None,
    atom_feat: Tensor | None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    is_atom = layer.graph_type == "atom graph"
    source_index = graph["source_index"]
    target_index = graph["target_index"]
    if is_atom:
        edge_feat_0 = edge_feat.index_select(0, directed2undirected)
        source_node_feat = node_feat.index_select(0, source_index)
        target_node_feat = node_feat.index_select(0, target_index)
        smooth_0 = smooth_weight.index_select(0, directed2undirected)
        smooth_projected, smooth_cache = _p53b_linear_forward(smooth_0, layer.learnable_envelope)
        refine_x = torch.cat([edge_feat_0, target_node_feat, source_node_feat], dim=1)
        smooth_cache_extra = {
            "kind": "atom",
            "linear": smooth_cache,
            "directed2undirected": directed2undirected,
            "smooth_rows": int(smooth_weight.shape[0]),
        }
    else:
        base_envelope, smooth_cache = _p53b_linear_forward(smooth_weight, layer.learnable_envelope)
        base_i = base_envelope.index_select(0, source_index)
        base_j = base_envelope.index_select(0, target_index)
        smooth_projected = base_i * base_j
        atom_index = graph["atom_list"]
        source_node_feat = node_feat.index_select(0, source_index)
        target_node_feat = node_feat.index_select(0, target_index)
        three_body_atom_feat = atom_feat.index_select(0, atom_index)
        refine_x = torch.cat([edge_feat, three_body_atom_feat, target_node_feat, source_node_feat], dim=1)
        smooth_cache_extra = {
            "kind": "line",
            "linear": smooth_cache,
            "base_envelope": base_envelope,
            "base_i": base_i,
            "base_j": base_j,
            "atom_index": atom_index,
            "smooth_rows": int(smooth_weight.shape[0]),
            "atom_rows": int(atom_feat.shape[0]),
        }

    nonlinear, nonlinear_cache = _p53b_apply_update(layer.edge_nonlinear_update, refine_x)
    smoothed = nonlinear * smooth_projected
    node_agg = _p53b_index_add(smoothed, target_index, int(node_feat.shape[0]))
    input_edge_ffn = smoothed if is_atom and layer.use_smoothed_for_delta_edge else nonlinear
    delta_node, node_ffn_cache = _p53b_mlp_forward(layer.node_FFN, node_agg)
    delta_edge_directed, edge_ffn_cache = _p53b_mlp_forward(layer.edge_FFN, input_edge_ffn)
    if is_atom:
        delta_edge, directed_cache = _p53b_directed_average_forward(
            delta_edge_directed,
            directed2undirected,
            int(edge_feat.shape[0]),
        )
    else:
        delta_edge = delta_edge_directed
        directed_cache = None
    node_out = delta_node + layer.node_res_weight.float() * node_feat
    edge_out = delta_edge + layer.edge_res_weight.float() * edge_feat
    return node_out, edge_out, {
        "is_atom": is_atom,
        "source_index": source_index,
        "target_index": target_index,
        "directed2undirected": directed2undirected,
        "edge_rows": int(edge_feat.shape[0]),
        "node_rows": int(node_feat.shape[0]),
        "nonlinear": nonlinear,
        "smooth": smooth_projected,
        "nonlinear_cache": nonlinear_cache,
        "node_ffn": node_ffn_cache,
        "edge_ffn": edge_ffn_cache,
        "smooth_extra": smooth_cache_extra,
        "directed": directed_cache,
        "use_smoothed_for_delta_edge": bool(layer.use_smoothed_for_delta_edge),
        "node_res": layer.node_res_weight.float(),
        "edge_res": layer.edge_res_weight.float(),
        "profile_prefix": getattr(layer, "profile_prefix", ""),
    }


def _p53b_refinement_backward(
    grad_node_out: Tensor,
    grad_edge_out: Tensor,
    cache: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    grad_node = grad_node_out.float() * cache["node_res"]
    grad_edge = grad_edge_out.float() * cache["edge_res"]
    grad_delta_edge = grad_edge_out.float()
    if cache["is_atom"]:
        grad_delta_edge = _p53b_directed_average_backward(grad_delta_edge, cache["directed"])

    grad_node_agg = _p53b_mlp_backward(grad_node_out, cache["node_ffn"])
    grad_smoothed = grad_node_agg.index_select(0, cache["target_index"])
    grad_nonlinear = grad_smoothed * cache["smooth"]
    grad_smooth_projected = grad_smoothed * cache["nonlinear"]
    grad_edge_ffn_in = _p53b_mlp_backward(grad_delta_edge, cache["edge_ffn"])
    if cache["is_atom"] and cache["use_smoothed_for_delta_edge"]:
        grad_smoothed = grad_smoothed + grad_edge_ffn_in
        grad_nonlinear = grad_nonlinear + grad_edge_ffn_in * cache["smooth"]
        grad_smooth_projected = grad_smooth_projected + grad_edge_ffn_in * cache["nonlinear"]
    else:
        grad_nonlinear = grad_nonlinear + grad_edge_ffn_in

    grad_refine_x = _p53b_apply_update_backward(grad_nonlinear, cache["nonlinear_cache"])
    dim = grad_node_out.shape[-1]
    smooth_extra = cache["smooth_extra"]
    if cache["is_atom"]:
        grad_edge_feat_0 = grad_refine_x[:, :dim]
        grad_target_node = grad_refine_x[:, dim : 2 * dim]
        grad_source_node = grad_refine_x[:, 2 * dim :]
        grad_node = grad_node + _p53b_gather_backward(grad_target_node, cache["target_index"], cache["node_rows"])
        grad_node = grad_node + _p53b_gather_backward(grad_source_node, cache["source_index"], cache["node_rows"])
        grad_edge = grad_edge + _p53b_gather_backward(
            grad_edge_feat_0,
            cache["directed2undirected"],
            cache["edge_rows"],
        )
        grad_smooth_0 = _p53b_linear_backward(grad_smooth_projected, smooth_extra["linear"])
        grad_smooth = _p53b_gather_backward(
            grad_smooth_0,
            smooth_extra["directed2undirected"],
            smooth_extra["smooth_rows"],
        )
        return grad_node, grad_edge, None, grad_smooth

    grad_edge = grad_edge + grad_refine_x[:, :dim]
    grad_atom = _p53b_gather_backward(
        grad_refine_x[:, dim : 2 * dim],
        smooth_extra["atom_index"],
        smooth_extra["atom_rows"],
    )
    grad_target_node = grad_refine_x[:, 2 * dim : 3 * dim]
    grad_source_node = grad_refine_x[:, 3 * dim :]
    grad_node = grad_node + _p53b_gather_backward(grad_target_node, cache["target_index"], cache["node_rows"])
    grad_node = grad_node + _p53b_gather_backward(grad_source_node, cache["source_index"], cache["node_rows"])
    grad_base_i = grad_smooth_projected * smooth_extra["base_j"]
    grad_base_j = grad_smooth_projected * smooth_extra["base_i"]
    grad_base = _p53b_gather_backward(grad_base_i, cache["source_index"], smooth_extra["smooth_rows"])
    grad_base = grad_base + _p53b_gather_backward(grad_base_j, cache["target_index"], smooth_extra["smooth_rows"])
    grad_smooth = _p53b_linear_backward(grad_base, smooth_extra["linear"])
    return grad_node, grad_edge, grad_atom, grad_smooth


def _p53b_full_block_forward(
    block: "Interaction_Block",
    batch_graph: Dict,
    node_feat: Tensor,
    edge_feat: Tensor,
    threebody_feat: Tensor | None,
    smooth_atom: Tensor | None,
    smooth_line: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None, dict[str, Any]]:
    atom_graph = batch_graph["atom_graph_dict"]
    line_graph = batch_graph["line_graph_dict"]
    directed2undirected = batch_graph["directed2undirected"]
    caches: dict[str, Any] = {}
    attn_edge_feat = edge_feat
    attn_threebody_feat = threebody_feat

    if isinstance(threebody_feat, Tensor):
        attn_edge_feat, attn_threebody_feat, caches["attn_line"] = _p53b_attention_layer_forward(
            block.attn_block_line_graph,
            edge_feat,
            threebody_feat,
            line_graph,
            None,
        )

    attn_node_feat, attn_edge_feat, caches["attn_atom"] = _p53b_attention_layer_forward(
        block.attn_block_atom_graph,
        node_feat,
        attn_edge_feat,
        atom_graph,
        directed2undirected,
    )

    update_threebody_feat = attn_threebody_feat
    if isinstance(threebody_feat, Tensor):
        update_edge_feat, update_threebody_feat, caches["refine_line"] = _p53b_refinement_forward(
            block.refine_block_line_graph,
            attn_edge_feat,
            attn_threebody_feat,
            smooth_line,
            line_graph,
            None,
            attn_node_feat,
        )
    else:
        update_edge_feat = attn_edge_feat

    update_node_feat, update_edge_feat, caches["refine_atom"] = _p53b_refinement_forward(
        block.refine_block_atom_graph,
        attn_node_feat,
        update_edge_feat,
        smooth_atom,
        atom_graph,
        directed2undirected,
        None,
    )
    return update_node_feat, update_edge_feat, update_threebody_feat, caches


class _P53BFullBlockManualVJP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        block: "Interaction_Block",
        batch_graph: Dict,
        node_feat: Tensor,
        edge_feat: Tensor,
        threebody_feat: Tensor | None,
        smooth_atom: Tensor | None,
        smooth_line: Tensor | None,
    ):
        ctx.has_threebody = isinstance(threebody_feat, Tensor)
        ctx.has_smooth_atom = isinstance(smooth_atom, Tensor)
        ctx.has_smooth_line = isinstance(smooth_line, Tensor)
        out_node, out_edge, out_threebody, caches = _p53b_full_block_forward(
            block,
            batch_graph,
            node_feat.detach(),
            edge_feat.detach(),
            threebody_feat.detach() if isinstance(threebody_feat, Tensor) else None,
            smooth_atom.detach() if isinstance(smooth_atom, Tensor) else None,
            smooth_line.detach() if isinstance(smooth_line, Tensor) else None,
        )
        ctx.caches = caches
        return out_node, out_edge, out_threebody

    @staticmethod
    def backward(ctx, grad_node_out: Tensor | None, grad_edge_out: Tensor | None, grad_threebody_out: Tensor | None):
        caches = ctx.caches
        if grad_node_out is None:
            raise RuntimeError("P53b full-block VJP requires grad_node_out")
        if grad_edge_out is None:
            raise RuntimeError("P53b full-block VJP requires grad_edge_out")

        grad_attn_node, grad_update_edge, _unused_atom_grad, grad_smooth_atom = _p53b_refinement_backward(
            grad_node_out,
            grad_edge_out,
            caches["refine_atom"],
        )

        grad_attn_threebody = grad_threebody_out
        grad_smooth_line = None
        if "refine_line" in caches:
            grad_refine_line_node, grad_attn_threebody_from_refine, grad_attn_node_from_refine, grad_smooth_line = (
                _p53b_refinement_backward(
                    grad_update_edge,
                    grad_attn_threebody if grad_attn_threebody is not None else grad_update_edge.new_zeros(caches["refine_line"]["edge_rows"], grad_update_edge.shape[-1]),
                    caches["refine_line"],
                )
            )
            grad_attn_edge = grad_refine_line_node
            grad_attn_threebody = grad_attn_threebody_from_refine
            grad_attn_node = grad_attn_node + grad_attn_node_from_refine
        else:
            grad_attn_edge = grad_update_edge

        grad_node, grad_attn_edge_from_atom = _p53b_attention_layer_backward(
            grad_attn_node,
            grad_attn_edge,
            caches["attn_atom"],
        )

        grad_edge = grad_attn_edge_from_atom
        grad_threebody = grad_attn_threebody
        if "attn_line" in caches:
            if grad_threebody is None:
                grad_threebody = grad_edge.new_zeros(caches["attn_line"]["edge_rows"], grad_edge.shape[-1])
            grad_edge_from_line, grad_threebody_from_line = _p53b_attention_layer_backward(
                grad_attn_edge_from_atom,
                grad_threebody,
                caches["attn_line"],
            )
            grad_edge = grad_edge_from_line
            grad_threebody = grad_threebody_from_line

        return (
            None,
            None,
            grad_node,
            grad_edge,
            grad_threebody if ctx.has_threebody else None,
            grad_smooth_atom if ctx.has_smooth_atom else None,
            grad_smooth_line if ctx.has_smooth_line else None,
        )


class _P58RefineLineSmoothReduce(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        nonlinear: Tensor,
        base_envelope: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "refine_line_smooth_reduce_forward"):
            raise RuntimeError("matris_op.refine_line_smooth_reduce_forward is unavailable")
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        nonlinear = nonlinear.contiguous()
        base_envelope = base_envelope.contiguous()
        out = matris_op.refine_line_smooth_reduce_forward(
            nonlinear,
            base_envelope,
            source_index,
            target_index,
            int(num_nodes),
        )
        ctx.save_for_backward(nonlinear, base_envelope, source_index, target_index)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        nonlinear, base_envelope, source_index, target_index = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "refine_line_smooth_reduce_backward"):
            raise RuntimeError("matris_op.refine_line_smooth_reduce_backward is unavailable")
        grad_nonlinear, grad_base = matris_op.refine_line_smooth_reduce_backward(
            grad_out.contiguous(),
            nonlinear,
            base_envelope,
            source_index,
            target_index,
        )
        return grad_nonlinear, grad_base, None, None, None


class _P58RefineLineSmoothReduceSorted(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        nonlinear: Tensor,
        base_envelope: Tensor,
        source_index: Tensor,
        target_offsets: Tensor,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "refine_line_smooth_reduce_sorted_forward"):
            raise RuntimeError("matris_op.refine_line_smooth_reduce_sorted_forward is unavailable")
        source_index = source_index.contiguous()
        target_offsets = target_offsets.contiguous()
        nonlinear = nonlinear.contiguous()
        base_envelope = base_envelope.contiguous()
        out = matris_op.refine_line_smooth_reduce_sorted_forward(
            nonlinear,
            base_envelope,
            source_index,
            target_offsets,
        )
        ctx.save_for_backward(nonlinear, base_envelope, source_index, target_offsets)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        nonlinear, base_envelope, source_index, target_offsets = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "refine_line_smooth_reduce_sorted_backward"):
            raise RuntimeError("matris_op.refine_line_smooth_reduce_sorted_backward is unavailable")
        grad_nonlinear, grad_base = matris_op.refine_line_smooth_reduce_sorted_backward(
            grad_out.contiguous(),
            nonlinear,
            base_envelope,
            source_index,
            target_offsets,
        )
        return grad_nonlinear, grad_base, None, None


class _P102RefineLineFFNPairManualVJP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        layer: nn.Module,
        node_input: Tensor,
        edge_input: Tensor,
    ) -> tuple[Tensor, Tensor]:
        delta_node, node_cache = _p53b_mlp_forward(layer.node_FFN, node_input.detach())
        delta_edge, edge_cache = _p53b_mlp_forward(layer.edge_FFN, edge_input.detach())
        ctx.node_cache = node_cache
        ctx.edge_cache = edge_cache
        return delta_node, delta_edge

    @staticmethod
    def backward(ctx, grad_delta_node: Tensor | None, grad_delta_edge: Tensor | None):
        grad_node_input = None
        grad_edge_input = None
        if grad_delta_node is not None:
            grad_node_input = _p53b_mlp_backward(grad_delta_node, ctx.node_cache)
        if grad_delta_edge is not None:
            grad_edge_input = _p53b_mlp_backward(grad_delta_edge, ctx.edge_cache)
        return None, grad_node_input, grad_edge_input


class _P103RefineLineEdgeUpdateEdgeFFNManualVJP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        layer: nn.Module,
        graph: Dict,
        node_feat: Tensor,
        edge_feat: Tensor,
        atom_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source_index = graph["source_index"]
        target_index = graph["target_index"]
        atom_index = graph["atom_list"]
        source_node_feat = node_feat.index_select(0, source_index)
        target_node_feat = node_feat.index_select(0, target_index)
        three_body_atom_feat = atom_feat.index_select(0, atom_index)
        refine_x = torch.cat([edge_feat, three_body_atom_feat, target_node_feat, source_node_feat], dim=1)
        nonlinear, nonlinear_cache = _p53b_apply_update(layer.edge_nonlinear_update, refine_x.detach())
        delta_edge, edge_ffn_cache = _p53b_mlp_forward(layer.edge_FFN, nonlinear)
        ctx.cache = {
            "source_index": source_index,
            "target_index": target_index,
            "atom_index": atom_index,
            "node_rows": int(node_feat.shape[0]),
            "edge_rows": int(edge_feat.shape[0]),
            "atom_rows": int(atom_feat.shape[0]),
            "nonlinear": nonlinear_cache,
            "edge_ffn": edge_ffn_cache,
        }
        return nonlinear, delta_edge

    @staticmethod
    def backward(ctx, grad_nonlinear_out: Tensor | None, grad_delta_edge: Tensor | None):
        cache = ctx.cache
        grad_nonlinear = None
        if grad_nonlinear_out is not None:
            grad_nonlinear = grad_nonlinear_out.float()
        if grad_delta_edge is not None:
            grad_edge_ffn_in = _p53b_mlp_backward(grad_delta_edge, cache["edge_ffn"])
            grad_nonlinear = grad_edge_ffn_in if grad_nonlinear is None else grad_nonlinear + grad_edge_ffn_in
        if grad_nonlinear is None:
            return None, None, None, None, None

        grad_refine_x = _p53b_apply_update_backward(grad_nonlinear, cache["nonlinear"])
        dim = grad_refine_x.shape[-1] // 4
        grad_edge = grad_refine_x[:, :dim]
        grad_atom = _p53b_gather_backward(
            grad_refine_x[:, dim : 2 * dim],
            cache["atom_index"],
            cache["atom_rows"],
        )
        grad_target_node = grad_refine_x[:, 2 * dim : 3 * dim]
        grad_source_node = grad_refine_x[:, 3 * dim :]
        grad_node = _p53b_gather_backward(grad_target_node, cache["target_index"], cache["node_rows"])
        grad_node = grad_node + _p53b_gather_backward(grad_source_node, cache["source_index"], cache["node_rows"])
        return None, None, grad_node, grad_edge, grad_atom


class _P104RefineLineBlockManualVJP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        layer: nn.Module,
        graph: Dict,
        node_feat: Tensor,
        edge_feat: Tensor,
        smooth_weight: Tensor,
        atom_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        with _p99b_record_function("P104.refine_line_block.forward"):
            node_out, edge_out, cache = _p53b_refinement_forward(
                layer,
                node_feat.detach(),
                edge_feat.detach(),
                smooth_weight.detach(),
                graph,
                None,
                atom_feat.detach(),
            )
        ctx.cache = cache
        return node_out, edge_out

    @staticmethod
    def backward(ctx, grad_node_out: Tensor | None, grad_edge_out: Tensor | None):
        cache = ctx.cache
        if grad_node_out is None and grad_edge_out is None:
            return None, None, None, None, None, None
        if grad_node_out is None:
            grad_node_out = grad_edge_out.new_zeros((cache["node_rows"], grad_edge_out.shape[-1]))
        if grad_edge_out is None:
            grad_edge_out = grad_node_out.new_zeros((cache["edge_rows"], grad_node_out.shape[-1]))
        with _p99b_record_function("P104.refine_line_block.backward"):
            grad_node, grad_edge, grad_atom, grad_smooth = _p53b_refinement_backward(
                grad_node_out,
                grad_edge_out,
                cache,
            )
        return None, None, grad_node, grad_edge, grad_smooth, grad_atom


def _p37_node_proxy(batch_graph: Dict, node_feat: Tensor, edge_feat: Tensor, profile_prefix: str) -> Tensor:
    proxy = node_feat
    if os.environ.get("MATRIS_P37_INTERACTION_STE_EDGE_PROXY", "1") != "1":
        return proxy
    if not (edge_feat.is_cuda and edge_feat.ndim == 2 and edge_feat.shape[-1] == node_feat.shape[-1]):
        return proxy
    edge_scale = _p37_env_float("MATRIS_P37_INTERACTION_STE_EDGE_SCALE", 1.0)
    if os.environ.get("MATRIS_P37_INTERACTION_STE_EDGE_PROXY_MODE", "aggregate") == "mean":
        return proxy + edge_feat.mean(dim=0, keepdim=True) * edge_scale
    atom_graph = batch_graph.get("atom_graph_dict", {})
    directed2undirected = batch_graph.get("directed2undirected")
    target_index = atom_graph.get("target_index")
    target_bincount = atom_graph.get("target_bincount")
    if directed2undirected is None or target_index is None:
        return proxy
    edge_directed = torch.index_select(edge_feat, 0, directed2undirected)
    edge_sum = aggregate(
        data=edge_directed,
        segment=target_index,
        bin_count=target_bincount,
        average=False,
        num_segment=len(node_feat),
        profile_name=f"{profile_prefix}.p37_node_proxy",
    )
    return proxy + edge_sum * edge_scale


class Graph_Attention_Layer(nn.Module):
    
    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        hidden_dim: int = 128,
        use_bias: bool = False,
        dropout: float = 0.0,
        mlp_type: str = "GateMLP", # MLP, GateMLP
        activation_type: str = "silu",
        norm_type: str = "layer",
        use_fp16: bool = False, 
    ):
        super().__init__()
        
        self.source_weight_linear = nn.Linear(
            in_features = edge_feat_dim, out_features = edge_feat_dim, bias = False
        )
        self.target_weight_linear = nn.Linear(
            in_features = edge_feat_dim, out_features = edge_feat_dim, bias = False
        )
        if mlp_type.lower() == "mlp":
            self.node_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=edge_feat_dim * 2 + node_feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=node_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                ),
                get_normalization(name=norm_type, dim=node_feat_dim) 
            )
            self.edge_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=node_feat_dim * 2 + edge_feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=edge_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                    use_fp16=use_fp16,
                ),
                get_normalization(name=norm_type, dim=edge_feat_dim) 
            )
        elif mlp_type.lower() == "gatemlp":
            self.node_nonlinear_update = GatedMLP(
                input_dim=edge_feat_dim * 2 + node_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=node_feat_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation_type,
            )
            self.edge_nonlinear_update = GatedMLP(
                input_dim=node_feat_dim * 2 + edge_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=edge_feat_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation_type,
                use_fp16=use_fp16,
            )
        else:
            raise NotImplementedError

        self.node_res_weight = torch.nn.Parameter(torch.ones(1, node_feat_dim), requires_grad=True)
        self.edge_res_weight = torch.nn.Parameter(torch.ones(1, edge_feat_dim), requires_grad=True)
    
    def forward(self, 
        node_feat: Tensor, 
        edge_feat: Tensor, 
        graph: Dict, # atom graph or line graph
        directed2undirected: Tensor = None,
    ): 
        source_node_index = graph['source_index']
        target_node_index = graph['target_index']
        profile_prefix = getattr(self, "profile_prefix", self.__class__.__name__)
        detail_profile = _attn_line_detail_profile_enabled(profile_prefix)
        record_function_profile = _attn_line_record_function_enabled(profile_prefix)
        self.last_profile = {}
        if (
            _use_p101_a3_lite_attn_line_vjp(profile_prefix)
            and directed2undirected is None
            and self.training is False
            and torch.is_grad_enabled()
            and node_feat.is_cuda
            and edge_feat.is_cuda
            and node_feat.ndim == 2
            and edge_feat.ndim == 2
            and node_feat.shape[-1] == 128
            and edge_feat.shape[-1] == 128
        ):
            return _P101A3LiteAttnLineManualVJP.apply(
                self,
                graph,
                node_feat,
                edge_feat,
            )
        if (
            _use_p99b_attn_line_module_vjp(profile_prefix)
            and directed2undirected is None
            and self.training is False
            and torch.is_grad_enabled()
            and node_feat.is_cuda
            and edge_feat.is_cuda
            and node_feat.ndim == 2
            and edge_feat.ndim == 2
            and node_feat.shape[-1] == 128
            and edge_feat.shape[-1] == 128
        ):
            return _P99BAttnLineModuleManualVJP.apply(
                self,
                graph,
                node_feat,
                edge_feat,
            )

        def timed_detail(stage_name: str, sync_values: tuple, fn):
            if not detail_profile and not record_function_profile:
                return fn()
            label = f"{profile_prefix}.{stage_name}"
            with torch.profiler.record_function(label):
                if detail_profile:
                    _sync_if_cuda_tensor(*sync_values)
                    start = time.perf_counter()
                    result = fn()
                    _sync_if_cuda_tensor(result, *sync_values)
                    self.last_profile[f"{label}_ms"] = (time.perf_counter() - start) * 1000.0
                else:
                    result = fn()
            return result

        def fused_node_update_residual_or_none(fusion_node_feat: Tensor) -> Tensor | None:
            if os.environ.get("MATRIS_P116_GATED_TAIL_SECOND_RESIDUAL_MACRO", "0") != "1":
                return None
            if not hasattr(self.node_nonlinear_update, "forward_with_residual_after_first_projection"):
                return None
            return self.node_nonlinear_update.forward_with_residual_after_first_projection(
                fusion_node_feat,
                node_feat,
                self.node_res_weight,
            )

        def node_update_residual_or_none(fusion_node_feat: Tensor) -> Tensor | None:
            if os.environ.get("MATRIS_P116_GATED_TAIL_SECOND_RESIDUAL_MACRO", "0") != "1":
                return None
            return timed_detail(
                "node_update_residual_fused",
                (fusion_node_feat, node_feat),
                lambda: fused_node_update_residual_or_none(fusion_node_feat),
            )

        broad_aggressive_mode = aggressive_broad_bwd_mode()
        aggressive_mode = (
            aggressive_line_attn_eval_mode()
            if directed2undirected is None and is_aggressive_line_attn_target(profile_prefix)
            else ""
        )
        line_edge_fused = None
        directed_edge_fused = None
        line_attention_macro = None
        if (
            directed2undirected is None
            and profile_prefix.endswith(".attn_line")
            and hasattr(self.edge_nonlinear_update, "forward_line_attention_macro")
        ):
            line_attention_macro = self.edge_nonlinear_update.forward_line_attention_macro(
                node_feat,
                edge_feat,
                source_node_index,
                target_node_index,
                self.source_weight_linear,
                self.target_weight_linear,
                len(node_feat),
            )
        if line_attention_macro is not None:
            attn_source_feat, attn_target_feat, attn_edge_feat = line_attention_macro
            fusion_node_feat = timed_detail(
                "node_update_input_concat",
                (node_feat, attn_target_feat, attn_source_feat),
                lambda: torch.cat([node_feat, attn_target_feat, attn_source_feat], dim=1),
            )
            fused_node_residual = node_update_residual_or_none(fusion_node_feat)
            if fused_node_residual is None:
                attn_node_feat = timed_detail(
                    "node_update",
                    (fusion_node_feat,),
                    lambda: self.node_nonlinear_update(fusion_node_feat),
                )
                attn_node_feat = timed_detail(
                    "residual",
                    (attn_node_feat, node_feat, attn_edge_feat, edge_feat),
                    lambda: attn_node_feat + self.node_res_weight * node_feat,
                )
            else:
                attn_node_feat = fused_node_residual
            attn_edge_feat = attn_edge_feat + self.edge_res_weight * edge_feat
            return attn_node_feat, attn_edge_feat
        if (
            directed2undirected is None
            and (
                (
                    (_use_line_attn_edge_bwd_fusion() or aggressive_mode == "no_param_grad")
                    and profile_prefix in ("interaction_block.8.attn_line", "interaction_block.9.attn_line")
                )
                or (use_p35_all_line_attn_edge_first_dataflow() and profile_prefix.endswith(".attn_line"))
            )
            and hasattr(self.edge_nonlinear_update, "forward_line_edge")
        ):
            line_edge_fused = self.edge_nonlinear_update.forward_line_edge(
                node_feat,
                edge_feat,
                source_node_index,
                target_node_index,
            )
        if (
            directed2undirected is not None
            and use_p35_atom_edge_first_dataflow()
            and profile_prefix.endswith(".attn_atom")
            and hasattr(self.edge_nonlinear_update, "forward_directed_edge")
        ):
            directed_edge_fused = self.edge_nonlinear_update.forward_directed_edge(
                node_feat,
                edge_feat,
                directed2undirected,
                source_node_index,
                target_node_index,
            )
        if line_edge_fused is not None:
            edge_feat_0 = edge_feat
            attn_edge_feat = line_edge_fused
        elif directed_edge_fused is not None:
            edge_feat_0 = torch.index_select(edge_feat, 0, directed2undirected)
            attn_edge_feat = directed_edge_fused
        elif aggressive_mode == "bypass_edge_update":
            edge_feat_0 = edge_feat
            attn_edge_feat = edge_feat_0
        else:
            # gather
            def gather_concat():
                if directed2undirected is not None:
                    source_node_feat = torch.index_select(node_feat, 0, source_node_index)
                    target_node_feat = torch.index_select(node_feat, 0, target_node_index)
                    # Atom Graph Update
                    edge_feat_0_local = torch.index_select(edge_feat, 0, directed2undirected) # [edge, dim] -> [2*edge, dim]
                    #======= combine feature =======
                    attn_edge_input = torch.cat([edge_feat_0_local, target_node_feat, source_node_feat], dim=1)
                    return edge_feat_0_local, attn_edge_input
                else:
                    # Line Graph Update
                    edge_feat_0_local = edge_feat
                    p79_attn_edge_input = p79_attn_line_gather_cat_or_none(
                        node_feat,
                        edge_feat_0_local,
                        source_node_index,
                        target_node_index,
                        profile_prefix,
                    )
                    if p79_attn_edge_input is not None:
                        return edge_feat_0_local, p79_attn_edge_input
                    source_node_feat = torch.index_select(node_feat, 0, source_node_index)
                    target_node_feat = torch.index_select(node_feat, 0, target_node_index)

                #======= combine feature =======
                attn_edge_input = torch.cat([edge_feat_0_local, target_node_feat, source_node_feat], dim=1)
                return edge_feat_0_local, attn_edge_input

            edge_feat_0, attn_edge_feat = timed_detail(
                "gather_concat",
                (node_feat, edge_feat, source_node_index, target_node_index),
                gather_concat,
            )
            attn_edge_feat = timed_detail(
                "edge_update",
                (attn_edge_feat,),
                lambda: self.edge_nonlinear_update(attn_edge_feat),
            )

        # ======= update atom feature ======= 
        def alpha_projection():
            if (
                use_p29_fused_alpha_input_grad_only()
                and self.training is False
                and torch.is_grad_enabled()
                and edge_feat_0.is_cuda
                and isinstance(self.source_weight_linear, nn.Linear)
                and isinstance(self.target_weight_linear, nn.Linear)
                and self.source_weight_linear.weight.shape[1] == self.target_weight_linear.weight.shape[1]
            ):
                return dual_linear_input_grad_only(
                    edge_feat_0,
                    self.source_weight_linear,
                    self.target_weight_linear,
                )
            if aggressive_mode == "no_param_grad" or broad_aggressive_mode in (
                "input_grad_only",
                "w8a8_saved_pre_all",
                "bypass_mlp",
            ):
                return (
                    linear_input_grad_only(edge_feat_0, self.source_weight_linear),
                    linear_input_grad_only(edge_feat_0, self.target_weight_linear),
                )
            if aggressive_mode == "bypass_edge_update":
                with torch.no_grad():
                    source_alpha_local = self.source_weight_linear(edge_feat_0.detach())
                    target_alpha_local = self.target_weight_linear(edge_feat_0.detach())
                return source_alpha_local.detach(), target_alpha_local.detach()
            return (
                self.source_weight_linear(edge_feat_0),
                self.target_weight_linear(edge_feat_0),
            )

        source_alpha_0, target_alpha_0 = timed_detail(
            "alpha_projection",
            (edge_feat_0,),
            alpha_projection,
        )

        fusion_node_feat = None
        if directed2undirected is None and profile_prefix.endswith(".attn_line"):
            fusion_node_feat = timed_detail(
                "attention_reduce_node_input",
                (source_alpha_0, target_alpha_0, attn_edge_feat, node_feat),
                lambda: fused_line_attention_node_input_or_none(
                    source_alpha_0,
                    target_alpha_0,
                    attn_edge_feat,
                    source_node_index,
                    target_node_index,
                    graph.get("target_segment_offsets"),
                    len(node_feat),
                    node_feat,
                    enable_hint=True,
                ),
            )
        if fusion_node_feat is not None:
            fused_node_residual = node_update_residual_or_none(fusion_node_feat)

            if fused_node_residual is None:
                attn_node_feat = timed_detail(
                    "node_update",
                    (fusion_node_feat,),
                    lambda: self.node_nonlinear_update(fusion_node_feat),
                )

                def residual_update():
                    return (
                        attn_node_feat + self.node_res_weight * node_feat,
                        attn_edge_feat + self.edge_res_weight * edge_feat,
                    )

                attn_node_feat, attn_edge_feat = timed_detail(
                    "residual",
                    (attn_node_feat, node_feat, attn_edge_feat, edge_feat),
                    residual_update,
                )
            else:
                attn_node_feat = fused_node_residual
                attn_edge_feat = timed_detail(
                    "residual",
                    (attn_node_feat, node_feat, attn_edge_feat, edge_feat),
                    lambda: attn_edge_feat + self.edge_res_weight * edge_feat,
                )
            return attn_node_feat, attn_edge_feat
        
        # Softmax
        num_segment = None #torch.unique(source_node_index).numel()
        def attention_reduce():
            reduce_detail = use_p81_attn_reduce_detail_profile()

            def detail_stage(name: str, fn):
                if reduce_detail and profile_prefix.endswith(".attn_line"):
                    with torch.profiler.record_function(f"{profile_prefix}.attention_reduce.{name}"):
                        return fn()
                return fn()

            def run_fused_line_attention():
                enable_hint = (
                    (directed2undirected is None and profile_prefix.endswith(".attn_line"))
                    or (directed2undirected is not None and profile_prefix.endswith(".attn_atom"))
                )
                target_offsets = graph.get("target_segment_offsets")
                if target_offsets is not None:
                    return fused_line_attention_or_none(
                        source_alpha_0,
                        target_alpha_0,
                        attn_edge_feat,
                        source_node_index,
                        target_node_index,
                        len(node_feat),
                        enable_hint=enable_hint,
                        atom_graph=directed2undirected is not None,
                        target_offsets=target_offsets,
                    )
                return fused_line_attention_or_none(
                    source_alpha_0,
                    target_alpha_0,
                    attn_edge_feat,
                    source_node_index,
                    target_node_index,
                    len(node_feat),
                    enable_hint=enable_hint,
                    atom_graph=directed2undirected is not None,
                )

            fused_line_attention = detail_stage(
                "fused_line_attention_or_none",
                run_fused_line_attention,
            )
            if fused_line_attention is None:
                source_alpha = detail_stage(
                    "source_softmax",
                    lambda: Dimwise_softmax(
                        source_alpha_0,
                        source_node_index,
                        num_segment,
                        profile_name=f"{profile_prefix}.source_softmax",
                    ),
                )
                target_alpha = detail_stage(
                    "target_softmax",
                    lambda: Dimwise_softmax(
                        target_alpha_0,
                        target_node_index,
                        num_segment,
                        profile_name=f"{profile_prefix}.target_softmax",
                        bin_count=graph.get("target_bincount"),
                    ),
                )
                
                source_weight = detail_stage(
                    "source_alpha_mul_value",
                    lambda: source_alpha * attn_edge_feat, # refer to sa_{ij} * e'_{ij} in MatRIS paper
                )
                target_attention_sum = detail_stage(
                    "target_weighted_sum_or_none",
                    lambda: segment_softmax_weighted_sum_sorted_or_none(
                        target_alpha_0,
                        attn_edge_feat,
                        target_node_index,
                        len(node_feat),
                        enable_hint=graph.get("target_bincount") is not None,
                    ),
                )
                if target_attention_sum is None:
                    target_weight = detail_stage(
                        "target_alpha_mul_value",
                        lambda: target_alpha * attn_edge_feat, # refer to ta_{ij} * e'_{ij} in MatRIS paper
                    )
                else:
                    target_weight = None
            else:
                return fused_line_attention
        
            attn_source_feat = detail_stage(
                "source_weight_sum",
                lambda: aggregate(data=source_weight,
                                  segment=source_node_index,
                                  bin_count=graph['source_bincount'],#bincount_source,
                                  average=False,
                                  num_segment=len(node_feat),
                                  profile_name=f"{profile_prefix}.source_weight_sum"),
            )

            if target_attention_sum is None:
                attn_target_feat = detail_stage(
                    "target_weight_sum",
                    lambda: aggregate(data=target_weight,
                                      segment=target_node_index,
                                      bin_count=graph['target_bincount'],#bincount_target,
                                      average=False,
                                      num_segment=len(node_feat),
                                      profile_name=f"{profile_prefix}.target_weight_sum"),
                )
            else:
                attn_target_feat = target_attention_sum
            return attn_source_feat, attn_target_feat

        attn_source_feat, attn_target_feat = timed_detail(
            "attention_reduce",
            (source_alpha_0, target_alpha_0, attn_edge_feat),
            attention_reduce,
        )
        
        if directed2undirected is not None:
            directed_average = directed2undirected_average_or_none(
                attn_edge_feat,
                directed2undirected,
                edge_feat.shape[0],
                enable_hint=True,
            )
            if directed_average is None:
                attn_edge_feat = aggregate(
                    data=attn_edge_feat,
                    segment=directed2undirected,
                    bin_count=None,
                    average=True,
                    num_segment=None,
                    profile_name=f"{profile_prefix}.directed2undirected_edge_average",
                ) #[2*edge, dim] -> [edge, dim]
            else:
                attn_edge_feat = directed_average
        # Compute Attention output
        def node_update_input_concat():
            if directed2undirected is None:
                p80_fusion_node_feat = p80_attn_line_node_cat_or_none(
                    node_feat,
                    attn_target_feat,
                    attn_source_feat,
                    profile_prefix,
                )
                if p80_fusion_node_feat is not None:
                    return p80_fusion_node_feat
            return torch.cat([node_feat, attn_target_feat, attn_source_feat], dim=1)

        fusion_node_feat = timed_detail(
            "node_update_input_concat",
            (node_feat, attn_target_feat, attn_source_feat),
            node_update_input_concat,
        )
        fused_node_residual = node_update_residual_or_none(fusion_node_feat)
        if fused_node_residual is None:
            attn_node_feat = timed_detail(
                "node_update",
                (fusion_node_feat,),
                lambda: self.node_nonlinear_update(fusion_node_feat),
            )

            # Resdual
            def residual_update():
                return (
                    attn_node_feat + self.node_res_weight * node_feat,
                    attn_edge_feat + self.edge_res_weight * edge_feat,
                )

            attn_node_feat, attn_edge_feat = timed_detail(
                "residual",
                (attn_node_feat, node_feat, attn_edge_feat, edge_feat),
                residual_update,
            )
        else:
            attn_node_feat = fused_node_residual
            attn_edge_feat = timed_detail(
                "residual",
                (attn_node_feat, node_feat, attn_edge_feat, edge_feat),
                lambda: attn_edge_feat + self.edge_res_weight * edge_feat,
            )

        return attn_node_feat, attn_edge_feat


class Refinement(nn.Module):
    
    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        hidden_dim: int = 128,
        num_basis: int = 7,
        dropout: float = 0.0,
        mlp_type: str = "GateMLP",    
        activation_type: str = "silu",
        norm_type: str = "layer",
        use_bias: bool = False,
        graph_type: Literal["atom graph", "line graph"] = "atom graph",
        atom_feat_dim: int = 128,
        use_smoothed_for_delta_edge: bool = False,
        use_fp16: bool = False, 
    ):
        super().__init__()
        self.graph_type = graph_type
        self.use_smoothed_for_delta_edge = use_smoothed_for_delta_edge
        
        if graph_type == "atom graph":
            input_dim = 2 * node_feat_dim + edge_feat_dim
        else:
            input_dim = atom_feat_dim + 2 * node_feat_dim + edge_feat_dim 
        if mlp_type.lower() == "mlp":
            self.edge_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=edge_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                    use_fp16=use_fp16,
                ),
                get_normalization(name=norm_type, dim=edge_feat_dim)
            )
        elif mlp_type.lower() == "gatemlp":
            self.edge_nonlinear_update = GatedMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=edge_feat_dim,
                dropout=dropout,
                norm_type=norm_type,
                activation=activation_type,
                use_fp16=use_fp16,
            )
        else:
            raise NotImplementedError
        
        self.node_FFN = MLP(
            input_dim=edge_feat_dim,
            hidden_dim=node_feat_dim,
            output_dim=node_feat_dim,
            bias=use_bias,
        )
        self.edge_FFN = MLP(
            input_dim=edge_feat_dim,
            hidden_dim=edge_feat_dim,
            output_dim=edge_feat_dim,
            bias=use_bias,
            use_fp16=use_fp16,
        )
        self.learnable_envelope = nn.Linear(
            in_features = num_basis, out_features = edge_feat_dim, bias = False
        )
        
        self.node_res_weight = torch.nn.Parameter(torch.ones(1, node_feat_dim), requires_grad=True)
        self.edge_res_weight = torch.nn.Parameter(torch.ones(1, edge_feat_dim), requires_grad=True)

    def forward(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        smooth_weight: Tensor,
        graph: Dict,
        directed2undirected: Tensor = None,
        atom_feat: Tensor = None, # Line graph
    ) -> Tensor:
        # Gather
        # when graph=="line graph", make sure atom_deat is not None.
        is_atom_graph = (self.graph_type == "atom graph")
        profile_prefix = getattr(self, "profile_prefix", self.__class__.__name__)

        if (
            not is_atom_graph
            and _use_p104_refine_line_r3_block_vjp(profile_prefix)
            and self.training is False
            and torch.is_grad_enabled()
            and atom_feat is not None
            and node_feat.is_cuda
            and edge_feat.is_cuda
            and smooth_weight.is_cuda
            and atom_feat.is_cuda
            and node_feat.dtype == torch.float32
            and edge_feat.dtype == torch.float32
            and smooth_weight.dtype == torch.float32
            and atom_feat.dtype == torch.float32
            and node_feat.ndim == 2
            and edge_feat.ndim == 2
            and smooth_weight.ndim == 2
            and atom_feat.ndim == 2
            and node_feat.shape[-1] == 128
            and edge_feat.shape[-1] == 128
            and atom_feat.shape[-1] == 128
        ):
            return _P104RefineLineBlockManualVJP.apply(
                self,
                graph,
                node_feat,
                edge_feat,
                smooth_weight,
                atom_feat,
            )
        
        p110_refine_atom_edge_update = (
            is_atom_graph
            and (
                os.environ.get("MATRIS_P110_REFINE_ATOM_EDGE_UPDATE", "0") == "1"
                or os.environ.get("MATRIS_P111_REFINE_ATOM_FUSED_FIRST", "0") == "1"
            )
            and hasattr(self.edge_nonlinear_update, "forward_refine_atom_edge")
        )
        if is_atom_graph and not p110_refine_atom_edge_update:
            edge_feat_0 = torch.index_select(edge_feat, 0, directed2undirected) 
        elif is_atom_graph:
            edge_feat_0 = None
        else:
            edge_feat_0 = edge_feat

        source_node_feat = None
        target_node_feat = None
        refine_fusion_feat_nonlinear = None
        delta_edge_feat_precomputed = None
        # Envelope 
        if is_atom_graph:
            smooth_weight = torch.index_select(smooth_weight, 0, directed2undirected)
            smooth_weight = self.learnable_envelope(smooth_weight)
            # Fusion feature
            if not p110_refine_atom_edge_update:
                source_node_feat = torch.index_select(node_feat, 0, graph['source_index'])
                target_node_feat = torch.index_select(node_feat, 0, graph['target_index'])
                refine_fusion_feat = torch.cat([edge_feat_0, target_node_feat, source_node_feat], dim=1)
            else:
                refine_fusion_feat = None
        else:
            base_envelope = self.learnable_envelope(smooth_weight)
            p58_smooth_reduce = (
                _use_p58_refine_line_smooth_reduce(profile_prefix)
                and self.training is False
                and torch.is_grad_enabled()
                and refine_fusion_feat_nonlinear is None
                and base_envelope.is_cuda
                and base_envelope.dtype == torch.float32
                and base_envelope.ndim == 2
                and base_envelope.shape[-1] == 128
                and smooth_weight.is_cuda
            )
            p63_edge_smooth_out = None
            p103_edge_update_edge_ffn = (
                _use_p103_refine_line_r2_edge_update_edge_ffn_vjp(profile_prefix)
                and self.training is False
                and torch.is_grad_enabled()
                and node_feat.is_cuda
                and edge_feat_0.is_cuda
                and atom_feat is not None
                and atom_feat.is_cuda
                and node_feat.dtype == torch.float32
                and edge_feat_0.dtype == torch.float32
                and atom_feat.dtype == torch.float32
                and node_feat.ndim == 2
                and edge_feat_0.ndim == 2
                and atom_feat.ndim == 2
                and node_feat.shape[-1] == 128
                and edge_feat_0.shape[-1] == 128
                and atom_feat.shape[-1] == 128
            )
            p63_edge_smooth_reduce = (
                (
                    os.environ.get("MATRIS_P63_REFINE_LINE_EDGE_SMOOTH_REDUCE", "0") == "1"
                    or os.environ.get("MATRIS_P64_REFINE_LINE_FUSED_BACKWARD", "0") == "1"
                    or os.environ.get("MATRIS_P65_REFINE_LINE_TILED_FUSED_BACKWARD", "0") == "1"
                    or os.environ.get("MATRIS_P65B_REFINE_LINE_PACKED_TILED_BWD", "0") == "1"
                    or os.environ.get("MATRIS_P69_REFINE_LINE_FIRST_TAIL_SMOOTH_REDUCE", "0") == "1"
                )
                and profile_prefix.endswith(".refine_line")
                and self.training is False
                and torch.is_grad_enabled()
                and base_envelope.is_cuda
                and base_envelope.dtype == torch.float32
                and base_envelope.ndim == 2
                and base_envelope.shape[-1] == 128
                and smooth_weight.is_cuda
                and hasattr(self.edge_nonlinear_update, "forward_refine_line_edge_smooth_reduce")
            )
            if p63_edge_smooth_reduce:
                p63_edge_smooth_out = self.edge_nonlinear_update.forward_refine_line_edge_smooth_reduce(
                    node_feat,
                    edge_feat_0,
                    atom_feat,
                    base_envelope,
                    graph["atom_list"],
                    graph["source_index"],
                    graph["target_index"],
                    len(node_feat),
                )
            p63_edge_smooth_used = p63_edge_smooth_out is not None
            if p63_edge_smooth_used:
                refine_node_feas, refine_fusion_feat_nonlinear = p63_edge_smooth_out
                refine_fusion_feat_smooth = None
                p58_smooth_reduce = False
            elif p103_edge_update_edge_ffn:
                refine_fusion_feat_nonlinear, delta_edge_feat_precomputed = (
                    _P103RefineLineEdgeUpdateEdgeFFNManualVJP.apply(
                    self,
                    graph,
                    node_feat,
                    edge_feat_0,
                    atom_feat,
                )
                )
            p58_smooth_reduce_sorted = (
                _use_p58_refine_line_smooth_reduce_sorted(profile_prefix)
                and self.training is False
                and torch.is_grad_enabled()
                and base_envelope.is_cuda
                and base_envelope.dtype == torch.float32
                and base_envelope.ndim == 2
                and base_envelope.shape[-1] == 128
                and smooth_weight.is_cuda
                and _refine_line_target_index_is_sorted(graph)
            )
            if not p63_edge_smooth_used and not p58_smooth_reduce and not p58_smooth_reduce_sorted:
                def gather_smooth_weights():
                    base_weights_i = torch.index_select(base_envelope, 0, graph['source_index'])
                    base_weights_j = torch.index_select(base_envelope, 0, graph['target_index'])
                    return base_weights_i * base_weights_j

                smooth_weight = gather_smooth_weights()
            # Fusion feature
            refine_fusion_feat = None
            if p63_edge_smooth_used:
                pass
            elif (
                (
                    use_p35_refine_line_edge_first_dataflow()
                    or _use_p58_refine_line_edge_update(profile_prefix)
                    or os.environ.get("MATRIS_P60_REFINE_LINE_FUSED_FIRST", "0") == "1"
                    or os.environ.get("MATRIS_P61_REFINE_LINE_FUSED_FIRST_ACTS", "0") == "1"
                    or os.environ.get("MATRIS_P61_REFINE_LINE_FIRST_TAIL", "0") == "1"
                    or os.environ.get("MATRIS_P62_REFINE_LINE_PACKED_FIRST_TAIL", "0") == "1"
                    or os.environ.get("MATRIS_P63_REFINE_LINE_EDGE_SMOOTH_REDUCE", "0") == "1"
                    or os.environ.get("MATRIS_P64_REFINE_LINE_FUSED_BACKWARD", "0") == "1"
                    or os.environ.get("MATRIS_P65_REFINE_LINE_TILED_FUSED_BACKWARD", "0") == "1"
                    or os.environ.get("MATRIS_P65B_REFINE_LINE_PACKED_TILED_BWD", "0") == "1"
                    or os.environ.get("MATRIS_P69_REFINE_LINE_FIRST_TAIL_SMOOTH_REDUCE", "0") == "1"
                )
                and profile_prefix.endswith(".refine_line")
                and hasattr(self.edge_nonlinear_update, "forward_refine_line_edge")
            ):
                refine_fusion_feat_nonlinear = self.edge_nonlinear_update.forward_refine_line_edge(
                    node_feat,
                    edge_feat_0,
                    atom_feat,
                    graph["atom_list"],
                    graph["source_index"],
                    graph["target_index"],
                )
            else:
                refine_fusion_feat_nonlinear = None
            if not p63_edge_smooth_used and refine_fusion_feat_nonlinear is None:
                def gather_concat_refine_line():
                    source_node_feat = torch.index_select(node_feat, 0, graph['source_index'])
                    target_node_feat = torch.index_select(node_feat, 0, graph['target_index'])
                    three_body_atom_feat = torch.index_select(atom_feat, 0, graph['atom_list'])
                    return torch.cat([edge_feat_0, three_body_atom_feat, target_node_feat, source_node_feat], dim=1)

                refine_fusion_feat = gather_concat_refine_line()
        
        # Nonlinear            
        if not is_atom_graph and 'p63_edge_smooth_used' in locals() and p63_edge_smooth_used:
            pass
        else:
            if refine_fusion_feat_nonlinear is None:
                if (
                    p110_refine_atom_edge_update
                ):
                    refine_fusion_feat_nonlinear = self.edge_nonlinear_update.forward_refine_atom_edge(
                        node_feat,
                        edge_feat,
                        directed2undirected,
                        graph['source_index'],
                        graph['target_index'],
                    )
                if refine_fusion_feat_nonlinear is None:
                    refine_fusion_feat_nonlinear = self.edge_nonlinear_update(refine_fusion_feat)
            if not is_atom_graph and 'p58_smooth_reduce' in locals() and p58_smooth_reduce:
                refine_fusion_feat_smooth = None
                refine_node_feas = _P58RefineLineSmoothReduce.apply(
                    refine_fusion_feat_nonlinear,
                    base_envelope,
                    graph['source_index'],
                    graph['target_index'],
                    len(node_feat),
                )
            elif not is_atom_graph and 'p58_smooth_reduce_sorted' in locals() and p58_smooth_reduce_sorted:
                refine_fusion_feat_smooth = None
                refine_node_feas = _P58RefineLineSmoothReduceSorted.apply(
                    refine_fusion_feat_nonlinear,
                    base_envelope,
                    graph['source_index'],
                    graph['target_segment_offsets'],
                )
            else:
                refine_fusion_feat_smooth = refine_fusion_feat_nonlinear * smooth_weight
             
                profile_prefix = getattr(self, "profile_prefix", self.__class__.__name__)
                refine_node_feas = aggregate(
                    refine_fusion_feat_smooth,
                    graph['target_index'],
                    graph['target_bincount'],
                    average=False,
                    num_segment=len(node_feat),
                    profile_name=f"{profile_prefix}.target_smooth_sum",
                )

        input2edgeFFN = (
            refine_fusion_feat_smooth
            if is_atom_graph and self.use_smoothed_for_delta_edge
            else refine_fusion_feat_nonlinear
        )
        
        profile_prefix = getattr(self, "profile_prefix", self.__class__.__name__)

        if delta_edge_feat_precomputed is not None:
            delta_node_feat = self.node_FFN(refine_node_feas)
            delta_edge_feat = delta_edge_feat_precomputed
        elif (
            not is_atom_graph
            and _use_p102_refine_line_r1_ffn_pair_vjp(profile_prefix)
            and self.training is False
            and torch.is_grad_enabled()
            and refine_node_feas.is_cuda
            and input2edgeFFN.is_cuda
            and refine_node_feas.dtype == torch.float32
            and input2edgeFFN.dtype == torch.float32
            and refine_node_feas.ndim == 2
            and input2edgeFFN.ndim == 2
            and refine_node_feas.shape[-1] == 128
            and input2edgeFFN.shape[-1] == 128
        ):
            delta_node_feat, delta_edge_feat = _P102RefineLineFFNPairManualVJP.apply(
                self,
                refine_node_feas,
                input2edgeFFN,
            )
        else:
            p68_grouped_pair = p68_grouped_ffn_pair_or_none(
                self.node_FFN,
                self.edge_FFN,
                refine_node_feas,
                input2edgeFFN,
                profile_prefix,
            )
            if p68_grouped_pair is not None:
                delta_node_feat, delta_edge_feat = p68_grouped_pair
            else:
                delta_node_feat = self.node_FFN(refine_node_feas)
                delta_edge_feat = self.edge_FFN(input2edgeFFN)
        
        if is_atom_graph:  
            directed_average = directed2undirected_average_or_none(
                delta_edge_feat,
                directed2undirected,
                edge_feat.shape[0],
                enable_hint=True,
            )
            if directed_average is None:
                delta_edge_feat = aggregate(
                    data=delta_edge_feat,
                    segment=directed2undirected,
                    bin_count=None,
                    average=True,
                    num_segment=None,
                    profile_name=f"{profile_prefix}.directed2undirected_delta_average",
                ) # [2*edge, dim] -> [edge, dim]
            else:
                delta_edge_feat = directed_average

        def residual_update():
            return (
                delta_node_feat + self.node_res_weight * node_feat,
                delta_edge_feat + self.edge_res_weight * edge_feat,
            )

        update_node_feat, update_edge_feat = residual_update()
        
        return update_node_feat, update_edge_feat


class Interaction_Block(nn.Module):
    """
    Interaction Block for MatRIS that processes both atom graphs and line graphs.
    
    This block performs attention-based message passing and refinement on two hierarchical graph structures:
    1. Atom graph: Nodes represent atoms, edges represent bonds
    2. Line graph: Nodes represent bonds, edges represent three-body interactions (angles)
    
    Attributes:
        attn_block_atom_graph (Graph_Attention_Layer): Attention layer for atom graph
        attn_block_line_graph (Graph_Attention_Layer): Attention layer for line graph
        refine_block_atom_graph (Refinement): Refinement layer for atom graph  
        refine_block_line_graph (Refinement): Refinement layer for line graph
    """
    
    def __init__(self,
                 node_feat_dim: int = 128,
                 edge_feat_dim: int = 128,
                 three_body_feat_dim: int = 128,
                 num_radial: int = 7,
                 num_angular: int = 7,
                 dropout: float = 0.0, 
                 use_bias: bool = False,
                 use_smoothed_for_delta_edge: bool = False,
                 mlp_type: str = "GateMLP",
                 norm_type: str = "layer",
                 activation_type: str = "silu",
                 ):
        """
        Initialize the Interaction Block.

        Args:
            node_feat_dim (int): Dimension of node features (atom features)
            edge_feat_dim (int): Dimension of edge features (bond features)  
            three_body_feat_dim (int): Dimension of three-body features (angle features)
            mlp_type (str): Type of MLP to use in the layers
            norm_type (str): Type of normalization to apply
            activation_type (str): Type of activation function to use
        """
        super().__init__()
        
        self.attn_block_atom_graph = Graph_Attention_Layer(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=node_feat_dim,
                use_bias=use_bias,
                mlp_type=mlp_type,
                norm_type=norm_type,
                activation_type=activation_type,
            )

        self.attn_block_line_graph = Graph_Attention_Layer(
                node_feat_dim=edge_feat_dim,
                edge_feat_dim=three_body_feat_dim,
                hidden_dim=edge_feat_dim,
                use_bias=use_bias,
                mlp_type=mlp_type,
                norm_type=norm_type,
                activation_type=activation_type,
                use_fp16=False,
            )
        
        self.refine_block_atom_graph = Refinement(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=node_feat_dim,  
                num_basis=num_radial,      
                dropout=dropout,            
                activation_type=activation_type,
                norm_type=norm_type,
                use_bias=use_bias,
                mlp_type=mlp_type,
                graph_type="atom graph",
                use_smoothed_for_delta_edge=use_smoothed_for_delta_edge,
            )
        
        self.refine_block_line_graph = Refinement(
                node_feat_dim=edge_feat_dim,
                edge_feat_dim=three_body_feat_dim,
                hidden_dim=edge_feat_dim,  
                num_basis=num_angular,     
                dropout=dropout,          
                activation_type=activation_type,
                norm_type=norm_type,
                use_bias=use_bias,
                mlp_type=mlp_type,
                graph_type="line graph",
                atom_feat_dim=node_feat_dim,
                use_fp16=False, 
            )
    
    def forward(
        self,
        batch_graph: Dict,
        node_feat: Tensor, 
        edge_feat: Tensor, 
        threebody_feat: Tensor | None,
        smooth_weight: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the Interaction Block.
        
        Args:
            batch_graph: Graph object containing:
                - atom_graph_dict: Atom graph structure
                - line_graph_dict: Bond graph (line graph) structure  
                - directed2undirected: Mapping from directed to undirected edges
                - bond_bases_bg: Smooth weights for bond graph
                - bond_bases_ag: Smooth weights for atom graph
            node_feat (Tensor): Node features [num_atoms, node_feat_dim]
            edge_feat (Tensor): Edge features [num_bonds, edge_feat_dim] 
            threebody_feat (Tensor): Three-body features [num_angles, three_body_feat_dim] or None
            bincount_atom_graph (Dict): Bincount information for atom graph
            bincount_line_graph (Dict): Bincount information for line graph
        """
        # Initialize variables to handle both cases (with and without threebody features)
        attn_edge_feat = edge_feat 
        attn_threebody_feat = threebody_feat
        update_edge_feat = edge_feat
        update_threebody_feat = threebody_feat 
        use_checkpoint = (
            isinstance(threebody_feat, torch.Tensor)
            and threebody_feat.shape[0] > THRESHOLD_VALUE
        )

        # Process line graph (bond graph) with attention if threebody features exist
        profile_prefix = getattr(self, "profile_prefix", self.__class__.__name__)
        if (
            _use_p53b_full_block_manual_vjp(profile_prefix)
            and self.training is False
            and torch.is_grad_enabled()
            and node_feat.is_cuda
            and edge_feat.is_cuda
            and not use_checkpoint
            and not getattr(self, "_p53b_full_block_manual_vjp_active", False)
        ):
            return _P53BFullBlockManualVJP.apply(
                self,
                batch_graph,
                node_feat,
                edge_feat,
                threebody_feat,
                smooth_weight.get("atom graph") if isinstance(smooth_weight, dict) else None,
                smooth_weight.get("line graph") if isinstance(smooth_weight, dict) else None,
            )
        if (
            _use_p53_one_block_custom_vjp(profile_prefix)
            and self.training is False
            and torch.is_grad_enabled()
            and node_feat.is_cuda
            and edge_feat.is_cuda
            and not use_checkpoint
            and not getattr(self, "_p53_one_block_custom_vjp_active", False)
        ):
            return _P53OneBlockCustomVJP.apply(
                self,
                batch_graph,
                node_feat,
                edge_feat,
                threebody_feat,
                smooth_weight.get("atom graph") if isinstance(smooth_weight, dict) else None,
                smooth_weight.get("line graph") if isinstance(smooth_weight, dict) else None,
            )
        if (
            _use_p37_interaction_ste(profile_prefix)
            and self.training is False
            and torch.is_grad_enabled()
            and node_feat.is_cuda
            and edge_feat.is_cuda
            and not getattr(self, "_p37_interaction_ste_active", False)
        ):
            self._p37_interaction_ste_active = True
            try:
                if os.environ.get("MATRIS_P37_INTERACTION_STE_FORWARD_MODE", "no_grad") == "grad":
                    actual_node_feat, actual_edge_feat, actual_threebody_feat = self.forward(
                        batch_graph=batch_graph,
                        node_feat=node_feat,
                        edge_feat=edge_feat,
                        threebody_feat=threebody_feat,
                        smooth_weight=smooth_weight,
                    )
                else:
                    with torch.no_grad():
                        actual_node_feat, actual_edge_feat, actual_threebody_feat = self.forward(
                            batch_graph=batch_graph,
                            node_feat=node_feat,
                            edge_feat=edge_feat,
                            threebody_feat=threebody_feat,
                            smooth_weight=smooth_weight,
                        )
            finally:
                self._p37_interaction_ste_active = False

            node_proxy = _p37_node_proxy(batch_graph, node_feat, edge_feat, profile_prefix)
            return (
                _p37_straight_through(actual_node_feat, node_proxy),
                _p37_straight_through(actual_edge_feat, edge_feat),
                _p37_straight_through(actual_threebody_feat, threebody_feat),
            )

        if threebody_feat is not None: 
            self.attn_block_line_graph.profile_prefix = f"{profile_prefix}.attn_line"
            if (
                use_p71_line_attn_eval_only_vjp(self.attn_block_line_graph.profile_prefix)
                and self.training is False
                and torch.is_grad_enabled()
                and edge_feat.is_cuda
                and threebody_feat.is_cuda
                and not use_checkpoint
            ):
                attn_edge_feat, attn_threebody_feat = p71_line_attention_eval_only_vjp(
                    self.attn_block_line_graph,
                    batch_graph['line_graph_dict'],
                    edge_feat,
                    threebody_feat,
                )
            else:
                attn_edge_feat, attn_threebody_feat = self.wrapper_attn_layer(
                    attn_layer=self.attn_block_line_graph,
                    node_feat=edge_feat,
                    edge_feat=threebody_feat,
                    graph=batch_graph['line_graph_dict'],
                    use_checkpoint=use_checkpoint, 
                )
        
        # Process atom graph with attention
        self.attn_block_atom_graph.profile_prefix = f"{profile_prefix}.attn_atom"
        attn_node_feat, attn_edge_feat = self.wrapper_attn_layer(
            attn_layer=self.attn_block_atom_graph,
            node_feat=node_feat, 
            edge_feat=attn_edge_feat, 
            graph=batch_graph['atom_graph_dict'], 
            directed2undirected=batch_graph['directed2undirected'],
        ) 
        
        # Refine line graph features if threebody features exist
        if threebody_feat is not None:
            self.refine_block_line_graph.profile_prefix = f"{profile_prefix}.refine_line"
            update_edge_feat, update_threebody_feat = self.wrapper_refine_layer(
                refine_layer=self.refine_block_line_graph,
                node_feat=attn_edge_feat,
                edge_feat=attn_threebody_feat,
                smooth_weight=smooth_weight['line graph'],
                graph=batch_graph['line_graph_dict'],
                atom_feat=attn_node_feat,
                use_checkpoint=use_checkpoint,
            )
        
        # Refine atom graph features
        self.refine_block_atom_graph.profile_prefix = f"{profile_prefix}.refine_atom"
        update_node_feat, update_edge_feat = self.wrapper_refine_layer(
            refine_layer=self.refine_block_atom_graph,
            node_feat=attn_node_feat,
            edge_feat=update_edge_feat,
            smooth_weight=smooth_weight['atom graph'],
            graph=batch_graph['atom_graph_dict'],
            directed2undirected=batch_graph['directed2undirected'],
        )
        self.last_profile = {}
        for module in (
            self.attn_block_line_graph,
            self.attn_block_atom_graph,
            self.refine_block_line_graph,
            self.refine_block_atom_graph,
        ):
            for key, value in getattr(module, "last_profile", {}).items():
                if isinstance(value, (int, float)):
                    self.last_profile[key] = float(value)
        
        return update_node_feat, update_edge_feat, update_threebody_feat
    
    def wrapper_attn_layer(self,
                            attn_layer: nn.Module,
                            node_feat: Tensor, 
                            edge_feat: Tensor, 
                            graph: Dict,
                            directed2undirected: Tensor = None,
                            use_checkpoint: bool = False,
                       ):
        if use_checkpoint:
            attn_node_feat, attn_edge_feat = checkpoint(
                attn_layer,
                node_feat, 
                edge_feat, 
                graph,
                directed2undirected,
                use_reentrant=False,
            ) 
        else:
            attn_node_feat, attn_edge_feat = attn_layer(
                node_feat=node_feat, 
                edge_feat=edge_feat, 
                graph=graph,
                directed2undirected=directed2undirected,
            )
        
        return attn_node_feat, attn_edge_feat 
        
    def wrapper_refine_layer(self, 
                            refine_layer: nn.Module,
                            node_feat: Tensor,
                            edge_feat: Tensor,
                            smooth_weight: Tensor,
                            graph: Dict,
                            directed2undirected: Tensor = None,
                            atom_feat: Tensor = None,
                            use_checkpoint: bool = False,
                        ):
        if use_checkpoint:
            update_node_feat, update_edge_feat = checkpoint(
                refine_layer,
                node_feat,
                edge_feat,
                smooth_weight,
                graph,
                directed2undirected,
                atom_feat,
                use_reentrant=False,
            )
        else:
            update_node_feat, update_edge_feat = refine_layer(
                    node_feat=node_feat,
                    edge_feat=edge_feat,
                    smooth_weight=smooth_weight,
                    graph=graph,
                    directed2undirected=directed2undirected,
                    atom_feat=atom_feat,
                )
        return update_node_feat, update_edge_feat 
        
    
