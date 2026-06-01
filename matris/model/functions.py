from __future__ import annotations

import contextlib
from collections.abc import Sequence

import json
import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path
from .op import fused_silu, fused_sigmoid

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional.
    triton = None
    tl = None

class FusedSiLU(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda":
            return fused_silu(x)
        else:
            return torch.nn.functional.silu(x) 

class FusedSigmoid(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda":
            return fused_sigmoid(x)
        else:
            return torch.nn.functional.sigmoid(x)

def get_activation(name: str) -> nn.Module:
    """Return an activation function"""
    activation_map = {
        "relu": nn.ReLU,
        "silu": FusedSiLU,  # Using fused version for better performance
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "sigmoid": FusedSigmoid,  # Using fused version for better performance
        "tanh": nn.Tanh,
    }
    
    name_lower = name.lower()
    if name_lower not in activation_map:
        raise NotImplementedError(
            f"Activation '{name}' is not implemented. "
            f"Supported activations: {list(activation_map.keys())}"
        )
    return activation_map[name_lower]()

def get_normalization(name: str, dim: int | None = None) -> nn.Module | None:
    """Return an normalization function"""
    if name is None:
        return None
        
    normalization_map = {
        "layer": nn.LayerNorm(dim),
        "rms": nn.RMSNorm(dim), # torch >= 2.6.0
        "batch": nn.BatchNorm1d(dim),
    }
    name_lower = name.lower()
    return normalization_map[name_lower]


def _graph_op_profile_path() -> str:
    return os.environ.get("MATRIS_GRAPH_OP_PROFILE_PATH", "").strip()


def _profiled_cuda_start(data: Tensor):
    if not _graph_op_profile_path() or data.device.type != "cuda":
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _profiled_cuda_finish(
    handle,
    *,
    op: str,
    name: str | None,
    data: Tensor,
    segment: Tensor,
    num_segment: int | None,
    average: bool | None = None,
) -> None:
    path = _graph_op_profile_path()
    if not path:
        return
    elapsed_ms = None
    if handle is not None:
        start, end = handle
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))

    with torch.no_grad():
        segment_cpu = segment.detach()
        segment_numel = int(segment_cpu.numel())
        segment_max_plus_one = 0
        if segment_numel:
            segment_max_plus_one = int(segment_cpu.max().item()) + 1
        resolved_num_segment = int(num_segment) if num_segment is not None else segment_max_plus_one
        record = {
            "op": op,
            "name": name or op,
            "elapsed_ms": elapsed_ms,
            "data_shape": list(data.shape),
            "data_dtype": str(data.dtype),
            "segment_numel": segment_numel,
            "segment_max_plus_one": segment_max_plus_one,
            "num_segment": resolved_num_segment,
            "feature_dim": int(data.shape[1]) if data.ndim == 2 else None,
            "average": average,
            "requires_grad": bool(data.requires_grad),
        }
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


if triton is not None:

    @triton.jit
    def _segment_softmax_sorted_forward_kernel(
        x_ptr,
        offsets_ptr,
        lengths_ptr,
        out_ptr,
        num_segments: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_r = tl.arange(0, BLOCK_R)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        start = tl.load(offsets_ptr + pid_s)
        length = tl.load(lengths_ptr + pid_s)
        mask = (offs_r[:, None] < length) & (offs_d[None, :] < dim)
        x = tl.load(
            x_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        max_v = tl.max(x, axis=0)
        exp_v = tl.exp(x - max_v[None, :])
        exp_v = tl.where(offs_r[:, None] < length, exp_v, 0.0)
        sum_v = tl.sum(exp_v, axis=0)
        y = exp_v / sum_v[None, :]
        tl.store(
            out_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            y,
            mask=mask,
        )

    @triton.jit
    def _segment_softmax_sorted_backward_kernel(
        grad_out_ptr,
        out_ptr,
        offsets_ptr,
        lengths_ptr,
        grad_x_ptr,
        num_segments: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_r = tl.arange(0, BLOCK_R)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        start = tl.load(offsets_ptr + pid_s)
        length = tl.load(lengths_ptr + pid_s)
        mask = (offs_r[:, None] < length) & (offs_d[None, :] < dim)
        y = tl.load(
            out_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_out = tl.load(
            grad_out_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(grad_out * y, axis=0)
        grad_x = y * (grad_out - dot[None, :])
        tl.store(
            grad_x_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            grad_x,
            mask=mask,
        )

    @triton.jit
    def _segment_softmax_weighted_sum_forward_kernel(
        alpha_logits_ptr,
        values_ptr,
        offsets_ptr,
        lengths_ptr,
        alpha_ptr,
        out_ptr,
        num_segments: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_r = tl.arange(0, BLOCK_R)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        start = tl.load(offsets_ptr + pid_s)
        length = tl.load(lengths_ptr + pid_s)
        valid_segment = length > 0
        mask = (offs_r[:, None] < length) & (offs_d[None, :] < dim)
        logits = tl.load(
            alpha_logits_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        max_v = tl.max(logits, axis=0)
        max_v = tl.where(valid_segment, max_v, 0.0)
        exp_v = tl.exp(logits - max_v[None, :])
        exp_v = tl.where(offs_r[:, None] < length, exp_v, 0.0)
        sum_v = tl.sum(exp_v, axis=0)
        alpha = exp_v / sum_v[None, :]
        alpha = tl.where(valid_segment, alpha, 0.0)
        values = tl.load(
            values_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(alpha * values, axis=0)
        tl.store(
            alpha_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            alpha,
            mask=mask,
        )
        tl.store(out_ptr + pid_s * dim + offs_d, out, mask=offs_d < dim)

    @triton.jit
    def _segment_softmax_weighted_sum_backward_kernel(
        grad_out_ptr,
        values_ptr,
        alpha_ptr,
        out_ptr,
        offsets_ptr,
        lengths_ptr,
        grad_alpha_logits_ptr,
        grad_values_ptr,
        num_segments: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_r = tl.arange(0, BLOCK_R)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        start = tl.load(offsets_ptr + pid_s)
        length = tl.load(lengths_ptr + pid_s)
        mask = (offs_r[:, None] < length) & (offs_d[None, :] < dim)
        grad_out = tl.load(grad_out_ptr + pid_s * dim + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
        out = tl.load(out_ptr + pid_s * dim + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
        values = tl.load(
            values_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        alpha = tl.load(
            alpha_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_values = alpha * grad_out[None, :]
        grad_alpha_logits = alpha * grad_out[None, :] * (values - out[None, :])
        tl.store(
            grad_values_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            grad_values,
            mask=mask,
        )
        tl.store(
            grad_alpha_logits_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            grad_alpha_logits,
            mask=mask,
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (max(value, 1) - 1).bit_length()


def _use_triton_sorted_segment_softmax() -> bool:
    return os.environ.get("MATRIS_USE_TRITON_SORTED_SEGMENT_SOFTMAX", "0") == "1"


def _use_torch_sorted_segment_reduce() -> bool:
    return os.environ.get("MATRIS_USE_TORCH_SORTED_SEGMENT_REDUCE", "0") == "1"


def _use_triton_target_attention_sum() -> bool:
    return os.environ.get("MATRIS_USE_TRITON_TARGET_ATTENTION_SUM", "0") == "1"


def _use_cuda_target_attention_sum() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_TARGET_ATTENTION_SUM", "0") == "1"


def _use_cuda_target_attention_bwd() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_TARGET_ATTENTION_BWD", "0") == "1"


def _use_cuda_directed2undirected_average() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE", "0") == "1"


def use_cuda_edge_vectors() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_EDGE_VECTORS", "0") == "1"


def _use_cuda_fused_line_attention() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_FUSED_LINE_ATTENTION", "0") == "1"


def _use_p82_fused_line_attention_forward_v2() -> bool:
    return os.environ.get("MATRIS_P82_FUSED_LINE_ATTENTION_FORWARD_V2", "0") == "1"


def _use_p82c_fused_line_attention_target_segment_max() -> bool:
    return os.environ.get("MATRIS_P82C_LINE_ATTENTION_TARGET_SEGMENT_MAX", "0") == "1"


def _p82c_line_attention_min_rows() -> int:
    return _env_int("MATRIS_P82C_LINE_ATTENTION_MIN_ROWS", 12000)


def _use_p83b_fused_line_attention_target_offsets() -> bool:
    return os.environ.get("MATRIS_P83B_LINE_ATTENTION_TARGET_OFFSETS", "0") == "1"


def _use_p83c_fused_line_attention_node_input() -> bool:
    return os.environ.get("MATRIS_P83C_LINE_ATTENTION_NODE_INPUT", "0") == "1"


def _use_cuda_fused_atom_attention() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION", "0") == "1"


def _use_line_attn_edge_bwd_fusion() -> bool:
    return os.environ.get("MATRIS_LINE_ATTN_EDGE_BWD_FUSION", "0") == "1"


def aggressive_line_attn_eval_mode() -> str:
    if os.environ.get("MATRIS_AGGRESSIVE_LINE_ATTN_EVAL", "0") != "1":
        return ""
    return os.environ.get("MATRIS_AGGRESSIVE_LINE_ATTN_MODE", "").strip()


def aggressive_broad_bwd_mode() -> str:
    if os.environ.get("MATRIS_AGGRESSIVE_BROAD_BWD", "0") != "1":
        return ""
    return os.environ.get("MATRIS_AGGRESSIVE_BROAD_BWD_MODE", "input_grad_only").strip()


def _use_aggressive_broad_input_grad_only() -> bool:
    return aggressive_broad_bwd_mode() in {
        "input_grad_only",
        "w8a8_saved_pre_all",
        "bypass_gated_mlp",
        "bypass_mlp",
    }


def _use_aggressive_broad_bypass_mlp() -> bool:
    return aggressive_broad_bwd_mode() == "bypass_mlp"


def _use_aggressive_broad_bypass_gated_mlp() -> bool:
    return aggressive_broad_bwd_mode() in {"bypass_gated_mlp", "bypass_mlp"}


def _use_p26_fused_first_input_grad_only() -> bool:
    return os.environ.get("MATRIS_P26_FUSED_FIRST_INPUT_GRAD_ONLY", "0") == "1"


def _use_p26_tail_input_grad_only() -> bool:
    return os.environ.get("MATRIS_P26_TAIL_INPUT_GRAD_ONLY", "0") == "1"


def _use_p27_tail_param_grad() -> bool:
    return os.environ.get("MATRIS_P27_TAIL_PARAM_GRAD", "0") == "1"


def _use_p28_mlp_input_grad_only() -> bool:
    return os.environ.get("MATRIS_P28_MLP_INPUT_GRAD_ONLY", "0") == "1"


def use_p29_fused_alpha_input_grad_only() -> bool:
    return os.environ.get("MATRIS_P29_FUSED_ALPHA_INPUT_GRAD_ONLY", "0") == "1"


def _use_p29_mlp_bwd_kernel() -> bool:
    return os.environ.get("MATRIS_P29_MLP_BWD_KERNEL", "0") == "1"


def use_p35_all_line_attn_edge_first_dataflow() -> bool:
    return os.environ.get("MATRIS_P35_ALL_LINE_ATTN_EDGE_FIRST_DATAFLOW", "0") == "1"


def use_p35_atom_edge_first_dataflow() -> bool:
    return os.environ.get("MATRIS_P35_ATOM_EDGE_FIRST_DATAFLOW", "0") == "1"


def use_p35_refine_line_edge_first_dataflow() -> bool:
    return os.environ.get("MATRIS_P35_REFINE_LINE_EDGE_FIRST_DATAFLOW", "0") == "1"


def use_p79_attn_line_gather_cat(profile_prefix: str) -> bool:
    return (
        os.environ.get("MATRIS_P79_ATTN_LINE_GATHER_CAT", "0") == "1"
        and profile_prefix.endswith(".attn_line")
    )


def use_p80_attn_line_node_cat(profile_prefix: str) -> bool:
    return (
        os.environ.get("MATRIS_P80_ATTN_LINE_NODE_CAT", "0") == "1"
        and profile_prefix.endswith(".attn_line")
    )


def use_p81_attn_reduce_detail_profile() -> bool:
    return os.environ.get("MATRIS_P81_ATTN_REDUCE_DETAIL", "0") == "1"


def _use_p44_line_edge_w8a8_big_bwd(rows: int) -> bool:
    if os.environ.get("MATRIS_P44_LINE_EDGE_W8A8_BIG_BWD", "0") != "1":
        return False
    min_rows = _env_int("MATRIS_P44_LINE_EDGE_W8A8_BIG_BWD_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P44_LINE_EDGE_W8A8_BIG_BWD_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _record_p44_line_edge_w8a8_big_bwd_hit() -> None:
    if os.environ.get("MATRIS_P44_LINE_EDGE_W8A8_BIG_BWD_STATS", "0") != "1":
        return
    import atexit

    hits = int(getattr(_record_p44_line_edge_w8a8_big_bwd_hit, "hits", 0)) + 1
    setattr(_record_p44_line_edge_w8a8_big_bwd_hit, "hits", hits)
    if not getattr(_record_p44_line_edge_w8a8_big_bwd_hit, "registered", False):
        def _print_stats() -> None:
            print(
                "[p44 line-edge w8a8 big bwd stats] "
                f"hits={getattr(_record_p44_line_edge_w8a8_big_bwd_hit, 'hits', 0)}",
                flush=True,
            )

        atexit.register(_print_stats)
        setattr(_record_p44_line_edge_w8a8_big_bwd_hit, "registered", True)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _p60_refine_line_census_path() -> str:
    return os.environ.get("MATRIS_P60_REFINE_LINE_CENSUS_PATH", "").strip()


def _record_p60_refine_line_census(
    *,
    module_name: str,
    stage: str,
    rows: int,
    input_dim: int | None = None,
    output_dim: int | None = None,
    core_hidden_dim: int | None = None,
    gate_hidden_dim: int | None = None,
    node_rows: int | None = None,
    edge_rows: int | None = None,
    atom_rows: int | None = None,
    reason: str = "",
) -> None:
    path_value = _p60_refine_line_census_path()
    if not path_value:
        return
    import atexit

    stats = getattr(_record_p60_refine_line_census, "_stats", None)
    if stats is None:
        stats = {}
        setattr(_record_p60_refine_line_census, "_stats", stats)

    if not getattr(_record_p60_refine_line_census, "_registered", False):
        def _dump_stats() -> None:
            path = Path(path_value)
            path.parent.mkdir(parents=True, exist_ok=True)
            records = []
            for record in stats.values():
                rows_values = record.pop("_rows_values", [])
                if rows_values:
                    record["rows_min"] = min(rows_values)
                    record["rows_max"] = max(rows_values)
                records.append(record)
            records.sort(key=lambda item: (item["module_name"], item["stage"], item["rows"]))
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")

        atexit.register(_dump_stats)
        setattr(_record_p60_refine_line_census, "_registered", True)

    key = (
        module_name,
        stage,
        int(rows),
        input_dim,
        output_dim,
        core_hidden_dim,
        gate_hidden_dim,
        reason,
    )
    record = stats.get(key)
    if record is None:
        record = {
            "module_name": module_name,
            "stage": stage,
            "rows": int(rows),
            "count": 0,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "core_hidden_dim": core_hidden_dim,
            "gate_hidden_dim": gate_hidden_dim,
            "node_rows_example": node_rows,
            "edge_rows_example": edge_rows,
            "atom_rows_example": atom_rows,
            "reason": reason,
            "_rows_values": [],
        }
        stats[key] = record
    record["count"] += 1
    record["_rows_values"].append(int(rows))


def _use_p60_refine_line_fused_first(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P60_REFINE_LINE_FUSED_FIRST", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    min_rows = _env_int("MATRIS_P60_REFINE_LINE_FUSED_FIRST_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P60_REFINE_LINE_FUSED_FIRST_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p61_refine_line_fused_first_acts(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P61_REFINE_LINE_FUSED_FIRST_ACTS", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    min_rows = _env_int("MATRIS_P61_REFINE_LINE_FUSED_FIRST_ACTS_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P61_REFINE_LINE_FUSED_FIRST_ACTS_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p61_refine_line_first_tail(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P61_REFINE_LINE_FIRST_TAIL", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    min_rows = _env_int("MATRIS_P61_REFINE_LINE_FIRST_TAIL_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P61_REFINE_LINE_FIRST_TAIL_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p62_refine_line_packed_first_tail(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P62_REFINE_LINE_PACKED_FIRST_TAIL", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    min_rows = _env_int("MATRIS_P62_REFINE_LINE_PACKED_FIRST_TAIL_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P62_REFINE_LINE_PACKED_FIRST_TAIL_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p63_refine_line_edge_smooth_reduce(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P63_REFINE_LINE_EDGE_SMOOTH_REDUCE", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    min_rows = _env_int("MATRIS_P63_REFINE_LINE_EDGE_SMOOTH_REDUCE_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P63_REFINE_LINE_EDGE_SMOOTH_REDUCE_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p69_refine_line_first_tail_smooth_reduce(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P69_REFINE_LINE_FIRST_TAIL_SMOOTH_REDUCE", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    min_rows = _env_int("MATRIS_P69_REFINE_LINE_FIRST_TAIL_SMOOTH_REDUCE_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P69_REFINE_LINE_FIRST_TAIL_SMOOTH_REDUCE_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p64_refine_line_fused_backward(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P64_REFINE_LINE_FUSED_BACKWARD", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    min_rows = _env_int("MATRIS_P64_REFINE_LINE_FUSED_BACKWARD_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P64_REFINE_LINE_FUSED_BACKWARD_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p65_refine_line_tiled_fused_backward(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P65_REFINE_LINE_TILED_FUSED_BACKWARD", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    min_rows = _env_int("MATRIS_P65_REFINE_LINE_TILED_FUSED_BACKWARD_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P65_REFINE_LINE_TILED_FUSED_BACKWARD_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p65b_refine_line_packed_tiled_backward(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P65B_REFINE_LINE_PACKED_TILED_BWD", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    min_rows = _env_int("MATRIS_P65B_REFINE_LINE_PACKED_TILED_BWD_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P65B_REFINE_LINE_PACKED_TILED_BWD_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _p67_ffn_census_path() -> str:
    return os.environ.get("MATRIS_P67_FFN_CENSUS_PATH", "").strip()


def _p67_ffn_scope_matches(module_name: str, scope_value: str) -> bool:
    scopes = [item.strip() for item in scope_value.split(",") if item.strip()]
    if not scopes:
        scopes = ["refine_line_edge_ffn"]
    for scope in scopes:
        if scope == "refine_line_edge_ffn" and ".refine_block_line_graph.edge_FFN" in module_name:
            return True
        if scope == "refine_line_node_ffn" and ".refine_block_line_graph.node_FFN" in module_name:
            return True
        if scope == "refine_line_ffn" and ".refine_block_line_graph." in module_name and module_name.endswith("_FFN"):
            return True
        if scope == "refine_atom_edge_ffn" and ".refine_block_atom_graph.edge_FFN" in module_name:
            return True
        if scope == "refine_atom_node_ffn" and ".refine_block_atom_graph.node_FFN" in module_name:
            return True
        if scope == "refine_atom_ffn" and ".refine_block_atom_graph." in module_name and module_name.endswith("_FFN"):
            return True
        if scope == "edge_ffn" and module_name.endswith(".edge_FFN"):
            return True
        if scope == "node_ffn" and module_name.endswith(".node_FFN"):
            return True
        if scope == "all_ffn" and module_name.endswith("_FFN"):
            return True
        if scope == "all":
            return True
    return False


def _p67_ffn_scope_rows_match(module_name: str, rows: int) -> bool:
    scope = os.environ.get("MATRIS_P67_FFN_FUSED_QUANT_SCOPE", "refine_line_edge_ffn").strip()
    if not _p67_ffn_scope_matches(module_name, scope):
        return False
    min_rows = _env_int("MATRIS_P67_FFN_FUSED_QUANT_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P67_FFN_FUSED_QUANT_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p67_ffn_fused_quant(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P67_FFN_FUSED_QUANT", "0") != "1":
        return False
    return _p67_ffn_scope_rows_match(module_name, rows)


def _record_p67_ffn_census(
    *,
    module_name: str,
    rows: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    use_fp16: bool,
    has_bias: bool,
    p67_eligible: bool,
    reason: str,
) -> None:
    path_value = _p67_ffn_census_path()
    if not path_value:
        return
    import atexit

    stats = getattr(_record_p67_ffn_census, "_stats", None)
    if stats is None:
        stats = {}
        setattr(_record_p67_ffn_census, "_stats", stats)

    if not getattr(_record_p67_ffn_census, "_registered", False):
        def _dump_stats() -> None:
            path = Path(path_value)
            path.parent.mkdir(parents=True, exist_ok=True)
            records = []
            for record in stats.values():
                item = dict(record)
                rows_values = item.pop("_rows_values", [])
                if rows_values:
                    item["rows_min"] = min(rows_values)
                    item["rows_max"] = max(rows_values)
                records.append(item)
            records.sort(key=lambda item: (item["module_name"], item["rows"], item["reason"]))
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")

        atexit.register(_dump_stats)
        setattr(_record_p67_ffn_census, "_registered", True)

    scope = os.environ.get("MATRIS_P67_FFN_FUSED_QUANT_SCOPE", "refine_line_edge_ffn").strip()
    key = (
        module_name,
        int(rows),
        int(input_dim),
        int(hidden_dim),
        int(output_dim),
        str(dtype),
        str(device),
        bool(use_fp16),
        bool(has_bias),
        bool(p67_eligible),
        reason,
    )
    record = stats.get(key)
    if record is None:
        record = {
            "module_name": module_name,
            "rows": int(rows),
            "count": 0,
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "output_dim": int(output_dim),
            "dtype": str(dtype),
            "device": str(device),
            "use_fp16": bool(use_fp16),
            "has_bias": bool(has_bias),
            "has_gather_scatter": False,
            "already_p28_input_grad_only": _use_p28_mlp_input_grad_only(),
            "p67_scope": scope,
            "p67_scope_match": _p67_ffn_scope_matches(module_name, scope),
            "p67_eligible": bool(p67_eligible),
            "reason": reason,
            "_rows_values": [],
        }
        stats[key] = record
    record["count"] += 1
    record["_rows_values"].append(int(rows))


def _use_line_edge_project_scatter_bwd(rows: int) -> bool:
    if os.environ.get("MATRIS_P32_LINE_EDGE_PROJECT_SCATTER_BWD", "0") != "1" and (
        os.environ.get("MATRIS_P36_LINE_EDGE_PROJECT_SCATTER_BWD", "0") != "1"
    ):
        return False
    min_rows = _env_int("MATRIS_P36_LINE_EDGE_PROJECT_SCATTER_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P36_LINE_EDGE_PROJECT_SCATTER_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p45_line_edge_project_scatter_tiled_bwd(rows: int) -> bool:
    if os.environ.get("MATRIS_P45_LINE_EDGE_PROJECT_SCATTER_TILED", "0") != "1":
        return False
    min_rows = _env_int("MATRIS_P45_LINE_EDGE_PROJECT_SCATTER_TILED_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P45_LINE_EDGE_PROJECT_SCATTER_TILED_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p46_line_edge_project_scatter_tile32_bwd(rows: int) -> bool:
    if os.environ.get("MATRIS_P46_LINE_EDGE_PROJECT_SCATTER_TILE32", "0") != "1":
        return False
    min_rows = _env_int("MATRIS_P46_LINE_EDGE_PROJECT_SCATTER_TILE32_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P46_LINE_EDGE_PROJECT_SCATTER_TILE32_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p58_refine_line_project_scatter_bwd(rows: int) -> bool:
    if os.environ.get("MATRIS_P58_REFINE_LINE_PROJECT_SCATTER_BWD", "0") != "1":
        return False
    min_rows = _env_int("MATRIS_P58_REFINE_LINE_PROJECT_SCATTER_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P58_REFINE_LINE_PROJECT_SCATTER_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p110_refine_atom_edge_update(rows: int) -> bool:
    if os.environ.get("MATRIS_P110_REFINE_ATOM_EDGE_UPDATE", "0") != "1":
        return False
    min_rows = _env_int("MATRIS_P110_REFINE_ATOM_EDGE_UPDATE_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P110_REFINE_ATOM_EDGE_UPDATE_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p111_refine_atom_fused_first(rows: int) -> bool:
    if os.environ.get("MATRIS_P111_REFINE_ATOM_FUSED_FIRST", "0") != "1":
        return False
    min_rows = _env_int("MATRIS_P111_REFINE_ATOM_FUSED_FIRST_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P111_REFINE_ATOM_FUSED_FIRST_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def _use_p112_refine_atom_fused_first_bwd(rows: int) -> bool:
    if os.environ.get("MATRIS_P112_REFINE_ATOM_FUSED_FIRST_BWD", "0") != "1":
        return False
    min_rows = _env_int("MATRIS_P112_REFINE_ATOM_FUSED_FIRST_BWD_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P112_REFINE_ATOM_FUSED_FIRST_BWD_MAX_ROWS", 2048)
    return min_rows <= rows <= max_rows


def _use_p48_line_edge_w8a8_macro_bwd(rows: int) -> bool:
    if os.environ.get("MATRIS_P48_LINE_EDGE_W8A8_MACRO_BWD", "0") != "1":
        return False
    min_rows = _env_int("MATRIS_P48_LINE_EDGE_W8A8_MACRO_BWD_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P48_LINE_EDGE_W8A8_MACRO_BWD_MAX_ROWS", 1000000000)
    return min_rows <= rows <= max_rows


def use_p49_line_attention_macro(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P49_LINE_ATTN_MACRO_BWD", "0") != "1":
        return False
    if ".attn_block_line_graph.edge_nonlinear_update" not in module_name:
        return False
    block_idx = _interaction_block_index_from_module(module_name)
    if block_idx is None:
        return False
    min_block = _env_int("MATRIS_P49_LINE_ATTN_MACRO_BWD_MIN_BLOCK", 0)
    max_block = _env_int("MATRIS_P49_LINE_ATTN_MACRO_BWD_MAX_BLOCK", 1000000000)
    min_rows = _env_int("MATRIS_P49_LINE_ATTN_MACRO_BWD_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P49_LINE_ATTN_MACRO_BWD_MAX_ROWS", 1000000000)
    return min_block <= block_idx <= max_block and min_rows <= rows <= max_rows


def _use_p50_line_attention_macro_alpha_fused() -> bool:
    return os.environ.get("MATRIS_P50_LINE_ATTN_MACRO_ALPHA_FUSED", "0") == "1"


def _use_p51_line_attention_alpha_cat_gemm() -> bool:
    return os.environ.get("MATRIS_P51_LINE_ATTN_ALPHA_CAT_GEMM", "0") == "1"


def _use_p52_w8a8_fused_first(module_name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P52_W8A8_FUSED_FIRST", "0") != "1":
        return False
    scope = os.environ.get("MATRIS_P52_W8A8_FUSED_FIRST_SCOPE", "line_edge").strip()
    if scope == "attn_line_edge":
        if ".attn_block_line_graph.edge_nonlinear_update" not in module_name:
            return False
    elif scope == "refine_line_edge":
        if ".refine_block_line_graph.edge_nonlinear_update" not in module_name:
            return False
    elif scope == "line_edge":
        if not (
            ".attn_block_line_graph.edge_nonlinear_update" in module_name
            or ".refine_block_line_graph.edge_nonlinear_update" in module_name
        ):
            return False
    elif scope != "all":
        return False

    block_idx = _interaction_block_index_from_module(module_name)
    min_block = _env_int("MATRIS_P52_W8A8_FUSED_FIRST_MIN_BLOCK", 0)
    max_block = _env_int("MATRIS_P52_W8A8_FUSED_FIRST_MAX_BLOCK", 1000000000)
    min_rows = _env_int("MATRIS_P52_W8A8_FUSED_FIRST_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P52_W8A8_FUSED_FIRST_MAX_ROWS", 1000000000)
    if block_idx is not None and not (min_block <= block_idx <= max_block):
        return False
    return min_rows <= rows <= max_rows


def _aggressive_bypass_project(x: Tensor, output_dim: int) -> Tensor:
    if x.shape[-1] == output_dim:
        return x
    if x.shape[-1] > output_dim:
        return x[..., :output_dim]
    pad = x.new_zeros(*x.shape[:-1], output_dim - x.shape[-1])
    return torch.cat([x, pad], dim=-1)


def is_aggressive_line_attn_target(name: str) -> bool:
    return name in ("interaction_block.8.attn_line", "interaction_block.9.attn_line")


def is_aggressive_edge_update_target(name: str) -> bool:
    return name in (
        "interaction_block.8.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.9.attn_block_line_graph.edge_nonlinear_update",
    )


def _interaction_block_index_from_module(name: str) -> int | None:
    if not name.startswith("interaction_block."):
        return None
    parts = name.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _is_p39_w8a8_true_bwd_line_edge_target(name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P39_W8A8_TRUE_BWD_LINE_EDGE", "0") != "1":
        return False
    if not name.startswith("interaction_block."):
        return False

    scope = os.environ.get("MATRIS_P39_W8A8_TRUE_BWD_SCOPE", "line_edge").strip()
    is_attn_line_edge = ".attn_block_line_graph.edge_nonlinear_update" in name
    is_refine_line_edge = ".refine_block_line_graph.edge_nonlinear_update" in name
    if scope == "attn_line_edge":
        if not is_attn_line_edge:
            return False
    elif scope == "refine_line_edge":
        if not is_refine_line_edge:
            return False
    else:
        if not (is_attn_line_edge or is_refine_line_edge):
            return False

    block_idx = _interaction_block_index_from_module(name)
    if block_idx is None:
        return False
    min_block = _env_int("MATRIS_P39_W8A8_TRUE_BWD_MIN_BLOCK", 0)
    max_block = _env_int("MATRIS_P39_W8A8_TRUE_BWD_MAX_BLOCK", 1000000000)
    min_rows = _env_int("MATRIS_P39_W8A8_TRUE_BWD_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P39_W8A8_TRUE_BWD_MAX_ROWS", 1000000000)
    return min_block <= block_idx <= max_block and min_rows <= rows <= max_rows


def _is_p87_refine_line_w8a8_saved_pre_target(name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE", "0") != "1":
        return False
    if ".refine_block_line_graph.edge_nonlinear_update" not in name:
        return False
    block_idx = _interaction_block_index_from_module(name)
    if block_idx is None:
        return False
    min_block = _env_int("MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE_MIN_BLOCK", 0)
    max_block = _env_int("MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE_MAX_BLOCK", 1000000000)
    min_rows = _env_int("MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE_MAX_ROWS", 1000000000)
    return min_block <= block_idx <= max_block and min_rows <= rows <= max_rows


def _is_p87_attn_line_w8a8_saved_pre_target(name: str, rows: int) -> bool:
    if os.environ.get("MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE", "0") != "1":
        return False
    if ".attn_block_line_graph.edge_nonlinear_update" not in name:
        return False
    block_idx = _interaction_block_index_from_module(name)
    if block_idx is None:
        return False
    min_block = _env_int("MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE_MIN_BLOCK", 8)
    max_block = _env_int("MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE_MAX_BLOCK", 9)
    min_rows = _env_int("MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE_MIN_ROWS", 0)
    max_rows = _env_int("MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE_MAX_ROWS", 1000000000)
    return min_block <= block_idx <= max_block and min_rows <= rows <= max_rows


def _load_matris_op():
    try:
        import matris_op  # type: ignore

        return matris_op
    except Exception:
        op_src = Path(__file__).resolve().parent / "op" / "src"
        if op_src.exists() and str(op_src) not in sys.path:
            sys.path.insert(0, str(op_src))
        try:
            import matris_op  # type: ignore

            return matris_op
        except Exception:
            return None


class _LineEdgeFirstProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "line_edge_gather_cat_forward"):
            raise RuntimeError("matris_op.line_edge_gather_cat_forward is unavailable")
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        x = matris_op.line_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            source_index,
            target_index,
        )
        projected = F.linear(x, weight, bias)
        ctx.save_for_backward(weight, source_index, target_index)
        ctx.node_rows = int(node_feat.shape[0])
        return projected

    @staticmethod
    def backward(ctx, grad_projected: Tensor):
        weight, source_index, target_index = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "line_edge_cat_grad_scatter_backward"):
            raise RuntimeError("matris_op.line_edge_cat_grad_scatter_backward is unavailable")
        use_project_scatter_tile32 = (
            _use_p46_line_edge_project_scatter_tile32_bwd(int(grad_projected.shape[0]))
            and hasattr(matris_op, "line_edge_project_grad_scatter_backward_tile32")
            and grad_projected.is_cuda
            and grad_projected.dtype == torch.float32
            and weight.is_cuda
            and weight.dtype == torch.float32
            and grad_projected.ndim == 2
            and weight.ndim == 2
            and weight.shape[1] == 384
            and grad_projected.shape[1] == weight.shape[0]
        )
        if use_project_scatter_tile32:
            grad_node, grad_edge = matris_op.line_edge_project_grad_scatter_backward_tile32(
                grad_projected.contiguous(),
                weight.contiguous(),
                source_index,
                target_index,
                ctx.node_rows,
            )
            return grad_node, grad_edge, None, None, None, None
        use_project_scatter_tiled = (
            _use_p45_line_edge_project_scatter_tiled_bwd(int(grad_projected.shape[0]))
            and hasattr(matris_op, "line_edge_project_grad_scatter_backward_tiled")
            and grad_projected.is_cuda
            and grad_projected.dtype == torch.float32
            and weight.is_cuda
            and weight.dtype == torch.float32
            and grad_projected.ndim == 2
            and weight.ndim == 2
            and weight.shape[1] == 384
            and grad_projected.shape[1] == weight.shape[0]
        )
        if use_project_scatter_tiled:
            grad_node, grad_edge = matris_op.line_edge_project_grad_scatter_backward_tiled(
                grad_projected.contiguous(),
                weight.contiguous(),
                source_index,
                target_index,
                ctx.node_rows,
            )
            return grad_node, grad_edge, None, None, None, None
        use_project_scatter = (
            _use_line_edge_project_scatter_bwd(int(grad_projected.shape[0]))
            and hasattr(matris_op, "line_edge_project_grad_scatter_backward")
            and grad_projected.is_cuda
            and grad_projected.dtype == torch.float32
            and weight.is_cuda
            and weight.dtype == torch.float32
            and grad_projected.ndim == 2
            and weight.ndim == 2
            and weight.shape[1] == 384
            and grad_projected.shape[1] == weight.shape[0]
        )
        if use_project_scatter:
            grad_node, grad_edge = matris_op.line_edge_project_grad_scatter_backward(
                grad_projected.contiguous(),
                weight.contiguous(),
                source_index,
                target_index,
                ctx.node_rows,
            )
        else:
            grad_cat = grad_projected.contiguous().matmul(weight)
            grad_node, grad_edge = matris_op.line_edge_cat_grad_scatter_backward(
                grad_cat.contiguous(),
                source_index,
                target_index,
                ctx.node_rows,
            )
        return grad_node, grad_edge, None, None, None, None


class _P79AttnLineGatherCat(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        source_index: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "line_edge_gather_cat_forward"):
            raise RuntimeError("matris_op.line_edge_gather_cat_forward is unavailable")
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        out = matris_op.line_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            source_index,
            target_index,
        )
        ctx.save_for_backward(source_index, target_index)
        ctx.node_rows = int(node_feat.shape[0])
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        source_index, target_index = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "line_edge_cat_grad_scatter_backward"):
            raise RuntimeError("matris_op.line_edge_cat_grad_scatter_backward is unavailable")
        grad_node, grad_edge = matris_op.line_edge_cat_grad_scatter_backward(
            grad_out.contiguous(),
            source_index,
            target_index,
            ctx.node_rows,
        )
        return grad_node, grad_edge, None, None


def p79_attn_line_gather_cat_or_none(
    node_feat: Tensor,
    edge_feat: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    profile_prefix: str,
) -> Tensor | None:
    if not use_p79_attn_line_gather_cat(profile_prefix):
        return None
    if not (
        node_feat.is_cuda
        and edge_feat.is_cuda
        and source_index.is_cuda
        and target_index.is_cuda
        and node_feat.dtype == torch.float32
        and edge_feat.dtype == torch.float32
        and source_index.dtype == torch.int64
        and target_index.dtype == torch.int64
        and node_feat.ndim == 2
        and edge_feat.ndim == 2
        and node_feat.shape[1] == 128
        and edge_feat.shape[1] == 128
        and source_index.ndim == 1
        and target_index.ndim == 1
        and source_index.shape[0] == edge_feat.shape[0]
        and target_index.shape[0] == edge_feat.shape[0]
        and (lambda op: op is not None and hasattr(op, "line_edge_gather_cat_forward"))(_load_matris_op())
        and (lambda op: op is not None and hasattr(op, "line_edge_cat_grad_scatter_backward"))(_load_matris_op())
    ):
        return None
    return _P79AttnLineGatherCat.apply(node_feat, edge_feat, source_index, target_index)


class _P80AttnLineNodeCat(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        target_feat: Tensor,
        source_feat: Tensor,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "line_node_triple_cat_forward"):
            raise RuntimeError("matris_op.line_node_triple_cat_forward is unavailable")
        return matris_op.line_node_triple_cat_forward(
            node_feat.contiguous(),
            target_feat.contiguous(),
            source_feat.contiguous(),
        )

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "line_node_triple_cat_backward"):
            raise RuntimeError("matris_op.line_node_triple_cat_backward is unavailable")
        grad_node, grad_target, grad_source = matris_op.line_node_triple_cat_backward(grad_out.contiguous())
        return grad_node, grad_target, grad_source


def p80_attn_line_node_cat_or_none(
    node_feat: Tensor,
    target_feat: Tensor,
    source_feat: Tensor,
    profile_prefix: str,
) -> Tensor | None:
    if not use_p80_attn_line_node_cat(profile_prefix):
        return None
    if not (
        node_feat.is_cuda
        and target_feat.is_cuda
        and source_feat.is_cuda
        and node_feat.dtype == torch.float32
        and target_feat.dtype == torch.float32
        and source_feat.dtype == torch.float32
        and node_feat.ndim == 2
        and target_feat.ndim == 2
        and source_feat.ndim == 2
        and node_feat.shape[1] == 128
        and target_feat.shape[1] == 128
        and source_feat.shape[1] == 128
        and node_feat.shape[0] == target_feat.shape[0]
        and node_feat.shape[0] == source_feat.shape[0]
        and (lambda op: op is not None and hasattr(op, "line_node_triple_cat_forward"))(_load_matris_op())
        and (lambda op: op is not None and hasattr(op, "line_node_triple_cat_backward"))(_load_matris_op())
    ):
        return None
    return _P80AttnLineNodeCat.apply(node_feat, target_feat, source_feat)


class _LineEdgeW8A8SecondTailInputGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        first_weight: Tensor,
        first_bias: Tensor | None,
        core_q_weight: Tensor,
        gate_q_weight: Tensor,
        core_weight_scale: Tensor,
        gate_weight_scale: Tensor,
        core_activation_scale: Tensor,
        gate_activation_scale: Tensor,
        core_bias: Tensor | None,
        gate_bias: Tensor | None,
        core_norm_weight: Tensor,
        core_norm_bias: Tensor,
        gate_norm_weight: Tensor,
        gate_norm_bias: Tensor,
        eps: float,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "line_edge_gather_cat_forward")
            and hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre")
        ):
            raise RuntimeError("p44 line-edge W8A8 big backward requires matris_op kernels")

        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        gathered = matris_op.line_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            source_index,
            target_index,
        )
        projected = F.linear(gathered, first_weight, first_bias)
        core_projected, gate_projected = projected.split([128, 128], dim=-1)
        core_second_in = F.silu(core_projected)
        gate_second_in = F.silu(gate_projected)

        core_2d = core_second_in.contiguous()
        gate_2d = gate_second_in.contiguous()
        core_weight_scale_fp32 = core_weight_scale.float().contiguous()
        gate_weight_scale_fp32 = gate_weight_scale.float().contiguous()
        core_activation_scale_fp32 = core_activation_scale.float().reshape(()).contiguous()
        gate_activation_scale_fp32 = gate_activation_scale.float().reshape(()).contiguous()
        core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
        gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
        core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core_2d.new_empty(0)
        gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate_2d.new_empty(0)
        core_norm_weight_fp32 = core_norm_weight.float().contiguous()
        core_norm_bias_fp32 = core_norm_bias.float().contiguous()
        gate_norm_weight_fp32 = gate_norm_weight.float().contiguous()
        gate_norm_bias_fp32 = gate_norm_bias.float().contiguous()
        out, core_pre, gate_pre = matris_op.quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre(
            core_2d,
            gate_2d,
            core_q_weight.contiguous(),
            gate_q_weight.contiguous(),
            core_weight_scale_fp32,
            gate_weight_scale_fp32,
            core_activation_scale_fp32,
            gate_activation_scale_fp32,
            core_bias_arg,
            gate_bias_arg,
            core_bias_fp32 is not None,
            gate_bias_fp32 is not None,
            core_norm_weight_fp32,
            core_norm_bias_fp32,
            gate_norm_weight_fp32,
            gate_norm_bias_fp32,
            float(eps),
        )
        ctx.save_for_backward(
            source_index,
            target_index,
            first_weight.float().contiguous(),
            core_projected.contiguous(),
            gate_projected.contiguous(),
            core_pre.contiguous(),
            gate_pre.contiguous(),
            core_q_weight.contiguous(),
            gate_q_weight.contiguous(),
            core_weight_scale_fp32,
            gate_weight_scale_fp32,
            core_norm_weight_fp32,
            core_norm_bias_fp32,
            gate_norm_weight_fp32,
            gate_norm_bias_fp32,
        )
        ctx.node_rows = int(node_feat.shape[0])
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            source_index,
            target_index,
            first_weight,
            core_projected,
            gate_projected,
            core_pre,
            gate_pre,
            core_q_weight,
            gate_q_weight,
            core_weight_scale,
            gate_weight_scale,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
        ) = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "line_edge_w8a8_tail_project_scatter_backward_n128"):
            raise RuntimeError("matris_op.line_edge_w8a8_tail_project_scatter_backward_n128 is unavailable")
        grad_node, grad_edge = matris_op.line_edge_w8a8_tail_project_scatter_backward_n128(
            grad_out.float().reshape(-1, 128).contiguous(),
            core_pre,
            gate_pre,
            core_projected,
            gate_projected,
            core_q_weight,
            gate_q_weight,
            core_weight_scale,
            gate_weight_scale,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
            first_weight,
            source_index,
            target_index,
            ctx.node_rows,
            ctx.eps,
        )
        return (
            grad_node,
            grad_edge,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _LineEdgeW8A8MacroTile32InputGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        first_weight: Tensor,
        first_bias: Tensor | None,
        core_q_weight: Tensor,
        gate_q_weight: Tensor,
        core_weight_scale: Tensor,
        gate_weight_scale: Tensor,
        core_activation_scale: Tensor,
        gate_activation_scale: Tensor,
        core_bias: Tensor | None,
        gate_bias: Tensor | None,
        core_norm_weight: Tensor,
        core_norm_bias: Tensor,
        gate_norm_weight: Tensor,
        gate_norm_bias: Tensor,
        eps: float,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "line_edge_gather_cat_forward")
            and hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre")
        ):
            raise RuntimeError("p48 line-edge macro backward requires matris_op forward kernels")

        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        gathered = matris_op.line_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            source_index,
            target_index,
        )
        projected = F.linear(gathered, first_weight, first_bias)
        core_projected, gate_projected = projected.split([128, 128], dim=-1)
        core_second_in = F.silu(core_projected)
        gate_second_in = F.silu(gate_projected)

        core_2d = core_second_in.contiguous()
        gate_2d = gate_second_in.contiguous()
        core_weight_scale_fp32 = core_weight_scale.float().contiguous()
        gate_weight_scale_fp32 = gate_weight_scale.float().contiguous()
        core_activation_scale_fp32 = core_activation_scale.float().reshape(()).contiguous()
        gate_activation_scale_fp32 = gate_activation_scale.float().reshape(()).contiguous()
        core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
        gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
        core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core_2d.new_empty(0)
        gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate_2d.new_empty(0)
        core_norm_weight_fp32 = core_norm_weight.float().contiguous()
        core_norm_bias_fp32 = core_norm_bias.float().contiguous()
        gate_norm_weight_fp32 = gate_norm_weight.float().contiguous()
        gate_norm_bias_fp32 = gate_norm_bias.float().contiguous()
        out, core_pre, gate_pre = matris_op.quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre(
            core_2d,
            gate_2d,
            core_q_weight.contiguous(),
            gate_q_weight.contiguous(),
            core_weight_scale_fp32,
            gate_weight_scale_fp32,
            core_activation_scale_fp32,
            gate_activation_scale_fp32,
            core_bias_arg,
            gate_bias_arg,
            core_bias_fp32 is not None,
            gate_bias_fp32 is not None,
            core_norm_weight_fp32,
            core_norm_bias_fp32,
            gate_norm_weight_fp32,
            gate_norm_bias_fp32,
            float(eps),
        )
        ctx.save_for_backward(
            source_index,
            target_index,
            first_weight.float().contiguous(),
            core_projected.contiguous(),
            gate_projected.contiguous(),
            core_pre.contiguous(),
            gate_pre.contiguous(),
            core_q_weight.contiguous(),
            gate_q_weight.contiguous(),
            core_weight_scale_fp32,
            gate_weight_scale_fp32,
            core_norm_weight_fp32,
            core_norm_bias_fp32,
            gate_norm_weight_fp32,
            gate_norm_bias_fp32,
        )
        ctx.node_rows = int(node_feat.shape[0])
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            source_index,
            target_index,
            first_weight,
            core_projected,
            gate_projected,
            core_pre,
            gate_pre,
            core_q_weight,
            gate_q_weight,
            core_weight_scale,
            gate_weight_scale,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
        ) = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128")
            and hasattr(matris_op, "line_edge_silu_project_grad_scatter_backward_tile32")
        ):
            raise RuntimeError("p48 line-edge macro backward requires matris_op backward kernels")

        grad_core, grad_gate = matris_op.w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128(
            grad_out.float().reshape(-1, 128).contiguous(),
            core_pre,
            gate_pre,
            core_q_weight,
            gate_q_weight,
            core_weight_scale,
            gate_weight_scale,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
            ctx.eps,
        )
        grad_node, grad_edge = matris_op.line_edge_silu_project_grad_scatter_backward_tile32(
            grad_core,
            grad_gate,
            core_projected,
            gate_projected,
            first_weight,
            source_index,
            target_index,
            ctx.node_rows,
        )
        return (
            grad_node,
            grad_edge,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _LineAttentionEdgeMacroInputGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        first_weight: Tensor,
        first_bias: Tensor | None,
        core_q_weight: Tensor,
        gate_q_weight: Tensor,
        core_weight_scale: Tensor,
        gate_weight_scale: Tensor,
        core_activation_scale: Tensor,
        gate_activation_scale: Tensor,
        core_bias: Tensor | None,
        gate_bias: Tensor | None,
        core_norm_weight: Tensor,
        core_norm_bias: Tensor,
        gate_norm_weight: Tensor,
        gate_norm_bias: Tensor,
        source_alpha_weight: Tensor,
        source_alpha_bias: Tensor | None,
        target_alpha_weight: Tensor,
        target_alpha_bias: Tensor | None,
        eps: float,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "line_edge_gather_cat_forward")
            and hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre")
            and hasattr(matris_op, "fused_line_attention_forward")
        ):
            raise RuntimeError("p49 line-attention macro forward requires matris_op kernels")

        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        gathered = matris_op.line_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            source_index,
            target_index,
        )
        projected = F.linear(gathered, first_weight, first_bias)
        core_projected, gate_projected = projected.split([128, 128], dim=-1)
        core_second_in = F.silu(core_projected)
        gate_second_in = F.silu(gate_projected)

        core_2d = core_second_in.contiguous()
        gate_2d = gate_second_in.contiguous()
        core_weight_scale_fp32 = core_weight_scale.float().contiguous()
        gate_weight_scale_fp32 = gate_weight_scale.float().contiguous()
        core_activation_scale_fp32 = core_activation_scale.float().reshape(()).contiguous()
        gate_activation_scale_fp32 = gate_activation_scale.float().reshape(()).contiguous()
        core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
        gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
        core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core_2d.new_empty(0)
        gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate_2d.new_empty(0)
        core_norm_weight_fp32 = core_norm_weight.float().contiguous()
        core_norm_bias_fp32 = core_norm_bias.float().contiguous()
        gate_norm_weight_fp32 = gate_norm_weight.float().contiguous()
        gate_norm_bias_fp32 = gate_norm_bias.float().contiguous()
        values, core_pre, gate_pre = matris_op.quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre(
            core_2d,
            gate_2d,
            core_q_weight.contiguous(),
            gate_q_weight.contiguous(),
            core_weight_scale_fp32,
            gate_weight_scale_fp32,
            core_activation_scale_fp32,
            gate_activation_scale_fp32,
            core_bias_arg,
            gate_bias_arg,
            core_bias_fp32 is not None,
            gate_bias_fp32 is not None,
            core_norm_weight_fp32,
            core_norm_bias_fp32,
            gate_norm_weight_fp32,
            gate_norm_bias_fp32,
            float(eps),
        )

        alpha_cat_mode = (
            _use_p51_line_attention_alpha_cat_gemm()
            and source_alpha_bias is None
            and target_alpha_bias is None
            and source_alpha_weight.shape == target_alpha_weight.shape
            and source_alpha_weight.shape == (128, 128)
            and edge_feat.shape[-1] == 128
        )
        if alpha_cat_mode:
            alpha_weight = torch.cat(
                [source_alpha_weight.float().contiguous(), target_alpha_weight.float().contiguous()],
                dim=0,
            ).contiguous()
            alpha_logits = F.linear(edge_feat.contiguous(), alpha_weight, None)
            source_logits, target_logits = alpha_logits.split([128, 128], dim=-1)
        else:
            alpha_weight = edge_feat.new_empty(0)
            source_logits = F.linear(edge_feat, source_alpha_weight, source_alpha_bias)
            target_logits = F.linear(edge_feat, target_alpha_weight, target_alpha_bias)
        source_out, target_out, source_alpha, target_alpha = matris_op.fused_line_attention_forward(
            source_logits.contiguous(),
            target_logits.contiguous(),
            values.contiguous(),
            source_index,
            target_index,
            int(num_nodes),
        )
        ctx.save_for_backward(
            source_index,
            target_index,
            first_weight.float().contiguous(),
            core_projected.contiguous(),
            gate_projected.contiguous(),
            core_pre.contiguous(),
            gate_pre.contiguous(),
            core_q_weight.contiguous(),
            gate_q_weight.contiguous(),
            core_weight_scale_fp32,
            gate_weight_scale_fp32,
            core_norm_weight_fp32,
            core_norm_bias_fp32,
            gate_norm_weight_fp32,
            gate_norm_bias_fp32,
            source_alpha_weight.float().contiguous(),
            target_alpha_weight.float().contiguous(),
            alpha_weight,
            values.contiguous(),
            source_out.contiguous(),
            target_out.contiguous(),
            source_alpha.contiguous(),
            target_alpha.contiguous(),
        )
        ctx.node_rows = int(node_feat.shape[0])
        ctx.edge_shape = tuple(edge_feat.shape)
        ctx.eps = float(eps)
        ctx.alpha_cat_mode = bool(alpha_cat_mode)
        return source_out, target_out, values

    @staticmethod
    def backward(
        ctx,
        grad_source_out: Tensor | None,
        grad_target_out: Tensor | None,
        grad_values_direct: Tensor | None,
    ):
        (
            source_index,
            target_index,
            first_weight,
            core_projected,
            gate_projected,
            core_pre,
            gate_pre,
            core_q_weight,
            gate_q_weight,
            core_weight_scale,
            gate_weight_scale,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
            source_alpha_weight,
            target_alpha_weight,
            alpha_weight,
            values,
            source_out,
            target_out,
            source_alpha,
            target_alpha,
        ) = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "fused_line_attention_backward")
            and hasattr(matris_op, "w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128")
            and hasattr(matris_op, "line_edge_silu_project_grad_scatter_backward_tile32")
        ):
            raise RuntimeError("p49 line-attention macro backward requires matris_op kernels")

        if grad_source_out is None:
            grad_source_out = source_out.new_zeros(source_out.shape)
        if grad_target_out is None:
            grad_target_out = target_out.new_zeros(target_out.shape)
        grad_source_logits, grad_target_logits, grad_values_attn = matris_op.fused_line_attention_backward(
            grad_source_out.contiguous(),
            grad_target_out.contiguous(),
            values,
            source_out,
            target_out,
            source_alpha,
            target_alpha,
            source_index,
            target_index,
        )
        if grad_values_direct is None:
            grad_values = grad_values_attn
        else:
            grad_values = grad_values_attn + grad_values_direct.contiguous()

        grad_core, grad_gate = matris_op.w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128(
            grad_values.float().reshape(-1, 128).contiguous(),
            core_pre,
            gate_pre,
            core_q_weight,
            gate_q_weight,
            core_weight_scale,
            gate_weight_scale,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
            ctx.eps,
        )
        if (
            _use_p50_line_attention_macro_alpha_fused()
            and hasattr(matris_op, "line_edge_silu_project_alpha_grad_scatter_backward_tile32")
        ):
            grad_node, grad_edge = matris_op.line_edge_silu_project_alpha_grad_scatter_backward_tile32(
                grad_core,
                grad_gate,
                core_projected,
                gate_projected,
                first_weight,
                grad_source_logits.contiguous(),
                grad_target_logits.contiguous(),
                source_alpha_weight,
                target_alpha_weight,
                source_index,
                target_index,
                ctx.node_rows,
            )
        else:
            grad_node, grad_edge_update = matris_op.line_edge_silu_project_grad_scatter_backward_tile32(
                grad_core,
                grad_gate,
                core_projected,
                gate_projected,
                first_weight,
                source_index,
                target_index,
                ctx.node_rows,
            )
            if ctx.alpha_cat_mode and alpha_weight.numel() != 0:
                grad_alpha_logits = torch.cat(
                    [grad_source_logits.contiguous(), grad_target_logits.contiguous()],
                    dim=-1,
                ).contiguous()
                grad_edge_alpha = grad_alpha_logits.matmul(alpha_weight)
            else:
                grad_edge_alpha = (
                    grad_source_logits.contiguous().matmul(source_alpha_weight)
                    + grad_target_logits.contiguous().matmul(target_alpha_weight)
                )
            grad_edge = grad_edge_update + grad_edge_alpha
        return (
            grad_node,
            grad_edge,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _DirectedEdgeFirstProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        edge_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed_edge_gather_cat_forward"):
            raise RuntimeError("matris_op.directed_edge_gather_cat_forward is unavailable")
        edge_index = edge_index.contiguous()
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        x = matris_op.directed_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            edge_index,
            source_index,
            target_index,
        )
        projected = F.linear(x, weight, bias)
        ctx.save_for_backward(weight, edge_index, source_index, target_index)
        ctx.node_rows = int(node_feat.shape[0])
        ctx.edge_rows = int(edge_feat.shape[0])
        return projected

    @staticmethod
    def backward(ctx, grad_projected: Tensor):
        weight, edge_index, source_index, target_index = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed_edge_cat_grad_scatter_backward"):
            raise RuntimeError("matris_op.directed_edge_cat_grad_scatter_backward is unavailable")
        grad_cat = grad_projected.contiguous().matmul(weight)
        grad_node, grad_edge = matris_op.directed_edge_cat_grad_scatter_backward(
            grad_cat.contiguous(),
            edge_index,
            source_index,
            target_index,
            ctx.node_rows,
            ctx.edge_rows,
        )
        return grad_node, grad_edge, None, None, None, None, None


class _RefineLineEdgeFirstProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        atom_feat: Tensor,
        atom_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "refine_line_edge_gather_cat_forward"):
            raise RuntimeError("matris_op.refine_line_edge_gather_cat_forward is unavailable")
        atom_index = atom_index.contiguous()
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        x = matris_op.refine_line_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            atom_feat.contiguous(),
            atom_index,
            source_index,
            target_index,
        )
        projected = F.linear(x, weight, bias)
        ctx.save_for_backward(weight, atom_index, source_index, target_index)
        ctx.node_rows = int(node_feat.shape[0])
        ctx.edge_rows = int(edge_feat.shape[0])
        ctx.atom_rows = int(atom_feat.shape[0])
        return projected

    @staticmethod
    def backward(ctx, grad_projected: Tensor):
        weight, atom_index, source_index, target_index = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "refine_line_edge_cat_grad_scatter_backward"):
            raise RuntimeError("matris_op.refine_line_edge_cat_grad_scatter_backward is unavailable")
        use_project_scatter_tile32 = (
            _use_p58_refine_line_project_scatter_bwd(int(grad_projected.shape[0]))
            and hasattr(matris_op, "refine_line_project_grad_scatter_backward_tile32")
            and grad_projected.is_cuda
            and grad_projected.dtype == torch.float32
            and weight.is_cuda
            and weight.dtype == torch.float32
            and grad_projected.ndim == 2
            and weight.ndim == 2
            and weight.shape[1] == 512
            and grad_projected.shape[1] == weight.shape[0]
        )
        if use_project_scatter_tile32:
            grad_node, grad_edge, grad_atom = matris_op.refine_line_project_grad_scatter_backward_tile32(
                grad_projected.contiguous(),
                weight.contiguous(),
                atom_index,
                source_index,
                target_index,
                ctx.node_rows,
                ctx.edge_rows,
                ctx.atom_rows,
            )
        else:
            grad_cat = grad_projected.contiguous().matmul(weight)
            grad_node, grad_edge, grad_atom = matris_op.refine_line_edge_cat_grad_scatter_backward(
                grad_cat.contiguous(),
                atom_index,
                source_index,
                target_index,
                ctx.node_rows,
                ctx.edge_rows,
                ctx.atom_rows,
            )
        return grad_node, grad_edge, grad_atom, None, None, None, None, None


class _RefineAtomEdgeFirstProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        edge_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed_edge_gather_cat_forward"):
            raise RuntimeError("matris_op.directed_edge_gather_cat_forward is unavailable")
        edge_index = edge_index.contiguous()
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        x = matris_op.directed_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            edge_index,
            source_index,
            target_index,
        )
        projected = F.linear(x, weight, bias)
        ctx.save_for_backward(weight, edge_index, source_index, target_index)
        ctx.node_rows = int(node_feat.shape[0])
        ctx.edge_rows = int(edge_feat.shape[0])
        return projected

    @staticmethod
    def backward(ctx, grad_projected: Tensor):
        weight, edge_index, source_index, target_index = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed_edge_cat_grad_scatter_backward"):
            raise RuntimeError("matris_op.directed_edge_cat_grad_scatter_backward is unavailable")
        grad_cat = grad_projected.contiguous().matmul(weight)
        grad_node, grad_edge = matris_op.directed_edge_cat_grad_scatter_backward(
            grad_cat.contiguous(),
            edge_index,
            source_index,
            target_index,
            ctx.node_rows,
            ctx.edge_rows,
        )
        return grad_node, grad_edge, None, None, None, None, None


class _P111RefineAtomFirstSilu(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        edge_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed_edge_gather_cat_forward"):
            raise RuntimeError("matris_op.directed_edge_gather_cat_forward is unavailable")
        edge_index = edge_index.contiguous()
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        x = matris_op.directed_edge_gather_cat_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            edge_index,
            source_index,
            target_index,
        )
        projected = F.linear(x, weight, bias)
        core_raw, gate_raw = projected.split([128, 128], dim=-1)
        core = F.silu(core_raw)
        gate = F.silu(gate_raw)
        ctx.save_for_backward(core_raw, gate_raw, weight.contiguous(), edge_index, source_index, target_index)
        ctx.node_rows = int(node_feat.shape[0])
        ctx.edge_rows = int(edge_feat.shape[0])
        return core, gate

    @staticmethod
    def backward(ctx, grad_core: Tensor | None, grad_gate: Tensor | None):
        core_raw, gate_raw, weight, edge_index, source_index, target_index = ctx.saved_tensors
        if grad_core is None:
            grad_core = core_raw.new_zeros(core_raw.shape)
        if grad_gate is None:
            grad_gate = gate_raw.new_zeros(gate_raw.shape)
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed_edge_cat_grad_scatter_backward"):
            raise RuntimeError("matris_op.directed_edge_cat_grad_scatter_backward is unavailable")
        rows = int(core_raw.shape[0])
        if (
            _use_p112_refine_atom_fused_first_bwd(rows)
            and hasattr(matris_op, "directed_edge_silu_project_grad_scatter_backward_tile32")
        ):
            grad_node, grad_edge = matris_op.directed_edge_silu_project_grad_scatter_backward_tile32(
                grad_core.contiguous(),
                grad_gate.contiguous(),
                core_raw,
                gate_raw,
                weight,
                edge_index,
                source_index,
                target_index,
                ctx.node_rows,
                ctx.edge_rows,
            )
            return grad_node, grad_edge, None, None, None, None, None
        grad_projected = torch.cat(
            [
                grad_core.float() * torch.sigmoid(core_raw) * (1.0 + core_raw * (1.0 - torch.sigmoid(core_raw))),
                grad_gate.float() * torch.sigmoid(gate_raw) * (1.0 + gate_raw * (1.0 - torch.sigmoid(gate_raw))),
            ],
            dim=-1,
        )
        grad_cat = grad_projected.contiguous().matmul(weight)
        grad_node, grad_edge = matris_op.directed_edge_cat_grad_scatter_backward(
            grad_cat.contiguous(),
            edge_index,
            source_index,
            target_index,
            ctx.node_rows,
            ctx.edge_rows,
        )
        return grad_node, grad_edge, None, None, None, None, None


class _P60RefineLineFirstSilu(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        atom_feat: Tensor,
        atom_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "refine_line_first_silu_forward"):
            raise RuntimeError("matris_op.refine_line_first_silu_forward is unavailable")
        bias_arg = bias.contiguous() if bias is not None else weight.new_empty(0)
        core, gate, core_raw, gate_raw = matris_op.refine_line_first_silu_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            atom_feat.contiguous(),
            atom_index.contiguous(),
            source_index.contiguous(),
            target_index.contiguous(),
            weight.contiguous(),
            bias_arg,
            bias is not None,
        )
        ctx.save_for_backward(
            core_raw,
            gate_raw,
            weight.contiguous(),
            atom_index.contiguous(),
            source_index.contiguous(),
            target_index.contiguous(),
        )
        ctx.node_rows = int(node_feat.shape[0])
        ctx.edge_rows = int(edge_feat.shape[0])
        ctx.atom_rows = int(atom_feat.shape[0])
        return core, gate

    @staticmethod
    def backward(ctx, grad_core: Tensor | None, grad_gate: Tensor | None):
        core_raw, gate_raw, weight, atom_index, source_index, target_index = ctx.saved_tensors
        if grad_core is None:
            grad_core = core_raw.new_zeros(core_raw.shape)
        if grad_gate is None:
            grad_gate = gate_raw.new_zeros(gate_raw.shape)
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "refine_line_first_silu_backward"):
            raise RuntimeError("matris_op.refine_line_first_silu_backward is unavailable")
        grad_node, grad_edge, grad_atom = matris_op.refine_line_first_silu_backward(
            grad_core.float().contiguous(),
            grad_gate.float().contiguous(),
            core_raw,
            gate_raw,
            weight,
            atom_index,
            source_index,
            target_index,
            ctx.node_rows,
            ctx.edge_rows,
            ctx.atom_rows,
        )
        return grad_node, grad_edge, grad_atom, None, None, None, None, None


class _P63RefineLineEdgeSmoothReduce(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_feat: Tensor,
        edge_feat: Tensor,
        atom_feat: Tensor,
        base_envelope: Tensor,
        atom_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        first_weight: Tensor,
        first_bias: Tensor | None,
        core_q_weight: Tensor,
        gate_q_weight: Tensor,
        core_weight_scale: Tensor,
        gate_weight_scale: Tensor,
        core_activation_scale: Tensor,
        gate_activation_scale: Tensor,
        core_bias: Tensor | None,
        gate_bias: Tensor | None,
        core_norm_weight: Tensor,
        core_norm_bias: Tensor,
        gate_norm_weight: Tensor,
        gate_norm_bias: Tensor,
        eps: float,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor]:
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "refine_line_first_silu_forward")
            and hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre")
            and hasattr(matris_op, "refine_line_smooth_reduce_forward")
        ):
            raise RuntimeError("p63 refine-line edge+smooth forward requires matris_op kernels")

        atom_index = atom_index.contiguous()
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        first_bias_arg = first_bias.contiguous() if first_bias is not None else first_weight.new_empty(0)
        if (
            os.environ.get("MATRIS_P69_REFINE_LINE_FIRST_TAIL_SMOOTH_REDUCE", "0") == "1"
            and hasattr(matris_op, "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre")
        ):
            core_weight_scale_fp32 = core_weight_scale.float().contiguous()
            gate_weight_scale_fp32 = gate_weight_scale.float().contiguous()
            core_activation_scale_fp32 = core_activation_scale.float().reshape(()).contiguous()
            gate_activation_scale_fp32 = gate_activation_scale.float().reshape(()).contiguous()
            core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
            gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
            core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else first_weight.new_empty(0)
            gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else first_weight.new_empty(0)
            core_norm_weight_fp32 = core_norm_weight.float().contiguous()
            core_norm_bias_fp32 = core_norm_bias.float().contiguous()
            gate_norm_weight_fp32 = gate_norm_weight.float().contiguous()
            gate_norm_bias_fp32 = gate_norm_bias.float().contiguous()
            backend = os.environ.get("MATRIS_W8A8_BACKEND", "")
            use_parallel_tail = backend == "cuda_wmma_tail_n128_parallel"
            if backend == "cuda_wmma_tail_n128_auto":
                threshold = _env_int(
                    "MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_ROWS",
                    _env_int("MATRIS_W8A8_TAIL_N128_AUTO_THRESHOLD", 4096),
                )
                when = os.environ.get("MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_WHEN", "")
                if when == "lt":
                    use_parallel_tail = edge_feat.shape[0] < threshold
                elif when == "always":
                    use_parallel_tail = True
                elif when == "never":
                    use_parallel_tail = False
                else:
                    use_parallel_tail = edge_feat.shape[0] >= threshold
            refine_node, nonlinear, core_raw, gate_raw, core_pre, gate_pre = (
                matris_op.refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre(
                    node_feat.contiguous(),
                    edge_feat.contiguous(),
                    atom_feat.contiguous(),
                    base_envelope.contiguous(),
                    atom_index,
                    source_index,
                    target_index,
                    first_weight.contiguous(),
                    first_bias_arg,
                    first_bias is not None,
                    core_q_weight.contiguous(),
                    gate_q_weight.contiguous(),
                    core_weight_scale_fp32,
                    gate_weight_scale_fp32,
                    core_activation_scale_fp32,
                    gate_activation_scale_fp32,
                    core_bias_arg,
                    gate_bias_arg,
                    core_bias_fp32 is not None,
                    gate_bias_fp32 is not None,
                    core_norm_weight_fp32,
                    core_norm_bias_fp32,
                    gate_norm_weight_fp32,
                    gate_norm_bias_fp32,
                    float(eps),
                    bool(use_parallel_tail),
                    int(num_nodes),
                )
            )
            ctx.save_for_backward(
                nonlinear.contiguous(),
                base_envelope.contiguous(),
                atom_index,
                source_index,
                target_index,
                first_weight.float().contiguous(),
                core_raw.contiguous(),
                gate_raw.contiguous(),
                core_pre.contiguous(),
                gate_pre.contiguous(),
                core_q_weight.contiguous(),
                gate_q_weight.contiguous(),
                core_weight_scale_fp32,
                gate_weight_scale_fp32,
                core_norm_weight_fp32,
                core_norm_bias_fp32,
                gate_norm_weight_fp32,
                gate_norm_bias_fp32,
            )
            ctx.node_rows = int(node_feat.shape[0])
            ctx.edge_rows = int(edge_feat.shape[0])
            ctx.atom_rows = int(atom_feat.shape[0])
            ctx.eps = float(eps)
            return refine_node, nonlinear
        if (
            os.environ.get("MATRIS_P66_REFINE_LINE_FIRST_TAIL_SAVED_FORWARD", "0") == "1"
            and hasattr(matris_op, "refine_line_first_tail_w8a8_forward_with_pre")
        ):
            core_weight_scale_fp32 = core_weight_scale.float().contiguous()
            gate_weight_scale_fp32 = gate_weight_scale.float().contiguous()
            core_activation_scale_fp32 = core_activation_scale.float().reshape(()).contiguous()
            gate_activation_scale_fp32 = gate_activation_scale.float().reshape(()).contiguous()
            core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
            gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
            core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else first_weight.new_empty(0)
            gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else first_weight.new_empty(0)
            core_norm_weight_fp32 = core_norm_weight.float().contiguous()
            core_norm_bias_fp32 = core_norm_bias.float().contiguous()
            gate_norm_weight_fp32 = gate_norm_weight.float().contiguous()
            gate_norm_bias_fp32 = gate_norm_bias.float().contiguous()
            backend = os.environ.get("MATRIS_W8A8_BACKEND", "")
            use_parallel_tail = backend == "cuda_wmma_tail_n128_parallel"
            if backend == "cuda_wmma_tail_n128_auto":
                threshold = _env_int(
                    "MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_ROWS",
                    _env_int("MATRIS_W8A8_TAIL_N128_AUTO_THRESHOLD", 4096),
                )
                when = os.environ.get("MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_WHEN", "")
                if when == "lt":
                    use_parallel_tail = edge_feat.shape[0] < threshold
                elif when == "always":
                    use_parallel_tail = True
                elif when == "never":
                    use_parallel_tail = False
                else:
                    use_parallel_tail = edge_feat.shape[0] >= threshold
            nonlinear, core_raw, gate_raw, core_pre, gate_pre = matris_op.refine_line_first_tail_w8a8_forward_with_pre(
                node_feat.contiguous(),
                edge_feat.contiguous(),
                atom_feat.contiguous(),
                atom_index,
                source_index,
                target_index,
                first_weight.contiguous(),
                first_bias_arg,
                first_bias is not None,
                core_q_weight.contiguous(),
                gate_q_weight.contiguous(),
                core_weight_scale_fp32,
                gate_weight_scale_fp32,
                core_activation_scale_fp32,
                gate_activation_scale_fp32,
                core_bias_arg,
                gate_bias_arg,
                core_bias_fp32 is not None,
                gate_bias_fp32 is not None,
                core_norm_weight_fp32,
                core_norm_bias_fp32,
                gate_norm_weight_fp32,
                gate_norm_bias_fp32,
                float(eps),
                bool(use_parallel_tail),
            )
            refine_node = matris_op.refine_line_smooth_reduce_forward(
                nonlinear.contiguous(),
                base_envelope.contiguous(),
                source_index,
                target_index,
                int(num_nodes),
            )
            ctx.save_for_backward(
                nonlinear.contiguous(),
                base_envelope.contiguous(),
                atom_index,
                source_index,
                target_index,
                first_weight.float().contiguous(),
                core_raw.contiguous(),
                gate_raw.contiguous(),
                core_pre.contiguous(),
                gate_pre.contiguous(),
                core_q_weight.contiguous(),
                gate_q_weight.contiguous(),
                core_weight_scale_fp32,
                gate_weight_scale_fp32,
                core_norm_weight_fp32,
                core_norm_bias_fp32,
                gate_norm_weight_fp32,
                gate_norm_bias_fp32,
            )
            ctx.node_rows = int(node_feat.shape[0])
            ctx.edge_rows = int(edge_feat.shape[0])
            ctx.atom_rows = int(atom_feat.shape[0])
            ctx.eps = float(eps)
            return refine_node, nonlinear

        core, gate, core_raw, gate_raw = matris_op.refine_line_first_silu_forward(
            node_feat.contiguous(),
            edge_feat.contiguous(),
            atom_feat.contiguous(),
            atom_index,
            source_index,
            target_index,
            first_weight.contiguous(),
            first_bias_arg,
            first_bias is not None,
        )

        core_weight_scale_fp32 = core_weight_scale.float().contiguous()
        gate_weight_scale_fp32 = gate_weight_scale.float().contiguous()
        core_activation_scale_fp32 = core_activation_scale.float().reshape(()).contiguous()
        gate_activation_scale_fp32 = gate_activation_scale.float().reshape(()).contiguous()
        core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
        gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
        core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core.new_empty(0)
        gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate.new_empty(0)
        core_norm_weight_fp32 = core_norm_weight.float().contiguous()
        core_norm_bias_fp32 = core_norm_bias.float().contiguous()
        gate_norm_weight_fp32 = gate_norm_weight.float().contiguous()
        gate_norm_bias_fp32 = gate_norm_bias.float().contiguous()
        nonlinear, core_pre, gate_pre = matris_op.quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre(
            core.contiguous(),
            gate.contiguous(),
            core_q_weight.contiguous(),
            gate_q_weight.contiguous(),
            core_weight_scale_fp32,
            gate_weight_scale_fp32,
            core_activation_scale_fp32,
            gate_activation_scale_fp32,
            core_bias_arg,
            gate_bias_arg,
            core_bias_fp32 is not None,
            gate_bias_fp32 is not None,
            core_norm_weight_fp32,
            core_norm_bias_fp32,
            gate_norm_weight_fp32,
            gate_norm_bias_fp32,
            float(eps),
        )
        refine_node = matris_op.refine_line_smooth_reduce_forward(
            nonlinear.contiguous(),
            base_envelope.contiguous(),
            source_index,
            target_index,
            int(num_nodes),
        )

        ctx.save_for_backward(
            nonlinear.contiguous(),
            base_envelope.contiguous(),
            atom_index,
            source_index,
            target_index,
            first_weight.float().contiguous(),
            core_raw.contiguous(),
            gate_raw.contiguous(),
            core_pre.contiguous(),
            gate_pre.contiguous(),
            core_q_weight.contiguous(),
            gate_q_weight.contiguous(),
            core_weight_scale_fp32,
            gate_weight_scale_fp32,
            core_norm_weight_fp32,
            core_norm_bias_fp32,
            gate_norm_weight_fp32,
            gate_norm_bias_fp32,
        )
        ctx.node_rows = int(node_feat.shape[0])
        ctx.edge_rows = int(edge_feat.shape[0])
        ctx.atom_rows = int(atom_feat.shape[0])
        ctx.eps = float(eps)
        return refine_node, nonlinear

    @staticmethod
    def backward(
        ctx,
        grad_refine_node: Tensor | None,
        grad_nonlinear_direct: Tensor | None,
    ):
        (
            nonlinear,
            base_envelope,
            atom_index,
            source_index,
            target_index,
            first_weight,
            core_raw,
            gate_raw,
            core_pre,
            gate_pre,
            core_q_weight,
            gate_q_weight,
            core_weight_scale,
            gate_weight_scale,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
        ) = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "refine_line_smooth_reduce_backward")
            and hasattr(matris_op, "w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128")
            and hasattr(matris_op, "refine_line_first_silu_backward")
        ):
            raise RuntimeError("p63 refine-line edge+smooth backward requires matris_op kernels")

        if grad_refine_node is None:
            grad_refine_node = base_envelope.new_zeros(base_envelope.shape)
        if os.environ.get("MATRIS_P65B_REFINE_LINE_PACKED_TILED_BWD", "0") == "1":
            has_direct_grad = grad_nonlinear_direct is not None
            grad_direct_arg = (
                grad_nonlinear_direct.float().contiguous()
                if has_direct_grad
                else nonlinear.new_empty((0,), dtype=torch.float32)
            )
            if not (
                hasattr(matris_op, "refine_line_smooth_w8a8_tail_actgrad_backward_n128")
                and hasattr(matris_op, "refine_line_first_silu_backward_packed")
            ):
                raise RuntimeError("p65b refine-line packed tiled backward requires matris_op kernels")
            grad_act, grad_base = matris_op.refine_line_smooth_w8a8_tail_actgrad_backward_n128(
                grad_refine_node.float().contiguous(),
                grad_direct_arg,
                nonlinear,
                base_envelope,
                source_index,
                target_index,
                core_pre,
                gate_pre,
                core_q_weight,
                gate_q_weight,
                core_weight_scale,
                gate_weight_scale,
                core_norm_weight,
                core_norm_bias,
                gate_norm_weight,
                gate_norm_bias,
                bool(has_direct_grad),
                ctx.eps,
            )
            grad_node, grad_edge, grad_atom = matris_op.refine_line_first_silu_backward_packed(
                grad_act.float().contiguous(),
                core_raw,
                gate_raw,
                first_weight,
                atom_index,
                source_index,
                target_index,
                ctx.node_rows,
                ctx.edge_rows,
                ctx.atom_rows,
            )
            return (
                grad_node,
                grad_edge,
                grad_atom,
                grad_base,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if os.environ.get("MATRIS_P65_REFINE_LINE_TILED_FUSED_BACKWARD", "0") == "1":
            has_direct_grad = grad_nonlinear_direct is not None
            grad_direct_arg = (
                grad_nonlinear_direct.float().contiguous()
                if has_direct_grad
                else nonlinear.new_empty((0,), dtype=torch.float32)
            )
            if not hasattr(matris_op, "refine_line_smooth_tail_first_silu_backward_tile"):
                raise RuntimeError("p65 refine-line tiled fused backward requires matris_op kernel")
            tile_rows = _env_int("MATRIS_P65_REFINE_LINE_TILED_FUSED_BACKWARD_TILE_ROWS", 32)
            grad_node, grad_edge, grad_atom, grad_base = matris_op.refine_line_smooth_tail_first_silu_backward_tile(
                grad_refine_node.float().contiguous(),
                grad_direct_arg,
                nonlinear,
                base_envelope,
                atom_index,
                source_index,
                target_index,
                first_weight,
                core_raw,
                gate_raw,
                core_pre,
                gate_pre,
                core_q_weight,
                gate_q_weight,
                core_weight_scale,
                gate_weight_scale,
                core_norm_weight,
                core_norm_bias,
                gate_norm_weight,
                gate_norm_bias,
                ctx.node_rows,
                ctx.edge_rows,
                ctx.atom_rows,
                bool(has_direct_grad),
                ctx.eps,
                int(tile_rows),
            )
            return (
                grad_node,
                grad_edge,
                grad_atom,
                grad_base,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if os.environ.get("MATRIS_P64_REFINE_LINE_FUSED_BACKWARD", "0") == "1":
            has_direct_grad = grad_nonlinear_direct is not None
            grad_direct_arg = (
                grad_nonlinear_direct.float().contiguous()
                if has_direct_grad
                else nonlinear.new_empty((0,), dtype=torch.float32)
            )
            p64_variant = os.environ.get("MATRIS_P64_REFINE_LINE_FUSED_BACKWARD_VARIANT", "split_tail")
            if p64_variant != "monolithic" and hasattr(
                matris_op, "refine_line_smooth_w8a8_tail_input_grad_backward_n128"
            ):
                grad_core, grad_gate, grad_base = matris_op.refine_line_smooth_w8a8_tail_input_grad_backward_n128(
                    grad_refine_node.float().contiguous(),
                    grad_direct_arg,
                    nonlinear,
                    base_envelope,
                    source_index,
                    target_index,
                    core_pre,
                    gate_pre,
                    core_q_weight,
                    gate_q_weight,
                    core_weight_scale,
                    gate_weight_scale,
                    core_norm_weight,
                    core_norm_bias,
                    gate_norm_weight,
                    gate_norm_bias,
                    bool(has_direct_grad),
                    ctx.eps,
                )
                grad_node, grad_edge, grad_atom = matris_op.refine_line_first_silu_backward(
                    grad_core.float().contiguous(),
                    grad_gate.float().contiguous(),
                    core_raw,
                    gate_raw,
                    first_weight,
                    atom_index,
                    source_index,
                    target_index,
                    ctx.node_rows,
                    ctx.edge_rows,
                    ctx.atom_rows,
                )
            else:
                if not hasattr(matris_op, "refine_line_edge_smooth_w8a8_backward_n128"):
                    raise RuntimeError("p64 refine-line fused backward requires matris_op kernel")
                grad_node, grad_edge, grad_atom, grad_base = matris_op.refine_line_edge_smooth_w8a8_backward_n128(
                    grad_refine_node.float().contiguous(),
                    grad_direct_arg,
                    nonlinear,
                    base_envelope,
                    atom_index,
                    source_index,
                    target_index,
                    first_weight,
                    core_raw,
                    gate_raw,
                    core_pre,
                    gate_pre,
                    core_q_weight,
                    gate_q_weight,
                    core_weight_scale,
                    gate_weight_scale,
                    core_norm_weight,
                    core_norm_bias,
                    gate_norm_weight,
                    gate_norm_bias,
                    ctx.node_rows,
                    ctx.edge_rows,
                    ctx.atom_rows,
                    bool(has_direct_grad),
                    ctx.eps,
                )
            return (
                grad_node,
                grad_edge,
                grad_atom,
                grad_base,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        grad_nonlinear, grad_base = matris_op.refine_line_smooth_reduce_backward(
            grad_refine_node.float().contiguous(),
            nonlinear,
            base_envelope,
            source_index,
            target_index,
        )
        if grad_nonlinear_direct is not None:
            grad_nonlinear = grad_nonlinear + grad_nonlinear_direct.float().contiguous()

        grad_core, grad_gate = matris_op.w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128(
            grad_nonlinear.float().reshape(-1, 128).contiguous(),
            core_pre,
            gate_pre,
            core_q_weight,
            gate_q_weight,
            core_weight_scale,
            gate_weight_scale,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
            ctx.eps,
        )
        grad_node, grad_edge, grad_atom = matris_op.refine_line_first_silu_backward(
            grad_core.float().contiguous(),
            grad_gate.float().contiguous(),
            core_raw,
            gate_raw,
            first_weight,
            atom_index,
            source_index,
            target_index,
            ctx.node_rows,
            ctx.edge_rows,
            ctx.atom_rows,
        )
        return (
            grad_node,
            grad_edge,
            grad_atom,
            grad_base,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _LinearInputGradOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(weight)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (weight,) = ctx.saved_tensors
        grad_x = grad_out.contiguous().matmul(weight)
        return grad_x, None, None


def linear_input_grad_only(x: Tensor, linear: nn.Linear) -> Tensor:
    return _LinearInputGradOnly.apply(x, linear.weight, linear.bias)


class _DualLinearInputGradOnly(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight_a: Tensor,
        bias_a: Tensor | None,
        weight_b: Tensor,
        bias_b: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        weight = torch.cat([weight_a, weight_b], dim=0).contiguous()
        bias = None
        if bias_a is not None or bias_b is not None:
            if bias_a is None:
                bias_a = weight.new_zeros(weight_a.shape[0])
            if bias_b is None:
                bias_b = weight.new_zeros(weight_b.shape[0])
            bias = torch.cat([bias_a, bias_b], dim=0).contiguous()
        out = F.linear(x, weight, bias)
        ctx.save_for_backward(weight)
        ctx.split = int(weight_a.shape[0])
        return out[:, : ctx.split], out[:, ctx.split :]

    @staticmethod
    def backward(ctx, grad_a: Tensor | None, grad_b: Tensor | None):
        (weight,) = ctx.saved_tensors
        if grad_a is None and grad_b is None:
            return None, None, None, None, None
        if grad_a is None:
            grad_a = weight.new_zeros(grad_b.shape[0], ctx.split)
        if grad_b is None:
            grad_b = weight.new_zeros(grad_a.shape[0], weight.shape[0] - ctx.split)
        grad = torch.cat([grad_a, grad_b], dim=-1)
        grad_x = grad.contiguous().matmul(weight)
        return grad_x, None, None, None, None


def dual_linear_input_grad_only(
    x: Tensor,
    linear_a: nn.Linear,
    linear_b: nn.Linear,
) -> tuple[Tensor, Tensor]:
    return _DualLinearInputGradOnly.apply(
        x,
        linear_a.weight,
        linear_a.bias,
        linear_b.weight,
        linear_b.bias,
    )


class _TritonSortedSegmentSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feas: Tensor, bin_count: Tensor) -> Tensor:
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        if feas.ndim != 2:
            raise RuntimeError("sorted segment softmax expects a 2D tensor")
        if not feas.is_cuda or not bin_count.is_cuda:
            raise RuntimeError("sorted segment softmax expects CUDA tensors")
        num_rows, dim = feas.shape
        if dim != 128:
            raise RuntimeError("sorted segment softmax currently specializes dim=128")

        lengths = bin_count.to(device=feas.device, dtype=torch.int64).contiguous()
        offsets = torch.cumsum(lengths, dim=0) - lengths
        max_len = int(lengths.max().item()) if lengths.numel() else 1
        block_r = _next_power_of_2(max_len)
        if block_r > 128:
            raise RuntimeError("sorted segment softmax currently supports max segment length <= 128")
        block_d = 32
        out = torch.empty_like(feas)
        grid = (lengths.numel(), triton.cdiv(dim, block_d))
        _segment_softmax_sorted_forward_kernel[grid](
            feas,
            offsets,
            lengths,
            out,
            lengths.numel(),
            dim,
            block_r,
            block_d,
            num_warps=4,
        )
        ctx.save_for_backward(out, offsets, lengths)
        ctx.dim = dim
        ctx.block_r = block_r
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        out, offsets, lengths = ctx.saved_tensors
        grad_x = torch.empty_like(grad_out)
        grid = (lengths.numel(), triton.cdiv(ctx.dim, ctx.block_d))
        _segment_softmax_sorted_backward_kernel[grid](
            grad_out.contiguous(),
            out,
            offsets,
            lengths,
            grad_x,
            lengths.numel(),
            ctx.dim,
            ctx.block_r,
            ctx.block_d,
            num_warps=4,
        )
        return grad_x, None


class _TritonSegmentSoftmaxWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_logits: Tensor, values: Tensor, lengths: Tensor) -> Tensor:
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        if alpha_logits.ndim != 2 or values.ndim != 2 or alpha_logits.shape != values.shape:
            raise RuntimeError("weighted segment softmax expects matching 2D tensors")
        if not alpha_logits.is_cuda or not values.is_cuda or not lengths.is_cuda:
            raise RuntimeError("weighted segment softmax expects CUDA tensors")
        _, dim = alpha_logits.shape
        if dim != 128:
            raise RuntimeError("weighted segment softmax currently specializes dim=128")

        lengths = lengths.to(device=alpha_logits.device, dtype=torch.int64).contiguous()
        offsets = torch.cumsum(lengths, dim=0) - lengths
        max_len = int(lengths.max().item()) if lengths.numel() else 1
        block_r = _next_power_of_2(max_len)
        if block_r > 128:
            raise RuntimeError("weighted segment softmax currently supports max segment length <= 128")
        block_d = 32
        alpha = torch.empty_like(alpha_logits)
        out = torch.empty((lengths.numel(), dim), device=alpha_logits.device, dtype=alpha_logits.dtype)
        grid = (lengths.numel(), triton.cdiv(dim, block_d))
        _segment_softmax_weighted_sum_forward_kernel[grid](
            alpha_logits,
            values,
            offsets,
            lengths,
            alpha,
            out,
            lengths.numel(),
            dim,
            block_r,
            block_d,
            num_warps=4,
        )
        ctx.save_for_backward(values, alpha, out, offsets, lengths)
        ctx.dim = dim
        ctx.block_r = block_r
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        values, alpha, out, offsets, lengths = ctx.saved_tensors
        grad_alpha_logits = torch.empty_like(alpha)
        grad_values = torch.empty_like(values)
        grid = (lengths.numel(), triton.cdiv(ctx.dim, ctx.block_d))
        _segment_softmax_weighted_sum_backward_kernel[grid](
            grad_out.contiguous(),
            values,
            alpha,
            out,
            offsets,
            lengths,
            grad_alpha_logits,
            grad_values,
            lengths.numel(),
            ctx.dim,
            ctx.block_r,
            ctx.block_d,
            num_warps=4,
        )
        return grad_alpha_logits, grad_values, None


class _CudaTargetAttentionSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_logits: Tensor, values: Tensor, lengths: Tensor) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "target_attention_sum_forward"):
            raise RuntimeError("matris_op.target_attention_sum_forward is unavailable")
        out, alpha = matris_op.target_attention_sum_forward(alpha_logits.contiguous(), values.contiguous(), lengths)
        ctx.save_for_backward(values, out, alpha, lengths)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        values, out, alpha, lengths = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "target_attention_sum_backward"):
            raise RuntimeError("matris_op.target_attention_sum_backward is unavailable")
        grad_logits, grad_values = matris_op.target_attention_sum_backward(
            grad_out.contiguous(),
            values,
            out,
            alpha,
            lengths,
        )
        return grad_logits, grad_values, None


class _CudaTargetAttentionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_logits: Tensor, values: Tensor, segment: Tensor, lengths: Tensor) -> Tensor:
        alpha_logits = alpha_logits.contiguous()
        values = values.contiguous()
        alpha = Dimwise_softmax(alpha_logits, segment)
        safe_lengths = torch.where(lengths == 0, torch.ones_like(lengths), lengths)
        out = aggregate(alpha * values, segment, bin_count=safe_lengths, average=False, num_segment=lengths.numel())
        ctx.save_for_backward(values, out, alpha, lengths)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        values, out, alpha, lengths = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "target_attention_sum_backward"):
            raise RuntimeError("matris_op.target_attention_sum_backward is unavailable")
        grad_logits, grad_values = matris_op.target_attention_sum_backward(
            grad_out.contiguous(),
            values,
            out,
            alpha,
            lengths,
        )
        return grad_logits, grad_values, None, None


class _CudaDirected2UndirectedAverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data: Tensor, segment: Tensor, num_segment: int) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed2undirected_average_forward"):
            raise RuntimeError("matris_op.directed2undirected_average_forward is unavailable")
        segment = segment.contiguous()
        out = matris_op.directed2undirected_average_forward(data.contiguous(), segment, int(num_segment))
        ctx.save_for_backward(segment)
        ctx.rows = int(data.shape[0])
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (segment,) = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed2undirected_average_backward"):
            raise RuntimeError("matris_op.directed2undirected_average_backward is unavailable")
        grad_data = matris_op.directed2undirected_average_backward(grad_out.contiguous(), segment, ctx.rows)
        return grad_data, None, None


def directed2undirected_average_or_none(
    data: Tensor,
    segment: Tensor,
    num_segment,
    enable_hint: bool,
) -> Tensor | None:
    if not _use_cuda_directed2undirected_average() or not enable_hint:
        return None
    if data.ndim != 2 or data.shape[1] != 128 or data.dtype != torch.float32 or not data.is_cuda:
        return None
    if segment.ndim != 1 or segment.numel() != data.shape[0] or segment.dtype != torch.int64 or not segment.is_cuda:
        return None
    resolved_num_segment = int(num_segment) if num_segment is not None else int(segment.max().item()) + 1
    if data.shape[0] != resolved_num_segment * 2:
        return None
    try:
        return _CudaDirected2UndirectedAverage.apply(data, segment, resolved_num_segment)
    except RuntimeError:
        return None


class _CudaEdgeVectors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords: Tensor, lattice: Tensor, image: Tensor, target: Tensor, source: Tensor) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "edge_vectors_forward"):
            raise RuntimeError("matris_op.edge_vectors_forward is unavailable")
        image = image.contiguous()
        target = target.contiguous()
        source = source.contiguous()
        out = matris_op.edge_vectors_forward(coords.contiguous(), lattice.contiguous(), image, target, source)
        ctx.save_for_backward(image, target, source)
        ctx.num_coords = int(coords.shape[0])
        ctx.lattice_rows = int(lattice.shape[0])
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        image, target, source = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "edge_vectors_backward"):
            raise RuntimeError("matris_op.edge_vectors_backward is unavailable")
        grad_coords, grad_lattice = matris_op.edge_vectors_backward(
            grad_out.contiguous(),
            image,
            target,
            source,
            ctx.num_coords,
            ctx.lattice_rows,
        )
        return grad_coords, grad_lattice, None, None, None


def edge_vectors_or_none(
    coords: Tensor,
    lattice: Tensor,
    image: Tensor,
    target: Tensor,
    source: Tensor,
) -> Tensor | None:
    if not use_cuda_edge_vectors():
        return None
    if (
        coords.ndim != 2
        or lattice.ndim != 2
        or image.ndim != 2
        or coords.shape[1] != 3
        or lattice.shape[1] != 3
        or image.shape[1] != lattice.shape[0]
        or coords.dtype != torch.float32
        or lattice.dtype != torch.float32
        or image.dtype != torch.float32
        or target.dtype != torch.int64
        or source.dtype != torch.int64
        or not coords.is_cuda
        or not lattice.is_cuda
        or not image.is_cuda
        or not target.is_cuda
        or not source.is_cuda
    ):
        return None
    try:
        return _CudaEdgeVectors.apply(coords, lattice, image, target, source)
    except RuntimeError:
        return None


class _CudaFusedLineAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source_logits: Tensor,
        target_logits: Tensor,
        values: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        num_segments: int,
        atom_graph: bool,
        target_offsets: Tensor | None,
    ):
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "fused_line_attention_forward"):
            raise RuntimeError("matris_op.fused_line_attention_forward is unavailable")
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        def run_forward():
            source_logits_c = source_logits.contiguous()
            target_logits_c = target_logits.contiguous()
            values_c = values.contiguous()
            if (
                not bool(atom_graph)
                and _use_p83b_fused_line_attention_target_offsets()
                and isinstance(target_offsets, Tensor)
                and target_offsets.is_cuda
                and target_offsets.dtype == torch.int64
                and target_offsets.ndim == 1
                and target_offsets.numel() == int(num_segments) + 1
                and hasattr(matris_op, "fused_line_attention_forward_target_offsets")
            ):
                return matris_op.fused_line_attention_forward_target_offsets(
                    source_logits_c,
                    target_logits_c,
                    values_c,
                    source_index,
                    target_index,
                    target_offsets.contiguous(),
                    int(num_segments),
                )
            if (
                not bool(atom_graph)
                and _use_p82c_fused_line_attention_target_segment_max()
                and source_logits_c.shape[0] >= _p82c_line_attention_min_rows()
                and hasattr(matris_op, "fused_line_attention_single_max_atomic")
                and hasattr(matris_op, "fused_line_attention_forward_with_max")
            ):
                source_max = matris_op.fused_line_attention_single_max_atomic(
                    source_logits_c,
                    source_index,
                    int(num_segments),
                )
                target_lengths = torch.bincount(target_index, minlength=int(num_segments))
                target_max = torch.segment_reduce(
                    target_logits_c,
                    reduce="max",
                    lengths=target_lengths,
                    axis=0,
                )
                return matris_op.fused_line_attention_forward_with_max(
                    source_logits_c,
                    target_logits_c,
                    values_c,
                    source_index,
                    target_index,
                    source_max,
                    target_max,
                    int(num_segments),
                )
            forward_op = matris_op.fused_line_attention_forward
            if (
                not bool(atom_graph)
                and _use_p82_fused_line_attention_forward_v2()
                and hasattr(matris_op, "fused_line_attention_forward_v2")
            ):
                forward_op = matris_op.fused_line_attention_forward_v2
            return forward_op(
                source_logits_c,
                target_logits_c,
                values_c,
                source_index,
                target_index,
                int(num_segments),
            )
        ctx.p81_attention_kind = "atom" if bool(atom_graph) else "line"
        if use_p81_attn_reduce_detail_profile():
            with torch.profiler.record_function(f"p81.fused_line_attention.{ctx.p81_attention_kind}.forward"):
                source_out, target_out, source_alpha, target_alpha = run_forward()
        else:
            source_out, target_out, source_alpha, target_alpha = run_forward()
        ctx.save_for_backward(values, source_out, target_out, source_alpha, target_alpha, source_index, target_index)
        return source_out, target_out

    @staticmethod
    def backward(ctx, grad_source_out: Tensor, grad_target_out: Tensor):
        values, source_out, target_out, source_alpha, target_alpha, source_index, target_index = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "fused_line_attention_backward"):
            raise RuntimeError("matris_op.fused_line_attention_backward is unavailable")
        def run_backward():
            return matris_op.fused_line_attention_backward(
                grad_source_out.contiguous(),
                grad_target_out.contiguous(),
                values,
                source_out,
                target_out,
                source_alpha,
                target_alpha,
                source_index,
                target_index,
            )
        if use_p81_attn_reduce_detail_profile():
            kind = getattr(ctx, "p81_attention_kind", "unknown")
            with torch.profiler.record_function(f"p81.fused_line_attention.{kind}.backward"):
                grad_source_logits, grad_target_logits, grad_values = run_backward()
        else:
            grad_source_logits, grad_target_logits, grad_values = run_backward()
        return grad_source_logits, grad_target_logits, grad_values, None, None, None, None, None


class _CudaFusedLineAttentionNodeInput(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source_logits: Tensor,
        target_logits: Tensor,
        values: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        target_offsets: Tensor,
        num_segments: int,
        node_feat: Tensor,
    ):
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "fused_line_attention_node_input_forward_target_offsets"):
            raise RuntimeError("matris_op.fused_line_attention_node_input_forward_target_offsets is unavailable")
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        target_offsets = target_offsets.contiguous()

        def run_forward():
            return matris_op.fused_line_attention_node_input_forward_target_offsets(
                source_logits.contiguous(),
                target_logits.contiguous(),
                values.contiguous(),
                source_index,
                target_index,
                target_offsets,
                node_feat.contiguous(),
                int(num_segments),
            )

        if use_p81_attn_reduce_detail_profile():
            with torch.profiler.record_function("p83c.fused_line_attention_node_input.line.forward"):
                fusion_node_feat, source_alpha, target_alpha = run_forward()
        else:
            fusion_node_feat, source_alpha, target_alpha = run_forward()
        ctx.save_for_backward(values, fusion_node_feat, source_alpha, target_alpha, source_index, target_index)
        return fusion_node_feat

    @staticmethod
    def backward(ctx, grad_fusion_node_feat: Tensor):
        values, fusion_node_feat, source_alpha, target_alpha, source_index, target_index = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "fused_line_attention_backward"):
            raise RuntimeError("matris_op.fused_line_attention_backward is unavailable")
        grad_fusion = grad_fusion_node_feat.contiguous()
        grad_node_feat = grad_fusion[:, :128].contiguous()
        grad_target_out = grad_fusion[:, 128:256].contiguous()
        grad_source_out = grad_fusion[:, 256:384].contiguous()
        target_out = fusion_node_feat[:, 128:256].contiguous()
        source_out = fusion_node_feat[:, 256:384].contiguous()

        def run_backward():
            return matris_op.fused_line_attention_backward(
                grad_source_out,
                grad_target_out,
                values,
                source_out,
                target_out,
                source_alpha,
                target_alpha,
                source_index,
                target_index,
            )

        if use_p81_attn_reduce_detail_profile():
            with torch.profiler.record_function("p83c.fused_line_attention_node_input.line.backward"):
                grad_source_logits, grad_target_logits, grad_values = run_backward()
        else:
            grad_source_logits, grad_target_logits, grad_values = run_backward()
        return grad_source_logits, grad_target_logits, grad_values, None, None, None, None, grad_node_feat


def fused_line_attention_or_none(
    source_logits: Tensor,
    target_logits: Tensor,
    values: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    num_segments: int,
    *,
    enable_hint: bool,
    atom_graph: bool = False,
    target_offsets: Tensor | None = None,
) -> tuple[Tensor, Tensor] | None:
    enabled = _use_cuda_fused_atom_attention() if atom_graph else _use_cuda_fused_line_attention()
    if not enabled or not enable_hint:
        return None
    if (
        source_logits.ndim != 2
        or source_logits.shape != target_logits.shape
        or source_logits.shape != values.shape
        or source_logits.shape[1] != 128
        or source_logits.dtype != torch.float32
        or target_logits.dtype != torch.float32
        or values.dtype != torch.float32
        or source_index.dtype != torch.int64
        or target_index.dtype != torch.int64
        or (target_offsets is not None and target_offsets.dtype != torch.int64)
        or not source_logits.is_cuda
        or not target_logits.is_cuda
        or not values.is_cuda
        or not source_index.is_cuda
        or not target_index.is_cuda
        or (target_offsets is not None and not target_offsets.is_cuda)
    ):
        return None
    try:
        source_out, target_out = _CudaFusedLineAttention.apply(
            source_logits,
            target_logits,
            values,
            source_index,
            target_index,
            int(num_segments),
            bool(atom_graph),
            target_offsets,
        )
        return source_out, target_out
    except RuntimeError:
        return None


def fused_line_attention_node_input_or_none(
    source_logits: Tensor,
    target_logits: Tensor,
    values: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    target_offsets: Tensor | None,
    num_segments: int,
    node_feat: Tensor,
    *,
    enable_hint: bool,
) -> Tensor | None:
    if not _use_p83c_fused_line_attention_node_input() or not _use_cuda_fused_line_attention() or not enable_hint:
        return None
    if (
        target_offsets is None
        or source_logits.ndim != 2
        or source_logits.shape != target_logits.shape
        or source_logits.shape != values.shape
        or source_logits.shape[1] != 128
        or node_feat.ndim != 2
        or node_feat.shape[0] != int(num_segments)
        or node_feat.shape[1] != 128
        or source_logits.dtype != torch.float32
        or target_logits.dtype != torch.float32
        or values.dtype != torch.float32
        or node_feat.dtype != torch.float32
        or source_index.dtype != torch.int64
        or target_index.dtype != torch.int64
        or target_offsets.dtype != torch.int64
        or target_offsets.ndim != 1
        or target_offsets.numel() != int(num_segments) + 1
        or not source_logits.is_cuda
        or not target_logits.is_cuda
        or not values.is_cuda
        or not node_feat.is_cuda
        or not source_index.is_cuda
        or not target_index.is_cuda
        or not target_offsets.is_cuda
    ):
        return None
    try:
        return _CudaFusedLineAttentionNodeInput.apply(
            source_logits,
            target_logits,
            values,
            source_index,
            target_index,
            target_offsets,
            int(num_segments),
            node_feat,
        )
    except RuntimeError:
        return None


def _triton_sorted_segment_softmax_or_none(
    feas: Tensor,
    segment: Tensor,
    num_segment,
    bin_count: Tensor | None,
) -> Tensor | None:
    if not _use_triton_sorted_segment_softmax() or bin_count is None or triton is None:
        return None
    if feas.ndim != 2 or feas.shape[1] != 128 or not feas.is_cuda:
        return None
    if segment.numel() > 1 and not bool(torch.all(segment[1:] >= segment[:-1]).item()):
        return None
    resolved_num_segment = int(num_segment) if num_segment is not None else int(segment.max().item()) + 1
    true_bin_count = torch.bincount(segment, minlength=resolved_num_segment).to(torch.int64)
    try:
        return _TritonSortedSegmentSoftmax.apply(feas, true_bin_count)
    except RuntimeError:
        return None


def _sorted_segment_reduce_or_none(
    data: Tensor,
    segment: Tensor,
    num_segment,
    average: bool,
    enable_hint: bool,
) -> Tensor | None:
    if not _use_torch_sorted_segment_reduce() or not enable_hint:
        return None
    if data.ndim != 2 or not data.is_cuda or segment.numel() == 0:
        return None
    if segment.numel() > 1 and not bool(torch.all(segment[1:] >= segment[:-1]).item()):
        return None
    resolved_num_segment = int(num_segment) if num_segment is not None else int(segment.max().item()) + 1
    lengths = torch.bincount(segment, minlength=resolved_num_segment).to(torch.int64)
    out = torch.segment_reduce(data, "sum", lengths=lengths)
    if average:
        safe_lengths = lengths.where(lengths != 0, lengths.new_ones(1)).to(data.dtype)
        out = out / safe_lengths.reshape(-1, 1)
    return out


def segment_softmax_weighted_sum_sorted_or_none(
    alpha_logits: Tensor,
    values: Tensor,
    segment: Tensor,
    num_segment,
    enable_hint: bool,
) -> Tensor | None:
    if not _use_triton_target_attention_sum() or not enable_hint or triton is None:
        if not (_use_cuda_target_attention_sum() or _use_cuda_target_attention_bwd()) or not enable_hint:
            return None
    if alpha_logits.ndim != 2 or alpha_logits.shape != values.shape or alpha_logits.shape[1] != 128:
        return None
    if not alpha_logits.is_cuda or segment.numel() == 0:
        return None
    if segment.numel() > 1 and not bool(torch.all(segment[1:] >= segment[:-1]).item()):
        return None
    resolved_num_segment = int(num_segment) if num_segment is not None else int(segment.max().item()) + 1
    lengths = torch.bincount(segment, minlength=resolved_num_segment).to(torch.int64)
    if _use_cuda_target_attention_sum():
        try:
            return _CudaTargetAttentionSum.apply(alpha_logits, values, lengths)
        except RuntimeError:
            return None
    if _use_cuda_target_attention_bwd():
        try:
            return _CudaTargetAttentionBackward.apply(alpha_logits, values, segment, lengths)
        except RuntimeError:
            return None
    if not _use_triton_target_attention_sum() or triton is None:
        return None
    try:
        return _TritonSegmentSoftmaxWeightedSum.apply(alpha_logits, values, lengths)
    except RuntimeError:
        return None

class SwishLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        bias: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            bias: Whether to use bias in the linear layer. Default: True.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.act = get_activation("silu")
    
    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas: shape (feas_num, in_dim)
            
        Returns:
            output: shape (feas_num, out_dim)
        """
        return self.act(self.linear(feas))

def Dimwise_softmax(
    feas: Tensor,
    segment: Tensor,
    num_segment=None,
    profile_name: str | None = None,
    bin_count: Tensor | None = None,
) -> Tensor:
    """Computes a sparsely evaluated softmax.
    
    Args:
        feas: The source tensor. shape: [num, dim]
        segment: specify the segment of each row [num, 1] 
    """
    triton_result = _triton_sorted_segment_softmax_or_none(feas, segment, num_segment, bin_count)
    if triton_result is not None:
        return triton_result

    profile_handle = _profiled_cuda_start(feas)
    num, dim = feas.shape
    original_dtype = feas.dtype
    reduce_dtype = torch.float32 if original_dtype in (torch.float16, torch.bfloat16) else original_dtype
    feas_for_reduce = feas.to(reduce_dtype)
    if num_segment is None:
        num_segment = int(segment.max()) + 1
    
    segment_expanded = segment.unsqueeze(1).expand(-1, dim) # [num, dim]
    
    feas_max = torch.empty(num_segment, dim, dtype=reduce_dtype, device=feas.device)
    feas_max.fill_(float("-inf"))
    feas_max = feas_max.scatter_reduce(
        0, segment_expanded, feas_for_reduce, reduce='amax', include_self=False,
    ) #[num_segment, dim]
    # Gather: [num_segment, dim] -> [num, dim]
    feas_max = feas_max[segment]
    out = (feas_for_reduce - feas_max).exp()
    
    # =========== scatter sum ============
    out_sum = torch.zeros(num_segment, dim, device=feas.device, dtype=reduce_dtype)
    out_sum = out_sum.scatter_reduce(
        0, segment_expanded, out, reduce='sum', include_self=False
    )
    # Gather: [num_segment, dim] -> [num, dim]
    out_sum = out_sum[segment]
    score = out / out_sum
    result = score.to(original_dtype)
    _profiled_cuda_finish(
        profile_handle,
        op="Dimwise_softmax",
        name=profile_name,
        data=feas,
        segment=segment,
        num_segment=num_segment,
        average=None,
    )
    return result

def aggregate(data: torch.Tensor, 
              segment: torch.Tensor, 
              bin_count: torch.Tensor = None, 
              average=True, 
              num_segment=None,
              profile_name: str | None = None) -> torch.Tensor:
    """Aggregate rows in data by specifying the segment.

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        segment (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows, if False, sum the rows.
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """
    segment_reduce_output = _sorted_segment_reduce_or_none(
        data,
        segment,
        num_segment,
        average,
        enable_hint=bin_count is not None or average,
    )
    if segment_reduce_output is not None:
        return segment_reduce_output

    profile_handle = _profiled_cuda_start(data)
    if bin_count is None:
        bin_count = torch.bincount(segment)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

    if (num_segment is not None) and (bin_count.shape[0] != num_segment):
        difference = num_segment - bin_count.shape[0]
        bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])
    # make sure this operation is done on the same device of data and owners
    output = data.new_zeros([bin_count.shape[0], data.shape[1]])
    output = output.index_add_(0, segment, data)
    if average:
        output = (output.T / bin_count).T
    _profiled_cuda_finish(
        profile_handle,
        op="aggregate",
        name=profile_name,
        data=data,
        segment=segment,
        num_segment=num_segment,
        average=average,
    )
    return output


class MLP(nn.Module):
        
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MLP layer.
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Number of hidden units. Can be an integer for a single
                hidden layer, a sequence of integers for multiple hidden layers,
                or None for no hidden layers. Default: (128, 128).
            output_dim: Dimension of output predictions. Default: 128.
            dropout: Dropout rate applied before each linear layer. Default: 0.0.
            activation: Activation function. Supported: "relu", "silu", "tanh", "gelu".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Dropout rate must be in [0.0, 1.0), got {dropout}")
        
        self.use_fp16 = use_fp16
        self.output_dim = output_dim
        activation_func = get_activation(activation)

        layers = []
        if hidden_dim in (None, 0):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        elif isinstance(hidden_dim, int):
            # Single hidden layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim, bias=bias),
                activation_func,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim, bias=bias),
            ])
        elif isinstance(hidden_dim, Sequence):
            # Multiple hidden layers
            layers.extend([
                nn.Linear(input_dim, hidden_dim[0], bias=bias),
                activation_func,
            ])
            # Additional hidden layers
            for i in range(len(hidden_dim) - 1):
                layers.extend([
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=bias),
                    activation_func,
                ])
            # Output layer
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim[-1], output_dim, bias=bias)])
        else:
            raise TypeError(
                f"hidden_dim must be an integer, sequence of integers, or None, "
                f"got {type(hidden_dim).__name__}"
            )
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, feas: Tensor) -> Tensor:
        """
            Args:
                feas: Input tensor of shape (features, input_dim)
            Returns:
                Output tensor of shape (features, output_dim)
        """
        if _use_aggressive_broad_bypass_mlp() and self.training is False and torch.is_grad_enabled():
            return _aggressive_bypass_project(feas, self.output_dim)
        if self.use_fp16 and feas.is_cuda:
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                out = self._forward_layers(feas)
            out = out.to(torch.float32)
        else:
            out = self._forward_layers(feas) 
        return out

    def _forward_layers(self, feas: Tensor) -> Tensor:
        if not (
            _use_aggressive_broad_input_grad_only()
            and self.training is False
            and torch.is_grad_enabled()
            and feas.is_cuda
        ):
            return self.layers(feas)
        out = feas
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                out = linear_input_grad_only(out, layer)
            else:
                out = layer(out)
        return out


class _TwoLinearSiluInputGradOnly(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight1: Tensor,
        bias1: Tensor | None,
        weight2: Tensor,
        bias2: Tensor | None,
    ) -> Tensor:
        hidden = F.linear(x, weight1, bias1)
        activated = F.silu(hidden)
        out = F.linear(activated, weight2, bias2)
        ctx.save_for_backward(weight1, weight2, hidden)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        weight1, weight2, hidden = ctx.saved_tensors
        matris_op = _load_matris_op()
        if (
            _use_p29_mlp_bwd_kernel()
            and matris_op is not None
            and hasattr(matris_op, "two_linear_silu_input_grad_backward_n128")
            and grad_out.is_cuda
            and weight1.is_cuda
            and weight2.is_cuda
            and hidden.is_cuda
            and grad_out.dtype == torch.float32
            and weight1.dtype == torch.float32
            and weight2.dtype == torch.float32
            and hidden.dtype == torch.float32
            and grad_out.is_contiguous()
            and weight1.is_contiguous()
            and weight2.is_contiguous()
            and hidden.is_contiguous()
            and grad_out.ndim == 2
            and hidden.shape == grad_out.shape
            and grad_out.shape[1] == 128
            and weight1.shape == (128, 128)
            and weight2.shape == (128, 128)
        ):
            grad_x = matris_op.two_linear_silu_input_grad_backward_n128(
                grad_out,
                weight2,
                hidden,
                weight1,
            )
            return grad_x, None, None, None, None
        grad_activated = grad_out.contiguous().matmul(weight2)
        if (
            matris_op is not None
            and hasattr(matris_op, "fuse_silu_bwd")
            and grad_activated.is_cuda
            and hidden.is_cuda
            and grad_activated.dtype == torch.float32
            and hidden.dtype == torch.float32
            and grad_activated.is_contiguous()
            and hidden.is_contiguous()
            and hidden.numel() % 4 == 0
        ):
            grad_hidden = matris_op.fuse_silu_bwd(grad_activated, hidden)
        else:
            sigmoid_hidden = torch.sigmoid(hidden)
            silu_grad = sigmoid_hidden * (1.0 + hidden * (1.0 - sigmoid_hidden))
            grad_hidden = grad_activated * silu_grad
        grad_x = grad_hidden.contiguous().matmul(weight1)
        return grad_x, None, None, None, None


def _p67_quantize_weight_per_channel(weight: Tensor) -> tuple[Tensor, Tensor]:
    weight_fp32 = weight.detach().float().contiguous()
    scale = weight_fp32.abs().amax(dim=1).clamp_min(1.0e-8).div(127.0).contiguous()
    q_weight = torch.round(weight_fp32 / scale.reshape(-1, 1)).clamp(-127, 127).to(torch.int8).contiguous()
    return q_weight, scale


def _p67_activation_scale(x: Tensor, env_name: str, default_value: float) -> Tensor:
    mode = os.environ.get("MATRIS_P67_FFN_FUSED_QUANT_ACT_SCALE_MODE", "dynamic").strip()
    if mode == "static":
        try:
            value = float(os.environ.get(env_name, str(default_value)))
        except ValueError:
            value = default_value
        return x.new_tensor(max(value, 1.0e-8), dtype=torch.float32)
    return x.detach().abs().amax().clamp_min(1.0e-8).div(127.0).float().reshape(())


def _p67_quant_linear_forward(
    matris_op,
    x: Tensor,
    q_weight: Tensor,
    weight_scale: Tensor,
    activation_scale: Tensor,
    bias: Tensor | None,
) -> Tensor:
    backend = os.environ.get("MATRIS_P67_FFN_FUSED_QUANT_BACKEND", "wmma").strip()
    if backend == "cutlass" and hasattr(matris_op, "quant_linear_w8a8_static_cutlass"):
        op = matris_op.quant_linear_w8a8_static_cutlass
    else:
        op = matris_op.quant_linear_w8a8_static_wmma
    bias_arg = bias.contiguous() if bias is not None else x.new_empty(0)
    return op(
        x.contiguous(),
        q_weight.contiguous(),
        weight_scale.float().contiguous(),
        activation_scale.reshape(()).contiguous(),
        bias_arg,
        bias is not None,
    )


class _P67FFNFusedQuantInputGradOnly(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight1: Tensor,
        bias1: Tensor | None,
        weight2: Tensor,
        bias2: Tensor | None,
        q_weight1: Tensor,
        weight_scale1: Tensor,
        q_weight2: Tensor,
        weight_scale2: Tensor,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_wmma"):
            raise RuntimeError("P67 requires matris_op.quant_linear_w8a8_static_wmma")

        x_2d = x.contiguous()
        activation_scale1 = _p67_activation_scale(
            x_2d,
            "MATRIS_P67_FFN_FUSED_QUANT_INPUT_SCALE",
            0.05,
        )
        hidden = _p67_quant_linear_forward(
            matris_op,
            x_2d,
            q_weight1,
            weight_scale1,
            activation_scale1,
            bias1,
        )
        hidden_act = F.silu(hidden)
        activation_scale2 = _p67_activation_scale(
            hidden_act,
            "MATRIS_P67_FFN_FUSED_QUANT_HIDDEN_SCALE",
            0.05,
        )
        out = _p67_quant_linear_forward(
            matris_op,
            hidden_act,
            q_weight2,
            weight_scale2,
            activation_scale2,
            bias2,
        )
        ctx.save_for_backward(weight1, weight2, hidden)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        weight1, weight2, hidden = ctx.saved_tensors
        matris_op = _load_matris_op()
        if (
            _use_p29_mlp_bwd_kernel()
            and matris_op is not None
            and hasattr(matris_op, "two_linear_silu_input_grad_backward_n128")
            and grad_out.is_cuda
            and weight1.is_cuda
            and weight2.is_cuda
            and hidden.is_cuda
            and grad_out.dtype == torch.float32
            and weight1.dtype == torch.float32
            and weight2.dtype == torch.float32
            and hidden.dtype == torch.float32
            and grad_out.is_contiguous()
            and weight1.is_contiguous()
            and weight2.is_contiguous()
            and hidden.is_contiguous()
            and grad_out.ndim == 2
            and hidden.shape == grad_out.shape
            and grad_out.shape[1] == 128
            and weight1.shape == (128, 128)
            and weight2.shape == (128, 128)
        ):
            grad_x = matris_op.two_linear_silu_input_grad_backward_n128(
                grad_out,
                weight2,
                hidden,
                weight1,
            )
            return grad_x, None, None, None, None, None, None, None, None
        grad_activated = grad_out.contiguous().matmul(weight2)
        if (
            matris_op is not None
            and hasattr(matris_op, "fuse_silu_bwd")
            and grad_activated.is_cuda
            and hidden.is_cuda
            and grad_activated.dtype == torch.float32
            and hidden.dtype == torch.float32
            and grad_activated.is_contiguous()
            and hidden.is_contiguous()
            and hidden.numel() % 4 == 0
        ):
            grad_hidden = matris_op.fuse_silu_bwd(grad_activated, hidden)
        else:
            sigmoid_hidden = torch.sigmoid(hidden)
            silu_grad = sigmoid_hidden * (1.0 + hidden * (1.0 - sigmoid_hidden))
            grad_hidden = grad_activated * silu_grad
        grad_x = grad_hidden.contiguous().matmul(weight1)
        return grad_x, None, None, None, None, None, None, None, None


def _p68_grouped_ffn_scope_matches(profile_prefix: str) -> bool:
    scope_value = os.environ.get("MATRIS_P68_GROUPED_FFN_PAIR_SCOPE", "refine_line").strip()
    scopes = [item.strip() for item in scope_value.split(",") if item.strip()]
    if not scopes:
        scopes = ["refine_line"]
    for scope in scopes:
        if scope == "refine_line" and profile_prefix.endswith(".refine_line"):
            return True
        if scope == "refine_atom" and profile_prefix.endswith(".refine_atom"):
            return True
        if scope == "all_refine" and (
            profile_prefix.endswith(".refine_line") or profile_prefix.endswith(".refine_atom")
        ):
            return True
        if scope == "all":
            return True
    return False


def _p68_activation_scale(x: Tensor, env_name: str, default_value: float) -> Tensor:
    mode = os.environ.get("MATRIS_P68_GROUPED_FFN_ACT_SCALE_MODE", "dynamic").strip()
    if mode == "static":
        try:
            value = float(os.environ.get(env_name, str(default_value)))
        except ValueError:
            value = default_value
        return x.new_tensor(max(value, 1.0e-8), dtype=torch.float32)
    return x.detach().abs().amax().clamp_min(1.0e-8).div(127.0).float().reshape(())


def _p68_grouped_pair_linear_forward(
    matris_op,
    x_a: Tensor,
    x_b: Tensor,
    q_weight_a: Tensor,
    q_weight_b: Tensor,
    weight_scale_a: Tensor,
    weight_scale_b: Tensor,
    activation_scale_a: Tensor,
    activation_scale_b: Tensor,
    bias_a: Tensor | None,
    bias_b: Tensor | None,
) -> tuple[Tensor, Tensor]:
    bias_a_arg = bias_a.contiguous() if bias_a is not None else x_a.new_empty(0)
    bias_b_arg = bias_b.contiguous() if bias_b is not None else x_b.new_empty(0)
    out = matris_op.quant_linear_w8a8_static_cutlass_grouped_pair(
        x_a.contiguous(),
        x_b.contiguous(),
        q_weight_a.contiguous(),
        q_weight_b.contiguous(),
        weight_scale_a.float().contiguous(),
        weight_scale_b.float().contiguous(),
        activation_scale_a.reshape(()).contiguous(),
        activation_scale_b.reshape(()).contiguous(),
        bias_a_arg,
        bias_b_arg,
        bias_a is not None,
        bias_b is not None,
    )
    return out[0], out[1]


def _p68_grouped_pair_ffn_forward(
    matris_op,
    x_a: Tensor,
    x_b: Tensor,
    q_weight1_a: Tensor,
    q_weight1_b: Tensor,
    weight_scale1_a: Tensor,
    weight_scale1_b: Tensor,
    activation_scale1_a: Tensor,
    activation_scale1_b: Tensor,
    bias1_a: Tensor | None,
    bias1_b: Tensor | None,
    q_weight2_a: Tensor,
    q_weight2_b: Tensor,
    weight_scale2_a: Tensor,
    weight_scale2_b: Tensor,
    activation_scale2_a: Tensor,
    activation_scale2_b: Tensor,
    bias2_a: Tensor | None,
    bias2_b: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    bias1_a_arg = bias1_a.contiguous() if bias1_a is not None else x_a.new_empty(0)
    bias1_b_arg = bias1_b.contiguous() if bias1_b is not None else x_b.new_empty(0)
    bias2_a_arg = bias2_a.contiguous() if bias2_a is not None else x_a.new_empty(0)
    bias2_b_arg = bias2_b.contiguous() if bias2_b is not None else x_b.new_empty(0)
    out = matris_op.quant_ffn_w8a8_static_cutlass_grouped_pair(
        x_a.contiguous(),
        x_b.contiguous(),
        q_weight1_a.contiguous(),
        q_weight1_b.contiguous(),
        weight_scale1_a.float().contiguous(),
        weight_scale1_b.float().contiguous(),
        activation_scale1_a.reshape(()).contiguous(),
        activation_scale1_b.reshape(()).contiguous(),
        bias1_a_arg,
        bias1_b_arg,
        bias1_a is not None,
        bias1_b is not None,
        q_weight2_a.contiguous(),
        q_weight2_b.contiguous(),
        weight_scale2_a.float().contiguous(),
        weight_scale2_b.float().contiguous(),
        activation_scale2_a.reshape(()).contiguous(),
        activation_scale2_b.reshape(()).contiguous(),
        bias2_a_arg,
        bias2_b_arg,
        bias2_a is not None,
        bias2_b is not None,
    )
    return out[0], out[1], out[2], out[3]


def _p68_two_linear_input_grad(
    grad_out: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    hidden: Tensor,
) -> Tensor:
    matris_op = _load_matris_op()
    grad_out_c = grad_out.contiguous()
    if (
        _use_p29_mlp_bwd_kernel()
        and matris_op is not None
        and hasattr(matris_op, "two_linear_silu_input_grad_backward_n128")
        and grad_out_c.is_cuda
        and weight1.is_cuda
        and weight2.is_cuda
        and hidden.is_cuda
        and grad_out_c.dtype == torch.float32
        and weight1.dtype == torch.float32
        and weight2.dtype == torch.float32
        and hidden.dtype == torch.float32
        and weight1.is_contiguous()
        and weight2.is_contiguous()
        and hidden.is_contiguous()
        and grad_out_c.ndim == 2
        and hidden.shape == grad_out_c.shape
        and grad_out_c.shape[1] == 128
        and weight1.shape == (128, 128)
        and weight2.shape == (128, 128)
    ):
        return matris_op.two_linear_silu_input_grad_backward_n128(
            grad_out_c,
            weight2,
            hidden,
            weight1,
        )
    grad_activated = grad_out_c.matmul(weight2)
    if (
        matris_op is not None
        and hasattr(matris_op, "fuse_silu_bwd")
        and grad_activated.is_cuda
        and hidden.is_cuda
        and grad_activated.dtype == torch.float32
        and hidden.dtype == torch.float32
        and grad_activated.is_contiguous()
        and hidden.is_contiguous()
        and hidden.numel() % 4 == 0
    ):
        grad_hidden = matris_op.fuse_silu_bwd(grad_activated, hidden)
    else:
        sigmoid_hidden = torch.sigmoid(hidden)
        silu_grad = sigmoid_hidden * (1.0 + hidden * (1.0 - sigmoid_hidden))
        grad_hidden = grad_activated * silu_grad
    return grad_hidden.contiguous().matmul(weight1)


class _P68GroupedFFNPairInputGradOnly(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_a: Tensor,
        x_b: Tensor,
        weight1_a: Tensor,
        bias1_a: Tensor | None,
        weight2_a: Tensor,
        bias2_a: Tensor | None,
        q_weight1_a: Tensor,
        weight_scale1_a: Tensor,
        q_weight2_a: Tensor,
        weight_scale2_a: Tensor,
        weight1_b: Tensor,
        bias1_b: Tensor | None,
        weight2_b: Tensor,
        bias2_b: Tensor | None,
        q_weight1_b: Tensor,
        weight_scale1_b: Tensor,
        q_weight2_b: Tensor,
        weight_scale2_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_cutlass_grouped_pair"):
            raise RuntimeError("P68 requires matris_op.quant_linear_w8a8_static_cutlass_grouped_pair")

        x_a_2d = x_a.contiguous()
        x_b_2d = x_b.contiguous()
        scale1_a = _p68_activation_scale(x_a_2d, "MATRIS_P68_GROUPED_FFN_INPUT_SCALE_A", 0.05)
        scale1_b = _p68_activation_scale(x_b_2d, "MATRIS_P68_GROUPED_FFN_INPUT_SCALE_B", 0.05)
        fused_middle = (
            os.environ.get("MATRIS_P68_GROUPED_FFN_PAIR_FUSED_MIDDLE", "0") == "1"
            and os.environ.get("MATRIS_P68_GROUPED_FFN_ACT_SCALE_MODE", "dynamic").strip() == "static"
            and hasattr(matris_op, "quant_ffn_w8a8_static_cutlass_grouped_pair")
        )
        if fused_middle:
            scale2_a = _p68_activation_scale(x_a_2d, "MATRIS_P68_GROUPED_FFN_HIDDEN_SCALE_A", 0.02)
            scale2_b = _p68_activation_scale(x_b_2d, "MATRIS_P68_GROUPED_FFN_HIDDEN_SCALE_B", 0.02)
            out_a, out_b, hidden_a, hidden_b = _p68_grouped_pair_ffn_forward(
                matris_op,
                x_a_2d,
                x_b_2d,
                q_weight1_a,
                q_weight1_b,
                weight_scale1_a,
                weight_scale1_b,
                scale1_a,
                scale1_b,
                bias1_a,
                bias1_b,
                q_weight2_a,
                q_weight2_b,
                weight_scale2_a,
                weight_scale2_b,
                scale2_a,
                scale2_b,
                bias2_a,
                bias2_b,
            )
            ctx.save_for_backward(weight1_a, weight2_a, hidden_a.contiguous(), weight1_b, weight2_b, hidden_b.contiguous())
            return out_a, out_b
        hidden_a, hidden_b = _p68_grouped_pair_linear_forward(
            matris_op,
            x_a_2d,
            x_b_2d,
            q_weight1_a,
            q_weight1_b,
            weight_scale1_a,
            weight_scale1_b,
            scale1_a,
            scale1_b,
            bias1_a,
            bias1_b,
        )
        hidden_act_a = F.silu(hidden_a)
        hidden_act_b = F.silu(hidden_b)
        scale2_a = _p68_activation_scale(hidden_act_a, "MATRIS_P68_GROUPED_FFN_HIDDEN_SCALE_A", 0.05)
        scale2_b = _p68_activation_scale(hidden_act_b, "MATRIS_P68_GROUPED_FFN_HIDDEN_SCALE_B", 0.05)
        out_a, out_b = _p68_grouped_pair_linear_forward(
            matris_op,
            hidden_act_a,
            hidden_act_b,
            q_weight2_a,
            q_weight2_b,
            weight_scale2_a,
            weight_scale2_b,
            scale2_a,
            scale2_b,
            bias2_a,
            bias2_b,
        )
        ctx.save_for_backward(weight1_a, weight2_a, hidden_a.contiguous(), weight1_b, weight2_b, hidden_b.contiguous())
        return out_a, out_b

    @staticmethod
    def backward(ctx, grad_out_a: Tensor | None, grad_out_b: Tensor | None):
        weight1_a, weight2_a, hidden_a, weight1_b, weight2_b, hidden_b = ctx.saved_tensors
        if grad_out_a is None:
            grad_x_a = None
        else:
            grad_x_a = _p68_two_linear_input_grad(grad_out_a, weight1_a, weight2_a, hidden_a)
        if grad_out_b is None:
            grad_x_b = None
        else:
            grad_x_b = _p68_two_linear_input_grad(grad_out_b, weight1_b, weight2_b, hidden_b)
        return (
            grad_x_a,
            grad_x_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def p68_grouped_ffn_pair_or_none(
    node_ffn: nn.Module,
    edge_ffn: nn.Module,
    node_input: Tensor,
    edge_input: Tensor,
    profile_prefix: str,
) -> tuple[Tensor, Tensor] | None:
    if os.environ.get("MATRIS_P68_GROUPED_FFN_PAIR", "0") != "1":
        return None
    if not _p68_grouped_ffn_scope_matches(profile_prefix):
        return None
    if node_ffn.training is not False or edge_ffn.training is not False:
        return None
    if not torch.is_grad_enabled():
        return None
    if not isinstance(node_ffn, FusedInputMLP) or not isinstance(edge_ffn, FusedInputMLP):
        return None
    if getattr(node_ffn, "use_fp16", False) or getattr(edge_ffn, "use_fp16", False):
        return None
    if not (node_input.is_cuda and edge_input.is_cuda):
        return None
    if node_input.dtype != torch.float32 or edge_input.dtype != torch.float32:
        return None
    if node_input.ndim != 2 or edge_input.ndim != 2:
        return None
    total_rows = int(node_input.shape[0]) + int(edge_input.shape[0])
    min_rows = _env_int("MATRIS_P68_GROUPED_FFN_PAIR_MIN_TOTAL_ROWS", 0)
    max_rows = _env_int("MATRIS_P68_GROUPED_FFN_PAIR_MAX_TOTAL_ROWS", 1000000000)
    if not (min_rows <= total_rows <= max_rows):
        return None
    if node_ffn.first.weight.shape != (128, 128) or node_ffn.second.weight.shape != (128, 128):
        return None
    if edge_ffn.first.weight.shape != (128, 128) or edge_ffn.second.weight.shape != (128, 128):
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_cutlass_grouped_pair"):
        return None

    q1_node, s1_node, q2_node, s2_node = node_ffn._p67_quantized_weights()
    q1_edge, s1_edge, q2_edge, s2_edge = edge_ffn._p67_quantized_weights()
    return _P68GroupedFFNPairInputGradOnly.apply(
        node_input,
        edge_input,
        node_ffn.first.weight,
        node_ffn.first.bias,
        node_ffn.second.weight,
        node_ffn.second.bias,
        q1_node,
        s1_node,
        q2_node,
        s2_node,
        edge_ffn.first.weight,
        edge_ffn.first.bias,
        edge_ffn.second.weight,
        edge_ffn.second.bias,
        q1_edge,
        s1_edge,
        q2_edge,
        s2_edge,
    )


class FusedInputMLP(nn.Module):
    """Two-linear SiLU MLP with eval-only input-gradient backward."""

    def __init__(
        self,
        first: nn.Linear,
        activation: nn.Module,
        dropout: nn.Dropout,
        second: nn.Linear,
        *,
        output_dim: int,
        use_fp16: bool = False,
        module_name: str = "",
    ) -> None:
        super().__init__()
        self.first = first
        self.activation = activation
        self.dropout = dropout
        self.second = second
        self.layers = nn.Sequential(self.first, self.activation, self.dropout, self.second)
        self.output_dim = output_dim
        self.use_fp16 = use_fp16
        self.module_name = module_name
        self._p67_quant_cache_key = None
        self._p67_q_weight1 = None
        self._p67_weight_scale1 = None
        self._p67_q_weight2 = None
        self._p67_weight_scale2 = None

    @staticmethod
    def can_fuse(original: nn.Module) -> bool:
        if not isinstance(original, MLP):
            return False
        layers = getattr(original, "layers", None)
        if not isinstance(layers, nn.Sequential) or len(layers) != 4:
            return False
        first, activation, dropout, second = list(layers.children())
        return (
            isinstance(first, nn.Linear)
            and isinstance(second, nn.Linear)
            and isinstance(dropout, nn.Dropout)
            and dropout.p == 0.0
            and isinstance(activation, FusedSiLU)
        )

    @classmethod
    def from_mlp(cls, original: MLP, module_name: str = "") -> "FusedInputMLP":
        if not cls.can_fuse(original):
            raise ValueError(f"Unsupported MLP structure for input-grad fusion: {module_name}")
        first, activation, dropout, second = list(original.layers.children())
        return cls(
            first,
            activation,
            dropout,
            second,
            output_dim=original.output_dim,
            use_fp16=getattr(original, "use_fp16", False),
            module_name=module_name,
        )

    def _p67_quantized_weights(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        key = (
            int(self.first.weight.data_ptr()),
            int(self.second.weight.data_ptr()),
            int(getattr(self.first.weight, "_version", 0)),
            int(getattr(self.second.weight, "_version", 0)),
            tuple(self.first.weight.shape),
            tuple(self.second.weight.shape),
            self.first.weight.device,
            self.second.weight.device,
        )
        if self._p67_quant_cache_key != key:
            self._p67_q_weight1, self._p67_weight_scale1 = _p67_quantize_weight_per_channel(self.first.weight)
            self._p67_q_weight2, self._p67_weight_scale2 = _p67_quantize_weight_per_channel(self.second.weight)
            self._p67_quant_cache_key = key
        return (
            self._p67_q_weight1,
            self._p67_weight_scale1,
            self._p67_q_weight2,
            self._p67_weight_scale2,
        )

    def _p67_reason(self, feas: Tensor) -> tuple[bool, str]:
        rows = int(feas.shape[0]) if feas.ndim == 2 else 0
        matris_op = _load_matris_op()
        if self.training is not False:
            return False, "training"
        if not torch.is_grad_enabled():
            return False, "grad_disabled"
        if self.use_fp16:
            return False, "use_fp16"
        if not feas.is_cuda:
            return False, "not_cuda"
        if feas.dtype != torch.float32:
            return False, "not_float32"
        if feas.ndim != 2:
            return False, "not_2d"
        if not _p67_ffn_scope_rows_match(self.module_name, rows):
            return False, "scope_or_rows"
        if not isinstance(self.activation, FusedSiLU):
            return False, "activation"
        if not isinstance(self.dropout, nn.Dropout) or self.dropout.p != 0.0:
            return False, "dropout"
        if not isinstance(self.first, nn.Linear) or not isinstance(self.second, nn.Linear):
            return False, "not_linear"
        if self.first.weight.shape != (128, 128) or self.second.weight.shape != (128, 128):
            return False, "shape_not_128"
        backend = os.environ.get("MATRIS_P67_FFN_FUSED_QUANT_BACKEND", "wmma").strip()
        if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_wmma"):
            return False, "matris_op_missing"
        if backend == "cutlass" and not hasattr(matris_op, "quant_linear_w8a8_static_cutlass"):
            return False, "cutlass_missing"
        if os.environ.get("MATRIS_P67_FFN_FUSED_QUANT", "0") != "1":
            return False, "eligible_disabled"
        return True, "eligible"

    def _forward_layers(self, feas: Tensor) -> Tensor:
        p67_active_or_census = (
            os.environ.get("MATRIS_P67_FFN_FUSED_QUANT", "0") == "1"
            or bool(_p67_ffn_census_path())
        )
        p67_eligible = False
        p67_reason = "disabled"
        if p67_active_or_census:
            p67_eligible, p67_reason = self._p67_reason(feas)
        if (
            p67_active_or_census
            and feas.ndim == 2
            and isinstance(self.first, nn.Linear)
            and isinstance(self.second, nn.Linear)
        ):
            _record_p67_ffn_census(
                module_name=self.module_name,
                rows=int(feas.shape[0]),
                input_dim=int(feas.shape[1]),
                hidden_dim=int(self.first.weight.shape[0]),
                output_dim=int(self.second.weight.shape[0]),
                dtype=feas.dtype,
                device=feas.device,
                use_fp16=bool(self.use_fp16),
                has_bias=self.first.bias is not None or self.second.bias is not None,
                p67_eligible=p67_eligible,
                reason=p67_reason,
            )
        if p67_eligible:
            q_weight1, weight_scale1, q_weight2, weight_scale2 = self._p67_quantized_weights()
            return _P67FFNFusedQuantInputGradOnly.apply(
                feas,
                self.first.weight,
                self.first.bias,
                self.second.weight,
                self.second.bias,
                q_weight1,
                weight_scale1,
                q_weight2,
                weight_scale2,
            )
        if (
            _use_p28_mlp_input_grad_only()
            and self.training is False
            and torch.is_grad_enabled()
            and not self.use_fp16
            and feas.is_cuda
            and feas.dtype == torch.float32
            and feas.ndim == 2
            and isinstance(self.activation, FusedSiLU)
            and isinstance(self.dropout, nn.Dropout)
            and self.dropout.p == 0.0
            and isinstance(self.first, nn.Linear)
            and isinstance(self.second, nn.Linear)
        ):
            return _TwoLinearSiluInputGradOnly.apply(
                feas,
                self.first.weight,
                self.first.bias,
                self.second.weight,
                self.second.bias,
            )
        return self.layers(feas)

    def forward(self, feas: Tensor) -> Tensor:
        if _use_aggressive_broad_bypass_mlp() and self.training is False and torch.is_grad_enabled():
            return _aggressive_bypass_project(feas, self.output_dim)
        if self.use_fp16 and feas.is_cuda:
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                out = self._forward_layers(feas)
            return out.to(torch.float32)
        return self._forward_layers(feas)


class GatedMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "silu",
        norm_type: str = "layer",
        bias: bool = True,
        use_fp16: bool = False,
    ) -> None:
        """
        Args:
            input_dim: The input dimension.
            hidden_dim: A list of integers or a single integer representing the number 
                of hidden units in each layer of the MLP. Default: None.
            output_dim: The output dimension.
            dropout: The dropout rate. Default: 0.0.
            activation: The name of the activation function. Must be one of "relu", 
                "silu", "tanh", or "gelu". Default: "silu".
            norm_type: The name of the normalization layer to use. Must be one of 
                "layer", "rms", "batch", "group", or None. Default: "layer".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        self.use_fp16 = use_fp16
        self.output_dim = output_dim
        self.activation_func = get_activation(activation)
        self.activation_gate = get_activation("sigmoid")
        self.gate_norm = get_normalization(name=norm_type, dim=output_dim)
        self.core_norm = get_normalization(name=norm_type, dim=output_dim)
        self.mlp_core = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )

    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas (Tensor): shape (feas_num, input_dim)
        Returns:
            output: shape (feas_num, output_dim)
        """
        if _use_aggressive_broad_bypass_gated_mlp() and self.training is False and torch.is_grad_enabled():
            return _aggressive_bypass_project(feas, self.output_dim)
        if self.gate_norm is None:
            core = self.activation_func(self.mlp_core(feas))
            gate = self.activation_gate(self.mlp_gate(feas))
        else:
            core = self.activation_func(self.core_norm(self.mlp_core(feas)))
            gate = self.activation_gate(self.gate_norm(self.mlp_gate(feas)))
        out = core * gate # gate mul
        return out


def _use_input_grad_only_gated_tail() -> bool:
    return os.environ.get("MATRIS_USE_INPUT_GRAD_ONLY_GATED_TAIL", "0") == "1"


def _use_cuda_input_grad_only_gated_tail_bwd() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_INPUT_GRAD_ONLY_GATED_TAIL_BWD", "0") == "1"


def _use_p113_gated_tail_second_silu_macro() -> bool:
    return os.environ.get("MATRIS_P113_GATED_TAIL_SECOND_SILU_MACRO", "0") == "1"


def _use_p116_gated_tail_second_residual_macro() -> bool:
    return os.environ.get("MATRIS_P116_GATED_TAIL_SECOND_RESIDUAL_MACRO", "0") == "1"


def _p113_record_function(name: str):
    if os.environ.get("MATRIS_P113_RECORD_FUNCTION", "0") == "1":
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


def _use_p77_fp32_gated_tail_forward(module_name: str) -> bool:
    use_p77 = os.environ.get("MATRIS_P77_FP32_GATED_TAIL_FORWARD", "0") == "1"
    use_p78 = os.environ.get("MATRIS_P78_FP32_GATED_TAIL_FORWARD", "0") == "1"
    if not (use_p77 or use_p78):
        return False
    if ".attn_block_line_graph.edge_nonlinear_update" in module_name:
        return True
    return use_p78 and ".attn_block_line_graph.node_nonlinear_update" in module_name


class _P77CudaFP32GatedTailForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core: Tensor,
        gate: Tensor,
        core_weight: Tensor,
        core_bias: Tensor,
        gate_weight: Tensor,
        gate_bias: Tensor,
        eps: float,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "fp32_gated_tail_forward_n128"):
            raise RuntimeError("matris_op.fp32_gated_tail_forward_n128 is unavailable")
        out = matris_op.fp32_gated_tail_forward_n128(
            core,
            gate,
            core_weight.contiguous(),
            core_bias.contiguous(),
            gate_weight.contiguous(),
            gate_bias.contiguous(),
            eps,
        )
        ctx.save_for_backward(core, gate, core_weight, core_bias, gate_weight, gate_bias)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        core, gate, core_weight, core_bias, gate_weight, gate_bias = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "input_grad_only_gated_tail_backward"):
            raise RuntimeError("matris_op.input_grad_only_gated_tail_backward is unavailable")
        grad_core, grad_gate = matris_op.input_grad_only_gated_tail_backward(
            grad_out.contiguous(),
            core,
            gate,
            core_weight.contiguous(),
            core_bias.contiguous(),
            gate_weight.contiguous(),
            gate_bias.contiguous(),
            ctx.eps,
        )
        return grad_core, grad_gate, None, None, None, None, None


class _CudaInputGradOnlyGatedTail(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core: Tensor,
        gate: Tensor,
        core_weight: Tensor,
        core_bias: Tensor,
        gate_weight: Tensor,
        gate_bias: Tensor,
        eps: float,
    ) -> Tensor:
        core_ln = F.layer_norm(core, (core.shape[-1],), core_weight, core_bias, eps)
        gate_ln = F.layer_norm(gate, (gate.shape[-1],), gate_weight, gate_bias, eps)
        out = F.silu(core_ln) * torch.sigmoid(gate_ln)
        ctx.save_for_backward(core, gate, core_weight, core_bias, gate_weight, gate_bias)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        core, gate, core_weight, core_bias, gate_weight, gate_bias = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "input_grad_only_gated_tail_backward"):
            raise RuntimeError("matris_op.input_grad_only_gated_tail_backward is unavailable")
        grad_core, grad_gate = matris_op.input_grad_only_gated_tail_backward(
            grad_out.contiguous(),
            core.contiguous(),
            gate.contiguous(),
            core_weight.contiguous(),
            core_bias.contiguous(),
            gate_weight.contiguous(),
            gate_bias.contiguous(),
            ctx.eps,
        )
        return grad_core, grad_gate, None, None, None, None, None


class _CudaParamGradGatedTail(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core: Tensor,
        gate: Tensor,
        core_weight: Tensor,
        core_bias: Tensor,
        gate_weight: Tensor,
        gate_bias: Tensor,
        eps: float,
    ) -> Tensor:
        core_ln = F.layer_norm(core, (core.shape[-1],), core_weight, core_bias, eps)
        gate_ln = F.layer_norm(gate, (gate.shape[-1],), gate_weight, gate_bias, eps)
        out = F.silu(core_ln) * torch.sigmoid(gate_ln)
        ctx.save_for_backward(core, gate, core_weight, core_bias, gate_weight, gate_bias)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        core, gate, core_weight, core_bias, gate_weight, gate_bias = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "param_grad_gated_tail_backward"):
            raise RuntimeError("matris_op.param_grad_gated_tail_backward is unavailable")
        (
            grad_core,
            grad_gate,
            grad_core_weight,
            grad_core_bias,
            grad_gate_weight,
            grad_gate_bias,
        ) = matris_op.param_grad_gated_tail_backward(
            grad_out.contiguous(),
            core.contiguous(),
            gate.contiguous(),
            core_weight.contiguous(),
            core_bias.contiguous(),
            gate_weight.contiguous(),
            gate_bias.contiguous(),
            ctx.eps,
        )
        return (
            grad_core,
            grad_gate,
            grad_core_weight,
            grad_core_bias,
            grad_gate_weight,
            grad_gate_bias,
            None,
        )


class _P113GatedTailSecondSiluMacro(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core_first: Tensor,
        gate_first: Tensor,
        core_second_weight: Tensor,
        core_second_bias: Tensor | None,
        gate_second_weight: Tensor,
        gate_second_bias: Tensor | None,
        core_norm_weight: Tensor,
        core_norm_bias: Tensor,
        gate_norm_weight: Tensor,
        gate_norm_bias: Tensor,
        eps: float,
        use_tail_bwd_v2: bool,
        module_name: str,
    ) -> Tensor:
        with _p113_record_function(f"P113.gated_tail_second.{module_name}.forward"):
            core_second_in = F.silu(core_first)
            gate_second_in = F.silu(gate_first)
            core_second = F.linear(core_second_in, core_second_weight, core_second_bias)
            gate_second = F.linear(gate_second_in, gate_second_weight, gate_second_bias)
            core_ln = F.layer_norm(core_second, (core_second.shape[-1],), core_norm_weight, core_norm_bias, eps)
            gate_ln = F.layer_norm(gate_second, (gate_second.shape[-1],), gate_norm_weight, gate_norm_bias, eps)
            out = F.silu(core_ln) * torch.sigmoid(gate_ln)
        ctx.save_for_backward(
            core_second,
            gate_second,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
            core_second_weight,
            gate_second_weight,
            core_first,
            gate_first,
        )
        ctx.eps = float(eps)
        ctx.use_tail_bwd_v2 = bool(use_tail_bwd_v2)
        ctx.module_name = str(module_name)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            core_second,
            gate_second,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
            core_second_weight,
            gate_second_weight,
            core_first,
            gate_first,
        ) = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "gated_tail_second_silu_input_grad_macro"):
            raise RuntimeError("matris_op.gated_tail_second_silu_input_grad_macro is unavailable")
        with _p113_record_function(f"P113.gated_tail_second.{ctx.module_name}.backward"):
            grad_core_first, grad_gate_first = matris_op.gated_tail_second_silu_input_grad_macro(
                grad_out.contiguous(),
                core_second.contiguous(),
                gate_second.contiguous(),
                core_norm_weight.contiguous(),
                core_norm_bias.contiguous(),
                gate_norm_weight.contiguous(),
                gate_norm_bias.contiguous(),
                ctx.eps,
                core_second_weight.contiguous(),
                gate_second_weight.contiguous(),
                core_first.contiguous(),
                gate_first.contiguous(),
                ctx.use_tail_bwd_v2,
            )
        return grad_core_first, grad_gate_first, None, None, None, None, None, None, None, None, None, None, None


class _P116GatedTailSecondSiluResidualMacro(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core_first: Tensor,
        gate_first: Tensor,
        core_second_weight: Tensor,
        core_second_bias: Tensor | None,
        gate_second_weight: Tensor,
        gate_second_bias: Tensor | None,
        core_norm_weight: Tensor,
        core_norm_bias: Tensor,
        gate_norm_weight: Tensor,
        gate_norm_bias: Tensor,
        eps: float,
        use_tail_bwd_v2: bool,
        old_feat: Tensor,
        res_weight: Tensor,
        module_name: str,
    ) -> Tensor:
        with _p113_record_function(f"P116.gated_tail_second_residual.{module_name}.forward"):
            core_second_in = F.silu(core_first)
            gate_second_in = F.silu(gate_first)
            core_second = F.linear(core_second_in, core_second_weight, core_second_bias)
            gate_second = F.linear(gate_second_in, gate_second_weight, gate_second_bias)
            core_ln = F.layer_norm(core_second, (core_second.shape[-1],), core_norm_weight, core_norm_bias, eps)
            gate_ln = F.layer_norm(gate_second, (gate_second.shape[-1],), gate_norm_weight, gate_norm_bias, eps)
            update = F.silu(core_ln) * torch.sigmoid(gate_ln)
            out = update + res_weight * old_feat
        ctx.save_for_backward(
            core_second,
            gate_second,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
            core_second_weight,
            gate_second_weight,
            core_first,
            gate_first,
            old_feat,
            res_weight,
        )
        ctx.eps = float(eps)
        ctx.use_tail_bwd_v2 = bool(use_tail_bwd_v2)
        ctx.module_name = str(module_name)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            core_second,
            gate_second,
            core_norm_weight,
            core_norm_bias,
            gate_norm_weight,
            gate_norm_bias,
            core_second_weight,
            gate_second_weight,
            core_first,
            gate_first,
            old_feat,
            res_weight,
        ) = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "gated_tail_second_silu_residual_input_grad_macro"):
            raise RuntimeError("matris_op.gated_tail_second_silu_residual_input_grad_macro is unavailable")
        with _p113_record_function(f"P116.gated_tail_second_residual.{ctx.module_name}.backward"):
            grad_core_first, grad_gate_first, grad_old, grad_res_weight = (
                matris_op.gated_tail_second_silu_residual_input_grad_macro(
                    grad_out.contiguous(),
                    core_second.contiguous(),
                    gate_second.contiguous(),
                    core_norm_weight.contiguous(),
                    core_norm_bias.contiguous(),
                    gate_norm_weight.contiguous(),
                    gate_norm_bias.contiguous(),
                    ctx.eps,
                    core_second_weight.contiguous(),
                    gate_second_weight.contiguous(),
                    core_first.contiguous(),
                    gate_first.contiguous(),
                    ctx.use_tail_bwd_v2,
                    old_feat.contiguous(),
                    res_weight.contiguous(),
                )
            )
        return (
            grad_core_first,
            grad_gate_first,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_old,
            grad_res_weight,
            None,
        )


class _InputGradOnlyGatedTail(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core: Tensor,
        gate: Tensor,
        core_weight: Tensor,
        core_bias: Tensor,
        gate_weight: Tensor,
        gate_bias: Tensor,
        eps: float,
    ) -> Tensor:
        core_f = core.float()
        gate_f = gate.float()
        core_mean = core_f.mean(dim=-1, keepdim=True)
        gate_mean = gate_f.mean(dim=-1, keepdim=True)
        core_var = (core_f - core_mean).pow(2).mean(dim=-1, keepdim=True)
        gate_var = (gate_f - gate_mean).pow(2).mean(dim=-1, keepdim=True)
        core_rstd = torch.rsqrt(core_var + eps)
        gate_rstd = torch.rsqrt(gate_var + eps)
        core_hat = (core_f - core_mean) * core_rstd
        gate_hat = (gate_f - gate_mean) * gate_rstd
        core_ln = core_hat * core_weight.float() + core_bias.float()
        gate_ln = gate_hat * gate_weight.float() + gate_bias.float()
        core_act = F.silu(core_ln)
        gate_act = torch.sigmoid(gate_ln)
        ctx.save_for_backward(
            core_hat,
            gate_hat,
            core_rstd,
            gate_rstd,
            core_ln,
            core_act,
            gate_act,
            core_weight.float(),
            gate_weight.float(),
        )
        ctx.core_shape = core.shape
        ctx.gate_shape = gate.shape
        return core_act * gate_act

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            core_hat,
            gate_hat,
            core_rstd,
            gate_rstd,
            core_ln,
            core_act,
            gate_act,
            core_weight,
            gate_weight,
        ) = ctx.saved_tensors
        grad = grad_out.float()
        core_sigmoid = torch.sigmoid(core_ln)
        core_silu_grad = core_sigmoid * (1.0 + core_ln * (1.0 - core_sigmoid))
        grad_core_ln = grad * gate_act * core_silu_grad
        grad_gate_ln = grad * core_act * gate_act * (1.0 - gate_act)

        def layernorm_input_grad(grad_y: Tensor, x_hat: Tensor, rstd: Tensor, weight: Tensor) -> Tensor:
            grad_norm = grad_y * weight
            n = grad_norm.shape[-1]
            sum_grad = grad_norm.sum(dim=-1, keepdim=True)
            sum_grad_xhat = (grad_norm * x_hat).sum(dim=-1, keepdim=True)
            return (grad_norm * n - sum_grad - x_hat * sum_grad_xhat) * (rstd / n)

        grad_core = layernorm_input_grad(grad_core_ln, core_hat, core_rstd, core_weight).reshape(ctx.core_shape)
        grad_gate = layernorm_input_grad(grad_gate_ln, gate_hat, gate_rstd, gate_weight).reshape(ctx.gate_shape)
        return grad_core, grad_gate, None, None, None, None, None


class FusedGatedMLPTail(nn.Module):
    """GatedMLP tail wrapper for norm, activation, sigmoid gate, and multiply."""

    def __init__(
        self,
        core_norm: nn.Module | None,
        gate_norm: nn.Module | None,
        activation_func: nn.Module,
        activation_gate: nn.Module,
        module_name: str = "",
    ) -> None:
        super().__init__()
        self.core_norm = core_norm
        self.gate_norm = gate_norm
        self.activation_func = activation_func
        self.activation_gate = activation_gate
        self.module_name = module_name

    def forward(self, core: Tensor, gate: Tensor) -> Tensor:
        broad_input_grad_only = _use_aggressive_broad_input_grad_only()
        p26_tail_input_grad_only = _use_p26_tail_input_grad_only()
        p27_tail_param_grad = _use_p27_tail_param_grad()
        if (
            _use_p77_fp32_gated_tail_forward(self.module_name)
            and torch.is_grad_enabled()
            and core.is_cuda
            and gate.is_cuda
            and core.dtype == torch.float32
            and gate.dtype == torch.float32
            and core.ndim == 2
            and core.shape[-1] in (128, 256)
            and gate.shape == core.shape
            and isinstance(self.core_norm, nn.LayerNorm)
            and isinstance(self.gate_norm, nn.LayerNorm)
            and self.core_norm.normalized_shape == self.gate_norm.normalized_shape
            and self.core_norm.eps == self.gate_norm.eps
            and self.core_norm.elementwise_affine
            and self.gate_norm.elementwise_affine
            and self.core_norm.weight is not None
            and self.core_norm.bias is not None
            and self.gate_norm.weight is not None
            and self.gate_norm.bias is not None
            and self.core_norm.weight.is_cuda
            and self.core_norm.bias.is_cuda
            and self.gate_norm.weight.is_cuda
            and self.gate_norm.bias.is_cuda
            and self.core_norm.weight.dtype == torch.float32
            and self.core_norm.bias.dtype == torch.float32
            and self.gate_norm.weight.dtype == torch.float32
            and self.gate_norm.bias.dtype == torch.float32
            and isinstance(self.activation_func, FusedSiLU)
            and isinstance(self.activation_gate, FusedSigmoid)
            and (lambda op: op is not None and hasattr(op, "fp32_gated_tail_forward_n128"))(_load_matris_op())
            and (lambda op: op is not None and hasattr(op, "input_grad_only_gated_tail_backward"))(_load_matris_op())
        ):
            return _P77CudaFP32GatedTailForward.apply(
                core,
                gate,
                self.core_norm.weight,
                self.core_norm.bias,
                self.gate_norm.weight,
                self.gate_norm.bias,
                self.core_norm.eps,
            )
        if (
            p27_tail_param_grad
            and self.training is False
            and torch.is_grad_enabled()
            and core.is_cuda
            and gate.is_cuda
            and core.dtype == torch.float32
            and gate.dtype == torch.float32
            and core.ndim == 2
            and core.shape[-1] in (128, 256)
            and gate.shape == core.shape
            and isinstance(self.core_norm, nn.LayerNorm)
            and isinstance(self.gate_norm, nn.LayerNorm)
            and self.core_norm.normalized_shape == self.gate_norm.normalized_shape
            and self.core_norm.eps == self.gate_norm.eps
            and self.core_norm.elementwise_affine
            and self.gate_norm.elementwise_affine
            and self.core_norm.weight is not None
            and self.core_norm.bias is not None
            and self.gate_norm.weight is not None
            and self.gate_norm.bias is not None
            and self.core_norm.weight.is_cuda
            and self.core_norm.bias.is_cuda
            and self.gate_norm.weight.is_cuda
            and self.gate_norm.bias.is_cuda
            and self.core_norm.weight.dtype == torch.float32
            and self.core_norm.bias.dtype == torch.float32
            and self.gate_norm.weight.dtype == torch.float32
            and self.gate_norm.bias.dtype == torch.float32
            and isinstance(self.activation_func, FusedSiLU)
            and isinstance(self.activation_gate, FusedSigmoid)
            and (lambda op: op is not None and hasattr(op, "param_grad_gated_tail_backward"))(_load_matris_op())
        ):
            return _CudaParamGradGatedTail.apply(
                core,
                gate,
                self.core_norm.weight,
                self.core_norm.bias,
                self.gate_norm.weight,
                self.gate_norm.bias,
                self.core_norm.eps,
            )

        if (
            (
                _use_cuda_input_grad_only_gated_tail_bwd()
                or broad_input_grad_only
                or p26_tail_input_grad_only
            )
            and self.training is False
            and torch.is_grad_enabled()
            and core.is_cuda
            and gate.is_cuda
            and core.dtype == torch.float32
            and gate.dtype == torch.float32
            and core.ndim == 2
            and core.shape[-1] in (128, 256)
            and gate.shape == core.shape
            and isinstance(self.core_norm, nn.LayerNorm)
            and isinstance(self.gate_norm, nn.LayerNorm)
            and self.core_norm.normalized_shape == self.gate_norm.normalized_shape
            and self.core_norm.eps == self.gate_norm.eps
            and self.core_norm.elementwise_affine
            and self.gate_norm.elementwise_affine
            and self.core_norm.weight is not None
            and self.core_norm.bias is not None
            and self.gate_norm.weight is not None
            and self.gate_norm.bias is not None
            and self.core_norm.weight.is_cuda
            and self.core_norm.bias.is_cuda
            and self.gate_norm.weight.is_cuda
            and self.gate_norm.bias.is_cuda
            and self.core_norm.weight.dtype == torch.float32
            and self.core_norm.bias.dtype == torch.float32
            and self.gate_norm.weight.dtype == torch.float32
            and self.gate_norm.bias.dtype == torch.float32
            and (
                broad_input_grad_only
                or p26_tail_input_grad_only
                or (
                    not self.core_norm.weight.requires_grad
                    and not self.core_norm.bias.requires_grad
                    and not self.gate_norm.weight.requires_grad
                    and not self.gate_norm.bias.requires_grad
                )
            )
            and isinstance(self.activation_func, FusedSiLU)
            and isinstance(self.activation_gate, FusedSigmoid)
            and (lambda op: op is not None and hasattr(op, "input_grad_only_gated_tail_backward"))(_load_matris_op())
        ):
            return _CudaInputGradOnlyGatedTail.apply(
                core,
                gate,
                self.core_norm.weight,
                self.core_norm.bias,
                self.gate_norm.weight,
                self.gate_norm.bias,
                self.core_norm.eps,
            )

        if (
            (
                _use_input_grad_only_gated_tail()
                or broad_input_grad_only
                or p26_tail_input_grad_only
            )
            and self.training is False
            and torch.is_grad_enabled()
            and core.is_cuda
            and gate.is_cuda
            and core.ndim == 2
            and gate.shape == core.shape
            and isinstance(self.core_norm, nn.LayerNorm)
            and isinstance(self.gate_norm, nn.LayerNorm)
            and self.core_norm.normalized_shape == self.gate_norm.normalized_shape
            and self.core_norm.eps == self.gate_norm.eps
            and self.core_norm.elementwise_affine
            and self.gate_norm.elementwise_affine
            and (
                broad_input_grad_only
                or p26_tail_input_grad_only
                or (
                    not self.core_norm.weight.requires_grad
                    and not self.core_norm.bias.requires_grad
                    and not self.gate_norm.weight.requires_grad
                    and not self.gate_norm.bias.requires_grad
                )
            )
            and isinstance(self.activation_func, FusedSiLU)
            and isinstance(self.activation_gate, FusedSigmoid)
        ):
            return _InputGradOnlyGatedTail.apply(
                core,
                gate,
                self.core_norm.weight,
                self.core_norm.bias,
                self.gate_norm.weight,
                self.gate_norm.bias,
                self.core_norm.eps,
            )

        if (
            not torch.is_grad_enabled()
            and core.is_cuda
            and gate.is_cuda
            and core.ndim == 2
            and gate.shape == core.shape
            and isinstance(self.core_norm, nn.LayerNorm)
            and isinstance(self.gate_norm, nn.LayerNorm)
            and self.core_norm.normalized_shape == self.gate_norm.normalized_shape
            and self.core_norm.eps == self.gate_norm.eps
            and self.core_norm.elementwise_affine
            and self.gate_norm.elementwise_affine
            and isinstance(self.activation_func, FusedSiLU)
            and isinstance(self.activation_gate, FusedSigmoid)
        ):
            if os.environ.get("MATRIS_DISABLE_TRITON_GATED_TAIL") != "1":
                from quant.layers import triton_gated_mlp_tail

                out = triton_gated_mlp_tail(
                    core,
                    gate,
                    self.core_norm.weight,
                    self.core_norm.bias,
                    self.gate_norm.weight,
                    self.gate_norm.bias,
                    self.core_norm.eps,
                )
                if out is not None:
                    return out

        if self.core_norm is not None:
            core = self.core_norm(core)
        if self.gate_norm is not None:
            gate = self.gate_norm(gate)
        return self.activation_func(core) * self.activation_gate(gate)


class FusedInputGatedMLP(nn.Module):
    """GatedMLP variant that fuses core/gate projection pairs."""

    def __init__(
        self,
        core_first: nn.Module,
        gate_first: nn.Module,
        fused_first: nn.Linear | None,
        core_tail: nn.Sequential,
        gate_tail: nn.Sequential,
        activation_func: nn.Module | None,
        activation_gate: nn.Module | None,
        core_norm: nn.Module | None,
        gate_norm: nn.Module | None,
        core_second: nn.Module | None = None,
        gate_second: nn.Module | None = None,
        fused_second: nn.Linear | None = None,
        core_second_prefix: nn.Sequential | None = None,
        gate_second_prefix: nn.Sequential | None = None,
        core_post_second_tail: nn.Sequential | None = None,
        gate_post_second_tail: nn.Sequential | None = None,
        fused_tail: FusedGatedMLPTail | None = None,
        use_fp16: bool = False,
        module_name: str = "",
    ) -> None:
        super().__init__()
        self.core_first = core_first
        self.gate_first = gate_first
        self.fused_first = fused_first
        self.core_tail = core_tail
        self.gate_tail = gate_tail
        self.core_second = core_second
        self.gate_second = gate_second
        self.fused_second = fused_second
        self.core_second_prefix = core_second_prefix
        self.gate_second_prefix = gate_second_prefix
        self.core_post_second_tail = core_post_second_tail
        self.gate_post_second_tail = gate_post_second_tail
        self.fused_tail = fused_tail
        self.activation_func = activation_func
        self.activation_gate = activation_gate
        self.core_norm = core_norm
        self.gate_norm = gate_norm
        self.use_fp16 = use_fp16
        self.module_name = module_name
        self.output_dim = self._linear_like_out_features(core_second) if core_second is not None else self._linear_like_out_features(core_first)
        self.core_hidden_dim = self._linear_like_out_features(core_first)
        self.gate_hidden_dim = self._linear_like_out_features(gate_first)
        self.core_output_dim = self._linear_like_out_features(core_second) if core_second is not None else None
        self.gate_output_dim = self._linear_like_out_features(gate_second) if gate_second is not None else None

    @staticmethod
    def _linear_like_out_features(module: nn.Module) -> int:
        if hasattr(module, "weight"):
            return module.weight.shape[0]
        if hasattr(module, "q_weight"):
            return module.q_weight.shape[0]
        if hasattr(module, "weight_low"):
            return module.weight_low.shape[0]
        raise AttributeError(f"Cannot infer out_features for {type(module).__name__}")

    @staticmethod
    def _is_linear_like(module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) or (
            hasattr(module, "weight") and hasattr(module, "_fake_quant_weight")
        ) or (
            hasattr(module, "q_weight") and hasattr(module, "scale")
        ) or (
            hasattr(module, "weight_low")
        )

    @staticmethod
    def can_fuse(original: nn.Module) -> bool:
        if not all(hasattr(original, name) for name in ("mlp_core", "mlp_gate")):
            return False
        core_layers = getattr(original.mlp_core, "layers", None)
        gate_layers = getattr(original.mlp_gate, "layers", None)
        if not isinstance(core_layers, nn.Sequential) or not isinstance(gate_layers, nn.Sequential):
            return False
        if len(core_layers) < 2 or len(core_layers) != len(gate_layers):
            return False
        return FusedInputGatedMLP._is_linear_like(core_layers[0]) and FusedInputGatedMLP._is_linear_like(gate_layers[0])

    @staticmethod
    def _is_matching_stateless_activation(core_layer: nn.Module, gate_layer: nn.Module) -> bool:
        return type(core_layer) is type(gate_layer) and len(list(core_layer.parameters())) == 0

    @staticmethod
    def _is_disabled_dropout(core_layer: nn.Module, gate_layer: nn.Module) -> bool:
        return (
            isinstance(core_layer, nn.Dropout)
            and isinstance(gate_layer, nn.Dropout)
            and core_layer.p == 0.0
            and gate_layer.p == 0.0
        )

    @staticmethod
    def can_fuse_second_tail(core_tail: nn.Sequential, gate_tail: nn.Sequential) -> bool:
        if len(core_tail) < 3 or len(core_tail) != len(gate_tail):
            return False
        core_layers = list(core_tail.children())
        gate_layers = list(gate_tail.children())
        return (
            FusedInputGatedMLP._is_matching_stateless_activation(core_layers[0], gate_layers[0])
            and FusedInputGatedMLP._is_disabled_dropout(core_layers[1], gate_layers[1])
            and FusedInputGatedMLP._is_linear_like(core_layers[2])
            and FusedInputGatedMLP._is_linear_like(gate_layers[2])
        )

    @classmethod
    def from_gated_mlp(
        cls,
        original: nn.Module,
        module_name: str = "",
        fuse_second: bool = False,
        fuse_tail: bool = False,
    ) -> "FusedInputGatedMLP":
        if not cls.can_fuse(original):
            raise ValueError(f"Unsupported GatedMLP structure for input fusion: {module_name}")

        core_layers = list(original.mlp_core.layers.children())
        gate_layers = list(original.mlp_gate.layers.children())
        fused_first = cls._make_fused_linear(core_layers[0], gate_layers[0])
        core_tail = nn.Sequential(*core_layers[1:])
        gate_tail = nn.Sequential(*gate_layers[1:])

        core_second = None
        gate_second = None
        fused_second = None
        core_second_prefix = None
        gate_second_prefix = None
        core_post_second_tail = None
        gate_post_second_tail = None
        if fuse_second and cls.can_fuse_second_tail(core_tail, gate_tail):
            core_tail_layers = list(core_tail.children())
            gate_tail_layers = list(gate_tail.children())
            core_second = core_tail_layers[2]
            gate_second = gate_tail_layers[2]
            fused_second = cls._make_block_diagonal_fused_linear(core_second, gate_second)
            core_second_prefix = nn.Sequential(*core_tail_layers[:2])
            gate_second_prefix = nn.Sequential(*gate_tail_layers[:2])
            core_post_second_tail = nn.Sequential(*core_tail_layers[3:])
            gate_post_second_tail = nn.Sequential(*gate_tail_layers[3:])
            core_tail = nn.Sequential()
            gate_tail = nn.Sequential()

        fused_tail = None
        activation_func = original.activation_func
        activation_gate = original.activation_gate
        core_norm = original.core_norm
        gate_norm = original.gate_norm
        if fuse_tail:
            fused_tail = FusedGatedMLPTail(
                core_norm=original.core_norm,
                gate_norm=original.gate_norm,
                activation_func=original.activation_func,
                activation_gate=original.activation_gate,
                module_name=module_name,
            )
            activation_func = None
            activation_gate = None
            core_norm = None
            gate_norm = None

        return cls(
            core_first=core_layers[0],
            gate_first=gate_layers[0],
            fused_first=fused_first,
            core_tail=core_tail,
            gate_tail=gate_tail,
            core_second=core_second,
            gate_second=gate_second,
            fused_second=fused_second,
            core_second_prefix=core_second_prefix,
            gate_second_prefix=gate_second_prefix,
            core_post_second_tail=core_post_second_tail,
            gate_post_second_tail=gate_post_second_tail,
            fused_tail=fused_tail,
            activation_func=activation_func,
            activation_gate=activation_gate,
            core_norm=core_norm,
            gate_norm=gate_norm,
            use_fp16=getattr(original, "use_fp16", False),
            module_name=module_name,
        )

    @staticmethod
    def _make_fused_linear(core_first: nn.Module, gate_first: nn.Module) -> nn.Linear | None:
        if not isinstance(core_first, nn.Linear) or not isinstance(gate_first, nn.Linear):
            return None
        if core_first.in_features != gate_first.in_features:
            return None

        bias = core_first.bias is not None or gate_first.bias is not None
        fused = nn.Linear(
            core_first.in_features,
            core_first.out_features + gate_first.out_features,
            bias=bias,
            device=core_first.weight.device,
            dtype=core_first.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(torch.cat([core_first.weight, gate_first.weight], dim=0))
            if fused.bias is not None:
                core_bias = (
                    core_first.bias
                    if core_first.bias is not None
                    else core_first.weight.new_zeros(core_first.out_features)
                )
                gate_bias = (
                    gate_first.bias
                    if gate_first.bias is not None
                    else gate_first.weight.new_zeros(gate_first.out_features)
                )
                fused.bias.copy_(torch.cat([core_bias, gate_bias], dim=0))
        return fused

    @staticmethod
    def _make_block_diagonal_fused_linear(core_second: nn.Module, gate_second: nn.Module) -> nn.Linear | None:
        if not isinstance(core_second, nn.Linear) or not isinstance(gate_second, nn.Linear):
            return None

        bias = core_second.bias is not None or gate_second.bias is not None
        fused = nn.Linear(
            core_second.in_features + gate_second.in_features,
            core_second.out_features + gate_second.out_features,
            bias=bias,
            device=core_second.weight.device,
            dtype=core_second.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.zero_()
            fused.weight[: core_second.out_features, : core_second.in_features].copy_(core_second.weight)
            fused.weight[
                core_second.out_features :,
                core_second.in_features :,
            ].copy_(gate_second.weight)
            if fused.bias is not None:
                core_bias = (
                    core_second.bias
                    if core_second.bias is not None
                    else core_second.weight.new_zeros(core_second.out_features)
                )
                gate_bias = (
                    gate_second.bias
                    if gate_second.bias is not None
                    else gate_second.weight.new_zeros(gate_second.out_features)
                )
                fused.bias.copy_(torch.cat([core_bias, gate_bias], dim=0))
        return fused

    @staticmethod
    def _weight_bias_stats(module: nn.Module) -> tuple[Tensor, Tensor | None, dict | None]:
        if isinstance(module, nn.Linear):
            return module.weight, module.bias, None

        dq_weight, stats = module._fake_quant_weight()
        bias = module.bias.float() if getattr(module, "bias", None) is not None else None
        return dq_weight, bias, stats

    @staticmethod
    def _set_fake_quant_stats(
        module: nn.Module,
        stats: dict | None,
        x: Tensor,
        out: Tensor,
    ) -> None:
        if stats is None:
            return
        stats["input_abs_max"] = x.float().abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        module.last_stats = stats

    @staticmethod
    def _maybe_fake_quant_activation(
        module: nn.Module,
        x: Tensor,
        stats: dict | None,
    ) -> tuple[Tensor, dict | None]:
        if stats is None or not hasattr(module, "_fake_quant_activation"):
            return x, stats
        dq_x, activation_stats = module._fake_quant_activation(x)
        stats.update(activation_stats)
        return dq_x, stats

    @staticmethod
    def _is_triton_w8a32(module: nn.Module) -> bool:
        return (
            hasattr(module, "q_weight")
            and hasattr(module, "scale")
            and getattr(module, "last_stats", {}).get("backend") == "triton_w8a32"
        )

    @staticmethod
    def _is_cached_low_precision(module: nn.Module) -> bool:
        return hasattr(module, "weight_low") and getattr(module, "last_stats", {}).get(
            "backend"
        ) == "linear_low_precision_cached"

    @staticmethod
    def _triton_w8a32_bias(module: nn.Module, rows: int) -> Tensor:
        bias = getattr(module, "bias", None)
        if bias is None:
            return module.scale.new_zeros(rows)
        return bias.float()

    def _triton_w8a32_fused_first(self, x: Tensor) -> tuple[Tensor, Tensor] | None:
        if not (self._is_triton_w8a32(self.core_first) and self._is_triton_w8a32(self.gate_first)):
            return None
        if self.core_first.q_weight.shape[1] != self.gate_first.q_weight.shape[1]:
            return None

        cache_key = "_triton_w8a32_first_cache"
        cache = getattr(self, cache_key, None)
        if cache is None or cache["device"] != x.device:
            q_weight = torch.cat([self.core_first.q_weight, self.gate_first.q_weight], dim=0).contiguous()
            scale = torch.cat([self.core_first.scale, self.gate_first.scale], dim=0).contiguous()
            bias = torch.cat(
                [
                    self._triton_w8a32_bias(self.core_first, self.core_first.q_weight.shape[0]),
                    self._triton_w8a32_bias(self.gate_first, self.gate_first.q_weight.shape[0]),
                ],
                dim=0,
            ).contiguous()
            cache = {
                "device": x.device,
                "q_weight": q_weight,
                "scale": scale,
                "bias": bias,
            }
            setattr(self, cache_key, cache)

        from quant.layers import triton_w8a32_linear

        projected = triton_w8a32_linear(
            x.float(),
            cache["q_weight"],
            cache["scale"],
            cache["bias"],
        )
        core_hidden, gate_hidden = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        self._set_fake_quant_stats(self.core_first, dict(self.core_first.last_stats), x, core_hidden)
        self._set_fake_quant_stats(self.gate_first, dict(self.gate_first.last_stats), x, gate_hidden)
        return core_hidden, gate_hidden

    def _p52_w8a8_static_fused_first(self, x: Tensor) -> tuple[Tensor, Tensor] | None:
        if not (
            self._is_triton_w8a8_static(self.core_first)
            and self._is_triton_w8a8_static(self.gate_first)
        ):
            return None
        if self.core_first.q_weight.shape[1] != self.gate_first.q_weight.shape[1]:
            return None
        if self.core_hidden_dim != self.core_first.q_weight.shape[0]:
            return None
        if self.gate_hidden_dim != self.gate_first.q_weight.shape[0]:
            return None

        rows = int(x.reshape(-1, x.shape[-1]).shape[0])
        if not _use_p52_w8a8_fused_first(self.module_name, rows):
            return None

        core_scale = self.core_first._activation_scale_for(x)
        gate_scale = self.gate_first._activation_scale_for(x)
        core_bias = self.core_first.bias.float() if self.core_first.bias is not None else None
        gate_bias = self.gate_first.bias.float() if self.gate_first.bias is not None else None
        backend = os.environ.get("MATRIS_P52_W8A8_FUSED_FIRST_BACKEND", "triton").strip()

        if backend == "cuda_wmma":
            from quant.layers import cuda_w8a8_static_wmma_dual_linear

            projected = cuda_w8a8_static_wmma_dual_linear(
                x,
                x,
                self.core_first.q_weight,
                self.core_first.scale,
                core_scale,
                core_bias,
                self.gate_first.q_weight,
                self.gate_first.scale,
                gate_scale,
                gate_bias,
            )
        elif backend == "cuda_cutlass":
            from quant.layers import cuda_w8a8_static_cutlass_dual_linear

            projected = cuda_w8a8_static_cutlass_dual_linear(
                x,
                x,
                self.core_first.q_weight,
                self.core_first.scale,
                core_scale,
                core_bias,
                self.gate_first.q_weight,
                self.gate_first.scale,
                gate_scale,
                gate_bias,
            )
        elif backend == "cuda_cutlass_grouped":
            from quant.layers import cuda_w8a8_static_cutlass_grouped_dual_linear

            projected = cuda_w8a8_static_cutlass_grouped_dual_linear(
                x,
                x,
                self.core_first.q_weight,
                self.core_first.scale,
                core_scale,
                core_bias,
                self.gate_first.q_weight,
                self.gate_first.scale,
                gate_scale,
                gate_bias,
            )
        else:
            from quant.layers import triton_w8a8_static_dual_linear

            projected = triton_w8a8_static_dual_linear(
                x,
                x,
                self.core_first.q_weight,
                self.core_first.q_weight_t,
                self.core_first.scale,
                core_scale,
                core_bias,
                self.gate_first.q_weight,
                self.gate_first.q_weight_t,
                self.gate_first.scale,
                gate_scale,
                gate_bias,
            )
        if projected is None:
            return None

        core_hidden, gate_hidden = projected
        self._set_fake_quant_stats(self.core_first, dict(self.core_first.last_stats), x, core_hidden)
        self._set_fake_quant_stats(self.gate_first, dict(self.gate_first.last_stats), x, gate_hidden)
        return core_hidden, gate_hidden

    def _triton_w8a32_fused_second(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor] | None:
        if self.core_second is None or self.gate_second is None:
            return None
        if not (self._is_triton_w8a32(self.core_second) and self._is_triton_w8a32(self.gate_second)):
            return None

        cache_key = "_triton_w8a32_second_cache"
        x = torch.cat([core.float(), gate.float()], dim=-1)
        cache = getattr(self, cache_key, None)
        if cache is None or cache["device"] != x.device:
            core_out, core_in = self.core_second.q_weight.shape
            gate_out, gate_in = self.gate_second.q_weight.shape
            q_weight = self.core_second.q_weight.new_zeros((core_out + gate_out, core_in + gate_in))
            q_weight[:core_out, :core_in] = self.core_second.q_weight
            q_weight[core_out:, core_in:] = self.gate_second.q_weight
            q_weight = q_weight.contiguous()
            scale = torch.cat([self.core_second.scale, self.gate_second.scale], dim=0).contiguous()
            bias = torch.cat(
                [
                    self._triton_w8a32_bias(self.core_second, core_out),
                    self._triton_w8a32_bias(self.gate_second, gate_out),
                ],
                dim=0,
            ).contiguous()
            cache = {
                "device": x.device,
                "q_weight": q_weight,
                "scale": scale,
                "bias": bias,
            }
            setattr(self, cache_key, cache)

        from quant.layers import triton_w8a32_linear

        projected = triton_w8a32_linear(
            x,
            cache["q_weight"],
            cache["scale"],
            cache["bias"],
        )
        core_out, gate_out = projected.split([self.core_output_dim, self.gate_output_dim], dim=-1)
        self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
        self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
        return core_out, gate_out

    @staticmethod
    def _is_triton_w8a8_static(module: nn.Module) -> bool:
        return (
            hasattr(module, "q_weight")
            and hasattr(module, "q_weight_t")
            and hasattr(module, "scale")
            and hasattr(module, "_activation_scale_for")
            and getattr(module, "last_stats", {}).get("backend") == "triton_w8a8_static"
        )

    @staticmethod
    def _p94_cached_dq_weight_t(module: nn.Module) -> Tensor:
        q_weight = module.q_weight
        scale = module.scale
        key = (
            int(q_weight.data_ptr()),
            int(scale.data_ptr()),
            int(getattr(q_weight, "_version", 0)),
            int(getattr(scale, "_version", 0)),
            int(q_weight.numel()),
            tuple(q_weight.shape),
            q_weight.device,
        )
        cache = getattr(module, "_p94_dq_weight_t_cache", None)
        if cache is not None and cache.get("key") == key:
            return cache["value"]
        from quant.layers import cached_w8a8_dq_weight_t

        value = cached_w8a8_dq_weight_t(q_weight, scale)
        setattr(module, "_p94_dq_weight_t_cache", {"key": key, "value": value})
        return value

    def _triton_w8a8_static_fused_second(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor] | None:
        if self.core_second is None or self.gate_second is None:
            return None
        if not (
            self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
        ):
            return None

        import os

        from quant.layers import triton_w8a8_static_dual_linear

        core_scale = self.core_second._activation_scale_for(core)
        gate_scale = self.gate_second._activation_scale_for(gate)
        core_bias = self.core_second.bias.float() if self.core_second.bias is not None else None
        gate_bias = self.gate_second.bias.float() if self.gate_second.bias is not None else None
        if os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_wmma":
            from quant.layers import cuda_w8a8_static_wmma_dual_linear

            dual_out = cuda_w8a8_static_wmma_dual_linear(
                core,
                gate,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
            )
            if dual_out is not None:
                core_out, gate_out = dual_out
                if getattr(self.core_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
                if getattr(self.gate_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
                return core_out, gate_out
        if os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_cutlass_dual":
            from quant.layers import cuda_w8a8_static_cutlass_dual_linear

            dual_out = cuda_w8a8_static_cutlass_dual_linear(
                core,
                gate,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
            )
            if dual_out is not None:
                core_out, gate_out = dual_out
                if getattr(self.core_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
                if getattr(self.gate_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
                return core_out, gate_out
        if os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_cutlass_grouped":
            from quant.layers import cuda_w8a8_static_cutlass_grouped_dual_linear

            dual_out = cuda_w8a8_static_cutlass_grouped_dual_linear(
                core,
                gate,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
            )
            if dual_out is not None:
                core_out, gate_out = dual_out
                if getattr(self.core_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
                if getattr(self.gate_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
                return core_out, gate_out
        if os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_cutlass":
            from quant.layers import cuda_w8a8_static_cutlass_linear

            core_out = cuda_w8a8_static_cutlass_linear(
                core,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
            )
            gate_out = cuda_w8a8_static_cutlass_linear(
                gate,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
            )
            if core_out is not None and gate_out is not None:
                if getattr(self.core_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
                if getattr(self.gate_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
                return core_out, gate_out

        core_out, gate_out = triton_w8a8_static_dual_linear(
            core,
            gate,
            self.core_second.q_weight,
            self.core_second.q_weight_t,
            self.core_second.scale,
            core_scale,
            core_bias,
            self.gate_second.q_weight,
            self.gate_second.q_weight_t,
            self.gate_second.scale,
            gate_scale,
            gate_bias,
        )
        if getattr(self.core_second, "collect_runtime_stats", False):
            self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
        if getattr(self.gate_second, "collect_runtime_stats", False):
            self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
        return core_out, gate_out

    @staticmethod
    def _is_silu_dropout0_prefix(prefix: nn.Sequential | None) -> bool:
        if prefix is None or len(prefix) != 2:
            return False
        layers = list(prefix.children())
        return isinstance(layers[0], FusedSiLU) and isinstance(layers[1], nn.Dropout) and layers[1].p == 0.0

    def _triton_w8a8_static_fused_silu_second(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor] | None:
        if torch.is_grad_enabled():
            return None
        import os

        if os.environ.get("MATRIS_ENABLE_TRITON_SILU_SECOND") != "1":
            return None
        if self.core_second is None or self.gate_second is None:
            return None
        if not (
            self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
        ):
            return None
        if not (
            self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
        ):
            return None

        from quant.layers import triton_w8a8_static_dual_silu_linear

        core_scale = (
            self.core_second.activation_static_scale.to(device=core.device, dtype=torch.float32).reshape(())
            if getattr(self.core_second, "activation_static_calibrated", False)
            else self.core_second._activation_scale_for(F.silu(core))
        )
        gate_scale = (
            self.gate_second.activation_static_scale.to(device=gate.device, dtype=torch.float32).reshape(())
            if getattr(self.gate_second, "activation_static_calibrated", False)
            else self.gate_second._activation_scale_for(F.silu(gate))
        )
        core_bias = self.core_second.bias.float() if self.core_second.bias is not None else None
        gate_bias = self.gate_second.bias.float() if self.gate_second.bias is not None else None
        return triton_w8a8_static_dual_silu_linear(
            core,
            gate,
            self.core_second.q_weight,
            self.core_second.q_weight_t,
            self.core_second.scale,
            core_scale,
            core_bias,
            self.gate_second.q_weight,
            self.gate_second.q_weight_t,
            self.gate_second.scale,
            gate_scale,
            gate_bias,
        )

    def _cuda_cutlass_w8a8_static_fused_second_tail(
        self,
        core: Tensor,
        gate: Tensor,
        *,
        prefix_applied: bool = False,
    ) -> Tensor | None:
        if self.training:
            return None
        import os

        backend = os.environ.get("MATRIS_W8A8_BACKEND")
        if backend not in (
            "cuda_cutlass_tail",
            "cuda_wmma_tail_n128",
            "cuda_wmma_tail_n128_parallel",
            "cuda_wmma_tail_n128_auto",
        ):
            return None
        if self.core_second is None or self.gate_second is None or self.fused_tail is None:
            return None
        if self.core_second_prefix is None or self.gate_second_prefix is None:
            return None
        if self.core_post_second_tail is None or self.gate_post_second_tail is None:
            return None
        if not (
            self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
        ):
            return None
        if len(self.core_post_second_tail) != 0 or len(self.gate_post_second_tail) != 0:
            return None

        tail = self.fused_tail
        if not (
            isinstance(tail.core_norm, nn.LayerNorm)
            and isinstance(tail.gate_norm, nn.LayerNorm)
            and tail.core_norm.normalized_shape == tail.gate_norm.normalized_shape
            and tail.core_norm.eps == tail.gate_norm.eps
            and tail.core_norm.elementwise_affine
            and tail.gate_norm.elementwise_affine
            and isinstance(tail.activation_func, FusedSiLU)
            and isinstance(tail.activation_gate, FusedSigmoid)
        ):
            return None
        if self.core_second.q_weight.shape[0] != self.gate_second.q_weight.shape[0]:
            return None

        raw_rows = int(core.reshape(-1, core.shape[-1]).shape[0])
        if (
            os.environ.get("MATRIS_P96B_SILU_SECOND_TAIL_AUTOGRAD", "0") == "1"
            and not prefix_applied
            and torch.is_grad_enabled()
            and backend in ("cuda_wmma_tail_n128", "cuda_wmma_tail_n128_parallel", "cuda_wmma_tail_n128_auto")
            and self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
            and getattr(self.core_second, "activation_static_calibrated", False)
            and getattr(self.gate_second, "activation_static_calibrated", False)
            and (
                _is_p87_refine_line_w8a8_saved_pre_target(self.module_name, raw_rows)
                or _is_p87_attn_line_w8a8_saved_pre_target(self.module_name, raw_rows)
            )
        ):
            from quant.layers import cuda_w8a8_static_wmma_dual_silu_gated_tail_n128_saved_pre_autograd

            core_dq_weight_t_precomputed = None
            gate_dq_weight_t_precomputed = None
            if os.environ.get("MATRIS_P94_PRECOMPUTE_DQ_WEIGHT_T", "0") == "1":
                core_dq_weight_t_precomputed = self._p94_cached_dq_weight_t(self.core_second)
                gate_dq_weight_t_precomputed = self._p94_cached_dq_weight_t(self.gate_second)
            core_scale = self.core_second.activation_static_scale.to(
                device=core.device,
                dtype=torch.float32,
            ).reshape(())
            gate_scale = self.gate_second.activation_static_scale.to(
                device=gate.device,
                dtype=torch.float32,
            ).reshape(())
            core_bias = self.core_second.bias.float() if self.core_second.bias is not None else None
            gate_bias = self.gate_second.bias.float() if self.gate_second.bias is not None else None
            if os.environ.get("MATRIS_P88_W8A8_RANGE_PROFILE", "0") == "1":
                profile_label = f"p88.w8a8_second_tail.{self.module_name}.forward_autograd"
                with torch.profiler.record_function(profile_label):
                    out = cuda_w8a8_static_wmma_dual_silu_gated_tail_n128_saved_pre_autograd(
                        core,
                        gate,
                        self.core_second.q_weight,
                        self.core_second.scale,
                        core_scale,
                        core_bias,
                        self.gate_second.q_weight,
                        self.gate_second.scale,
                        gate_scale,
                        gate_bias,
                        tail.core_norm.weight,
                        tail.core_norm.bias,
                        tail.gate_norm.weight,
                        tail.gate_norm.bias,
                        tail.core_norm.eps,
                        core_dq_weight_t_precomputed,
                        gate_dq_weight_t_precomputed,
                        self.module_name,
                    )
            else:
                out = cuda_w8a8_static_wmma_dual_silu_gated_tail_n128_saved_pre_autograd(
                    core,
                    gate,
                    self.core_second.q_weight,
                    self.core_second.scale,
                    core_scale,
                    core_bias,
                    self.gate_second.q_weight,
                    self.gate_second.scale,
                    gate_scale,
                    gate_bias,
                    tail.core_norm.weight,
                    tail.core_norm.bias,
                    tail.gate_norm.weight,
                    tail.gate_norm.bias,
                    tail.core_norm.eps,
                    core_dq_weight_t_precomputed,
                    gate_dq_weight_t_precomputed,
                    self.module_name,
                )
            if out is not None:
                if ".refine_block_line_graph.edge_nonlinear_update" in self.module_name:
                    _record_p60_refine_line_census(
                        module_name=self.module_name,
                        stage="p96b_w8a8_silu_second_tail_autograd",
                        rows=raw_rows,
                        input_dim=int(core.shape[-1]),
                        output_dim=int(out.shape[-1]),
                        core_hidden_dim=self.core_hidden_dim,
                        gate_hidden_dim=self.gate_hidden_dim,
                    )
                return out

        if not prefix_applied:
            core = self.core_second_prefix(core)
            gate = self.gate_second_prefix(gate)
        rows = int(core.reshape(-1, core.shape[-1]).shape[0])
        if ".refine_block_line_graph.edge_nonlinear_update" in self.module_name:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="w8a8_second_tail_candidate",
                rows=rows,
                input_dim=int(core.shape[-1]),
                output_dim=int(self.core_second.q_weight.shape[0]),
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
            )
        stats_path = os.environ.get("MATRIS_W8A8_SHAPE_STATS_PATH")
        if stats_path:
            import atexit
            import json
            from pathlib import Path

            stats = getattr(FusedInputGatedMLP, "_w8a8_shape_stats", None)
            if stats is None:
                stats = {}
                setattr(FusedInputGatedMLP, "_w8a8_shape_stats", stats)

            if not getattr(FusedInputGatedMLP, "_w8a8_shape_stats_atexit", False):
                def _dump_shape_stats() -> None:
                    path = Path(stats_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    ordered = [
                        {"shape": key, "count": count}
                        for key, count in sorted(stats.items(), key=lambda item: (-item[1], item[0]))
                    ]
                    path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

                atexit.register(_dump_shape_stats)
                setattr(FusedInputGatedMLP, "_w8a8_shape_stats_atexit", True)

            key = (
                f"module={self.module_name},rows={rows},core_k={int(core.shape[-1])},gate_k={int(gate.shape[-1])},"
                f"core_n={int(self.core_second.q_weight.shape[0])},gate_n={int(self.gate_second.q_weight.shape[0])}"
            )
            stats[key] = stats.get(key, 0) + 1

        core_scale = self.core_second._activation_scale_for(core)
        gate_scale = self.gate_second._activation_scale_for(gate)
        core_bias = self.core_second.bias.float() if self.core_second.bias is not None else None
        gate_bias = self.gate_second.bias.float() if self.gate_second.bias is not None else None
        if backend in ("cuda_wmma_tail_n128", "cuda_wmma_tail_n128_parallel", "cuda_wmma_tail_n128_auto"):
            from quant.layers import (
                cuda_w8a8_static_wmma_dual_gated_tail_n128,
                cuda_w8a8_static_wmma_dual_gated_tail_n128_input_grad_autograd,
                cuda_w8a8_static_wmma_dual_gated_tail_n128_saved_pre_autograd,
            )

            fused_bwd_max_rows_env = os.environ.get("MATRIS_W8A8_FUSED_BACKWARD_MAX_ROWS", "1024")
            try:
                fused_bwd_max_rows = int(fused_bwd_max_rows_env)
            except ValueError:
                fused_bwd_max_rows = 1024
            fused_bwd_rows = int(core.reshape(-1, core.shape[-1]).shape[0])
            aggressive_saved_pre = (
                torch.is_grad_enabled()
                and aggressive_line_attn_eval_mode() == "no_param_grad"
                and is_aggressive_edge_update_target(self.module_name)
            )
            broad_saved_pre = (
                torch.is_grad_enabled()
                and _use_aggressive_broad_input_grad_only()
                and self.module_name.startswith("interaction_block.")
            )
            p39_true_bwd_line_edge = (
                torch.is_grad_enabled()
                and _is_p39_w8a8_true_bwd_line_edge_target(self.module_name, fused_bwd_rows)
            )
            p87_refine_line_saved_pre = (
                torch.is_grad_enabled()
                and _is_p87_refine_line_w8a8_saved_pre_target(self.module_name, fused_bwd_rows)
            )
            p87_attn_line_saved_pre = (
                torch.is_grad_enabled()
                and _is_p87_attn_line_w8a8_saved_pre_target(self.module_name, fused_bwd_rows)
            )
            if (
                torch.is_grad_enabled()
                and (
                    (
                        os.environ.get("MATRIS_W8A8_ENABLE_FUSED_BACKWARD") == "1"
                        and fused_bwd_rows <= fused_bwd_max_rows
                    )
                    or aggressive_saved_pre
                    or broad_saved_pre
                    or p39_true_bwd_line_edge
                    or p87_refine_line_saved_pre
                    or p87_attn_line_saved_pre
                )
                and (
                    broad_saved_pre
                    or p39_true_bwd_line_edge
                    or p87_refine_line_saved_pre
                    or p87_attn_line_saved_pre
                    or self.module_name
                    in (
                        "interaction_block.8.attn_block_line_graph.edge_nonlinear_update",
                        "interaction_block.9.attn_block_line_graph.edge_nonlinear_update",
                    )
                )
            ):
                fused_bwd_fn = (
                    cuda_w8a8_static_wmma_dual_gated_tail_n128_saved_pre_autograd
                    if broad_saved_pre
                    or aggressive_saved_pre
                    or p39_true_bwd_line_edge
                    or p87_refine_line_saved_pre
                    or p87_attn_line_saved_pre
                    or os.environ.get("MATRIS_W8A8_FUSED_BACKWARD_MODE") == "saved_pre_torchmm"
                    else cuda_w8a8_static_wmma_dual_gated_tail_n128_input_grad_autograd
                )
                fused_bwd_extra_args = ()
                if fused_bwd_fn is cuda_w8a8_static_wmma_dual_gated_tail_n128_saved_pre_autograd:
                    core_dq_weight_t_precomputed = None
                    gate_dq_weight_t_precomputed = None
                    if os.environ.get("MATRIS_P94_PRECOMPUTE_DQ_WEIGHT_T", "0") == "1":
                        core_dq_weight_t_precomputed = self._p94_cached_dq_weight_t(self.core_second)
                        gate_dq_weight_t_precomputed = self._p94_cached_dq_weight_t(self.gate_second)
                    fused_bwd_extra_args = (
                        core_dq_weight_t_precomputed,
                        gate_dq_weight_t_precomputed,
                        self.module_name,
                    )
                if os.environ.get("MATRIS_P88_W8A8_RANGE_PROFILE", "0") == "1":
                    profile_label = f"p88.w8a8_second_tail.{self.module_name}.forward_autograd"
                    with torch.profiler.record_function(profile_label):
                        out = fused_bwd_fn(
                            core,
                            gate,
                            self.core_second.q_weight,
                            self.core_second.scale,
                            core_scale,
                            core_bias,
                            self.gate_second.q_weight,
                            self.gate_second.scale,
                            gate_scale,
                            gate_bias,
                            tail.core_norm.weight,
                            tail.core_norm.bias,
                            tail.gate_norm.weight,
                            tail.gate_norm.bias,
                            tail.core_norm.eps,
                            *fused_bwd_extra_args,
                        )
                else:
                    out = fused_bwd_fn(
                        core,
                        gate,
                        self.core_second.q_weight,
                        self.core_second.scale,
                        core_scale,
                        core_bias,
                        self.gate_second.q_weight,
                        self.gate_second.scale,
                        gate_scale,
                        gate_bias,
                        tail.core_norm.weight,
                        tail.core_norm.bias,
                        tail.gate_norm.weight,
                        tail.gate_norm.bias,
                        tail.core_norm.eps,
                        *fused_bwd_extra_args,
                    )
                if out is not None:
                    if ".refine_block_line_graph.edge_nonlinear_update" in self.module_name:
                        _record_p60_refine_line_census(
                            module_name=self.module_name,
                            stage="w8a8_second_tail_autograd",
                            rows=rows,
                            input_dim=int(core.shape[-1]),
                            output_dim=int(out.shape[-1]),
                            core_hidden_dim=self.core_hidden_dim,
                            gate_hidden_dim=self.gate_hidden_dim,
                        )
                    return out

            out = cuda_w8a8_static_wmma_dual_gated_tail_n128(
                core,
                gate,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
                tail.core_norm.weight,
                tail.core_norm.bias,
                tail.gate_norm.weight,
                tail.gate_norm.bias,
                tail.core_norm.eps,
            )
            if out is not None and ".refine_block_line_graph.edge_nonlinear_update" in self.module_name:
                _record_p60_refine_line_census(
                    module_name=self.module_name,
                    stage="w8a8_second_tail_forward",
                    rows=rows,
                    input_dim=int(core.shape[-1]),
                    output_dim=int(out.shape[-1]),
                    core_hidden_dim=self.core_hidden_dim,
                    gate_hidden_dim=self.gate_hidden_dim,
                )
            return out

        from quant.layers import cuda_w8a8_static_cutlass_dual_gated_tail

        out = cuda_w8a8_static_cutlass_dual_gated_tail(
            core,
            gate,
            self.core_second.q_weight,
            self.core_second.scale,
            core_scale,
            core_bias,
            self.gate_second.q_weight,
            self.gate_second.scale,
            gate_scale,
            gate_bias,
            tail.core_norm.weight,
            tail.core_norm.bias,
            tail.gate_norm.weight,
            tail.gate_norm.bias,
            tail.core_norm.eps,
        )
        if out is not None and ".refine_block_line_graph.edge_nonlinear_update" in self.module_name:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="w8a8_second_tail_forward",
                rows=rows,
                input_dim=int(core.shape[-1]),
                output_dim=int(out.shape[-1]),
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
            )
        return out

    @staticmethod
    def _cached_low_precision_bias(module: nn.Module, rows: int) -> Tensor:
        bias = getattr(module, "bias_low", None)
        if bias is None:
            return module.weight_low.new_zeros(rows)
        return bias

    def _cached_low_precision_fused_first(self, x: Tensor) -> tuple[Tensor, Tensor] | None:
        if not (self._is_cached_low_precision(self.core_first) and self._is_cached_low_precision(self.gate_first)):
            return None
        if self.core_first.weight_low.shape[1] != self.gate_first.weight_low.shape[1]:
            return None

        cache_key = "_cached_low_precision_first_cache"
        cache = getattr(self, cache_key, None)
        if cache is None or cache["device"] != x.device:
            weight = torch.cat([self.core_first.weight_low, self.gate_first.weight_low], dim=0).contiguous()
            bias = torch.cat(
                [
                    self._cached_low_precision_bias(self.core_first, self.core_first.weight_low.shape[0]),
                    self._cached_low_precision_bias(self.gate_first, self.gate_first.weight_low.shape[0]),
                ],
                dim=0,
            ).contiguous()
            cache = {"device": x.device, "weight": weight, "bias": bias, "dtype": weight.dtype}
            setattr(self, cache_key, cache)

        projected = F.linear(x.to(cache["dtype"]), cache["weight"], cache["bias"]).float()
        core_hidden, gate_hidden = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        self._set_fake_quant_stats(self.core_first, dict(self.core_first.last_stats), x, core_hidden)
        self._set_fake_quant_stats(self.gate_first, dict(self.gate_first.last_stats), x, gate_hidden)
        return core_hidden, gate_hidden

    def _cached_low_precision_fused_second(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor] | None:
        if self.core_second is None or self.gate_second is None:
            return None
        if not (self._is_cached_low_precision(self.core_second) and self._is_cached_low_precision(self.gate_second)):
            return None

        x = torch.cat([core, gate], dim=-1)
        cache_key = "_cached_low_precision_second_cache"
        cache = getattr(self, cache_key, None)
        if cache is None or cache["device"] != x.device:
            core_out, core_in = self.core_second.weight_low.shape
            gate_out, gate_in = self.gate_second.weight_low.shape
            weight = self.core_second.weight_low.new_zeros((core_out + gate_out, core_in + gate_in))
            weight[:core_out, :core_in] = self.core_second.weight_low
            weight[core_out:, core_in:] = self.gate_second.weight_low
            weight = weight.contiguous()
            bias = torch.cat(
                [
                    self._cached_low_precision_bias(self.core_second, core_out),
                    self._cached_low_precision_bias(self.gate_second, gate_out),
                ],
                dim=0,
            ).contiguous()
            cache = {"device": x.device, "weight": weight, "bias": bias, "dtype": weight.dtype}
            setattr(self, cache_key, cache)

        projected = F.linear(x.to(cache["dtype"]), cache["weight"], cache["bias"]).float()
        core_out, gate_out = projected.split([self.core_output_dim, self.gate_output_dim], dim=-1)
        self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
        self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
        return core_out, gate_out

    def _fused_first_projection(self, feas: Tensor) -> tuple[Tensor, Tensor]:
        if self.fused_first is not None:
            if (
                (
                    _use_aggressive_broad_input_grad_only()
                    or _use_p26_fused_first_input_grad_only()
                )
                and self.training is False
                and torch.is_grad_enabled()
                and feas.is_cuda
                and isinstance(self.fused_first, nn.Linear)
            ):
                projected = linear_input_grad_only(feas, self.fused_first)
            else:
                projected = self.fused_first(feas)
            return projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)

        low_precision_projected = self._cached_low_precision_fused_first(feas)
        if low_precision_projected is not None:
            return low_precision_projected

        p52_w8a8_projected = self._p52_w8a8_static_fused_first(feas)
        if p52_w8a8_projected is not None:
            return p52_w8a8_projected

        triton_projected = self._triton_w8a32_fused_first(feas)
        if triton_projected is not None:
            return triton_projected

        core_weight, core_bias, core_stats = self._weight_bias_stats(self.core_first)
        gate_weight, gate_bias, gate_stats = self._weight_bias_stats(self.gate_first)
        weight = torch.cat([core_weight.float(), gate_weight.float()], dim=0)

        bias = None
        if core_bias is not None or gate_bias is not None:
            if core_bias is None:
                core_bias = weight.new_zeros(core_weight.shape[0])
            if gate_bias is None:
                gate_bias = weight.new_zeros(gate_weight.shape[0])
            bias = torch.cat([core_bias.float(), gate_bias.float()], dim=0)

        x = feas.float() if core_stats is not None or gate_stats is not None else feas
        raw_x = x
        if core_stats is not None:
            x, core_stats = self._maybe_fake_quant_activation(self.core_first, raw_x, core_stats)
        if gate_stats is not None and gate_stats is not core_stats:
            _, gate_stats = self._maybe_fake_quant_activation(self.gate_first, raw_x, gate_stats)
        if (
            (
                _use_aggressive_broad_input_grad_only()
                or _use_p26_fused_first_input_grad_only()
            )
            and self.training is False
            and torch.is_grad_enabled()
            and x.is_cuda
        ):
            projected = _LinearInputGradOnly.apply(x, weight, bias)
        else:
            projected = F.linear(x, weight, bias)
        core_hidden, gate_hidden = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        self._set_fake_quant_stats(self.core_first, core_stats, x, core_hidden)
        self._set_fake_quant_stats(self.gate_first, gate_stats, x, gate_hidden)
        return core_hidden, gate_hidden

    def _fused_second_projection(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor]:
        if self.core_second is None or self.gate_second is None:
            raise RuntimeError("Second projection fusion requested without second projection modules.")

        if self.fused_second is not None:
            x = torch.cat([core, gate], dim=-1)
            if (
                _use_aggressive_broad_input_grad_only()
                and self.training is False
                and torch.is_grad_enabled()
                and x.is_cuda
                and isinstance(self.fused_second, nn.Linear)
            ):
                projected = linear_input_grad_only(x, self.fused_second)
            else:
                projected = self.fused_second(x)
            return projected.split([self.core_output_dim, self.gate_output_dim], dim=-1)

        low_precision_projected = self._cached_low_precision_fused_second(core, gate)
        if low_precision_projected is not None:
            return low_precision_projected

        triton_w8a8_projected = self._triton_w8a8_static_fused_second(core, gate)
        if triton_w8a8_projected is not None:
            return triton_w8a8_projected

        triton_projected = self._triton_w8a32_fused_second(core, gate)
        if triton_projected is not None:
            return triton_projected

        core_weight, core_bias, core_stats = self._weight_bias_stats(self.core_second)
        gate_weight, gate_bias, gate_stats = self._weight_bias_stats(self.gate_second)
        core_x = core.float() if core_stats is not None else core
        gate_x = gate.float() if gate_stats is not None else gate
        if core_stats is not None:
            core_x, core_stats = self._maybe_fake_quant_activation(self.core_second, core_x, core_stats)
        if gate_stats is not None and gate_stats is not core_stats:
            gate_x, gate_stats = self._maybe_fake_quant_activation(self.gate_second, gate_x, gate_stats)
        x = torch.cat([core_x, gate_x], dim=-1)
        weight = x.new_zeros(
            core_weight.shape[0] + gate_weight.shape[0],
            core_weight.shape[1] + gate_weight.shape[1],
        )
        weight[: core_weight.shape[0], : core_weight.shape[1]] = core_weight.float()
        weight[core_weight.shape[0] :, core_weight.shape[1] :] = gate_weight.float()

        bias = None
        if core_bias is not None or gate_bias is not None:
            if core_bias is None:
                core_bias = weight.new_zeros(core_weight.shape[0])
            if gate_bias is None:
                gate_bias = weight.new_zeros(gate_weight.shape[0])
            bias = torch.cat([core_bias.float(), gate_bias.float()], dim=0)

        if _use_aggressive_broad_input_grad_only() and self.training is False and torch.is_grad_enabled() and x.is_cuda:
            projected = _LinearInputGradOnly.apply(x, weight, bias)
        else:
            projected = F.linear(x, weight, bias)
        core_out, gate_out = projected.split([core_weight.shape[0], gate_weight.shape[0]], dim=-1)
        self._set_fake_quant_stats(self.core_second, core_stats, core_x, core_out)
        self._set_fake_quant_stats(self.gate_second, gate_stats, gate_x, gate_out)
        return core_out, gate_out

    def _forward_after_first_projection(self, core: Tensor, gate: Tensor) -> Tensor:
        if self.core_second is None:
            core = self.core_tail(core)
            gate = self.gate_tail(gate)
        else:
            fused_second_tail = self._cuda_cutlass_w8a8_static_fused_second_tail(core, gate)
            if fused_second_tail is not None:
                return fused_second_tail
            fused_silu_second = self._triton_w8a8_static_fused_silu_second(core, gate)
            if fused_silu_second is None:
                if (
                    _use_p113_gated_tail_second_silu_macro()
                    and self.training is False
                    and torch.is_grad_enabled()
                    and self.core_second_prefix is not None
                    and self.gate_second_prefix is not None
                    and self._is_silu_dropout0_prefix(self.core_second_prefix)
                    and self._is_silu_dropout0_prefix(self.gate_second_prefix)
                    and self.core_post_second_tail is not None
                    and self.gate_post_second_tail is not None
                    and len(self.core_post_second_tail) == 0
                    and len(self.gate_post_second_tail) == 0
                    and self.fused_tail is not None
                    and isinstance(self.core_second, nn.Linear)
                    and isinstance(self.gate_second, nn.Linear)
                    and isinstance(self.fused_tail.core_norm, nn.LayerNorm)
                    and isinstance(self.fused_tail.gate_norm, nn.LayerNorm)
                    and self.fused_tail.core_norm.normalized_shape == self.fused_tail.gate_norm.normalized_shape
                    and self.fused_tail.core_norm.eps == self.fused_tail.gate_norm.eps
                    and self.fused_tail.core_norm.elementwise_affine
                    and self.fused_tail.gate_norm.elementwise_affine
                    and self.fused_tail.core_norm.weight is not None
                    and self.fused_tail.core_norm.bias is not None
                    and self.fused_tail.gate_norm.weight is not None
                    and self.fused_tail.gate_norm.bias is not None
                    and isinstance(self.fused_tail.activation_func, FusedSiLU)
                    and isinstance(self.fused_tail.activation_gate, FusedSigmoid)
                    and core.is_cuda
                    and gate.is_cuda
                    and core.dtype == torch.float32
                    and gate.dtype == torch.float32
                    and core.ndim == 2
                    and gate.shape == core.shape
                    and core.shape[-1] in (128, 256)
                    and self.core_second.weight.shape == (core.shape[-1], core.shape[-1])
                    and self.gate_second.weight.shape == (core.shape[-1], core.shape[-1])
                    and self.core_second.weight.dtype == torch.float32
                    and self.gate_second.weight.dtype == torch.float32
                    and self.core_second.weight.is_cuda
                    and self.gate_second.weight.is_cuda
                    and (self.core_second.bias is None or (self.core_second.bias.is_cuda and self.core_second.bias.dtype == torch.float32))
                    and (self.gate_second.bias is None or (self.gate_second.bias.is_cuda and self.gate_second.bias.dtype == torch.float32))
                    and self.fused_tail.core_norm.weight.is_cuda
                    and self.fused_tail.core_norm.bias.is_cuda
                    and self.fused_tail.gate_norm.weight.is_cuda
                    and self.fused_tail.gate_norm.bias.is_cuda
                    and self.fused_tail.core_norm.weight.dtype == torch.float32
                    and self.fused_tail.core_norm.bias.dtype == torch.float32
                    and self.fused_tail.gate_norm.weight.dtype == torch.float32
                    and self.fused_tail.gate_norm.bias.dtype == torch.float32
                    and (lambda op: op is not None and hasattr(op, "gated_tail_second_silu_input_grad_macro"))(_load_matris_op())
                ):
                    return _P113GatedTailSecondSiluMacro.apply(
                        core,
                        gate,
                        self.core_second.weight,
                        self.core_second.bias,
                        self.gate_second.weight,
                        self.gate_second.bias,
                        self.fused_tail.core_norm.weight,
                        self.fused_tail.core_norm.bias,
                        self.fused_tail.gate_norm.weight,
                        self.fused_tail.gate_norm.bias,
                        self.fused_tail.core_norm.eps,
                        bool(core.shape[-1] == 128 and os.environ.get("MATRIS_P113_USE_TAIL_BWD_V2", "1") == "1"),
                        self.module_name,
                    )
                core = self.core_second_prefix(core)
                gate = self.gate_second_prefix(gate)
                core, gate = self._fused_second_projection(core, gate)
            else:
                core, gate = fused_silu_second
            core = self.core_post_second_tail(core)
            gate = self.gate_post_second_tail(gate)
        if self.fused_tail is not None:
            return self.fused_tail(core, gate)
        if self.core_norm is not None:
            core = self.core_norm(core)
        if self.gate_norm is not None:
            gate = self.gate_norm(gate)
        if self.activation_func is None or self.activation_gate is None:
            raise RuntimeError("GatedMLP tail is missing activation modules.")
        return self.activation_func(core) * self.activation_gate(gate)

    def _forward_after_first_activation(self, core: Tensor, gate: Tensor) -> Tensor:
        if self.core_second is None:
            core = self.core_tail(core)
            gate = self.gate_tail(gate)
        else:
            fused_second_tail = self._cuda_cutlass_w8a8_static_fused_second_tail(
                core,
                gate,
                prefix_applied=True,
            )
            if fused_second_tail is not None:
                return fused_second_tail
            core, gate = self._fused_second_projection(core, gate)
            core = self.core_post_second_tail(core)
            gate = self.gate_post_second_tail(gate)
        if self.fused_tail is not None:
            return self.fused_tail(core, gate)
        if self.core_norm is not None:
            core = self.core_norm(core)
        if self.gate_norm is not None:
            gate = self.gate_norm(gate)
        if self.activation_func is None or self.activation_gate is None:
            raise RuntimeError("GatedMLP tail is missing activation modules.")
        return self.activation_func(core) * self.activation_gate(gate)

    def forward_with_residual_after_first_projection(
        self,
        feas: Tensor,
        old_feat: Tensor,
        res_weight: Tensor,
    ) -> Tensor | None:
        if not (
            _use_p116_gated_tail_second_residual_macro()
            and _use_p113_gated_tail_second_silu_macro()
            and self.training is False
            and torch.is_grad_enabled()
            and self.fused_first is not None
            and self.core_second is not None
            and self.gate_second is not None
            and self.core_second_prefix is not None
            and self.gate_second_prefix is not None
            and self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
            and self.core_post_second_tail is not None
            and self.gate_post_second_tail is not None
            and len(self.core_post_second_tail) == 0
            and len(self.gate_post_second_tail) == 0
            and self.fused_tail is not None
            and isinstance(self.core_second, nn.Linear)
            and isinstance(self.gate_second, nn.Linear)
            and isinstance(self.fused_tail.core_norm, nn.LayerNorm)
            and isinstance(self.fused_tail.gate_norm, nn.LayerNorm)
            and self.fused_tail.core_norm.normalized_shape == self.fused_tail.gate_norm.normalized_shape
            and self.fused_tail.core_norm.eps == self.fused_tail.gate_norm.eps
            and self.fused_tail.core_norm.elementwise_affine
            and self.fused_tail.gate_norm.elementwise_affine
            and self.fused_tail.core_norm.weight is not None
            and self.fused_tail.core_norm.bias is not None
            and self.fused_tail.gate_norm.weight is not None
            and self.fused_tail.gate_norm.bias is not None
            and isinstance(self.fused_tail.activation_func, FusedSiLU)
            and isinstance(self.fused_tail.activation_gate, FusedSigmoid)
            and feas.is_cuda
            and old_feat.is_cuda
            and res_weight.is_cuda
            and feas.dtype == torch.float32
            and old_feat.dtype == torch.float32
            and res_weight.dtype == torch.float32
            and old_feat.ndim == 2
            and res_weight.ndim == 2
            and res_weight.shape[0] == 1
            and old_feat.shape[1] == res_weight.shape[1]
        ):
            return None
        projected = self._fused_first_projection(feas)
        core, gate = projected
        if not (
            core.is_cuda
            and gate.is_cuda
            and core.dtype == torch.float32
            and gate.dtype == torch.float32
            and core.ndim == 2
            and gate.shape == core.shape
            and old_feat.shape == core.shape
            and core.shape[-1] in (128, 256)
            and self.core_second.weight.shape == (core.shape[-1], core.shape[-1])
            and self.gate_second.weight.shape == (core.shape[-1], core.shape[-1])
            and self.core_second.weight.dtype == torch.float32
            and self.gate_second.weight.dtype == torch.float32
            and self.core_second.weight.is_cuda
            and self.gate_second.weight.is_cuda
            and (self.core_second.bias is None or (self.core_second.bias.is_cuda and self.core_second.bias.dtype == torch.float32))
            and (self.gate_second.bias is None or (self.gate_second.bias.is_cuda and self.gate_second.bias.dtype == torch.float32))
            and self.fused_tail.core_norm.weight.is_cuda
            and self.fused_tail.core_norm.bias.is_cuda
            and self.fused_tail.gate_norm.weight.is_cuda
            and self.fused_tail.gate_norm.bias.is_cuda
            and self.fused_tail.core_norm.weight.dtype == torch.float32
            and self.fused_tail.core_norm.bias.dtype == torch.float32
            and self.fused_tail.gate_norm.weight.dtype == torch.float32
            and self.fused_tail.gate_norm.bias.dtype == torch.float32
            and (lambda op: op is not None and hasattr(op, "gated_tail_second_silu_residual_input_grad_macro"))(_load_matris_op())
        ):
            return None
        return _P116GatedTailSecondSiluResidualMacro.apply(
            core,
            gate,
            self.core_second.weight,
            self.core_second.bias,
            self.gate_second.weight,
            self.gate_second.bias,
            self.fused_tail.core_norm.weight,
            self.fused_tail.core_norm.bias,
            self.fused_tail.gate_norm.weight,
            self.fused_tail.gate_norm.bias,
            self.fused_tail.core_norm.eps,
            bool(core.shape[-1] == 128 and os.environ.get("MATRIS_P113_USE_TAIL_BWD_V2", "1") == "1"),
            old_feat,
            res_weight,
            self.module_name,
        )

    def _forward_impl(self, feas: Tensor) -> Tensor:
        core, gate = self._fused_first_projection(feas)
        return self._forward_after_first_projection(core, gate)

    def _forward_line_edge_w8a8_big_bwd(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        source_index: Tensor,
        target_index: Tensor,
    ) -> Tensor | None:
        rows = int(edge_feat.shape[0])
        use_p48_macro = _use_p48_line_edge_w8a8_macro_bwd(rows)
        use_p44_big = _use_p44_line_edge_w8a8_big_bwd(rows)
        if not (
            (use_p48_macro or use_p44_big)
            and self.training is False
            and torch.is_grad_enabled()
            and self.fused_first is not None
            and self.core_second is not None
            and self.gate_second is not None
            and self.fused_tail is not None
            and self.core_second_prefix is not None
            and self.gate_second_prefix is not None
            and self.core_hidden_dim == 128
            and self.gate_hidden_dim == 128
            and self.fused_first.in_features == 384
            and self.fused_first.out_features == 256
            and self.fused_first.weight.shape == (256, 384)
            and self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
            and self.core_second.q_weight.shape == (128, 128)
            and self.gate_second.q_weight.shape == (128, 128)
            and getattr(self.core_second, "activation_static_calibrated", False)
            and getattr(self.gate_second, "activation_static_calibrated", False)
        ):
            return None
        core_prefix_layers = list(self.core_second_prefix.children())
        gate_prefix_layers = list(self.gate_second_prefix.children())
        if not (
            len(core_prefix_layers) == 2
            and len(gate_prefix_layers) == 2
            and isinstance(core_prefix_layers[0], FusedSiLU)
            and isinstance(gate_prefix_layers[0], FusedSiLU)
            and isinstance(core_prefix_layers[1], nn.Dropout)
            and isinstance(gate_prefix_layers[1], nn.Dropout)
            and core_prefix_layers[1].p == 0.0
            and gate_prefix_layers[1].p == 0.0
        ):
            return None
        tail = self.fused_tail
        if not (
            isinstance(tail.core_norm, nn.LayerNorm)
            and isinstance(tail.gate_norm, nn.LayerNorm)
            and tail.core_norm.normalized_shape == (128,)
            and tail.gate_norm.normalized_shape == (128,)
            and tail.core_norm.eps == tail.gate_norm.eps
            and tail.core_norm.elementwise_affine
            and tail.gate_norm.elementwise_affine
            and tail.core_norm.weight is not None
            and tail.core_norm.bias is not None
            and tail.gate_norm.weight is not None
            and tail.gate_norm.bias is not None
            and isinstance(tail.activation_func, FusedSiLU)
            and isinstance(tail.activation_gate, FusedSigmoid)
        ):
            return None
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "line_edge_gather_cat_forward")
            and hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre")
        ):
            return None
        if use_p48_macro:
            if not (
                hasattr(matris_op, "w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128")
                and hasattr(matris_op, "line_edge_silu_project_grad_scatter_backward_tile32")
            ):
                return None
        elif not hasattr(matris_op, "line_edge_w8a8_tail_project_scatter_backward_n128"):
            return None
        core_scale = self.core_second.activation_static_scale.to(
            device=edge_feat.device,
            dtype=torch.float32,
        ).reshape(())
        gate_scale = self.gate_second.activation_static_scale.to(
            device=edge_feat.device,
            dtype=torch.float32,
        ).reshape(())
        core_bias = self.core_second.bias.float() if self.core_second.bias is not None else None
        gate_bias = self.gate_second.bias.float() if self.gate_second.bias is not None else None
        if use_p48_macro:
            return _LineEdgeW8A8MacroTile32InputGrad.apply(
                node_feat,
                edge_feat,
                source_index,
                target_index,
                self.fused_first.weight,
                self.fused_first.bias,
                self.core_second.q_weight,
                self.gate_second.q_weight,
                self.core_second.scale,
                self.gate_second.scale,
                core_scale,
                gate_scale,
                core_bias,
                gate_bias,
                tail.core_norm.weight,
                tail.core_norm.bias,
                tail.gate_norm.weight,
                tail.gate_norm.bias,
                tail.core_norm.eps,
            )
        _record_p44_line_edge_w8a8_big_bwd_hit()
        return _LineEdgeW8A8SecondTailInputGrad.apply(
            node_feat,
            edge_feat,
            source_index,
            target_index,
            self.fused_first.weight,
            self.fused_first.bias,
            self.core_second.q_weight,
            self.gate_second.q_weight,
            self.core_second.scale,
            self.gate_second.scale,
            core_scale,
            gate_scale,
            core_bias,
            gate_bias,
            tail.core_norm.weight,
            tail.core_norm.bias,
            tail.gate_norm.weight,
            tail.gate_norm.bias,
            tail.core_norm.eps,
        )

    def forward_line_edge(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        source_index: Tensor,
        target_index: Tensor,
    ) -> Tensor | None:
        if self.fused_first is None:
            return None
        if not (
            node_feat.is_cuda
            and edge_feat.is_cuda
            and source_index.is_cuda
            and target_index.is_cuda
            and node_feat.dtype == torch.float32
            and edge_feat.dtype == torch.float32
            and source_index.dtype == torch.int64
            and target_index.dtype == torch.int64
            and node_feat.ndim == 2
            and edge_feat.ndim == 2
            and node_feat.shape[1] == 128
            and edge_feat.shape[1] == 128
            and source_index.ndim == 1
            and target_index.ndim == 1
            and source_index.shape[0] == edge_feat.shape[0]
            and target_index.shape[0] == edge_feat.shape[0]
            and self.fused_first.in_features == 384
            and self.fused_first.out_features == self.core_hidden_dim + self.gate_hidden_dim
        ):
            return None
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "line_edge_gather_cat_forward")
            and hasattr(matris_op, "line_edge_cat_grad_scatter_backward")
        ):
            return None
        p44_out = self._forward_line_edge_w8a8_big_bwd(
            node_feat,
            edge_feat,
            source_index,
            target_index,
        )
        if p44_out is not None:
            return p44_out
        projected = _LineEdgeFirstProjection.apply(
            node_feat,
            edge_feat,
            source_index,
            target_index,
            self.fused_first.weight,
            self.fused_first.bias,
        )
        core, gate = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        return self._forward_after_first_projection(core, gate)

    def forward_refine_atom_edge(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        edge_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
    ) -> Tensor | None:
        rows = int(source_index.shape[0]) if source_index.ndim > 0 else 0
        use_p110 = _use_p110_refine_atom_edge_update(rows)
        use_p111 = _use_p111_refine_atom_fused_first(rows)
        if not (use_p110 or use_p111):
            return None
        if self.fused_first is None:
            return None
        if not (
            node_feat.is_cuda
            and edge_feat.is_cuda
            and edge_index.is_cuda
            and source_index.is_cuda
            and target_index.is_cuda
            and node_feat.dtype == torch.float32
            and edge_feat.dtype == torch.float32
            and edge_index.dtype == torch.int64
            and source_index.dtype == torch.int64
            and target_index.dtype == torch.int64
            and node_feat.ndim == 2
            and edge_feat.ndim == 2
            and node_feat.shape[1] == 128
            and edge_feat.shape[1] == 128
            and edge_index.ndim == 1
            and source_index.ndim == 1
            and target_index.ndim == 1
            and edge_index.shape[0] == source_index.shape[0]
            and target_index.shape[0] == source_index.shape[0]
            and self.fused_first.in_features == 384
            and self.fused_first.out_features == self.core_hidden_dim + self.gate_hidden_dim
        ):
            return None
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "directed_edge_gather_cat_forward")
            and hasattr(matris_op, "directed_edge_cat_grad_scatter_backward")
        ):
            return None
        if (
            use_p111
            and self.training is False
            and torch.is_grad_enabled()
            and isinstance(self.fused_first, nn.Linear)
            and self.fused_first.weight.shape == (256, 384)
            and self.core_hidden_dim == 128
            and self.gate_hidden_dim == 128
            and self.core_second_prefix is not None
            and self.gate_second_prefix is not None
            and self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
        ):
            core, gate = _P111RefineAtomFirstSilu.apply(
                node_feat,
                edge_feat,
                edge_index,
                source_index,
                target_index,
                self.fused_first.weight,
                self.fused_first.bias,
            )
            return self._forward_after_first_activation(core, gate)
        projected = _RefineAtomEdgeFirstProjection.apply(
            node_feat,
            edge_feat,
            edge_index,
            source_index,
            target_index,
            self.fused_first.weight,
            self.fused_first.bias,
        )
        core, gate = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        return self._forward_after_first_projection(core, gate)

    def forward_line_attention_macro(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        source_alpha_linear: nn.Linear,
        target_alpha_linear: nn.Linear,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        rows = int(edge_feat.shape[0])
        if not (
            use_p49_line_attention_macro(self.module_name, rows)
            and self.training is False
            and torch.is_grad_enabled()
            and self.fused_first is not None
            and self.core_second is not None
            and self.gate_second is not None
            and self.fused_tail is not None
            and self.core_second_prefix is not None
            and self.gate_second_prefix is not None
            and self.core_hidden_dim == 128
            and self.gate_hidden_dim == 128
            and self.fused_first.in_features == 384
            and self.fused_first.out_features == 256
            and self.fused_first.weight.shape == (256, 384)
            and isinstance(source_alpha_linear, nn.Linear)
            and isinstance(target_alpha_linear, nn.Linear)
            and source_alpha_linear.weight.shape == (128, 128)
            and target_alpha_linear.weight.shape == (128, 128)
            and self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
            and self.core_second.q_weight.shape == (128, 128)
            and self.gate_second.q_weight.shape == (128, 128)
            and getattr(self.core_second, "activation_static_calibrated", False)
            and getattr(self.gate_second, "activation_static_calibrated", False)
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
            and node_feat.shape[1] == 128
            and edge_feat.shape[1] == 128
        ):
            return None
        core_prefix_layers = list(self.core_second_prefix.children())
        gate_prefix_layers = list(self.gate_second_prefix.children())
        if not (
            len(core_prefix_layers) == 2
            and len(gate_prefix_layers) == 2
            and isinstance(core_prefix_layers[0], FusedSiLU)
            and isinstance(gate_prefix_layers[0], FusedSiLU)
            and isinstance(core_prefix_layers[1], nn.Dropout)
            and isinstance(gate_prefix_layers[1], nn.Dropout)
            and core_prefix_layers[1].p == 0.0
            and gate_prefix_layers[1].p == 0.0
        ):
            return None
        tail = self.fused_tail
        if not (
            isinstance(tail.core_norm, nn.LayerNorm)
            and isinstance(tail.gate_norm, nn.LayerNorm)
            and tail.core_norm.normalized_shape == (128,)
            and tail.gate_norm.normalized_shape == (128,)
            and tail.core_norm.eps == tail.gate_norm.eps
            and tail.core_norm.elementwise_affine
            and tail.gate_norm.elementwise_affine
            and tail.core_norm.weight is not None
            and tail.core_norm.bias is not None
            and tail.gate_norm.weight is not None
            and tail.gate_norm.bias is not None
            and isinstance(tail.activation_func, FusedSiLU)
            and isinstance(tail.activation_gate, FusedSigmoid)
        ):
            return None
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "line_edge_gather_cat_forward")
            and hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre")
            and hasattr(matris_op, "fused_line_attention_forward")
            and hasattr(matris_op, "fused_line_attention_backward")
            and hasattr(matris_op, "w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128")
            and hasattr(matris_op, "line_edge_silu_project_grad_scatter_backward_tile32")
        ):
            return None
        core_scale = self.core_second.activation_static_scale.to(
            device=edge_feat.device,
            dtype=torch.float32,
        ).reshape(())
        gate_scale = self.gate_second.activation_static_scale.to(
            device=edge_feat.device,
            dtype=torch.float32,
        ).reshape(())
        core_bias = self.core_second.bias.float() if self.core_second.bias is not None else None
        gate_bias = self.gate_second.bias.float() if self.gate_second.bias is not None else None
        return _LineAttentionEdgeMacroInputGrad.apply(
            node_feat,
            edge_feat,
            source_index,
            target_index,
            self.fused_first.weight,
            self.fused_first.bias,
            self.core_second.q_weight,
            self.gate_second.q_weight,
            self.core_second.scale,
            self.gate_second.scale,
            core_scale,
            gate_scale,
            core_bias,
            gate_bias,
            tail.core_norm.weight,
            tail.core_norm.bias,
            tail.gate_norm.weight,
            tail.gate_norm.bias,
            source_alpha_linear.weight,
            source_alpha_linear.bias,
            target_alpha_linear.weight,
            target_alpha_linear.bias,
            tail.core_norm.eps,
            int(num_nodes),
        )

    def forward_directed_edge(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        edge_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
    ) -> Tensor | None:
        if self.fused_first is None:
            return None
        if not (
            node_feat.is_cuda
            and edge_feat.is_cuda
            and edge_index.is_cuda
            and source_index.is_cuda
            and target_index.is_cuda
            and node_feat.dtype == torch.float32
            and edge_feat.dtype == torch.float32
            and edge_index.dtype == torch.int64
            and source_index.dtype == torch.int64
            and target_index.dtype == torch.int64
            and node_feat.ndim == 2
            and edge_feat.ndim == 2
            and node_feat.shape[1] == 128
            and edge_feat.shape[1] == 128
            and edge_index.ndim == 1
            and source_index.ndim == 1
            and target_index.ndim == 1
            and edge_index.shape[0] == source_index.shape[0]
            and target_index.shape[0] == source_index.shape[0]
            and self.fused_first.in_features == 384
            and self.fused_first.out_features == self.core_hidden_dim + self.gate_hidden_dim
        ):
            return None
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "directed_edge_gather_cat_forward")
            and hasattr(matris_op, "directed_edge_cat_grad_scatter_backward")
        ):
            return None
        projected = _DirectedEdgeFirstProjection.apply(
            node_feat,
            edge_feat,
            edge_index,
            source_index,
            target_index,
            self.fused_first.weight,
            self.fused_first.bias,
        )
        core, gate = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        return self._forward_after_first_projection(core, gate)

    def forward_refine_line_edge_smooth_reduce(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        atom_feat: Tensor,
        base_envelope: Tensor,
        atom_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor] | None:
        rows = int(edge_feat.shape[0]) if edge_feat.ndim > 0 else 0
        if self.fused_first is None:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p63_skip",
                rows=rows,
                reason="no_fused_first",
            )
            return None
        eligible = (
            node_feat.is_cuda
            and edge_feat.is_cuda
            and atom_feat.is_cuda
            and base_envelope.is_cuda
            and atom_index.is_cuda
            and source_index.is_cuda
            and target_index.is_cuda
            and node_feat.dtype == torch.float32
            and edge_feat.dtype == torch.float32
            and atom_feat.dtype == torch.float32
            and base_envelope.dtype == torch.float32
            and atom_index.dtype == torch.int64
            and source_index.dtype == torch.int64
            and target_index.dtype == torch.int64
            and node_feat.ndim == 2
            and edge_feat.ndim == 2
            and atom_feat.ndim == 2
            and base_envelope.ndim == 2
            and node_feat.shape[1] == 128
            and edge_feat.shape[1] == 128
            and atom_feat.shape[1] == 128
            and base_envelope.shape[1] == 128
            and int(base_envelope.shape[0]) == int(num_nodes)
            and atom_index.ndim == 1
            and source_index.ndim == 1
            and target_index.ndim == 1
            and atom_index.shape[0] == edge_feat.shape[0]
            and source_index.shape[0] == edge_feat.shape[0]
            and target_index.shape[0] == edge_feat.shape[0]
            and self.fused_first.in_features == 512
            and self.fused_first.out_features == self.core_hidden_dim + self.gate_hidden_dim
        )
        if not eligible:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p63_skip",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                reason="precondition",
            )
            return None
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "refine_line_first_silu_forward")
            and hasattr(matris_op, "refine_line_first_silu_backward")
            and hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre")
            and hasattr(matris_op, "w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128")
            and hasattr(matris_op, "refine_line_smooth_reduce_forward")
            and hasattr(matris_op, "refine_line_smooth_reduce_backward")
        ):
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p63_skip",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                reason="missing_p63_kernels",
            )
            return None

        backend = os.environ.get("MATRIS_W8A8_BACKEND", "")
        tail = self.fused_tail
        dynamic_activation_scale = os.environ.get(
            "MATRIS_P63_REFINE_LINE_EDGE_SMOOTH_REDUCE_DYNAMIC_SCALE",
            "0",
        ) == "1"
        if dynamic_activation_scale:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p63_skip",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                reason="dynamic_scale_not_implemented",
            )
            return None
        use_p64 = _use_p64_refine_line_fused_backward(self.module_name, rows)
        use_p65 = _use_p65_refine_line_tiled_fused_backward(self.module_name, rows)
        use_p65b = _use_p65b_refine_line_packed_tiled_backward(self.module_name, rows)
        use_p69 = _use_p69_refine_line_first_tail_smooth_reduce(self.module_name, rows)
        use_p63 = (
            (_use_p63_refine_line_edge_smooth_reduce(self.module_name, rows) or use_p64 or use_p65 or use_p65b or use_p69)
            and self.training is False
            and torch.is_grad_enabled()
            and backend in ("cuda_wmma_tail_n128", "cuda_wmma_tail_n128_parallel", "cuda_wmma_tail_n128_auto")
            and isinstance(self.fused_first, nn.Linear)
            and self.fused_first.weight.shape == (256, 512)
            and self.core_hidden_dim == 128
            and self.gate_hidden_dim == 128
            and self.core_second is not None
            and self.gate_second is not None
            and self.core_second_prefix is not None
            and self.gate_second_prefix is not None
            and self.core_post_second_tail is not None
            and self.gate_post_second_tail is not None
            and len(self.core_post_second_tail) == 0
            and len(self.gate_post_second_tail) == 0
            and self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
            and self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
            and self.core_second.q_weight.shape == (128, 128)
            and self.gate_second.q_weight.shape == (128, 128)
            and getattr(self.core_second, "activation_static_calibrated", False)
            and getattr(self.gate_second, "activation_static_calibrated", False)
            and tail is not None
            and isinstance(tail.core_norm, nn.LayerNorm)
            and isinstance(tail.gate_norm, nn.LayerNorm)
            and tail.core_norm.normalized_shape == tail.gate_norm.normalized_shape
            and tail.core_norm.eps == tail.gate_norm.eps
            and tail.core_norm.elementwise_affine
            and tail.gate_norm.elementwise_affine
            and isinstance(tail.activation_func, FusedSiLU)
            and isinstance(tail.activation_gate, FusedSigmoid)
            and (not use_p69 or hasattr(matris_op, "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre"))
            and (not use_p65 or hasattr(matris_op, "refine_line_smooth_tail_first_silu_backward_tile"))
            and (
                not use_p65b
                or (
                    hasattr(matris_op, "refine_line_smooth_w8a8_tail_actgrad_backward_n128")
                    and hasattr(matris_op, "refine_line_first_silu_backward_packed")
                )
            )
        )
        if not use_p63:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p63_skip",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                reason="guard",
            )
            return None

        _record_p60_refine_line_census(
            module_name=self.module_name,
            stage=(
                "p69_first_tail_smooth_reduce_forward"
                if use_p69
                else (
                    "p65b_packed_tail_tiled_first_backward"
                    if use_p65b
                    else (
                        "p65_tiled_smooth_tail_first_backward"
                        if use_p65
                        else ("p64_edge_smooth_fused_backward" if use_p64 else "p63_edge_smooth_reduce_macro")
                    )
                )
            ),
            rows=rows,
            input_dim=int(self.fused_first.in_features),
            output_dim=self.output_dim,
            core_hidden_dim=self.core_hidden_dim,
            gate_hidden_dim=self.gate_hidden_dim,
            node_rows=int(node_feat.shape[0]),
            edge_rows=rows,
            atom_rows=int(atom_feat.shape[0]),
        )
        return _P63RefineLineEdgeSmoothReduce.apply(
            node_feat,
            edge_feat,
            atom_feat,
            base_envelope,
            atom_index,
            source_index,
            target_index,
            self.fused_first.weight,
            self.fused_first.bias,
            self.core_second.q_weight,
            self.gate_second.q_weight,
            self.core_second.scale,
            self.gate_second.scale,
            self.core_second.activation_static_scale.to(device=edge_feat.device, dtype=torch.float32).reshape(()),
            self.gate_second.activation_static_scale.to(device=edge_feat.device, dtype=torch.float32).reshape(()),
            self.core_second.bias,
            self.gate_second.bias,
            tail.core_norm.weight,
            tail.core_norm.bias,
            tail.gate_norm.weight,
            tail.gate_norm.bias,
            float(tail.core_norm.eps),
            int(num_nodes),
        )

    def forward_refine_line_edge(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        atom_feat: Tensor,
        atom_index: Tensor,
        source_index: Tensor,
        target_index: Tensor,
    ) -> Tensor | None:
        rows = int(edge_feat.shape[0]) if edge_feat.ndim > 0 else 0
        _record_p60_refine_line_census(
            module_name=self.module_name,
            stage="enter",
            rows=rows,
            input_dim=int(self.fused_first.in_features) if self.fused_first is not None else None,
            output_dim=self.output_dim,
            core_hidden_dim=self.core_hidden_dim,
            gate_hidden_dim=self.gate_hidden_dim,
            node_rows=int(node_feat.shape[0]) if node_feat.ndim > 0 else None,
            edge_rows=rows,
            atom_rows=int(atom_feat.shape[0]) if atom_feat.ndim > 0 else None,
        )
        if self.fused_first is None:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="skip",
                rows=rows,
                reason="no_fused_first",
            )
            return None
        eligible = (
            node_feat.is_cuda
            and edge_feat.is_cuda
            and atom_feat.is_cuda
            and atom_index.is_cuda
            and source_index.is_cuda
            and target_index.is_cuda
            and node_feat.dtype == torch.float32
            and edge_feat.dtype == torch.float32
            and atom_feat.dtype == torch.float32
            and atom_index.dtype == torch.int64
            and source_index.dtype == torch.int64
            and target_index.dtype == torch.int64
            and node_feat.ndim == 2
            and edge_feat.ndim == 2
            and atom_feat.ndim == 2
            and node_feat.shape[1] == 128
            and edge_feat.shape[1] == 128
            and atom_feat.shape[1] == 128
            and atom_index.ndim == 1
            and source_index.ndim == 1
            and target_index.ndim == 1
            and atom_index.shape[0] == edge_feat.shape[0]
            and source_index.shape[0] == edge_feat.shape[0]
            and target_index.shape[0] == edge_feat.shape[0]
            and self.fused_first.in_features == 512
            and self.fused_first.out_features == self.core_hidden_dim + self.gate_hidden_dim
        )
        if not eligible:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="skip",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                reason="precondition",
            )
            return None
        matris_op = _load_matris_op()
        if matris_op is None or not (
            hasattr(matris_op, "refine_line_edge_gather_cat_forward")
            and hasattr(matris_op, "refine_line_edge_cat_grad_scatter_backward")
        ):
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="skip",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                reason="missing_refine_line_gather_kernel",
            )
            return None
        use_p60_fused_first = (
            _use_p60_refine_line_fused_first(self.module_name, rows)
            and self.training is False
            and torch.is_grad_enabled()
            and isinstance(self.fused_first, nn.Linear)
            and self.fused_first.weight.shape == (256, 512)
            and self.core_hidden_dim == 128
            and self.gate_hidden_dim == 128
            and self.core_second_prefix is not None
            and self.gate_second_prefix is not None
            and self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
            and hasattr(matris_op, "refine_line_first_silu_forward")
            and hasattr(matris_op, "refine_line_first_silu_backward")
        )
        use_p61_fused_first_acts = (
            _use_p61_refine_line_fused_first_acts(self.module_name, rows)
            and self.training is False
            and torch.is_grad_enabled()
            and isinstance(self.fused_first, nn.Linear)
            and self.fused_first.weight.shape == (256, 512)
            and self.core_hidden_dim == 128
            and self.gate_hidden_dim == 128
            and self.core_second_prefix is not None
            and self.gate_second_prefix is not None
            and self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
            and hasattr(matris_op, "refine_line_first_silu_forward_acts")
        )
        backend = os.environ.get("MATRIS_W8A8_BACKEND", "")
        tail = self.fused_tail
        p61_dynamic_activation_scale = os.environ.get("MATRIS_P61_REFINE_LINE_FIRST_TAIL_DYNAMIC_SCALE", "1") != "0"
        p61_tile_scale_partials = os.environ.get("MATRIS_P61_REFINE_LINE_FIRST_TAIL_TILE_SCALE", "0") == "1"
        p62_dynamic_activation_scale = os.environ.get("MATRIS_P62_REFINE_LINE_PACKED_FIRST_TAIL_DYNAMIC_SCALE", "1") != "0"
        p62_tile_scale_partials = os.environ.get("MATRIS_P62_REFINE_LINE_PACKED_FIRST_TAIL_TILE_SCALE", "1") != "0"
        use_p62_packed_first_tail = (
            _use_p62_refine_line_packed_first_tail(self.module_name, rows)
            and self.training is False
            and torch.is_grad_enabled()
            and backend in ("cuda_wmma_tail_n128", "cuda_wmma_tail_n128_parallel", "cuda_wmma_tail_n128_auto")
            and isinstance(self.fused_first, nn.Linear)
            and self.fused_first.weight.shape == (256, 512)
            and self.core_hidden_dim == 128
            and self.gate_hidden_dim == 128
            and self.core_second is not None
            and self.gate_second is not None
            and self.core_second_prefix is not None
            and self.gate_second_prefix is not None
            and self.core_post_second_tail is not None
            and self.gate_post_second_tail is not None
            and len(self.core_post_second_tail) == 0
            and len(self.gate_post_second_tail) == 0
            and self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
            and self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
            and self.core_second.q_weight.shape == (128, 128)
            and self.gate_second.q_weight.shape == (128, 128)
            and (
                p62_dynamic_activation_scale
                or (
                    getattr(self.core_second, "activation_static_calibrated", False)
                    and getattr(self.gate_second, "activation_static_calibrated", False)
                )
            )
            and tail is not None
            and isinstance(tail.core_norm, nn.LayerNorm)
            and isinstance(tail.gate_norm, nn.LayerNorm)
            and tail.core_norm.normalized_shape == tail.gate_norm.normalized_shape
            and tail.core_norm.eps == tail.gate_norm.eps
            and tail.core_norm.elementwise_affine
            and tail.gate_norm.elementwise_affine
            and isinstance(tail.activation_func, FusedSiLU)
            and isinstance(tail.activation_gate, FusedSigmoid)
            and hasattr(matris_op, "refine_line_first_tail_w8a8_packed_forward")
        )
        use_p61_first_tail = (
            _use_p61_refine_line_first_tail(self.module_name, rows)
            and self.training is False
            and torch.is_grad_enabled()
            and backend in ("cuda_wmma_tail_n128", "cuda_wmma_tail_n128_parallel", "cuda_wmma_tail_n128_auto")
            and isinstance(self.fused_first, nn.Linear)
            and self.fused_first.weight.shape == (256, 512)
            and self.core_hidden_dim == 128
            and self.gate_hidden_dim == 128
            and self.core_second is not None
            and self.gate_second is not None
            and self.core_second_prefix is not None
            and self.gate_second_prefix is not None
            and self.core_post_second_tail is not None
            and self.gate_post_second_tail is not None
            and len(self.core_post_second_tail) == 0
            and len(self.gate_post_second_tail) == 0
            and self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
            and self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
            and self.core_second.q_weight.shape == (128, 128)
            and self.gate_second.q_weight.shape == (128, 128)
            and hasattr(self.core_second, "activation_static_scale")
            and hasattr(self.gate_second, "activation_static_scale")
            and tail is not None
            and isinstance(tail.core_norm, nn.LayerNorm)
            and isinstance(tail.gate_norm, nn.LayerNorm)
            and tail.core_norm.normalized_shape == tail.gate_norm.normalized_shape
            and tail.core_norm.eps == tail.gate_norm.eps
            and tail.core_norm.elementwise_affine
            and tail.gate_norm.elementwise_affine
            and isinstance(tail.activation_func, FusedSiLU)
            and isinstance(tail.activation_gate, FusedSigmoid)
            and hasattr(matris_op, "refine_line_first_tail_w8a8_forward")
        )
        if use_p62_packed_first_tail:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p62_packed_first_tail_w8a8",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                node_rows=int(node_feat.shape[0]),
                edge_rows=rows,
                atom_rows=int(atom_feat.shape[0]),
            )
            first_bias = (
                self.fused_first.bias.contiguous()
                if self.fused_first.bias is not None
                else self.fused_first.weight.new_empty(0)
            )
            core_bias = (
                self.core_second.bias.float().contiguous()
                if self.core_second.bias is not None
                else self.fused_first.weight.new_empty(0)
            )
            gate_bias = (
                self.gate_second.bias.float().contiguous()
                if self.gate_second.bias is not None
                else self.fused_first.weight.new_empty(0)
            )
            use_parallel_tail = backend == "cuda_wmma_tail_n128_parallel"
            if backend == "cuda_wmma_tail_n128_auto":
                threshold = _env_int("MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_ROWS", _env_int("MATRIS_W8A8_TAIL_N128_AUTO_THRESHOLD", 4096))
                when = os.environ.get("MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_WHEN", "")
                if when == "lt":
                    use_parallel_tail = rows < threshold
                elif when == "always":
                    use_parallel_tail = True
                elif when == "never":
                    use_parallel_tail = False
                else:
                    use_parallel_tail = rows >= threshold
            out = matris_op.refine_line_first_tail_w8a8_packed_forward(
                node_feat.contiguous(),
                edge_feat.contiguous(),
                atom_feat.contiguous(),
                atom_index.contiguous(),
                source_index.contiguous(),
                target_index.contiguous(),
                self.fused_first.weight.contiguous(),
                first_bias,
                self.fused_first.bias is not None,
                self.core_second.q_weight.contiguous(),
                self.gate_second.q_weight.contiguous(),
                self.core_second.scale.float().contiguous(),
                self.gate_second.scale.float().contiguous(),
                self.core_second.activation_static_scale.to(device=edge_feat.device, dtype=torch.float32).reshape(()).contiguous(),
                self.gate_second.activation_static_scale.to(device=edge_feat.device, dtype=torch.float32).reshape(()).contiguous(),
                core_bias,
                gate_bias,
                self.core_second.bias is not None,
                self.gate_second.bias is not None,
                tail.core_norm.weight.float().contiguous(),
                tail.core_norm.bias.float().contiguous(),
                tail.gate_norm.weight.float().contiguous(),
                tail.gate_norm.bias.float().contiguous(),
                float(tail.core_norm.eps),
                bool(use_parallel_tail),
                bool(p62_dynamic_activation_scale),
                bool(p62_tile_scale_partials),
            )
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p62_packed_first_tail_w8a8_forward",
                rows=rows,
                input_dim=128,
                output_dim=int(out.shape[-1]),
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
            )
            return out
        if use_p61_first_tail:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p61_first_tail_w8a8",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                node_rows=int(node_feat.shape[0]),
                edge_rows=rows,
                atom_rows=int(atom_feat.shape[0]),
            )
            first_bias = (
                self.fused_first.bias.contiguous()
                if self.fused_first.bias is not None
                else self.fused_first.weight.new_empty(0)
            )
            core_bias = (
                self.core_second.bias.float().contiguous()
                if self.core_second.bias is not None
                else self.fused_first.weight.new_empty(0)
            )
            gate_bias = (
                self.gate_second.bias.float().contiguous()
                if self.gate_second.bias is not None
                else self.fused_first.weight.new_empty(0)
            )
            use_parallel_tail = backend == "cuda_wmma_tail_n128_parallel"
            if backend == "cuda_wmma_tail_n128_auto":
                threshold = _env_int("MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_ROWS", _env_int("MATRIS_W8A8_TAIL_N128_AUTO_THRESHOLD", 4096))
                when = os.environ.get("MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_WHEN", "")
                if when == "lt":
                    use_parallel_tail = rows < threshold
                elif when == "always":
                    use_parallel_tail = True
                elif when == "never":
                    use_parallel_tail = False
                else:
                    use_parallel_tail = rows >= threshold
            out = matris_op.refine_line_first_tail_w8a8_forward(
                node_feat.contiguous(),
                edge_feat.contiguous(),
                atom_feat.contiguous(),
                atom_index.contiguous(),
                source_index.contiguous(),
                target_index.contiguous(),
                self.fused_first.weight.contiguous(),
                first_bias,
                self.fused_first.bias is not None,
                self.core_second.q_weight.contiguous(),
                self.gate_second.q_weight.contiguous(),
                self.core_second.scale.float().contiguous(),
                self.gate_second.scale.float().contiguous(),
                self.core_second.activation_static_scale.to(device=edge_feat.device, dtype=torch.float32).reshape(()).contiguous(),
                self.gate_second.activation_static_scale.to(device=edge_feat.device, dtype=torch.float32).reshape(()).contiguous(),
                core_bias,
                gate_bias,
                self.core_second.bias is not None,
                self.gate_second.bias is not None,
                tail.core_norm.weight.float().contiguous(),
                tail.core_norm.bias.float().contiguous(),
                tail.gate_norm.weight.float().contiguous(),
                tail.gate_norm.bias.float().contiguous(),
                float(tail.core_norm.eps),
                bool(use_parallel_tail),
                bool(p61_dynamic_activation_scale),
                bool(p61_tile_scale_partials),
            )
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p61_first_tail_w8a8_forward",
                rows=rows,
                input_dim=128,
                output_dim=int(out.shape[-1]),
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
            )
            return out
        if use_p61_fused_first_acts:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p61_fused_first_acts",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                node_rows=int(node_feat.shape[0]),
                edge_rows=rows,
                atom_rows=int(atom_feat.shape[0]),
            )
            bias_arg = (
                self.fused_first.bias.contiguous()
                if self.fused_first.bias is not None
                else self.fused_first.weight.new_empty(0)
            )
            core, gate = matris_op.refine_line_first_silu_forward_acts(
                node_feat.contiguous(),
                edge_feat.contiguous(),
                atom_feat.contiguous(),
                atom_index.contiguous(),
                source_index.contiguous(),
                target_index.contiguous(),
                self.fused_first.weight.contiguous(),
                bias_arg,
                self.fused_first.bias is not None,
            )
            return self._forward_after_first_activation(core, gate)
        if use_p60_fused_first:
            _record_p60_refine_line_census(
                module_name=self.module_name,
                stage="p60_fused_first",
                rows=rows,
                input_dim=int(self.fused_first.in_features),
                output_dim=self.output_dim,
                core_hidden_dim=self.core_hidden_dim,
                gate_hidden_dim=self.gate_hidden_dim,
                node_rows=int(node_feat.shape[0]),
                edge_rows=rows,
                atom_rows=int(atom_feat.shape[0]),
            )
            core, gate = _P60RefineLineFirstSilu.apply(
                node_feat,
                edge_feat,
                atom_feat,
                atom_index,
                source_index,
                target_index,
                self.fused_first.weight,
                self.fused_first.bias,
            )
            return self._forward_after_first_activation(core, gate)
        _record_p60_refine_line_census(
            module_name=self.module_name,
            stage="fused_first",
            rows=rows,
            input_dim=int(self.fused_first.in_features),
            output_dim=self.output_dim,
            core_hidden_dim=self.core_hidden_dim,
            gate_hidden_dim=self.gate_hidden_dim,
            node_rows=int(node_feat.shape[0]),
            edge_rows=rows,
            atom_rows=int(atom_feat.shape[0]),
        )
        projected = _RefineLineEdgeFirstProjection.apply(
            node_feat,
            edge_feat,
            atom_feat,
            atom_index,
            source_index,
            target_index,
            self.fused_first.weight,
            self.fused_first.bias,
        )
        core, gate = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        return self._forward_after_first_projection(core, gate)

    def forward(self, feas: Tensor) -> Tensor:
        if _use_aggressive_broad_bypass_gated_mlp() and self.training is False and torch.is_grad_enabled():
            return _aggressive_bypass_project(feas, self.output_dim)
        if self.use_fp16 and feas.is_cuda:
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                out = self._forward_impl(feas)
            return out.to(torch.float32)
        return self._forward_impl(feas)


class MOE_Layer(nn.Module):
    
    def __init__(
        self,
        num_expert: int = 64,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MOE layer.

        Args:
            
        """
        super().__init__()
        
        raise NotImplementedError
         
    def forward(self, feas: Tensor) -> Tensor:
        return None


class GraphPooling(nn.Module):
    def __init__(self, average: bool = False) -> None:
        
        super().__init__()
        self.average = average

    def forward(self, node_feat: Tensor, segment: Tensor) -> Tensor:
        """
        Args:
            atom_feat (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, node_feat_dim or 1]
            segment (Tensor): graph indices for each atom.
                [num_batch_atoms]
        
        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, node_feat_dim or 1]
        """
        bin_count = torch.bincount(segment)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

        output = node_feat.new_zeros([bin_count.shape[0], node_feat.shape[1]])
        output = output.index_add_(0, segment, node_feat)
        if self.average:
            output = (output.T / bin_count).T
        return output


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.tensor:
    if ang_mom not in [2]:
        raise NotImplementedError

    if ang_mom == 2:
        change_mat = torch.tensor(
            [
                [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                [
                    -(6 ** (-0.5)),
                    0,
                    0,
                    0,
                    2 * 6 ** (-0.5),
                    0,
                    0,
                    0,
                    -(6 ** (-0.5)),
                ],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
            ],
            device=device,
        ).detach()

    return change_mat


def irreps_sum(ang_mom: int) -> int:
    """
    Returns the sum of the dimensions of the irreps up to the specified angular momentum.

    :param ang_mom: max angular momenttum to sum up dimensions of irreps
    """
    total = 0
    for i in range(ang_mom + 1):
        total += 2 * i + 1

    return total


def reshape_stress(L0out, L2out, batch_size=1):
    _max_rank = 2
    pred_irreps = torch.zeros(
        (batch_size, irreps_sum(_max_rank)),
        device = L0out.device,
    )
    # L=0
    L=0
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L0out.view(batch_size, -1)
    
    L=2
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L2out.view(batch_size, -1) 
    
    pred = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(_max_rank, device = L0out.device),
        pred_irreps,
    )
    
    return pred.view(batch_size, 3,3)


class Sphere(nn.Module):
    
    def __init__(self, lmax=2):
        super(Sphere, self).__init__()
        self.lmax = lmax
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.lmax, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        sh_0_0 = torch.ones_like(x)
        if lmax == 0:
            return torch.stack([ sh_0_0, ], dim=-1)
        
        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)
