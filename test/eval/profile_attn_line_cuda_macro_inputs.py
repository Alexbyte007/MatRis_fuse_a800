from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
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
    build_calculator,
    configure_precision,
    run_activation_calibration,
    select_group_aligned_keys,
)
from profile_matriscalculator_pipeline import profile_one  # noqa: E402
from profile_p76_attn_line_kernel_breakdown import parse_trace  # noqa: E402
from matris.model.functions import _load_matris_op  # noqa: E402


PURE_FUSE_ENV = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
}

CUDA_OPS_TO_CHECK = (
    "line_edge_gather_cat_forward",
    "line_edge_cat_grad_scatter_backward",
    "line_edge_project_grad_scatter_backward",
    "line_edge_project_grad_scatter_backward_tiled",
    "line_edge_project_grad_scatter_backward_tile32",
    "line_edge_silu_project_grad_scatter_backward_tile32",
    "line_edge_silu_project_alpha_grad_scatter_backward_tile32",
    "fused_line_attention_forward",
    "fused_line_attention_forward_v2",
    "fused_line_attention_forward_target_offsets",
    "fused_line_attention_node_input_forward_target_offsets",
    "fused_line_attention_backward",
    "two_linear_silu_input_grad_backward_n128",
    "input_grad_only_gated_tail_backward",
    "input_grad_only_gated_tail_backward_n128_v2",
    "line_node_triple_cat_forward",
    "line_node_triple_cat_backward",
)

BLOCK_RE = re.compile(r"interaction_block\.(\d+)\.attn_block_line_graph$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect attn_line shape/stride/segment statistics plus CUDA profiler attribution "
            "for planning A-CUDA attn_line macro kernels."
        )
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--profile-limit", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--output-dir", default="results/attn_line_cuda_macro_inputs_20260524")
    parser.add_argument("--no-set-pure-fuse-env", action="store_true")
    return parser.parse_args()


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "requires_grad": bool(tensor.requires_grad),
    }


def index_summary(index: torch.Tensor, rows: int) -> dict[str, Any]:
    result = tensor_meta(index)
    result["numel"] = int(index.numel())
    result["rows"] = int(rows)
    if index.numel() == 0:
        result.update({"min": None, "max": None, "degree": numeric_summary([]), "nonzero_segments": 0})
        return result
    with torch.no_grad():
        cpu_index = index.detach().to("cpu")
        degree = torch.bincount(cpu_index, minlength=max(0, int(rows))).numpy().astype(np.float64)
        nonzero = degree[degree > 0]
        result.update(
            {
                "min": int(cpu_index.min().item()),
                "max": int(cpu_index.max().item()),
                "degree": numeric_summary(nonzero.tolist()),
                "nonzero_segments": int(nonzero.size),
            }
        )
    return result


def bincount_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    result = tensor_meta(tensor)
    if tensor.numel() == 0:
        result["values"] = numeric_summary([])
        return result
    values = tensor.detach().to("cpu").float().numpy().tolist()
    result["values"] = numeric_summary(values)
    return result


class AttnLineInputRecorder:
    def __init__(self, model: torch.nn.Module) -> None:
        self.records: list[dict[str, Any]] = []
        self.current_graph_id: int | None = None
        self.handles = []
        for name, module in model.named_modules():
            match = BLOCK_RE.fullmatch(name)
            if match is None:
                continue
            block_index = int(match.group(1))
            self.handles.append(module.register_forward_pre_hook(self._make_hook(block_index, name), with_kwargs=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, block_index: int, module_name: str):
        def hook(module, inputs, kwargs):
            node_feat = kwargs.get("node_feat") if isinstance(kwargs, dict) else None
            edge_feat = kwargs.get("edge_feat") if isinstance(kwargs, dict) else None
            graph = kwargs.get("graph") if isinstance(kwargs, dict) else None
            if node_feat is None and len(inputs) >= 1:
                node_feat = inputs[0]
            if edge_feat is None and len(inputs) >= 2:
                edge_feat = inputs[1]
            if graph is None and len(inputs) >= 3:
                graph = inputs[2]
            if not isinstance(node_feat, torch.Tensor) or not isinstance(edge_feat, torch.Tensor) or not isinstance(graph, dict):
                return
            source_index = graph.get("source_index")
            target_index = graph.get("target_index")
            if not isinstance(source_index, torch.Tensor) or not isinstance(target_index, torch.Tensor):
                return
            source_bincount = graph.get("source_bincount")
            target_bincount = graph.get("target_bincount")
            record = {
                "graph_id": self.current_graph_id,
                "block_index": block_index,
                "module": module_name,
                "node_feat": tensor_meta(node_feat),
                "edge_feat": tensor_meta(edge_feat),
                "source_index": index_summary(source_index, int(node_feat.shape[0])),
                "target_index": index_summary(target_index, int(node_feat.shape[0])),
                "source_bincount": bincount_summary(source_bincount if isinstance(source_bincount, torch.Tensor) else None),
                "target_bincount": bincount_summary(target_bincount if isinstance(target_bincount, torch.Tensor) else None),
                "line_nodes": int(node_feat.shape[0]),
                "line_edges": int(edge_feat.shape[0]),
                "feature_dim": int(node_feat.shape[-1]) if node_feat.ndim == 2 else None,
                "edge_feature_dim": int(edge_feat.shape[-1]) if edge_feat.ndim == 2 else None,
            }
            self.records.append(record)

        return hook


def summarize_shape_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_block[int(record["block_index"])].append(record)

    summary = {}
    for block, items in sorted(by_block.items()):
        node_rows = [float(item["line_nodes"]) for item in items]
        edge_rows = [float(item["line_edges"]) for item in items]
        src_degree_mean = [float(item["source_index"]["degree"]["mean"]) for item in items]
        src_degree_max = [float(item["source_index"]["degree"]["max"]) for item in items]
        tgt_degree_mean = [float(item["target_index"]["degree"]["mean"]) for item in items]
        tgt_degree_max = [float(item["target_index"]["degree"]["max"]) for item in items]
        summary[str(block)] = {
            "num_records": len(items),
            "line_nodes": numeric_summary(node_rows),
            "line_edges": numeric_summary(edge_rows),
            "source_degree_mean_per_graph": numeric_summary(src_degree_mean),
            "source_degree_max_per_graph": numeric_summary(src_degree_max),
            "target_degree_mean_per_graph": numeric_summary(tgt_degree_mean),
            "target_degree_max_per_graph": numeric_summary(tgt_degree_max),
            "node_feat_contiguous_all": all(bool(item["node_feat"]["is_contiguous"]) for item in items),
            "edge_feat_contiguous_all": all(bool(item["edge_feat"]["is_contiguous"]) for item in items),
            "source_index_contiguous_all": all(bool(item["source_index"]["is_contiguous"]) for item in items),
            "target_index_contiguous_all": all(bool(item["target_index"]["is_contiguous"]) for item in items),
            "node_feat_stride_set": sorted({tuple(item["node_feat"]["stride"]) for item in items}),
            "edge_feat_stride_set": sorted({tuple(item["edge_feat"]["stride"]) for item in items}),
            "node_dtype_set": sorted({item["node_feat"]["dtype"] for item in items}),
            "edge_dtype_set": sorted({item["edge_feat"]["dtype"] for item in items}),
            "requires_grad_node_any": any(bool(item["node_feat"]["requires_grad"]) for item in items),
            "requires_grad_edge_any": any(bool(item["edge_feat"]["requires_grad"]) for item in items),
        }
    return summary


def collect_op_availability() -> dict[str, Any]:
    matris_op = _load_matris_op()
    return {
        "matris_op_available": matris_op is not None,
        "ops": {name: bool(matris_op is not None and hasattr(matris_op, name)) for name in CUDA_OPS_TO_CHECK},
    }


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Attn Line CUDA Macro Inputs",
        "",
        "## Existing CUDA Op Availability",
        "",
        "| op | available |",
        "|---|---:|",
    ]
    for name, available in payload["op_availability"]["ops"].items():
        lines.append(f"| `{name}` | {available} |")

    lines.extend(
        [
            "",
            "## Shape Summary By Block",
            "",
            "| block | records | line nodes mean/p90/max | line edges mean/p90/max | source degree max p90/max | target degree max p90/max | contiguous |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for block, stats in payload["shape_summary_by_block"].items():
        nodes = stats["line_nodes"]
        edges = stats["line_edges"]
        src_max = stats["source_degree_max_per_graph"]
        tgt_max = stats["target_degree_max_per_graph"]
        contiguous = (
            f"node={stats['node_feat_contiguous_all']}, edge={stats['edge_feat_contiguous_all']}, "
            f"src={stats['source_index_contiguous_all']}, tgt={stats['target_index_contiguous_all']}"
        )
        lines.append(
            f"| {block} | {stats['num_records']} | "
            f"{nodes['mean']:.1f}/{nodes['p90']:.1f}/{nodes['max']:.0f} | "
            f"{edges['mean']:.1f}/{edges['p90']:.1f}/{edges['max']:.0f} | "
            f"{src_max['p90']:.1f}/{src_max['max']:.0f} | "
            f"{tgt_max['p90']:.1f}/{tgt_max['max']:.0f} | {contiguous} |"
        )

    kernel_summary = payload.get("kernel_profile", {}).get("by_stage", {})
    lines.extend(
        [
            "",
            "## Kernel Stage Summary",
            "",
            "| stage | range ms | kernel ms | kernel count | top categories |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for stage, stats in kernel_summary.items():
        cats = ", ".join(f"{cat}={ms:.3f}" for cat, ms in list(stats.get("categories_ms", {}).items())[:5])
        lines.append(
            f"| `{stage}` | {stats['dur_ms']:.6f} | {stats['kernel_ms']:.6f} | {stats['kernel_count']} | {cats} |"
        )

    lines.extend(["", "## Initial Routing Hint", "", payload["routing_hint"], ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_routing_hint(kernel_profile: dict[str, Any]) -> str:
    by_stage = kernel_profile.get("by_stage", {})
    gather = by_stage.get("gather_concat", {}).get("kernel_ms", 0.0)
    alpha = by_stage.get("alpha_projection", {}).get("kernel_ms", 0.0)
    reduce_ms = by_stage.get("attention_reduce", {}).get("kernel_ms", 0.0)
    edge_update = by_stage.get("edge_update", {}).get("kernel_ms", 0.0)
    node_update = by_stage.get("node_update", {}).get("kernel_ms", 0.0)
    if reduce_ms >= max(gather + alpha, edge_update, node_update):
        return "Routing hint: attention_reduce/fused attention dominates this profile; prioritize A-CUDA2 before widening A-CUDA1."
    if gather + alpha >= max(reduce_ms, edge_update, node_update):
        return "Routing hint: gather_concat + alpha_projection is a top cost; prioritize A-CUDA1."
    if edge_update >= node_update:
        return "Routing hint: edge_update is prominent; A-CUDA1 should include edge_update input-grad reuse before A-CUDA2."
    return "Routing hint: node_update is prominent; keep A-CUDA4 in mind, but still start with A-CUDA1/A-CUDA2 correctness hooks."


def main() -> None:
    args = parse_args()
    if not args.no_set_pure_fuse_env:
        for key, value in PURE_FUSE_ENV.items():
            os.environ.setdefault(key, value)
        os.environ.pop("MATRIS_W8A8_BACKEND", None)
        os.environ.pop("MATRIS_W8A8_DISABLE_FAST_WRAPPER", None)
    os.environ["MATRIS_CALCULATOR_STAGE_PROFILE"] = "1"
    os.environ["MATRIS_ATTNLINE_DETAIL_PROFILE"] = "1"
    os.environ["MATRIS_ATTNLINE_RECORD_FUNCTION"] = "1"
    os.environ.setdefault("MATRIS_P99B_RECORD_FUNCTION", "1")

    configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config=dict(src=args.dataset_src))
    calculator = build_calculator(args)
    calculator.model.eval()
    run_activation_calibration(structures, calculator, args)
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)

    recorder = AttnLineInputRecorder(calculator.model)
    records = []
    warmup_keys = keys[: max(0, args.warmup_steps)]
    for graph_id in warmup_keys:
        try:
            recorder.current_graph_id = int(graph_id)
            profile_one(structures, int(graph_id), calculator, args)
        except Exception:
            pass
    recorder.records.clear()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_keys = keys[: max(0, min(args.profile_limit, len(keys)))]
    trace_path = output_dir / "trace.json"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        for graph_id in profile_keys:
            try:
                recorder.current_graph_id = int(graph_id)
                with torch.profiler.record_function(f"a_cuda_probe.graph_{int(graph_id)}"):
                    records.append(profile_one(structures, int(graph_id), calculator, args))
                prof.step()
            except Exception as exc:
                records.append({"graph_id": int(graph_id), "error": str(exc)})
    prof.export_chrome_trace(str(trace_path))

    for graph_id in keys[len(profile_keys) :]:
        try:
            recorder.current_graph_id = int(graph_id)
            records.append(profile_one(structures, int(graph_id), calculator, args))
        except Exception as exc:
            records.append({"graph_id": int(graph_id), "error": str(exc)})
    recorder.close()

    kernel_profile = parse_trace(trace_path)
    payload = {
        "metadata": {
            "limit": args.limit,
            "profile_limit": len(profile_keys),
            "warmup_steps": args.warmup_steps,
            "sample_seed": args.sample_seed,
            "precision_mode": args.precision_mode,
            "quant_mode": args.quant_mode,
            "fusion_mode": args.fusion_mode,
            "trace": str(trace_path),
        },
        "op_availability": collect_op_availability(),
        "shape_summary_by_block": summarize_shape_records(recorder.records),
        "shape_records": recorder.records,
        "kernel_profile": kernel_profile,
        "routing_hint": make_routing_hint(kernel_profile),
        "endpoint_records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    compact = {
        "shape_summary_by_block": payload["shape_summary_by_block"],
        "op_availability": payload["op_availability"],
        "routing_hint": payload["routing_hint"],
        "kernel_by_stage": payload["kernel_profile"].get("by_stage", {}),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
