from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from fairchem.core.datasets import AseDBDataset
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from check_attn_line_a3_correctness import (  # noqa: E402
    PURE_FUSE_ENV,
    env_without_candidate,
    patched_env,
    prepare_attn_line_inputs,
)
from infer_salex_lmdb_quant import build_calculator, configure_precision, select_group_aligned_keys  # noqa: E402
from matris.model.functions import _load_matris_op  # noqa: E402
from matris.model.interaction_block import (  # noqa: E402
    _p101_attention_layer_forward,
    _p106_attention_node_input_backward_edge_direct_or_none,
    _p53b_apply_update_backward,
    _p53b_linear_backward,
    _p53b_tail_backward,
    _p53b_silu_grad,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "P108 feasibility microbench: compare current P105 fused line-edge "
            "projection/alpha/scatter backward against dense torch.mm materialization."
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
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--block-index", type=int, default=9)
    parser.add_argument("--warmup-iters", type=int, default=8)
    parser.add_argument("--bench-iters", type=int, default=32)
    parser.add_argument("--hybrid-row-threshold", type=int, default=8192)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--output-json", default="results/p108_projection_gemm_feasibility/summary.json")
    parser.add_argument("--output-md", default="results/p108_projection_gemm_feasibility/summary.md")
    parser.add_argument("--no-set-pure-fuse-env", action="store_true")
    return parser.parse_args()


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    p50 = sorted_values[mid] if len(sorted_values) % 2 else 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "p50": float(p50),
        "max": float(max(values)),
    }


def cuda_time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requires a CUDA device")
    sink = None
    for _ in range(max(0, warmup)):
        sink = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        sink = fn()
    end.record()
    torch.cuda.synchronize()
    # Keep outputs live until after synchronization.
    if isinstance(sink, tuple) and len(sink) == -1:
        raise AssertionError("unreachable")
    return float(start.elapsed_time(end) / max(1, iters))


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def collect_case(calculator, graph, graph_id: int, args: argparse.Namespace) -> dict[str, Any] | None:
    disabled_candidate_env = {
        **env_without_candidate(),
        "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": None,
        "MATRIS_P101_USE_NODE_INPUT_ATTENTION": None,
        "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": None,
        "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": None,
    }
    prepared = prepare_attn_line_inputs(
        calculator,
        graph,
        args.block_index,
        args.task,
        disabled_candidate_env,
    )
    if prepared is None:
        return None

    layer = prepared["layer"]
    line_graph = prepared["graph"]
    source_index = line_graph["source_index"].contiguous()
    target_index = line_graph["target_index"].contiguous()
    node_rows = int(prepared["node_feat"].shape[0])

    with patched_env(
        {
            **env_without_candidate(),
            "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
            "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
        }
    ):
        node_out, edge_out, cache = _p101_attention_layer_forward(
            layer,
            prepared["node_feat"].detach(),
            prepared["edge_feat"].detach(),
            line_graph,
        )
        generator = torch.Generator(device=node_out.device)
        generator.manual_seed(1234 + int(graph_id) + 1000 * int(args.block_index))
        grad_node_out = torch.randn(node_out.shape, generator=generator, device=node_out.device, dtype=node_out.dtype)
        grad_edge_out = torch.randn(edge_out.shape, generator=generator, device=edge_out.device, dtype=edge_out.dtype)

        grad_node_x = _p53b_apply_update_backward(grad_node_out, cache["node_cache"])
        fused_node_input = _p106_attention_node_input_backward_edge_direct_or_none(
            grad_node_x,
            cache["attn"],
            grad_edge_out.float(),
        )
        if fused_node_input is None:
            raise RuntimeError("P106 edge-direct attention backward was not available")
        _, grad_source_logits, grad_target_logits, grad_edge_values = fused_node_input

    edge_cache = cache["edge_cache"]
    first_weight = edge_cache["first"][0].contiguous()
    source_weight = cache["source_linear"][0].contiguous()
    target_weight = cache["target_linear"][0].contiguous()
    core_first = edge_cache["core_first"].contiguous()
    gate_first = edge_cache["gate_first"].contiguous()
    grad_core, grad_gate = _p53b_tail_backward(grad_edge_values, edge_cache["tail"])
    grad_core_second_in = _p53b_linear_backward(grad_core, edge_cache["core_second"]).contiguous()
    grad_gate_second_in = _p53b_linear_backward(grad_gate, edge_cache["gate_second"]).contiguous()

    return {
        "graph_id": int(graph_id),
        "rows": int(core_first.shape[0]),
        "node_rows": node_rows,
        "source_index": source_index,
        "target_index": target_index,
        "grad_core_second_in": grad_core_second_in,
        "grad_gate_second_in": grad_gate_second_in,
        "core_first": core_first,
        "gate_first": gate_first,
        "first_weight": first_weight,
        "grad_source_logits": grad_source_logits.contiguous(),
        "grad_target_logits": grad_target_logits.contiguous(),
        "source_weight": source_weight,
        "target_weight": target_weight,
    }


def run_p105_fused(case: dict[str, Any]):
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "line_edge_silu_project_alpha_grad_scatter_backward_tile32"):
        raise RuntimeError("matris_op.line_edge_silu_project_alpha_grad_scatter_backward_tile32 is unavailable")
    return matris_op.line_edge_silu_project_alpha_grad_scatter_backward_tile32(
        case["grad_core_second_in"],
        case["grad_gate_second_in"],
        case["core_first"],
        case["gate_first"],
        case["first_weight"],
        case["grad_source_logits"],
        case["grad_target_logits"],
        case["source_weight"],
        case["target_weight"],
        case["source_index"],
        case["target_index"],
        int(case["node_rows"]),
    )


def dense_project(case: dict[str, Any]) -> torch.Tensor:
    grad_core_first = case["grad_core_second_in"] * _p53b_silu_grad(case["core_first"])
    grad_gate_first = case["grad_gate_second_in"] * _p53b_silu_grad(case["gate_first"])
    grad_hidden = torch.cat([grad_core_first, grad_gate_first], dim=-1)
    grad_cat = grad_hidden.matmul(case["first_weight"])
    grad_alpha = case["grad_source_logits"].matmul(case["source_weight"])
    grad_alpha = grad_alpha + case["grad_target_logits"].matmul(case["target_weight"])
    grad_cat[:, :128] = grad_cat[:, :128] + grad_alpha
    return grad_cat


def dense_project_scatter(case: dict[str, Any]):
    grad_cat = dense_project(case)
    grad_edge = grad_cat[:, :128].contiguous()
    grad_target = grad_cat[:, 128:256].contiguous()
    grad_source = grad_cat[:, 256:384].contiguous()
    grad_node = grad_cat.new_zeros((int(case["node_rows"]), 128))
    grad_node.index_add_(0, case["target_index"], grad_target)
    grad_node.index_add_(0, case["source_index"], grad_source)
    return grad_node, grad_edge


def dense_project_cuda_scatter(case: dict[str, Any]):
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "line_edge_cat_grad_scatter_backward"):
        raise RuntimeError("matris_op.line_edge_cat_grad_scatter_backward is unavailable")
    grad_cat = dense_project(case)
    return matris_op.line_edge_cat_grad_scatter_backward(
        grad_cat,
        case["source_index"],
        case["target_index"],
        int(case["node_rows"]),
    )


def hybrid_threshold(case: dict[str, Any], threshold: int):
    if int(case["rows"]) > int(threshold):
        return dense_project_cuda_scatter(case)
    return run_p105_fused(case)


def benchmark_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ref_node, ref_edge = run_p105_fused(case)
    dense_node, dense_edge = dense_project_scatter(case)
    dense_cuda_node, dense_cuda_edge = dense_project_cuda_scatter(case)
    hybrid_node, hybrid_edge = hybrid_threshold(case, int(args.hybrid_row_threshold))
    torch.cuda.synchronize()
    diff = {
        "grad_node_max_abs": max_abs_diff(ref_node, dense_node),
        "grad_edge_max_abs": max_abs_diff(ref_edge, dense_edge),
        "cuda_scatter_grad_node_max_abs": max_abs_diff(ref_node, dense_cuda_node),
        "cuda_scatter_grad_edge_max_abs": max_abs_diff(ref_edge, dense_cuda_edge),
        "hybrid_grad_node_max_abs": max_abs_diff(ref_node, hybrid_node),
        "hybrid_grad_edge_max_abs": max_abs_diff(ref_edge, hybrid_edge),
    }
    times = {
        "p105_fused_ms": cuda_time_ms(
            lambda: run_p105_fused(case),
            warmup=args.warmup_iters,
            iters=args.bench_iters,
        ),
        "dense_project_only_ms": cuda_time_ms(
            lambda: dense_project(case),
            warmup=args.warmup_iters,
            iters=args.bench_iters,
        ),
        "dense_project_scatter_ms": cuda_time_ms(
            lambda: dense_project_scatter(case),
            warmup=args.warmup_iters,
            iters=args.bench_iters,
        ),
        "dense_project_cuda_scatter_ms": cuda_time_ms(
            lambda: dense_project_cuda_scatter(case),
            warmup=args.warmup_iters,
            iters=args.bench_iters,
        ),
        "hybrid_threshold_ms": cuda_time_ms(
            lambda: hybrid_threshold(case, int(args.hybrid_row_threshold)),
            warmup=args.warmup_iters,
            iters=args.bench_iters,
        ),
    }
    return {
        "graph_id": int(case["graph_id"]),
        "rows": int(case["rows"]),
        "node_rows": int(case["node_rows"]),
        "diff": diff,
        "times_ms": times,
        "ratios": {
            "dense_project_only_vs_p105": times["dense_project_only_ms"] / times["p105_fused_ms"],
            "dense_project_scatter_vs_p105": times["dense_project_scatter_ms"] / times["p105_fused_ms"],
            "dense_project_cuda_scatter_vs_p105": times["dense_project_cuda_scatter_ms"] / times["p105_fused_ms"],
            "hybrid_threshold_vs_p105": times["hybrid_threshold_ms"] / times["p105_fused_ms"],
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "num_records": len(records),
        "rows": numeric_summary([float(item["rows"]) for item in records]),
        "node_rows": numeric_summary([float(item["node_rows"]) for item in records]),
        "p105_fused_ms": numeric_summary([float(item["times_ms"]["p105_fused_ms"]) for item in records]),
        "dense_project_only_ms": numeric_summary(
            [float(item["times_ms"]["dense_project_only_ms"]) for item in records]
        ),
        "dense_project_scatter_ms": numeric_summary(
            [float(item["times_ms"]["dense_project_scatter_ms"]) for item in records]
        ),
        "dense_project_cuda_scatter_ms": numeric_summary(
            [float(item["times_ms"]["dense_project_cuda_scatter_ms"]) for item in records]
        ),
        "hybrid_threshold_ms": numeric_summary(
            [float(item["times_ms"]["hybrid_threshold_ms"]) for item in records]
        ),
        "dense_project_only_vs_p105": numeric_summary(
            [float(item["ratios"]["dense_project_only_vs_p105"]) for item in records]
        ),
        "dense_project_scatter_vs_p105": numeric_summary(
            [float(item["ratios"]["dense_project_scatter_vs_p105"]) for item in records]
        ),
        "dense_project_cuda_scatter_vs_p105": numeric_summary(
            [float(item["ratios"]["dense_project_cuda_scatter_vs_p105"]) for item in records]
        ),
        "hybrid_threshold_vs_p105": numeric_summary(
            [float(item["ratios"]["hybrid_threshold_vs_p105"]) for item in records]
        ),
        "grad_node_max_abs": numeric_summary([float(item["diff"]["grad_node_max_abs"]) for item in records]),
        "grad_edge_max_abs": numeric_summary([float(item["diff"]["grad_edge_max_abs"]) for item in records]),
        "cuda_scatter_grad_node_max_abs": numeric_summary(
            [float(item["diff"]["cuda_scatter_grad_node_max_abs"]) for item in records]
        ),
        "cuda_scatter_grad_edge_max_abs": numeric_summary(
            [float(item["diff"]["cuda_scatter_grad_edge_max_abs"]) for item in records]
        ),
        "hybrid_grad_node_max_abs": numeric_summary(
            [float(item["diff"]["hybrid_grad_node_max_abs"]) for item in records]
        ),
        "hybrid_grad_edge_max_abs": numeric_summary(
            [float(item["diff"]["hybrid_grad_edge_max_abs"]) for item in records]
        ),
    }
    if records:
        mean_p105 = float(summary["p105_fused_ms"]["mean"])
        mean_project = float(summary["dense_project_only_ms"]["mean"])
        mean_scatter = float(summary["dense_project_scatter_ms"]["mean"])
        mean_cuda_scatter = float(summary["dense_project_cuda_scatter_ms"]["mean"])
        mean_hybrid = float(summary["hybrid_threshold_ms"]["mean"])
        summary["mean_delta_ms"] = {
            "dense_project_only_minus_p105": mean_project - mean_p105,
            "dense_project_scatter_minus_p105": mean_scatter - mean_p105,
            "dense_project_cuda_scatter_minus_p105": mean_cuda_scatter - mean_p105,
            "hybrid_threshold_minus_p105": mean_hybrid - mean_p105,
        }
    return summary


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = payload["summary"]
    lines = [
        "# P108 Projection-GEMM Feasibility",
        "",
        f"- records: `{summary['num_records']}`",
        f"- block_index: `{payload['args']['block_index']}`",
        f"- hybrid_row_threshold: `{payload['args']['hybrid_row_threshold']}`",
        f"- warmup/iters: `{payload['args']['warmup_iters']}` / `{payload['args']['bench_iters']}`",
        "",
        "| metric | mean ms/ratio | std | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in (
        "p105_fused_ms",
        "dense_project_only_ms",
        "dense_project_scatter_ms",
        "dense_project_cuda_scatter_ms",
        "hybrid_threshold_ms",
        "dense_project_only_vs_p105",
        "dense_project_scatter_vs_p105",
        "dense_project_cuda_scatter_vs_p105",
        "hybrid_threshold_vs_p105",
    ):
        item = summary[key]
        lines.append(
            f"| `{key}` | `{float(item['mean']):.6f}` | `{float(item['std']):.6f}` | "
            f"`{float(item['min']):.6f}` | `{float(item['max']):.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Correctness Drift",
            "",
            f"- grad_node max_abs max: `{float(summary['grad_node_max_abs']['max']):.9g}`",
            f"- grad_edge max_abs max: `{float(summary['grad_edge_max_abs']['max']):.9g}`",
            f"- cuda scatter grad_node max_abs max: `{float(summary['cuda_scatter_grad_node_max_abs']['max']):.9g}`",
            f"- cuda scatter grad_edge max_abs max: `{float(summary['cuda_scatter_grad_edge_max_abs']['max']):.9g}`",
            f"- hybrid grad_node max_abs max: `{float(summary['hybrid_grad_node_max_abs']['max']):.9g}`",
            f"- hybrid grad_edge max_abs max: `{float(summary['hybrid_grad_edge_max_abs']['max']):.9g}`",
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.no_set_pure_fuse_env:
        os.environ.update(PURE_FUSE_ENV)
        os.environ.pop("MATRIS_W8A8_BACKEND", None)
        os.environ.pop("MATRIS_W8A8_DISABLE_FAST_WRAPPER", None)

    configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config={"src": args.dataset_src})
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    calculator = build_calculator(args)
    calculator.model.eval()

    records: list[dict[str, Any]] = []
    for graph_id in tqdm(keys, desc="p108_projection_gemm", leave=False):
        atom = structures.get_atoms(int(graph_id))
        calculator._adjust_pbc(atom)
        structure = AseAtomsAdaptor.get_structure(atom)
        graph_cpu = calculator.model.graph_converter(structure)
        graph = graph_cpu[0].to(args.device) if isinstance(graph_cpu, list) else graph_cpu.to(args.device)
        case = collect_case(calculator, graph, int(graph_id), args)
        if case is None:
            continue
        records.append(benchmark_case(case, args))

    payload = {
        "summary": summarize(records),
        "records": records,
        "args": vars(args),
        "env": {key: os.environ.get(key) for key in PURE_FUSE_ENV},
    }
    write_outputs(payload, args)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
