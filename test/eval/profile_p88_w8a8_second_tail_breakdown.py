from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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


P8D_QUANT_MODE = "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8"
FUSION_MODE = "p28_p26_all_ffn_mlp_input_grad_only"


def refine_core(block: int) -> str:
    return f"interaction_block.{block}.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"


def refine_gate(block: int) -> str:
    return f"interaction_block.{block}.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"


def attn_core(block: int) -> str:
    return f"interaction_block.{block}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"


def attn_gate(block: int) -> str:
    return f"interaction_block.{block}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"


TARGET_GLOBS = tuple(
    target
    for block in (8, 9)
    for target in (refine_core(block), refine_gate(block), attn_core(block), attn_gate(block))
)
REFINE_89_TARGET_GLOBS = tuple(
    target
    for block in (8, 9)
    for target in (refine_core(block), refine_gate(block))
)
ATTN_89_TARGET_GLOBS = tuple(
    target
    for block in (8, 9)
    for target in (attn_core(block), attn_gate(block))
)
TARGET_GROUPS = {
    "combo_8_9": TARGET_GLOBS,
    "refine_8_9": REFINE_89_TARGET_GLOBS,
    "attn_8_9": ATTN_89_TARGET_GLOBS,
}


BASE_ENV_FLAGS = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_P78_FP32_GATED_TAIL_FORWARD": "1",
    "MATRIS_P79_ATTN_LINE_GATHER_CAT": "1",
    "MATRIS_P83C_LINE_ATTENTION_NODE_INPUT": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
    "MATRIS_W8A8_BACKEND": "cuda_wmma_tail_n128_parallel",
    "MATRIS_W8A8_DISABLE_FAST_WRAPPER": "1",
    "MATRIS_P87_REFINE_LINE_W8A8_SAVED_PRE": "1",
    "MATRIS_P87_ATTN_LINE_W8A8_SAVED_PRE": "1",
    "MATRIS_P88_W8A8_RANGE_PROFILE": "1",
}


P88_RANGE_RE = re.compile(r"p88\.w8a8_(saved_pre|second_tail)\.(.+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P88A W8A8 saved-pre second-tail kernel breakdown.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default=P8D_QUANT_MODE)
    parser.add_argument("--fusion-mode", default=FUSION_MODE)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument(
        "--target-group",
        default="combo_8_9",
        choices=sorted(TARGET_GROUPS),
        help="W8A8 second-tail target group to profile.",
    )
    parser.add_argument(
        "--target-globs",
        default="",
        help="Optional comma-separated target globs. Overrides --target-group.",
    )
    parser.add_argument("--output-dir", default="results/p88_w8a8_second_tail_breakdown")
    return parser.parse_args()


def selected_target_globs(args: argparse.Namespace) -> tuple[str, ...]:
    if args.target_globs.strip():
        return tuple(item.strip() for item in args.target_globs.split(",") if item.strip())
    return TARGET_GROUPS[args.target_group]


def is_kernel_event(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    if "kernel" in category:
        return True
    args = event.get("args", {})
    return isinstance(args, dict) and args.get("External id") is not None and "stream" in args


def fine_category(kernel_name: str) -> str:
    name = kernel_name.lower()
    if "quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel" in name:
        return "forward_pack_wmma_dequant_tail_parallel"
    if "quant_linear_w8a8_wmma_dual_gated_tail_n128" in name:
        return "forward_pack_wmma_dequant_tail"
    if "w8a8_dual_gated_tail_saved_pre_dq_input_grad_backward_n128" in name:
        return "backward_fused_tail_input_grad_dq"
    if "w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128" in name:
        return "backward_fused_tail_input_grad_int8"
    if "w8a8_dual_input_grad_matmul" in name:
        return "backward_custom_input_grad_matmul"
    if "input_grad_only_gated_tail_backward" in name:
        return "backward_tail_only"
    if "quant" in name and "w8a8" in name:
        return "other_w8a8"
    if any(token in name for token in ("gemm", "cublas", "cutlass", "matmul", "sgemm")):
        return "gemm_or_library_matmul"
    if "bmm" in name:
        return "bmm"
    if any(token in name for token in ("copy", "memcpy", "contiguous", "cast")):
        return "copy_layout_cast"
    if any(token in name for token in ("cat", "concat", "stack")):
        return "cat_stack_layout"
    if any(token in name for token in ("fill", "zero", "empty")):
        return "fill_zero_alloc"
    if any(token in name for token in ("silu", "sigmoid", "mul", "add", "elementwise")):
        return "elementwise"
    if any(token in name for token in ("layer_norm", "native_layer_norm", "norm")):
        return "norm"
    return "other"


def ms(us: float) -> float:
    return us / 1000.0


def parse_trace(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", [])
    ranges: list[dict[str, Any]] = []
    kernels: list[dict[str, Any]] = []
    global_kernel_category = defaultdict(lambda: {"dur_us": 0.0, "count": 0})
    global_kernel_name = defaultdict(lambda: {"dur_us": 0.0, "count": 0, "category": ""})

    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        match = P88_RANGE_RE.fullmatch(name)
        if match:
            ranges.append(
                {
                    "kind": match.group(1),
                    "name": name,
                    "detail": match.group(2),
                    "ts": float(event.get("ts", 0.0)),
                    "dur": float(event.get("dur", 0.0)),
                }
            )
            continue
        if is_kernel_event(event):
            category = fine_category(name)
            kernel = {
                "name": name,
                "ts": float(event.get("ts", 0.0)),
                "dur": float(event.get("dur", 0.0)),
                "category": category,
            }
            kernels.append(kernel)
            global_kernel_category[category]["dur_us"] += kernel["dur"]
            global_kernel_category[category]["count"] += 1
            global_kernel_name[name]["dur_us"] += kernel["dur"]
            global_kernel_name[name]["count"] += 1
            global_kernel_name[name]["category"] = category

    by_range = defaultdict(lambda: {"dur_us": 0.0, "kernel_us": 0.0, "count": 0, "categories": defaultdict(float)})
    by_module = defaultdict(lambda: {"dur_us": 0.0, "kernel_us": 0.0, "count": 0, "categories": defaultdict(float)})
    by_kernel_in_p88 = defaultdict(lambda: {"dur_us": 0.0, "count": 0, "category": ""})

    for range_event in ranges:
        key = f"{range_event['kind']}.{range_event['detail']}"
        by_range[key]["dur_us"] += range_event["dur"]
        start = range_event["ts"]
        end = start + range_event["dur"]
        for kernel in kernels:
            if start <= kernel["ts"] <= end:
                dur = kernel["dur"]
                category = kernel["category"]
                by_range[key]["kernel_us"] += dur
                by_range[key]["count"] += 1
                by_range[key]["categories"][category] += dur
                by_kernel_in_p88[kernel["name"]]["dur_us"] += dur
                by_kernel_in_p88[kernel["name"]]["count"] += 1
                by_kernel_in_p88[kernel["name"]]["category"] = category
                if range_event["kind"] == "second_tail" and key.endswith(".forward_autograd"):
                    module_name = range_event["detail"].removesuffix(".forward_autograd")
                    by_module[module_name]["dur_us"] += range_event["dur"]
                    by_module[module_name]["kernel_us"] += dur
                    by_module[module_name]["count"] += 1
                    by_module[module_name]["categories"][category] += dur

    def category_rows(source: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        total_us = sum(float(value["dur_us"]) for value in source.values())
        return [
            {
                "category": category,
                "kernel_ms": ms(float(value["dur_us"])),
                "kernel_count": int(value["count"]),
                "pct": float(value["dur_us"]) / total_us * 100.0 if total_us else 0.0,
            }
            for category, value in sorted(source.items(), key=lambda item: float(item[1]["dur_us"]), reverse=True)
        ]

    def range_rows(source: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for name, value in sorted(source.items(), key=lambda item: float(item[1]["kernel_us"]), reverse=True):
            rows.append(
                {
                    "name": name,
                    "range_ms": ms(float(value["dur_us"])),
                    "kernel_ms": ms(float(value["kernel_us"])),
                    "kernel_count": int(value["count"]),
                    "categories_ms": {
                        category: ms(float(dur))
                        for category, dur in sorted(
                            value["categories"].items(),
                            key=lambda item: float(item[1]),
                            reverse=True,
                        )
                    },
                }
            )
        return rows

    def kernel_rows(source: dict[str, dict[str, Any]], limit: int = 32) -> list[dict[str, Any]]:
        return [
            {
                "kernel": name,
                "category": value["category"],
                "kernel_ms": ms(float(value["dur_us"])),
                "kernel_count": int(value["count"]),
            }
            for name, value in sorted(source.items(), key=lambda item: float(item[1]["dur_us"]), reverse=True)[:limit]
        ]

    return {
        "trace": str(trace_path),
        "num_p88_ranges": len(ranges),
        "num_kernel_events": len(kernels),
        "range_rows": range_rows(by_range),
        "module_forward_rows": range_rows(by_module),
        "p88_top_kernels": kernel_rows(by_kernel_in_p88),
        "global_w8a8_category_rows": [
            row
            for row in category_rows(global_kernel_category)
            if "w8a8" in row["category"]
            or row["category"].startswith("forward_")
            or row["category"].startswith("backward_")
        ],
        "global_top_w8a8_kernels": [
            row
            for row in kernel_rows(global_kernel_name, 64)
            if "w8a8" in row["kernel"].lower()
            or row["category"].startswith("forward_")
            or row["category"].startswith("backward_")
        ][:32],
    }


def cats_text(row: dict[str, Any], limit: int = 5) -> str:
    return ", ".join(f"{key}={value:.3f}" for key, value in list(row.get("categories_ms", {}).items())[:limit])


def write_table(lines: list[str], rows: list[dict[str, Any]], *, kind: str) -> None:
    if kind == "range":
        lines.append("| range | range ms | kernel ms | count | top categories |")
        lines.append("|---|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                f"| `{row['name']}` | {row['range_ms']:.6f} | {row['kernel_ms']:.6f} | "
                f"{row['kernel_count']} | {cats_text(row)} |"
            )
    elif kind == "category":
        lines.append("| category | kernel ms | count | pct |")
        lines.append("|---|---:|---:|---:|")
        for row in rows:
            lines.append(f"| {row['category']} | {row['kernel_ms']:.6f} | {row['kernel_count']} | {row['pct']:.2f} |")
    else:
        lines.append("| category | kernel ms | count | kernel |")
        lines.append("|---|---:|---:|---|")
        for row in rows:
            kernel = row["kernel"].replace("|", "\\|")
            lines.append(f"| {row['category']} | {row['kernel_ms']:.6f} | {row['kernel_count']} | `{kernel}` |")


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    range_rows = summary["range_rows"]
    forward_kernel = next((row for row in range_rows if row["name"] == "saved_pre.forward.kernel_with_pre"), None)
    save_row = next((row for row in range_rows if row["name"] == "saved_pre.forward.save_for_backward"), None)
    backward_rows = [row for row in range_rows if row["name"].startswith("saved_pre.backward.")]
    backward_kernel_ms = sum(row["kernel_ms"] for row in backward_rows)
    lines = [
        "# P88A W8A8 Second-Tail Kernel Breakdown",
        "",
        f"- trace: `{summary['trace']}`",
        f"- p88 ranges: `{summary['num_p88_ranges']}`",
        f"- kernel events: `{summary['num_kernel_events']}`",
        "",
        "## Quick Read",
        "",
        f"- forward kernel_with_pre kernel time: `{forward_kernel['kernel_ms']:.6f} ms`" if forward_kernel else "- forward kernel_with_pre: `n/a`",
        f"- save_for_backward kernel time: `{save_row['kernel_ms']:.6f} ms`" if save_row else "- save_for_backward: `n/a`",
        f"- backward saved-pre kernel time: `{backward_kernel_ms:.6f} ms`",
        "",
        "## P88 Ranges",
        "",
    ]
    write_table(lines, summary["range_rows"], kind="range")
    lines.extend(["", "## Module Forward Ranges", ""])
    write_table(lines, summary["module_forward_rows"], kind="range")
    lines.extend(["", "## P88 Top Kernels", ""])
    write_table(lines, summary["p88_top_kernels"], kind="kernel")
    lines.extend(["", "## Global W8A8 Categories", ""])
    write_table(lines, summary["global_w8a8_category_rows"], kind="category")
    lines.extend(["", "## Global W8A8 Kernels", ""])
    write_table(lines, summary["global_top_w8a8_kernels"], kind="kernel")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("P88A is intended for CUDA kernel breakdown.")

    target_globs = selected_target_globs(args)
    env_flags = dict(BASE_ENV_FLAGS)
    env_flags["MATRIS_QUANT_INCLUDE_GLOBS"] = ",".join(target_globs)
    for key, value in env_flags.items():
        os.environ[key] = value

    configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config=dict(src=args.dataset_src))
    calculator = build_calculator(args)
    run_activation_calibration(structures, calculator, args)
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)

    warmup_keys = keys[: max(0, args.warmup_steps)]
    for graph_id in warmup_keys:
        try:
            profile_one(structures, int(graph_id), calculator, args)
        except Exception:
            pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.json"
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        for graph_id in keys:
            with torch.profiler.record_function(f"p88_sample.graph_{int(graph_id)}"):
                profile_one(structures, int(graph_id), calculator, args)
            prof.step()
    prof.export_chrome_trace(str(trace_path))

    summary = parse_trace(trace_path)
    summary["metadata"] = {
        "limit": args.limit,
        "warmup_steps": args.warmup_steps,
        "sample_seed": args.sample_seed,
        "quant_mode": args.quant_mode,
        "fusion_mode": args.fusion_mode,
        "target_group": args.target_group,
        "target_globs": target_globs,
        "env": env_flags,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "summary.md")
    print(json.dumps({"range_rows": summary["range_rows"][:12], "module_forward_rows": summary["module_forward_rows"]}, indent=2))


if __name__ == "__main__":
    main()
