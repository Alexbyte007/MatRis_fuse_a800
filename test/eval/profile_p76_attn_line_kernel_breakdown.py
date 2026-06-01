from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

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


STAGE_RE = re.compile(r"interaction_block\.(\d+)\.attn_line\.(.+)")
P81_RE = re.compile(r"p81\.(.+)")
P83C_RE = re.compile(r"p83c\.(.+)")
TARGET_STAGES = {
    "gather_concat",
    "edge_update",
    "alpha_projection",
    "attention_reduce",
    "attention_reduce_node_input",
    "attention_reduce.fused_line_attention_or_none",
    "attention_reduce.source_softmax",
    "attention_reduce.target_softmax",
    "attention_reduce.source_alpha_mul_value",
    "attention_reduce.target_alpha_mul_value",
    "attention_reduce.target_weighted_sum_or_none",
    "attention_reduce.source_weight_sum",
    "attention_reduce.target_weight_sum",
    "node_update_input_concat",
    "node_update",
    "residual",
}
P81_STAGES = {
    "fused_line_attention.forward",
    "fused_line_attention.backward",
    "fused_line_attention.line.forward",
    "fused_line_attention.line.backward",
    "fused_line_attention.atom.forward",
    "fused_line_attention.atom.backward",
    "fused_line_attention.unknown.backward",
}
P83C_STAGES = {
    "fused_line_attention_node_input.line.forward",
    "fused_line_attention_node_input.line.backward",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P76 attn_line CUDA kernel breakdown.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--output-dir", default="results/p76_attn_line_kernel_breakdown")
    return parser.parse_args()


def is_kernel_event(event: dict) -> bool:
    category = str(event.get("cat", "")).lower()
    if "kernel" in category:
        return True
    args = event.get("args", {})
    return isinstance(args, dict) and args.get("External id") is not None and "stream" in args


def categorize_kernel(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("gemm", "cublas", "cutlass", "matmul", "sgemm")):
        return "gemm"
    if any(token in lowered for token in ("indexselect", "index_select", "gather", "scatter")):
        return "gather_scatter"
    if any(token in lowered for token in ("softmax", "reduce", "segment", "indexadd", "index_add")):
        return "softmax_reduce"
    if any(token in lowered for token in ("layer_norm", "rms_norm", "norm")):
        return "norm"
    if any(token in lowered for token in ("silu", "sigmoid", "elementwise", "activation", "mul", "add")):
        return "elementwise"
    if any(token in lowered for token in ("copy", "memcpy", "cat", "concat", "fill", "zero")):
        return "memory_layout"
    return "other"


def parse_trace(trace_path: Path) -> dict:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", [])
    stages = []
    kernels = []
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        match = STAGE_RE.fullmatch(name)
        if match and match.group(2) in TARGET_STAGES:
            stages.append(
                {
                    "block": int(match.group(1)),
                    "stage": match.group(2),
                    "name": name,
                    "ts": float(event.get("ts", 0.0)),
                    "dur": float(event.get("dur", 0.0)),
                }
            )
            continue
        p81_match = P81_RE.fullmatch(name)
        if p81_match and p81_match.group(1) in P81_STAGES:
            stages.append(
                {
                    "block": -1,
                    "stage": f"p81.{p81_match.group(1)}",
                    "name": name,
                    "ts": float(event.get("ts", 0.0)),
                    "dur": float(event.get("dur", 0.0)),
                }
            )
            continue
        p83c_match = P83C_RE.fullmatch(name)
        if p83c_match and p83c_match.group(1) in P83C_STAGES:
            stages.append(
                {
                    "block": -1,
                    "stage": f"p83c.{p83c_match.group(1)}",
                    "name": name,
                    "ts": float(event.get("ts", 0.0)),
                    "dur": float(event.get("dur", 0.0)),
                }
            )
            continue
        if is_kernel_event(event):
            kernels.append(
                {
                    "name": name,
                    "ts": float(event.get("ts", 0.0)),
                    "dur": float(event.get("dur", 0.0)),
                    "category": categorize_kernel(name),
                }
            )

    by_stage = defaultdict(lambda: {"dur_us": 0.0, "kernel_us": 0.0, "kernel_count": 0, "categories": defaultdict(float)})
    by_stage_block = defaultdict(lambda: {"dur_us": 0.0, "kernel_us": 0.0, "kernel_count": 0, "categories": defaultdict(float)})
    by_kernel_name = defaultdict(float)

    for stage in stages:
        key = stage["stage"]
        block_key = f"block_{stage['block']}.{stage['stage']}"
        by_stage[key]["dur_us"] += stage["dur"]
        by_stage_block[block_key]["dur_us"] += stage["dur"]
        start = stage["ts"]
        end = start + stage["dur"]
        for kernel in kernels:
            kernel_start = kernel["ts"]
            if start <= kernel_start <= end:
                by_stage[key]["kernel_us"] += kernel["dur"]
                by_stage[key]["kernel_count"] += 1
                by_stage[key]["categories"][kernel["category"]] += kernel["dur"]
                by_stage_block[block_key]["kernel_us"] += kernel["dur"]
                by_stage_block[block_key]["kernel_count"] += 1
                by_stage_block[block_key]["categories"][kernel["category"]] += kernel["dur"]
                by_kernel_name[(key, kernel["name"])] += kernel["dur"]

    def clean_table(table: dict) -> dict:
        out = {}
        for key, value in table.items():
            categories = dict(sorted(value["categories"].items(), key=lambda item: item[1], reverse=True))
            out[key] = {
                "dur_ms": value["dur_us"] / 1000.0,
                "kernel_ms": value["kernel_us"] / 1000.0,
                "kernel_count": value["kernel_count"],
                "categories_ms": {cat: dur / 1000.0 for cat, dur in categories.items()},
            }
        return dict(sorted(out.items(), key=lambda item: item[1]["kernel_ms"], reverse=True))

    top_kernels_by_stage = defaultdict(list)
    for (stage, name), dur in sorted(by_kernel_name.items(), key=lambda item: item[1], reverse=True):
        if len(top_kernels_by_stage[stage]) < 12:
            top_kernels_by_stage[stage].append({"name": name, "ms": dur / 1000.0, "category": categorize_kernel(name)})

    return {
        "num_stage_ranges": len(stages),
        "num_kernel_events": len(kernels),
        "by_stage": clean_table(by_stage),
        "by_stage_block": clean_table(by_stage_block),
        "top_kernels_by_stage": dict(top_kernels_by_stage),
    }


def write_markdown(summary: dict, output_path: Path) -> None:
    lines = [
        "# P76 Attn Line Kernel Breakdown",
        "",
        "This profile attributes CUDA kernels to `attn_line` stage ranges recorded with PyTorch profiler.",
        "Absolute timing includes profiler overhead and should be used for attribution only.",
        "",
        "## Stage Summary",
        "",
        "| stage | range ms | kernel ms | kernel count | top categories |",
        "|---|---:|---:|---:|---|",
    ]
    for stage, stats in summary["by_stage"].items():
        cats = ", ".join(f"{cat}={ms:.3f}" for cat, ms in list(stats["categories_ms"].items())[:5])
        lines.append(
            f"| {stage} | {stats['dur_ms']:.6f} | {stats['kernel_ms']:.6f} | {stats['kernel_count']} | {cats} |"
        )
    lines.extend(["", "## Top Kernels By Stage", ""])
    for stage, kernels in summary["top_kernels_by_stage"].items():
        lines.append(f"### {stage}")
        lines.append("")
        lines.append("| category | ms | kernel |")
        lines.append("|---|---:|---|")
        for item in kernels:
            safe_name = item["name"].replace("|", "\\|")
            lines.append(f"| {item['category']} | {item['ms']:.6f} | `{safe_name}` |")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ["MATRIS_CALCULATOR_STAGE_PROFILE"] = "1"
    os.environ["MATRIS_ATTNLINE_DETAIL_PROFILE"] = "1"
    os.environ["MATRIS_ATTNLINE_RECORD_FUNCTION"] = "1"

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
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        for graph_id in keys:
            with torch.profiler.record_function(f"p76_sample.graph_{int(graph_id)}"):
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
        "trace": str(trace_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "summary.md")
    print(json.dumps(summary["by_stage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
