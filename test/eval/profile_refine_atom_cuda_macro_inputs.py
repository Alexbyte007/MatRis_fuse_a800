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
from profile_matriscalculator_pipeline import ModuleTimer, profile_one, summarize  # noqa: E402
from profile_p76_attn_line_kernel_breakdown import categorize_kernel, is_kernel_event  # noqa: E402


PURE_FUSE_ENV = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
}

P108_BEST_ENV = {
    "MATRIS_P83B_LINE_ATTENTION_TARGET_OFFSETS": "1",
    "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
    "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
    "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
    "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_OP_BWD": "1",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_THRESHOLD": "4096",
}

STAGE_RE = re.compile(r"interaction_block\.(\d+)\.refine_atom\.(.+)")
TARGET_STAGES = {
    "gather_source_node",
    "gather_target_node",
    "gather_directed_smooth",
    "learnable_envelope",
    "gather_concat",
    "edge_nonlinear_update",
    "smooth_multiply",
    "target_smooth_sum",
    "p68_grouped_ffn_pair_or_none",
    "node_FFN",
    "edge_FFN",
    "directed2undirected_delta_average",
    "residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile refine_atom under the current P108 best path for AT-CUDA planning."
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--profile-limit", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--output-dir", default="results/refine_atom_profile_p108_best_20260525")
    parser.add_argument("--no-set-pure-fuse-env", action="store_true")
    parser.add_argument("--no-set-p108-best-env", action="store_true")
    return parser.parse_args()


def clean_stage_table(table: dict) -> dict[str, Any]:
    out = {}
    for key, value in table.items():
        categories = dict(sorted(value["categories"].items(), key=lambda item: item[1], reverse=True))
        out[key] = {
            "range_ms": value["dur_us"] / 1000.0,
            "kernel_ms": value["kernel_us"] / 1000.0,
            "kernel_count": int(value["kernel_count"]),
            "categories_ms": {cat: dur / 1000.0 for cat, dur in categories.items()},
        }
    return dict(sorted(out.items(), key=lambda item: item[1]["kernel_ms"], reverse=True))


def parse_trace(trace_path: Path) -> dict[str, Any]:
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
    all_kernel_names = defaultdict(float)
    for kernel in kernels:
        all_kernel_names[kernel["name"]] += kernel["dur"]

    for stage in stages:
        key = stage["stage"]
        block_key = f"block_{stage['block']}.{stage['stage']}"
        by_stage[key]["dur_us"] += stage["dur"]
        by_stage_block[block_key]["dur_us"] += stage["dur"]
        start = stage["ts"]
        end = start + stage["dur"]
        for kernel in kernels:
            if start <= kernel["ts"] <= end:
                by_stage[key]["kernel_us"] += kernel["dur"]
                by_stage[key]["kernel_count"] += 1
                by_stage[key]["categories"][kernel["category"]] += kernel["dur"]
                by_stage_block[block_key]["kernel_us"] += kernel["dur"]
                by_stage_block[block_key]["kernel_count"] += 1
                by_stage_block[block_key]["categories"][kernel["category"]] += kernel["dur"]
                by_kernel_name[(key, kernel["name"])] += kernel["dur"]

    top_kernels_by_stage = defaultdict(list)
    for (stage, name), dur in sorted(by_kernel_name.items(), key=lambda item: item[1], reverse=True):
        if len(top_kernels_by_stage[stage]) < 16:
            top_kernels_by_stage[stage].append(
                {"name": name, "ms": dur / 1000.0, "category": categorize_kernel(name)}
            )
    top_kernels_global = [
        {"name": name, "ms": dur / 1000.0, "category": categorize_kernel(name)}
        for name, dur in sorted(all_kernel_names.items(), key=lambda item: item[1], reverse=True)[:40]
    ]

    return {
        "num_stage_ranges": len(stages),
        "num_kernel_events": len(kernels),
        "by_stage": clean_stage_table(by_stage),
        "by_stage_block": clean_stage_table(by_stage_block),
        "top_kernels_by_stage": dict(top_kernels_by_stage),
        "top_kernels_global": top_kernels_global,
    }


def make_routing_hint(kernel_profile: dict[str, Any], stage_summary: dict[str, Any]) -> str:
    by_stage = kernel_profile.get("by_stage", {})
    gather = (
        by_stage.get("gather_source_node", {}).get("kernel_ms", 0.0)
        + by_stage.get("gather_target_node", {}).get("kernel_ms", 0.0)
        + by_stage.get("gather_directed_smooth", {}).get("kernel_ms", 0.0)
        + by_stage.get("gather_concat", {}).get("kernel_ms", 0.0)
    )
    edge_update = by_stage.get("edge_nonlinear_update", {}).get("kernel_ms", 0.0)
    smooth = (
        by_stage.get("learnable_envelope", {}).get("kernel_ms", 0.0)
        + by_stage.get("smooth_multiply", {}).get("kernel_ms", 0.0)
        + by_stage.get("target_smooth_sum", {}).get("kernel_ms", 0.0)
    )
    ffn = by_stage.get("node_FFN", {}).get("kernel_ms", 0.0) + by_stage.get("edge_FFN", {}).get("kernel_ms", 0.0)
    d2u = by_stage.get("directed2undirected_delta_average", {}).get("kernel_ms", 0.0)
    if edge_update + gather >= max(smooth, ffn, d2u):
        return "Routing hint: AT-CUDA1 is aligned with this profile; refine_atom gather_concat + edge_nonlinear_update dominate attribution."
    if d2u >= max(edge_update + gather, smooth, ffn):
        return "Routing hint: directed2undirected delta average is prominent; AT-CUDA2 should be considered with grad_edge/stress checks."
    if ffn >= max(edge_update + gather, smooth, d2u):
        return "Routing hint: FFN input-grad is prominent; only fuse it if it can be merged into a wider refine_atom edge macro."
    refine_backward = stage_summary.get("module_backward.refine_atom_ms", {}).get("mean", 0.0)
    return f"Routing hint: fine-grained stage attribution is weak; use whole refine_atom backward mean {refine_backward:.3f} ms and kernel ranking first."


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# refine_atom P108 Profile",
        "",
        "Absolute profiler timings include overhead; use them for attribution and ranking.",
        "",
        "## Python/Module Stage Ranking",
        "",
        "| rank | stage | mean ms | pct latency |",
        "|---:|---|---:|---:|",
    ]
    for idx, item in enumerate(payload["module_stage_ranking"][:25], start=1):
        lines.append(f"| {idx} | `{item['name']}` | {item['mean_ms']:.6f} | {item['pct_of_latency']:.2f}% |")

    lines.extend(
        [
            "",
            "## refine_atom Stage Attribution",
            "",
            "| stage | range ms | kernel ms | kernel count | top categories |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for stage, stats in payload["kernel_profile"]["by_stage"].items():
        cats = ", ".join(f"{cat}={ms:.3f}" for cat, ms in list(stats.get("categories_ms", {}).items())[:5])
        lines.append(
            f"| `{stage}` | {stats['range_ms']:.6f} | {stats['kernel_ms']:.6f} | {stats['kernel_count']} | {cats} |"
        )

    lines.extend(["", "## Top Kernels By refine_atom Stage", ""])
    for stage, kernels in payload["kernel_profile"]["top_kernels_by_stage"].items():
        lines.append(f"### {stage}")
        lines.append("")
        lines.append("| category | ms | kernel |")
        lines.append("|---|---:|---|")
        for item in kernels:
            safe_name = item["name"].replace("|", "\\|")
            lines.append(f"| {item['category']} | {item['ms']:.6f} | `{safe_name}` |")
        lines.append("")

    lines.extend(
        [
            "## Top Global Kernels",
            "",
            "| rank | category | ms | kernel |",
            "|---:|---|---:|---|",
        ]
    )
    for idx, item in enumerate(payload["kernel_profile"]["top_kernels_global"][:25], start=1):
        safe_name = item["name"].replace("|", "\\|")
        lines.append(f"| {idx} | {item['category']} | {item['ms']:.6f} | `{safe_name}` |")

    lines.extend(["", "## Routing Hint", "", payload["routing_hint"], ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.no_set_pure_fuse_env:
        os.environ.update(PURE_FUSE_ENV)
        os.environ.pop("MATRIS_W8A8_BACKEND", None)
        os.environ.pop("MATRIS_W8A8_DISABLE_FAST_WRAPPER", None)
    if not args.no_set_p108_best_env:
        os.environ.update(P108_BEST_ENV)
    os.environ["MATRIS_CALCULATOR_STAGE_PROFILE"] = "1"
    os.environ["MATRIS_REFINEATOM_DETAIL_PROFILE"] = "1"
    os.environ["MATRIS_REFINEATOM_RECORD_FUNCTION"] = "1"

    configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config=dict(src=args.dataset_src))
    calculator = build_calculator(args)
    calculator.model.eval()
    run_activation_calibration(structures, calculator, args)
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    module_timer = ModuleTimer(calculator.model, args.device) if args.device == "cuda" else None

    for graph_id in keys[: max(0, args.warmup_steps)]:
        try:
            profile_one(structures, int(graph_id), calculator, args, module_timer)
        except Exception:
            pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.json"
    profile_keys = keys[: max(0, min(args.profile_limit, len(keys)))]
    records = []
    failures = []
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        for graph_id in profile_keys:
            try:
                with torch.profiler.record_function(f"refine_atom_probe.graph_{int(graph_id)}"):
                    records.append(profile_one(structures, int(graph_id), calculator, args, module_timer))
                prof.step()
            except Exception as exc:
                failures.append({"graph_id": int(graph_id), "error": str(exc)})
    prof.export_chrome_trace(str(trace_path))

    for graph_id in keys[len(profile_keys) :]:
        try:
            records.append(profile_one(structures, int(graph_id), calculator, args, module_timer))
        except Exception as exc:
            failures.append({"graph_id": int(graph_id), "error": str(exc)})
    if module_timer is not None:
        module_timer.close()

    stage_summary = summarize(records)
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
            "p108_best_env": not args.no_set_p108_best_env,
            "trace": str(trace_path),
        },
        "module_summary": stage_summary,
        "module_stage_ranking": stage_summary.get("stage_ranking", []),
        "kernel_profile": kernel_profile,
        "routing_hint": make_routing_hint(kernel_profile, stage_summary.get("stages", {})),
        "records": records,
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    compact = {
        "routing_hint": payload["routing_hint"],
        "top_module_stages": payload["module_stage_ranking"][:12],
        "refine_atom_stage_by_kernel": payload["kernel_profile"]["by_stage"],
        "top_global_kernels": payload["kernel_profile"]["top_kernels_global"][:12],
        "failures": failures,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
