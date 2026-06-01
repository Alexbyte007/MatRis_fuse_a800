from __future__ import annotations

import argparse
import json
import os
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
from profile_matriscalculator_pipeline import ModuleTimer, profile_one, summarize  # noqa: E402
from profile_p113_best_kernel_ranking import categorize_kernel, is_kernel_event  # noqa: E402


PURE_FUSE_ENV = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
}

P113_BEST_ENV = {
    "MATRIS_P83B_LINE_ATTENTION_TARGET_OFFSETS": "1",
    "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
    "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
    "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
    "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_OP_BWD": "1",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_THRESHOLD": "4096",
    "MATRIS_P113_GATED_TAIL_SECOND_SILU_MACRO": "1",
}

INTERESTING_CATEGORIES = {"elementwise", "norm", "copy_layout", "fill_zero", "gated_tail_custom"}
ROLE_SUFFIXES = (".attn_line", ".attn_atom", ".refine_line", ".refine_atom")
STAGE_PREFIX = "interaction_block."
P113_PREFIX = "P113.gated_tail_second."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute P113 best-path elementwise/norm kernels to module ranges.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--profile-limit", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--output-dir", default="results/p113_elementwise_attribution_limit12_profile4_20260526")
    return parser.parse_args()


def is_stage_range(name: str) -> bool:
    if name.startswith(P113_PREFIX):
        return True
    if not name.startswith(STAGE_PREFIX):
        return False
    return any(part in name for part in ROLE_SUFFIXES)


def stage_role(name: str) -> str:
    if name.startswith(P113_PREFIX):
        module_name = name.removeprefix(P113_PREFIX)
        for token, role in (
            (".attn_block_line_graph.", "attn_line"),
            (".attn_block_atom_graph.", "attn_atom"),
            (".refine_block_line_graph.", "refine_line"),
            (".refine_block_atom_graph.", "refine_atom"),
        ):
            if token in module_name:
                return role
        return "p113_unknown"
    for suffix in ROLE_SUFFIXES:
        token = suffix + "."
        if token in name or name.endswith(suffix):
            return suffix[1:]
    return "unknown"


def clean_stage_name(name: str) -> str:
    if name.startswith(P113_PREFIX):
        return name
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "interaction_block":
        return ".".join(parts[:4]) + ("." + ".".join(parts[4:]) if len(parts) > 4 else "")
    return name


def parse_trace(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    ranges: list[dict[str, Any]] = []
    kernels: list[dict[str, Any]] = []
    for event in payload.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        item = {
            "name": name,
            "ts": float(event.get("ts", 0.0)),
            "dur": float(event.get("dur", 0.0)),
        }
        if is_kernel_event(event):
            item["category"] = categorize_kernel(name)
            kernels.append(item)
        elif is_stage_range(name):
            item["role"] = stage_role(name)
            item["depth"] = name.count(".")
            ranges.append(item)

    by_range = defaultdict(
        lambda: {
            "role": "",
            "range_us": 0.0,
            "kernel_us": 0.0,
            "kernel_count": 0,
            "categories": defaultdict(float),
            "kernels": defaultdict(float),
        }
    )
    unattributed = defaultdict(float)

    for kernel in kernels:
        category = str(kernel["category"])
        if category not in INTERESTING_CATEGORIES:
            continue
        start = float(kernel["ts"])
        parents = [r for r in ranges if float(r["ts"]) <= start <= float(r["ts"]) + float(r["dur"])]
        if not parents:
            unattributed[(category, str(kernel["name"]))] += float(kernel["dur"])
            continue
        parent = max(parents, key=lambda r: (int(r["depth"]), -float(r["dur"])))
        key = clean_stage_name(str(parent["name"]))
        row = by_range[key]
        row["role"] = str(parent["role"])
        row["range_us"] += float(parent["dur"])
        row["kernel_us"] += float(kernel["dur"])
        row["kernel_count"] += 1
        row["categories"][category] += float(kernel["dur"])
        row["kernels"][str(kernel["name"])] += float(kernel["dur"])

    def top_kernel_rows(kernel_table: dict[str, float], limit: int = 8) -> list[dict[str, Any]]:
        return [
            {
                "kernel": name,
                "category": categorize_kernel(name),
                "ms": dur / 1000.0,
            }
            for name, dur in sorted(kernel_table.items(), key=lambda item: item[1], reverse=True)[:limit]
        ]

    range_rows = []
    for name, row in by_range.items():
        categories = dict(sorted(row["categories"].items(), key=lambda item: item[1], reverse=True))
        range_rows.append(
            {
                "range": name,
                "role": row["role"],
                "range_ms_sum": row["range_us"] / 1000.0,
                "interesting_kernel_ms": row["kernel_us"] / 1000.0,
                "interesting_kernel_count": int(row["kernel_count"]),
                "categories_ms": {cat: dur / 1000.0 for cat, dur in categories.items()},
                "top_kernels": top_kernel_rows(row["kernels"]),
            }
        )
    range_rows.sort(key=lambda item: item["interesting_kernel_ms"], reverse=True)

    by_role = defaultdict(lambda: {"ms": 0.0, "count": 0, "categories": defaultdict(float)})
    for row in range_rows:
        role_row = by_role[row["role"]]
        role_row["ms"] += float(row["interesting_kernel_ms"])
        role_row["count"] += int(row["interesting_kernel_count"])
        for cat, value in row["categories_ms"].items():
            role_row["categories"][cat] += float(value)

    role_rows = [
        {
            "role": role,
            "interesting_kernel_ms": value["ms"],
            "interesting_kernel_count": int(value["count"]),
            "categories_ms": dict(sorted(value["categories"].items(), key=lambda item: item[1], reverse=True)),
        }
        for role, value in sorted(by_role.items(), key=lambda item: item[1]["ms"], reverse=True)
    ]

    unattributed_rows = [
        {"category": key[0], "kernel": key[1], "ms": dur / 1000.0}
        for key, dur in sorted(unattributed.items(), key=lambda item: item[1], reverse=True)[:30]
    ]

    return {
        "trace": str(trace_path),
        "num_ranges": len(ranges),
        "num_kernels": len(kernels),
        "interesting_categories": sorted(INTERESTING_CATEGORIES),
        "role_rows": role_rows,
        "range_rows": range_rows[:80],
        "unattributed_top": unattributed_rows,
    }


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# P113 Elementwise Attribution",
        "",
        f"- trace: `{payload['kernel_profile']['trace']}`",
        f"- stage ranges: `{payload['kernel_profile']['num_ranges']}`",
        f"- kernel events: `{payload['kernel_profile']['num_kernels']}`",
        "",
        "## Role Totals",
        "",
        "| role | interesting kernel ms | count | categories |",
        "|---|---:|---:|---|",
    ]
    for row in payload["kernel_profile"]["role_rows"]:
        cats = ", ".join(f"{cat}={ms:.3f}" for cat, ms in row["categories_ms"].items())
        lines.append(f"| `{row['role']}` | {row['interesting_kernel_ms']:.6f} | {row['interesting_kernel_count']} | {cats} |")

    lines.extend(["", "## Top Ranges", "", "| range | role | interesting kernel ms | categories |", "|---|---|---:|---|"])
    for row in payload["kernel_profile"]["range_rows"][:35]:
        cats = ", ".join(f"{cat}={ms:.3f}" for cat, ms in row["categories_ms"].items())
        lines.append(f"| `{row['range']}` | `{row['role']}` | {row['interesting_kernel_ms']:.6f} | {cats} |")

    lines.extend(["", "## Top Kernels In Top Ranges", ""])
    for row in payload["kernel_profile"]["range_rows"][:12]:
        lines.append(f"### {row['range']}")
        lines.append("")
        lines.append("| category | ms | kernel |")
        lines.append("|---|---:|---|")
        for kernel in row["top_kernels"]:
            safe_name = str(kernel["kernel"]).replace("|", "\\|")
            lines.append(f"| {kernel['category']} | {kernel['ms']:.6f} | `{safe_name}` |")
        lines.append("")

    lines.extend(["## Unattributed Interesting Kernels", "", "| category | ms | kernel |", "|---|---:|---|"])
    for row in payload["kernel_profile"]["unattributed_top"]:
        safe_name = str(row["kernel"]).replace("|", "\\|")
        lines.append(f"| {row['category']} | {row['ms']:.6f} | `{safe_name}` |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("This profiler is intended for CUDA.")
    os.environ.update(PURE_FUSE_ENV)
    os.environ.update(P113_BEST_ENV)
    os.environ["MATRIS_CALCULATOR_STAGE_PROFILE"] = "1"
    os.environ["MATRIS_ATTNLINE_DETAIL_PROFILE"] = "1"
    os.environ["MATRIS_ATTNLINE_RECORD_FUNCTION"] = "1"
    os.environ["MATRIS_REFINELINE_DETAIL_PROFILE"] = "1"
    os.environ["MATRIS_REFINELINE_RECORD_FUNCTION"] = "1"
    os.environ["MATRIS_REFINEATOM_DETAIL_PROFILE"] = "1"
    os.environ["MATRIS_REFINEATOM_RECORD_FUNCTION"] = "1"
    os.environ["MATRIS_P81_ATTN_REDUCE_DETAIL"] = "1"
    os.environ["MATRIS_P99B_RECORD_FUNCTION"] = "1"
    os.environ["MATRIS_P113_RECORD_FUNCTION"] = "1"

    configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config=dict(src=args.dataset_src))
    calculator = build_calculator(args)
    calculator.model.eval()
    run_activation_calibration(structures, calculator, args)
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    module_timer = ModuleTimer(calculator.model, args.device)

    for graph_id in keys[: max(0, args.warmup_steps)]:
        try:
            profile_one(structures, int(graph_id), calculator, args, module_timer)
        except Exception:
            pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.json"
    records = []
    failures = []
    profile_keys = keys[: max(0, min(args.profile_limit, len(keys)))]
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        for graph_id in profile_keys:
            try:
                with torch.profiler.record_function(f"p113_attr.graph_{int(graph_id)}"):
                    records.append(profile_one(structures, int(graph_id), calculator, args, module_timer))
                prof.step()
            except Exception as exc:
                failures.append({"graph_id": int(graph_id), "error": str(exc)})
    prof.export_chrome_trace(str(trace_path))
    module_timer.close()

    stage_summary = summarize(records)
    payload = {
        "metadata": {
            "limit": args.limit,
            "profile_limit": len(profile_keys),
            "warmup_steps": args.warmup_steps,
            "sample_seed": args.sample_seed,
            "precision_mode": args.precision_mode,
            "quant_mode": args.quant_mode,
            "fusion_mode": args.fusion_mode,
            "env": {**PURE_FUSE_ENV, **P113_BEST_ENV},
        },
        "module_summary": stage_summary,
        "module_stage_ranking": stage_summary.get("stage_ranking", []),
        "kernel_profile": parse_trace(trace_path),
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    compact = {
        "role_rows": payload["kernel_profile"]["role_rows"],
        "top_ranges": payload["kernel_profile"]["range_rows"][:12],
        "failures": failures,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
