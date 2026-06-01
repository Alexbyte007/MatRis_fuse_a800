from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P113 best-path global CUDA kernel ranking.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--output-dir", default="results/p113_best_kernel_ranking_limit8_20260526")
    return parser.parse_args()


def is_kernel_event(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    if "kernel" in category:
        return True
    args = event.get("args", {})
    return isinstance(args, dict) and args.get("External id") is not None and "stream" in args


def categorize_kernel(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("cublas", "gemm", "sgemm", "matmul", "ampere_", "cutlass")):
        return "gemm"
    if any(token in lowered for token in ("gather", "scatter", "indexselect", "index_select", "indexadd", "index_add")):
        return "scatter_gather"
    if any(token in lowered for token in ("copy", "memcpy", "contiguous", "clone", "cat", "concat", "to_copy")):
        return "copy_layout"
    if any(token in lowered for token in ("fill", "zero", "empty")):
        return "fill_zero"
    if any(token in lowered for token in ("softmax", "reduce", "segment", "sum")):
        return "reduce_softmax"
    if any(token in lowered for token in ("norm", "layer_norm", "rms")):
        return "norm"
    if any(token in lowered for token in ("silu", "sigmoid", "mul", "add", "sub", "div", "pow", "where")):
        return "elementwise"
    if "fused_line_attention" in lowered or "attention" in lowered:
        return "attention_custom"
    if "gated_tail" in lowered or "tail" in lowered:
        return "gated_tail_custom"
    return "other"


def ms(us: float) -> float:
    return us / 1000.0


def parse_trace(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = [event for event in payload.get("traceEvents", []) if event.get("ph") == "X"]
    kernels: list[dict[str, Any]] = []
    ranges: list[dict[str, Any]] = []

    for event in events:
        name = str(event.get("name", ""))
        item = {
            "name": name,
            "ts": float(event.get("ts", 0.0)),
            "dur": float(event.get("dur", 0.0)),
        }
        if is_kernel_event(event):
            item["category"] = categorize_kernel(name)
            kernels.append(item)
        elif any(token in name for token in ("gated_tail_second", "gated_tail_backward", "p113_sample")):
            ranges.append(item)

    by_category = defaultdict(lambda: {"dur_us": 0.0, "count": 0})
    by_kernel = defaultdict(lambda: {"dur_us": 0.0, "count": 0, "category": ""})
    for kernel in kernels:
        cat = str(kernel["category"])
        by_category[cat]["dur_us"] += float(kernel["dur"])
        by_category[cat]["count"] += 1
        by_kernel[kernel["name"]]["dur_us"] += float(kernel["dur"])
        by_kernel[kernel["name"]]["count"] += 1
        by_kernel[kernel["name"]]["category"] = cat

    def category_rows() -> list[dict[str, Any]]:
        total = sum(float(value["dur_us"]) for value in by_category.values())
        return [
            {
                "category": cat,
                "kernel_ms": ms(float(value["dur_us"])),
                "kernel_count": int(value["count"]),
                "pct": float(value["dur_us"]) / total * 100.0 if total else 0.0,
            }
            for cat, value in sorted(by_category.items(), key=lambda item: float(item[1]["dur_us"]), reverse=True)
        ]

    def kernel_rows(limit: int = 80) -> list[dict[str, Any]]:
        return [
            {
                "kernel": name,
                "category": str(value["category"]),
                "kernel_ms": ms(float(value["dur_us"])),
                "kernel_count": int(value["count"]),
            }
            for name, value in sorted(by_kernel.items(), key=lambda item: float(item[1]["dur_us"]), reverse=True)[:limit]
        ]

    return {
        "trace": str(trace_path),
        "num_kernel_events": len(kernels),
        "num_p113_related_ranges": len(ranges),
        "category_rows": category_rows(),
        "top_kernels": kernel_rows(),
        "p113_related_ranges": sorted(ranges, key=lambda item: float(item["dur"]), reverse=True)[:80],
    }


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# P113 Best Kernel Ranking",
        "",
        f"- trace: `{summary['trace']}`",
        f"- kernel events: `{summary['num_kernel_events']}`",
        f"- P113-related CPU ranges: `{summary['num_p113_related_ranges']}`",
        "",
        "## Kernel Categories",
        "",
        "| category | kernel ms | count | pct |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["category_rows"]:
        lines.append(f"| {row['category']} | {row['kernel_ms']:.6f} | {row['kernel_count']} | {row['pct']:.2f} |")
    lines.extend(["", "## Top Kernels", "", "| category | kernel ms | count | kernel |", "|---|---:|---:|---|"])
    for row in summary["top_kernels"][:40]:
        kernel = str(row["kernel"]).replace("|", "\\|")
        lines.append(f"| {row['category']} | {row['kernel_ms']:.6f} | {row['kernel_count']} | `{kernel}` |")
    lines.extend(["", "## P113 Related CPU Ranges", "", "| range ms | name |", "|---:|---|"])
    for row in summary["p113_related_ranges"][:40]:
        name = str(row["name"]).replace("|", "\\|")
        lines.append(f"| {ms(float(row['dur'])):.6f} | `{name}` |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("This profiler is intended for CUDA.")
    configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config=dict(src=args.dataset_src))
    calculator = build_calculator(args)
    run_activation_calibration(structures, calculator, args)
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)

    for graph_id in keys[: max(0, args.warmup_steps)]:
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
            with torch.profiler.record_function(f"p113_sample.graph_{int(graph_id)}"):
                profile_one(structures, int(graph_id), calculator, args)
            prof.step()
    prof.export_chrome_trace(str(trace_path))

    summary = parse_trace(trace_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "summary.md")
    print(json.dumps({"category_rows": summary["category_rows"][:10], "top_kernels": summary["top_kernels"][:10]}, indent=2))


if __name__ == "__main__":
    main()
