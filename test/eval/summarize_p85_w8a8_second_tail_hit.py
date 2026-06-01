from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


EDGE_UPDATE_RE = re.compile(r"interaction_block\.(\d+)\.attn_line\.edge_update")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P85A attribution for W8A8 second-projection + gated-tail hit rate."
    )
    parser.add_argument("--trace", required=True, help="Chrome trace JSON from P76 profiler.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-kernels", type=int, default=32)
    return parser.parse_args()


def is_kernel_event(event: dict) -> bool:
    category = str(event.get("cat", "")).lower()
    if "kernel" in category:
        return True
    args = event.get("args", {})
    return isinstance(args, dict) and args.get("External id") is not None and "stream" in args


def fine_category(kernel_name: str) -> str:
    name = kernel_name.lower()
    if (
        "quant_linear_w8a8_wmma_dual_gated_tail_n128" in name
        or "quant_linear_w8a8_static_wmma_dual_gated_tail_n128" in name
    ):
        return "w8a8_second_tail_forward"
    if (
        "w8a8_dual_gated_tail" in name
        or "w8a8_dual_input_grad" in name
        or "saved_pre" in name
    ):
        return "w8a8_second_tail_backward"
    if "fp32_gated_tail_forward_kernel" in name:
        return "fp32_gated_tail_forward"
    if "input_grad_only_gated_tail_backward" in name:
        return "fp32_tail_backward"
    if "silu_kernel" in name:
        return "torch_silu"
    if "catarraybatchedcopy" in name or "cat" in name or "concat" in name:
        return "torch_cat_layout"
    if "direct_copy_kernel" in name or "copy" in name or "memcpy" in name:
        return "torch_copy_or_contiguous"
    if "quant" in name and "w8a8" in name:
        return "other_w8a8"
    if any(token in name for token in ("gemm", "cublas", "cutlass", "matmul", "sgemm")):
        return "gemm"
    if "layer_norm" in name or "native_layer_norm" in name or "rms_norm" in name:
        return "torch_norm"
    if any(token in name for token in ("sigmoid", "mul", "add", "elementwise", "activation")):
        return "other_elementwise"
    return "other"


def ms(us: float) -> float:
    return us / 1000.0


def parse_trace(trace_path: Path, top_kernels: int) -> dict:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", [])
    stages: list[dict] = []
    kernels: list[dict] = []
    global_by_category = defaultdict(lambda: {"dur_us": 0.0, "count": 0})
    global_by_kernel = defaultdict(lambda: {"dur_us": 0.0, "count": 0, "category": ""})

    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        match = EDGE_UPDATE_RE.fullmatch(name)
        if match:
            stages.append(
                {
                    "block": int(match.group(1)),
                    "name": name,
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
            global_by_category[category]["dur_us"] += kernel["dur"]
            global_by_category[category]["count"] += 1
            global_by_kernel[name]["dur_us"] += kernel["dur"]
            global_by_kernel[name]["count"] += 1
            global_by_kernel[name]["category"] = category

    by_category = defaultdict(lambda: {"dur_us": 0.0, "count": 0})
    by_block = defaultdict(lambda: {"dur_us": 0.0, "kernel_us": 0.0, "count": 0, "categories": defaultdict(float)})
    by_kernel = defaultdict(lambda: {"dur_us": 0.0, "count": 0, "category": ""})
    total_stage_us = 0.0
    total_kernel_us = 0.0
    total_kernel_count = 0

    for stage in stages:
        total_stage_us += stage["dur"]
        block = stage["block"]
        by_block[block]["dur_us"] += stage["dur"]
        start = stage["ts"]
        end = start + stage["dur"]
        for kernel in kernels:
            if start <= kernel["ts"] <= end:
                cat = kernel["category"]
                dur = kernel["dur"]
                total_kernel_us += dur
                total_kernel_count += 1
                by_category[cat]["dur_us"] += dur
                by_category[cat]["count"] += 1
                by_block[block]["kernel_us"] += dur
                by_block[block]["count"] += 1
                by_block[block]["categories"][cat] += dur
                by_kernel[kernel["name"]]["dur_us"] += dur
                by_kernel[kernel["name"]]["count"] += 1
                by_kernel[kernel["name"]]["category"] = cat

    def category_rows(source: dict, total_us: float | None = None) -> list[dict]:
        denom = total_us if total_us is not None else sum(value["dur_us"] for value in source.values())
        return [
            {
                "category": cat,
                "kernel_ms": ms(value["dur_us"]),
                "kernel_count": value["count"],
                "pct": value["dur_us"] / denom * 100.0 if denom else 0.0,
            }
            for cat, value in sorted(source.items(), key=lambda item: item[1]["dur_us"], reverse=True)
        ]

    def kernel_rows(source: dict, limit: int) -> list[dict]:
        return [
            {
                "category": value["category"],
                "kernel_ms": ms(value["dur_us"]),
                "kernel_count": value["count"],
                "kernel": name,
            }
            for name, value in sorted(source.items(), key=lambda item: item[1]["dur_us"], reverse=True)[:limit]
        ]

    block_rows = []
    for block, value in sorted(by_block.items()):
        cats = {
            cat: ms(dur)
            for cat, dur in sorted(value["categories"].items(), key=lambda item: item[1], reverse=True)
        }
        block_rows.append(
            {
                "block": block,
                "range_ms": ms(value["dur_us"]),
                "kernel_ms": ms(value["kernel_us"]),
                "kernel_count": value["count"],
                "categories_ms": cats,
            }
        )

    edge_categories = category_rows(by_category, total_kernel_us)
    global_categories = category_rows(global_by_category)
    hit = {
        "edge_forward_w8a8_kernel_count": by_category["w8a8_second_tail_forward"]["count"],
        "edge_forward_w8a8_kernel_ms": ms(by_category["w8a8_second_tail_forward"]["dur_us"]),
        "edge_fp32_tail_kernel_count": by_category["fp32_gated_tail_forward"]["count"],
        "edge_fp32_tail_kernel_ms": ms(by_category["fp32_gated_tail_forward"]["dur_us"]),
        "global_w8a8_backward_kernel_count": global_by_category["w8a8_second_tail_backward"]["count"],
        "global_w8a8_backward_kernel_ms": ms(global_by_category["w8a8_second_tail_backward"]["dur_us"]),
        "global_fp32_tail_backward_kernel_count": global_by_category["fp32_tail_backward"]["count"],
        "global_fp32_tail_backward_kernel_ms": ms(global_by_category["fp32_tail_backward"]["dur_us"]),
    }

    return {
        "trace": str(trace_path),
        "num_edge_update_ranges": len(stages),
        "total_edge_update_range_ms": ms(total_stage_us),
        "total_edge_update_kernel_ms": ms(total_kernel_us),
        "total_edge_update_kernel_count": total_kernel_count,
        "hit": hit,
        "edge_category_rows": edge_categories,
        "edge_block_rows": block_rows,
        "edge_top_kernels": kernel_rows(by_kernel, top_kernels),
        "global_category_rows": global_categories,
        "global_top_w8a8_kernels": [
            row
            for row in kernel_rows(global_by_kernel, top_kernels * 2)
            if "w8a8" in row["category"] or "w8a8" in row["kernel"].lower()
        ][:top_kernels],
    }


def write_table(lines: list[str], rows: list[dict], *, include_pct: bool = True) -> None:
    if include_pct:
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


def write_markdown(summary: dict, output_path: Path) -> None:
    hit = summary["hit"]
    lines = [
        "# P85A W8A8 Second-Tail Hit Summary",
        "",
        f"- trace: `{summary['trace']}`",
        f"- edge_update ranges: `{summary['num_edge_update_ranges']}`",
        f"- edge_update range time: `{summary['total_edge_update_range_ms']:.6f} ms`",
        f"- edge_update kernel time: `{summary['total_edge_update_kernel_ms']:.6f} ms`",
        "",
        "## Hit Check",
        "",
        f"- W8A8 fused second-tail forward in edge_update: `{hit['edge_forward_w8a8_kernel_count']}` kernels, `{hit['edge_forward_w8a8_kernel_ms']:.6f} ms`",
        f"- FP32 gated-tail forward remaining in edge_update: `{hit['edge_fp32_tail_kernel_count']}` kernels, `{hit['edge_fp32_tail_kernel_ms']:.6f} ms`",
        f"- W8A8 fused/input-grad backward globally: `{hit['global_w8a8_backward_kernel_count']}` kernels, `{hit['global_w8a8_backward_kernel_ms']:.6f} ms`",
        f"- FP32 tail backward globally: `{hit['global_fp32_tail_backward_kernel_count']}` kernels, `{hit['global_fp32_tail_backward_kernel_ms']:.6f} ms`",
        "",
        "## Edge Update Categories",
        "",
    ]
    write_table(lines, summary["edge_category_rows"])
    lines.extend(["", "## Edge Update Top Kernels", ""])
    write_table(lines, summary["edge_top_kernels"], include_pct=False)
    lines.extend(["", "## Global W8A8 Kernels", ""])
    write_table(lines, summary["global_top_w8a8_kernels"], include_pct=False)
    lines.extend(["", "## Per Block", ""])
    lines.append("| block | range ms | kernel ms | count | top categories |")
    lines.append("|---:|---:|---:|---:|---|")
    for row in summary["edge_block_rows"]:
        cats = ", ".join(f"{cat}={value:.3f}" for cat, value in list(row["categories_ms"].items())[:5])
        lines.append(
            f"| {row['block']} | {row['range_ms']:.6f} | {row['kernel_ms']:.6f} | "
            f"{row['kernel_count']} | {cats} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = parse_trace(Path(args.trace), args.top_kernels)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "summary.md")
    print(json.dumps(summary["hit"], indent=2))


if __name__ == "__main__":
    main()
