from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


EDGE_UPDATE_RE = re.compile(r"interaction_block\.(\d+)\.attn_line\.edge_update")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P84A fine-grained kernel attribution for attn_line.edge_update."
    )
    parser.add_argument("--trace", required=True, help="Chrome trace JSON from profile_p76_attn_line_kernel_breakdown.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-kernels", type=int, default=24)
    return parser.parse_args()


def is_kernel_event(event: dict) -> bool:
    category = str(event.get("cat", "")).lower()
    if "kernel" in category:
        return True
    args = event.get("args", {})
    return isinstance(args, dict) and args.get("External id") is not None and "stream" in args


def fine_category(kernel_name: str) -> str:
    name = kernel_name.lower()
    if "fp32_gated_tail_forward_kernel" in name:
        return "fp32_gated_tail_forward"
    if "silu_kernel" in name:
        return "torch_silu"
    if "catarraybatchedcopy" in name or "cat" in name or "concat" in name:
        return "torch_cat_layout"
    if "direct_copy_kernel" in name or "copy" in name or "memcpy" in name:
        return "torch_copy_or_contiguous"
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
            kernels.append(
                {
                    "name": name,
                    "ts": float(event.get("ts", 0.0)),
                    "dur": float(event.get("dur", 0.0)),
                    "category": fine_category(name),
                }
            )

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

    category_rows = []
    non_gemm_us = sum(value["dur_us"] for key, value in by_category.items() if key != "gemm")
    for cat, value in sorted(by_category.items(), key=lambda item: item[1]["dur_us"], reverse=True):
        category_rows.append(
            {
                "category": cat,
                "kernel_ms": ms(value["dur_us"]),
                "kernel_count": value["count"],
                "pct_of_edge_kernel": value["dur_us"] / total_kernel_us * 100.0 if total_kernel_us else 0.0,
                "pct_of_non_gemm": value["dur_us"] / non_gemm_us * 100.0 if cat != "gemm" and non_gemm_us else None,
            }
        )

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

    kernel_rows = []
    for name, value in sorted(by_kernel.items(), key=lambda item: item[1]["dur_us"], reverse=True)[:top_kernels]:
        kernel_rows.append(
            {
                "category": value["category"],
                "kernel_ms": ms(value["dur_us"]),
                "kernel_count": value["count"],
                "kernel": name,
            }
        )

    return {
        "trace": str(trace_path),
        "num_edge_update_ranges": len(stages),
        "total_edge_update_range_ms": ms(total_stage_us),
        "total_edge_update_kernel_ms": ms(total_kernel_us),
        "total_edge_update_kernel_count": total_kernel_count,
        "total_non_gemm_kernel_ms": ms(non_gemm_us),
        "category_rows": category_rows,
        "block_rows": block_rows,
        "top_kernels": kernel_rows,
    }


def write_markdown(summary: dict, output_path: Path) -> None:
    lines = [
        "# P84A Attn-Line Edge Update Internal Breakdown",
        "",
        f"- trace: `{summary['trace']}`",
        f"- edge_update ranges: `{summary['num_edge_update_ranges']}`",
        f"- range time: `{summary['total_edge_update_range_ms']:.6f} ms`",
        f"- kernel time: `{summary['total_edge_update_kernel_ms']:.6f} ms`",
        f"- non-GEMM kernel time: `{summary['total_non_gemm_kernel_ms']:.6f} ms`",
        "",
        "## Category Split",
        "",
        "| category | kernel ms | count | % edge kernels | % non-GEMM |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["category_rows"]:
        pct_non_gemm = "" if row["pct_of_non_gemm"] is None else f"{row['pct_of_non_gemm']:.2f}"
        lines.append(
            f"| {row['category']} | {row['kernel_ms']:.6f} | {row['kernel_count']} | "
            f"{row['pct_of_edge_kernel']:.2f} | {pct_non_gemm} |"
        )

    lines.extend(
        [
            "",
            "## Top Kernels",
            "",
            "| category | kernel ms | count | kernel |",
            "|---|---:|---:|---|",
        ]
    )
    for row in summary["top_kernels"]:
        kernel = row["kernel"].replace("|", "\\|")
        lines.append(f"| {row['category']} | {row['kernel_ms']:.6f} | {row['kernel_count']} | `{kernel}` |")

    lines.extend(
        [
            "",
            "## Per Block",
            "",
            "| block | range ms | kernel ms | count | top categories |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["block_rows"]:
        cats = ", ".join(f"{cat}={value:.3f}" for cat, value in list(row["categories_ms"].items())[:5])
        lines.append(
            f"| {row['block']} | {row['range_ms']:.6f} | {row['kernel_ms']:.6f} | "
            f"{row['kernel_count']} | {cats} |"
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `fp32_gated_tail_forward` is the P77/P78 fused tail kernel: LayerNorm(core/gate), SiLU, sigmoid, and gate multiply.",
            "- `torch_silu` is the unfused SiLU before the second projection inside the two-branch GatedMLP.",
            "- `torch_copy_or_contiguous` is mostly split/copy/contiguous glue around branch tensors.",
            "- `torch_cat_layout` is materialized cat/layout before a fused second projection.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = parse_trace(Path(args.trace), args.top_kernels)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "summary.md")
    print(json.dumps(summary["category_rows"], indent=2))


if __name__ == "__main__":
    main()
