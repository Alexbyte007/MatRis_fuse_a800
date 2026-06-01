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
from profile_p88_w8a8_second_tail_breakdown import (  # noqa: E402
    BASE_ENV_FLAGS,
    FUSION_MODE,
    P8D_QUANT_MODE,
    TARGET_GLOBS,
    TARGET_GROUPS,
    cats_text,
    fine_category,
    is_kernel_event,
    ms,
    selected_target_globs,
)


P90_RANGE_RE = re.compile(r"p(90|91)\.w8a8_saved_pre\.backward\.(.+)")
P88_RANGE_RE = re.compile(r"p88\.w8a8_saved_pre\.(.+)")
AUTOGRAD_BACKWARD_TOKENS = (
    "cudaw8a8staticwmmadualgatedtailn128savedprefunctionbackward",
    "cudaw8a8staticwmmadualsilugatedtailn128savedprefunctionbackward",
    "w8a8staticwmmadualgatedtailn128savedprefunctionbackward",
    "w8a8staticwmmadualsilugatedtailn128savedprefunctionbackward",
)
SYNC_TOKENS = ("cudadevicesynchronize", "cudastreamsynchronize", "cudaeventsynchronize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P90 W8A8 saved-pre backward-chain bottleneck breakdown.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default=P8D_QUANT_MODE)
    parser.add_argument("--fusion-mode", default=FUSION_MODE)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--enable-p89-v2", action="store_true")
    parser.add_argument("--enable-p91-driver", action="store_true")
    parser.add_argument("--enable-p91-cublas-pair", action="store_true")
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
    parser.add_argument("--output-dir", default="results/p90_w8a8_backward_chain_breakdown")
    return parser.parse_args()


def is_cuda_api_event(event: dict[str, Any]) -> bool:
    if is_kernel_event(event):
        return False
    name = str(event.get("name", "")).lower()
    category = str(event.get("cat", "")).lower()
    return name.startswith("cuda") or "cuda_runtime" in category or "cuda driver" in category


def is_sync_event(event: dict[str, Any]) -> bool:
    name = str(event.get("name", "")).replace("_", "").lower()
    return any(token in name for token in SYNC_TOKENS)


def event_overlaps(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    inner_start = float(inner.get("ts", 0.0))
    inner_end = inner_start + float(inner.get("dur", 0.0))
    outer_start = float(outer.get("ts", 0.0))
    outer_end = outer_start + float(outer.get("dur", 0.0))
    return inner_start < outer_end and inner_end > outer_start


def event_starts_inside(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    inner_start = float(inner.get("ts", 0.0))
    outer_start = float(outer.get("ts", 0.0))
    outer_end = outer_start + float(outer.get("dur", 0.0))
    return outer_start <= inner_start <= outer_end


def collect_trace_events(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    raw_events = [event for event in payload.get("traceEvents", []) if event.get("ph") == "X"]
    ranges_p90: list[dict[str, Any]] = []
    ranges_p88: list[dict[str, Any]] = []
    kernels: list[dict[str, Any]] = []
    cuda_api: list[dict[str, Any]] = []
    sync_events: list[dict[str, Any]] = []
    autograd_backward_ranges: list[dict[str, Any]] = []
    autograd_candidates = defaultdict(lambda: {"dur_us": 0.0, "count": 0})

    for event in raw_events:
        name = str(event.get("name", ""))
        duration = float(event.get("dur", 0.0))
        match_p90 = P90_RANGE_RE.fullmatch(name)
        if match_p90:
            detail = match_p90.group(2)
            if match_p90.group(1) == "91":
                detail = f"p91.{detail}"
            ranges_p90.append(
                {
                    "name": name,
                    "detail": detail,
                    "cat": str(event.get("cat", "")),
                    "ts": float(event.get("ts", 0.0)),
                    "dur": duration,
                }
            )
            continue
        match_p88 = P88_RANGE_RE.fullmatch(name)
        if match_p88:
            ranges_p88.append(
                {
                    "name": name,
                    "detail": match_p88.group(1),
                    "cat": str(event.get("cat", "")),
                    "ts": float(event.get("ts", 0.0)),
                    "dur": duration,
                }
            )
            continue
        if is_kernel_event(event):
            kernels.append(
                {
                    "name": name,
                    "ts": float(event.get("ts", 0.0)),
                    "dur": duration,
                    "category": fine_category(name),
                }
            )
            continue
        if is_cuda_api_event(event):
            api_event = {"name": name, "ts": float(event.get("ts", 0.0)), "dur": duration}
            cuda_api.append(api_event)
            if is_sync_event(event):
                sync_events.append(api_event)
        lowered = name.lower()
        if any(token in lowered for token in AUTOGRAD_BACKWARD_TOKENS):
            autograd_candidates[name]["dur_us"] += duration
            autograd_candidates[name]["count"] += 1
            if str(event.get("cat", "")) == "cpu_op" and name.startswith("_CudaW8A8"):
                autograd_backward_ranges.append({"name": name, "ts": float(event.get("ts", 0.0)), "dur": duration})

    return {
        "raw_event_count": len(raw_events),
        "ranges_p90": ranges_p90,
        "ranges_p88": ranges_p88,
        "kernels": kernels,
        "cuda_api": cuda_api,
        "sync_events": sync_events,
        "autograd_backward_ranges": autograd_backward_ranges,
        "autograd_candidates": autograd_candidates,
    }


def summarize_ranges(ranges: list[dict[str, Any]], kernels: list[dict[str, Any]], cuda_api: list[dict[str, Any]], sync_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(
        lambda: {
            "cpu_us": 0.0,
            "gpu_annotation_us": 0.0,
            "cuda_api_us": 0.0,
            "sync_us": 0.0,
            "cpu_count": 0,
            "gpu_annotation_count": 0,
            "cuda_api_count": 0,
            "sync_count": 0,
        }
    )
    for range_event in ranges:
        key = str(range_event["detail"])
        row = grouped[key]
        if str(range_event.get("cat", "")) == "gpu_user_annotation":
            row["gpu_annotation_us"] += float(range_event["dur"])
            row["gpu_annotation_count"] += 1
            continue
        row["cpu_us"] += float(range_event["dur"])
        row["cpu_count"] += 1
        for api_event in cuda_api:
            if event_starts_inside(api_event, range_event):
                row["cuda_api_us"] += float(api_event["dur"])
                row["cuda_api_count"] += 1
        for sync_event in sync_events:
            if event_starts_inside(sync_event, range_event):
                row["sync_us"] += float(sync_event["dur"])
                row["sync_count"] += 1

    rows: list[dict[str, Any]] = []
    for name, value in sorted(grouped.items(), key=lambda item: float(item[1]["cpu_us"]) + float(item[1]["gpu_annotation_us"]), reverse=True):
        cpu_ms = ms(float(value["cpu_us"]))
        gpu_annotation_ms = ms(float(value["gpu_annotation_us"]))
        cuda_api_ms = ms(float(value["cuda_api_us"]))
        sync_ms = ms(float(value["sync_us"]))
        rows.append(
            {
                "name": name,
                "cpu_ms": cpu_ms,
                "gpu_annotation_ms": gpu_annotation_ms,
                "cuda_api_ms": cuda_api_ms,
                "sync_ms": sync_ms,
                "cpu_self_without_cuda_api_ms": max(0.0, cpu_ms - cuda_api_ms),
                "cpu_count": int(value["cpu_count"]),
                "gpu_annotation_count": int(value["gpu_annotation_count"]),
                "cuda_api_count": int(value["cuda_api_count"]),
                "sync_count": int(value["sync_count"]),
            }
        )
    return rows


def top_kernel_rows(kernels: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    grouped = defaultdict(lambda: {"dur_us": 0.0, "count": 0, "category": ""})
    for kernel in kernels:
        name = kernel["name"]
        grouped[name]["dur_us"] += float(kernel["dur"])
        grouped[name]["count"] += 1
        grouped[name]["category"] = kernel["category"]
    return [
        {
            "kernel": name,
            "category": value["category"],
            "kernel_ms": ms(float(value["dur_us"])),
            "kernel_count": int(value["count"]),
        }
        for name, value in sorted(grouped.items(), key=lambda item: float(item[1]["dur_us"]), reverse=True)[:limit]
    ]


def top_cuda_api_rows(events: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    grouped = defaultdict(lambda: {"dur_us": 0.0, "count": 0})
    for event in events:
        grouped[event["name"]]["dur_us"] += float(event["dur"])
        grouped[event["name"]]["count"] += 1
    return [
        {"name": name, "api_ms": ms(float(value["dur_us"])), "count": int(value["count"])}
        for name, value in sorted(grouped.items(), key=lambda item: float(item[1]["dur_us"]), reverse=True)[:limit]
    ]


def top_autograd_rows(candidates: dict[str, dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"name": name, "range_ms": ms(float(value["dur_us"])), "count": int(value["count"])}
        for name, value in sorted(candidates.items(), key=lambda item: float(item[1]["dur_us"]), reverse=True)[:limit]
    ]


def kernels_started_in_ranges(kernels: list[dict[str, Any]], ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for range_event in ranges:
        for kernel in kernels:
            if event_starts_inside(kernel, range_event):
                selected.append(kernel)
    return selected


def find_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["name"] == name), None)


def make_quick_read(
    p90_rows: list[dict[str, Any]],
    p88_rows: list[dict[str, Any]],
    autograd_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tail = find_row(p90_rows, "tail_backward_op") or find_row(p90_rows, "tail_backward_op_n128_v2")
    core_linear = find_row(p90_rows, "input_grad_core_linear")
    gate_linear = find_row(p90_rows, "input_grad_gate_linear")
    core_reshape = find_row(p90_rows, "input_grad_core_reshape")
    gate_reshape = find_row(p90_rows, "input_grad_gate_reshape")
    grad_out_flat = find_row(p90_rows, "grad_out_flatten_contiguous")
    saved_unpack = find_row(p90_rows, "saved_tensors_unpack")
    load_op = find_row(p90_rows, "load_matris_op")
    p91_driver = find_row(p90_rows, "p91.driver")
    save_for_backward = find_row(p88_rows, "forward.save_for_backward")
    forward_prepare = find_row(p88_rows, "forward.prepare")

    input_grad_linear_cpu_ms = sum(row["cpu_ms"] for row in (core_linear, gate_linear) if row)
    input_grad_linear_gpu_ms = sum(row["gpu_annotation_ms"] for row in (core_linear, gate_linear) if row)
    reshape_ms = sum(row["cpu_ms"] for row in (core_reshape, gate_reshape) if row)
    observed_backward_cpu_ms = sum(row["cpu_ms"] for row in p90_rows)
    observed_backward_gpu_annotation_ms = sum(row["gpu_annotation_ms"] for row in p90_rows)
    observed_cuda_api_ms = sum(row["cuda_api_ms"] for row in p90_rows)
    sync_ms = sum(row["sync_ms"] for row in p90_rows)
    autograd_function_cpu_ms = next(
        (
            row["range_ms"]
            for row in autograd_rows
            if row["name"]
            in (
                "_CudaW8A8StaticWmmaDualGatedTailN128SavedPreFunctionBackward",
                "_CudaW8A8StaticWmmaDualSiluGatedTailN128SavedPreFunctionBackward",
            )
        ),
        0.0,
    )

    return {
        "tail_backward_cpu_ms": tail["cpu_ms"] if tail else 0.0,
        "tail_backward_gpu_annotation_ms": tail["gpu_annotation_ms"] if tail else 0.0,
        "input_grad_linear_cpu_ms": input_grad_linear_cpu_ms,
        "input_grad_linear_gpu_annotation_ms": input_grad_linear_gpu_ms,
        "reshape_range_ms": reshape_ms,
        "grad_out_flatten_contiguous_range_ms": grad_out_flat["cpu_ms"] if grad_out_flat else 0.0,
        "saved_tensors_unpack_range_ms": saved_unpack["cpu_ms"] if saved_unpack else 0.0,
        "load_matris_op_range_ms": load_op["cpu_ms"] if load_op else 0.0,
        "p91_driver_cpu_ms": p91_driver["cpu_ms"] if p91_driver else 0.0,
        "p91_driver_gpu_annotation_ms": p91_driver["gpu_annotation_ms"] if p91_driver else 0.0,
        "save_for_backward_cpu_ms": save_for_backward["cpu_ms"] if save_for_backward else 0.0,
        "save_for_backward_gpu_annotation_ms": save_for_backward["gpu_annotation_ms"] if save_for_backward else 0.0,
        "forward_prepare_cpu_ms": forward_prepare["cpu_ms"] if forward_prepare else 0.0,
        "autograd_function_cpu_ms": autograd_function_cpu_ms,
        "unmarked_autograd_cpu_ms": max(0.0, autograd_function_cpu_ms - observed_backward_cpu_ms),
        "observed_p90_backward_cpu_ms": observed_backward_cpu_ms,
        "observed_p90_backward_gpu_annotation_ms": observed_backward_gpu_annotation_ms,
        "observed_p90_cuda_api_ms": observed_cuda_api_ms,
        "observed_p90_sync_ms": sync_ms,
    }


def write_range_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.append("| range | CPU ms | GPU annotated ms | cuda api ms | CPU self minus cuda api ms | sync ms | CPU ranges | GPU ranges |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['cpu_ms']:.6f} | {row['gpu_annotation_ms']:.6f} | "
            f"{row['cuda_api_ms']:.6f} | {row['cpu_self_without_cuda_api_ms']:.6f} | {row['sync_ms']:.6f} | "
            f"{row['cpu_count']} | {row['gpu_annotation_count']} |"
        )


def write_kernel_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.append("| category | kernel ms | count | kernel |")
    lines.append("|---|---:|---:|---|")
    for row in rows:
        kernel = row["kernel"].replace("|", "\\|")
        lines.append(f"| {row['category']} | {row['kernel_ms']:.6f} | {row['kernel_count']} | `{kernel}` |")


def write_api_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.append("| api/event | ms | count |")
    lines.append("|---|---:|---:|")
    for row in rows:
        lines.append(f"| `{row['name']}` | {row['api_ms']:.6f} | {row['count']} |")


def write_autograd_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.append("| autograd event | range ms | count |")
    lines.append("|---|---:|---:|")
    for row in rows:
        lines.append(f"| `{row['name']}` | {row['range_ms']:.6f} | {row['count']} |")


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    quick = summary["quick_read"]
    lines = [
        "# P90 W8A8 Saved-Pre Backward-Chain Breakdown",
        "",
        f"- trace: `{summary['trace']}`",
        f"- limit: `{summary['metadata']['limit']}`",
        f"- p89 tail bwd v2: `{summary['metadata']['enable_p89_v2']}`",
        f"- p91 backward driver: `{summary['metadata']['enable_p91_driver']}`",
        f"- p91 cuBLAS pair: `{summary['metadata']['enable_p91_cublas_pair']}`",
        "",
        "## Quick Read",
        "",
        f"- autograd function CPU range: `{quick['autograd_function_cpu_ms']:.6f} ms`.",
        f"- observed P90 marked CPU range: `{quick['observed_p90_backward_cpu_ms']:.6f} ms`; "
        f"unmarked/autograd glue CPU: `{quick['unmarked_autograd_cpu_ms']:.6f} ms`.",
        f"- observed P90 GPU-annotated range: `{quick['observed_p90_backward_gpu_annotation_ms']:.6f} ms`; "
        f"cuda api inside marked CPU ranges: `{quick['observed_p90_cuda_api_ms']:.6f} ms`.",
        f"- tail backward: CPU `{quick['tail_backward_cpu_ms']:.6f} ms`, GPU `{quick['tail_backward_gpu_annotation_ms']:.6f} ms`.",
        f"- input-grad GEMM/core+gate F.linear: CPU `{quick['input_grad_linear_cpu_ms']:.6f} ms`, "
        f"GPU `{quick['input_grad_linear_gpu_annotation_ms']:.6f} ms`.",
        f"- P91 driver: CPU `{quick['p91_driver_cpu_ms']:.6f} ms`, "
        f"GPU `{quick['p91_driver_gpu_annotation_ms']:.6f} ms`.",
        f"- reshape/copy markers: grad_out flatten `{quick['grad_out_flatten_contiguous_range_ms']:.6f} ms`, "
        f"input-grad reshapes `{quick['reshape_range_ms']:.6f} ms`.",
        f"- saved tensor overhead: forward save_for_backward CPU `{quick['save_for_backward_cpu_ms']:.6f} ms`, "
        f"backward unpack `{quick['saved_tensors_unpack_range_ms']:.6f} ms`.",
        f"- sync inside marked P90 ranges: `{quick['observed_p90_sync_ms']:.6f} ms`.",
        "",
        "## P90 Backward Ranges",
        "",
    ]
    write_range_table(lines, summary["p90_range_rows"])
    lines.extend(["", "## P88 Saved-Pre Context Ranges", ""])
    write_range_table(lines, summary["p88_range_rows"])
    lines.extend(["", "## Top Kernels In P90 Ranges", ""])
    write_kernel_table(lines, summary["p90_top_kernels"])
    lines.extend(["", "## CUDA API / Sync Events", ""])
    write_api_table(lines, summary["top_cuda_api_events"])
    lines.extend(["", "## Autograd Wrapper Candidates", ""])
    write_autograd_table(lines, summary["autograd_candidates"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(trace_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    events = collect_trace_events(trace_path)
    target_globs = selected_target_globs(args)
    p90_rows = summarize_ranges(events["ranges_p90"], events["kernels"], events["cuda_api"], events["sync_events"])
    p88_rows = summarize_ranges(events["ranges_p88"], events["kernels"], events["cuda_api"], events["sync_events"])
    autograd_rows = top_autograd_rows(events["autograd_candidates"], 24)
    p90_kernels = kernels_started_in_ranges(events["kernels"], events["autograd_backward_ranges"])

    summary = {
        "trace": str(trace_path),
        "raw_event_count": events["raw_event_count"],
        "num_p90_ranges": len(events["ranges_p90"]),
        "num_p88_ranges": len(events["ranges_p88"]),
        "num_kernel_events": len(events["kernels"]),
        "num_cuda_api_events": len(events["cuda_api"]),
        "p90_range_rows": p90_rows,
        "p88_range_rows": p88_rows,
        "quick_read": make_quick_read(p90_rows, p88_rows, autograd_rows),
        "p90_top_kernels": top_kernel_rows(p90_kernels, 24),
        "top_cuda_api_events": top_cuda_api_rows(events["cuda_api"], 24),
        "top_sync_events": top_cuda_api_rows(events["sync_events"], 24),
        "autograd_candidates": autograd_rows,
        "metadata": {
            "limit": args.limit,
            "warmup_steps": args.warmup_steps,
            "sample_seed": args.sample_seed,
            "quant_mode": args.quant_mode,
            "fusion_mode": args.fusion_mode,
            "target_group": args.target_group,
            "target_globs": target_globs,
            "enable_p89_v2": args.enable_p89_v2,
            "enable_p91_driver": args.enable_p91_driver,
            "enable_p91_cublas_pair": args.enable_p91_cublas_pair,
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("P90 is intended for CUDA profiling.")

    target_globs = selected_target_globs(args)
    env_flags = dict(BASE_ENV_FLAGS)
    env_flags["MATRIS_QUANT_INCLUDE_GLOBS"] = ",".join(target_globs)
    env_flags.update(
        {
            "MATRIS_P30_SAVED_PRE_FUSED_INPUT_GRAD": "0",
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL": "0",
            "MATRIS_P33_CUBLAS_GROUPED_INPUT_GRAD": "0",
            "MATRIS_P34_BMM_INPUT_GRAD": "0",
            "MATRIS_P40_SAVED_PRE_DQ_FUSED_INPUT_GRAD": "0",
            "MATRIS_P41_CUBLAS_PAIR_INPUT_GRAD": "0",
            "MATRIS_P43_TAIL_STACK_BMM_INPUT_GRAD": "0",
            "MATRIS_P89_TAIL_BWD_V2": "1" if args.enable_p89_v2 else "0",
            "MATRIS_P91_BACKWARD_DRIVER": "1" if args.enable_p91_driver else "0",
            "MATRIS_P91_CUBLAS_PAIR": "1" if args.enable_p91_cublas_pair else "0",
        }
    )
    for key, value in env_flags.items():
        os.environ[key] = value

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
            with torch.profiler.record_function(f"p90_sample.graph_{int(graph_id)}"):
                profile_one(structures, int(graph_id), calculator, args)
            prof.step()
    prof.export_chrome_trace(str(trace_path))

    summary = build_summary(trace_path, args)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "summary.md")
    print(json.dumps(summary["quick_read"], indent=2))


if __name__ == "__main__":
    main()
