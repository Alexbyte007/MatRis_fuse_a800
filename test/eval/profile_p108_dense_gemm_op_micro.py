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
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from benchmark_p108_projection_gemm_feasibility import collect_case  # noqa: E402
from check_attn_line_a3_correctness import PURE_FUSE_ENV  # noqa: E402
from infer_salex_lmdb_quant import build_calculator, configure_precision, select_group_aligned_keys  # noqa: E402
from matris.model.functions import _load_matris_op  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile only the P108 dense-GEMM CUDA op on real attn_line cases.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--profile-cases", type=int, default=8)
    parser.add_argument("--block-index", type=int, default=9)
    parser.add_argument("--min-rows", type=int, default=4097)
    parser.add_argument("--warmup-iters", type=int, default=8)
    parser.add_argument("--profile-iters", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--output-dir", default="results/p108_dense_gemm_op_micro_profile")
    parser.add_argument("--no-set-pure-fuse-env", action="store_true")
    return parser.parse_args()


def run_dense_op(case: dict[str, Any]):
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm"):
        raise RuntimeError("P108 dense-GEMM op unavailable")
    return matris_op.line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm(
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


def is_kernel_event(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    if "kernel" in category:
        return True
    args = event.get("args", {})
    return isinstance(args, dict) and args.get("External id") is not None and "stream" in args


def categorize_kernel(name: str) -> str:
    lowered = name.lower()
    if "line_edge_fill_silu_grad_hidden" in lowered:
        return "fill_silu_grad_hidden"
    if "line_edge_cat_grad_scatter_backward" in lowered:
        return "cat_grad_scatter"
    if "sgemm" in lowered or "gemm" in lowered or "cublas" in lowered or "cutlass" in lowered:
        return "sgemm"
    if "memcpy" in lowered or "copy" in lowered:
        return "copy"
    if "fill" in lowered or "zero" in lowered:
        return "fill_zero"
    return "other"


def parse_trace(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    ranges = []
    kernels = []
    for event in payload.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        if name.startswith("p108_dense_op."):
            ranges.append(
                {
                    "name": name,
                    "ts": float(event.get("ts", 0.0)),
                    "dur": float(event.get("dur", 0.0)),
                }
            )
        elif is_kernel_event(event):
            kernels.append(
                {
                    "name": name,
                    "ts": float(event.get("ts", 0.0)),
                    "dur": float(event.get("dur", 0.0)),
                    "category": categorize_kernel(name),
                }
            )

    by_category = defaultdict(float)
    by_kernel = defaultdict(lambda: {"dur_us": 0.0, "count": 0, "category": ""})
    range_kernel_us = 0.0
    range_dur_us = 0.0
    for item in ranges:
        start = item["ts"]
        end = start + item["dur"]
        range_dur_us += item["dur"]
        for kernel in kernels:
            if start <= kernel["ts"] <= end:
                by_category[kernel["category"]] += kernel["dur"]
                by_kernel[kernel["name"]]["dur_us"] += kernel["dur"]
                by_kernel[kernel["name"]]["count"] += 1
                by_kernel[kernel["name"]]["category"] = kernel["category"]
                range_kernel_us += kernel["dur"]

    top_kernels = [
        {"name": name, "ms": value["dur_us"] / 1000.0, "count": value["count"], "category": value["category"]}
        for name, value in sorted(by_kernel.items(), key=lambda item: item[1]["dur_us"], reverse=True)
    ]
    return {
        "range_count": len(ranges),
        "range_ms": range_dur_us / 1000.0,
        "kernel_ms": range_kernel_us / 1000.0,
        "categories_ms": {key: value / 1000.0 for key, value in sorted(by_category.items(), key=lambda item: item[1], reverse=True)},
        "top_kernels": top_kernels[:32],
    }


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    profile = payload["profile"]
    lines = [
        "# P108 Dense-GEMM Op Micro Profile",
        "",
        f"- cases: `{len(payload['cases'])}`",
        f"- range_count: `{profile['range_count']}`",
        f"- total range ms: `{profile['range_ms']:.6f}`",
        f"- total kernel ms: `{profile['kernel_ms']:.6f}`",
        "",
        "## Categories",
        "",
        "| category | ms | share |",
        "|---|---:|---:|",
    ]
    total = float(profile["kernel_ms"]) or 1.0
    for name, ms in profile["categories_ms"].items():
        lines.append(f"| `{name}` | `{float(ms):.6f}` | `{float(ms) / total * 100.0:.2f}%` |")
    lines.extend(["", "## Top Kernels", "", "| kernel | category | calls | ms |", "|---|---|---:|---:|"])
    for item in profile["top_kernels"]:
        lines.append(f"| `{item['name']}` | `{item['category']}` | `{item['count']}` | `{item['ms']:.6f}` |")
    output_path.write_text("\n".join(lines), encoding="utf-8")


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

    cases = []
    for graph_id in tqdm(keys, desc="collect_p108_cases", leave=False):
        atom = structures.get_atoms(int(graph_id))
        calculator._adjust_pbc(atom)
        structure = AseAtomsAdaptor.get_structure(atom)
        graph_cpu = calculator.model.graph_converter(structure)
        graph = graph_cpu[0].to(args.device) if isinstance(graph_cpu, list) else graph_cpu.to(args.device)
        case = collect_case(calculator, graph, int(graph_id), args)
        if case is not None and int(case["rows"]) >= int(args.min_rows):
            cases.append(case)
        if len(cases) >= int(args.profile_cases):
            break
    if not cases:
        raise RuntimeError("No cases matched profile filter")

    for case in cases:
        for _ in range(max(0, args.warmup_iters)):
            run_dense_op(case)
    torch.cuda.synchronize()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.json"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        for case_index, case in enumerate(cases):
            for iter_index in range(max(1, args.profile_iters)):
                with torch.profiler.record_function(
                    f"p108_dense_op.case_{case_index}.rows_{int(case['rows'])}.iter_{iter_index}"
                ):
                    run_dense_op(case)
                prof.step()
    prof.export_chrome_trace(str(trace_path))
    profile = parse_trace(trace_path)
    payload = {
        "args": vars(args),
        "cases": [{"graph_id": int(case["graph_id"]), "rows": int(case["rows"]), "node_rows": int(case["node_rows"])} for case in cases],
        "trace": str(trace_path),
        "profile": profile,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
