from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
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

from infer_salex_lmdb_quant import (  # noqa: E402
    autocast_context,
    build_calculator,
    configure_precision,
    select_group_aligned_keys,
    sync_if_needed,
)
import matris.model.interaction_block as interaction_block  # noqa: E402
from matris.model.functions import _load_matris_op  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P82B line attention max-reduction microbenchmark.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-captures", type=int, default=80)
    parser.add_argument("--max-bench-cases", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output-json", default="results/p82b_line_attention_max_microbench.json")
    return parser.parse_args()


def index_stats(index_cpu: torch.Tensor, num_segments: int) -> dict:
    if index_cpu.numel() <= 1:
        monotonic_ratio = 1.0
        is_monotonic = True
    else:
        monotonic = index_cpu[1:] >= index_cpu[:-1]
        monotonic_ratio = float(monotonic.float().mean().item())
        is_monotonic = bool(monotonic.all().item())
    unique = torch.unique(index_cpu)
    unique_consecutive = torch.unique_consecutive(index_cpu)
    bincount = torch.bincount(index_cpu, minlength=num_segments)
    nonzero = bincount[bincount > 0].float()
    if nonzero.numel() == 0:
        degree = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    else:
        degree = {
            "mean": float(nonzero.mean().item()),
            "p50": float(torch.quantile(nonzero, 0.50).item()),
            "p95": float(torch.quantile(nonzero, 0.95).item()),
            "max": float(nonzero.max().item()),
        }
    return {
        "is_monotonic": is_monotonic,
        "monotonic_ratio": monotonic_ratio,
        "is_grouped": int(unique.numel()) == int(unique_consecutive.numel()),
        "unique_segments": int(unique.numel()),
        "unique_consecutive_runs": int(unique_consecutive.numel()),
        "degree": degree,
    }


def summarize_capture(source_index: torch.Tensor, target_index: torch.Tensor, num_segments: int) -> dict:
    rows = int(source_index.numel())
    return {
        "rows": rows,
        "num_segments": int(num_segments),
        "source": index_stats(source_index, num_segments),
        "target": index_stats(target_index, num_segments),
    }


def time_cuda(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def scatter_reduce_pair(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_out = torch.full((num_segments, source_logits.shape[1]), -torch.inf, device=source_logits.device)
    target_out = torch.full((num_segments, target_logits.shape[1]), -torch.inf, device=target_logits.device)
    source_expand = source_index.view(-1, 1).expand(-1, source_logits.shape[1])
    target_expand = target_index.view(-1, 1).expand(-1, target_logits.shape[1])
    source_out.scatter_reduce_(0, source_expand, source_logits, reduce="amax", include_self=True)
    target_out.scatter_reduce_(0, target_expand, target_logits, reduce="amax", include_self=True)
    return source_out, target_out


def scatter_reduce_single(logits: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = torch.full((num_segments, logits.shape[1]), -torch.inf, device=logits.device)
    expand = index.view(-1, 1).expand(-1, logits.shape[1])
    out.scatter_reduce_(0, expand, logits, reduce="amax", include_self=True)
    return out


def segment_reduce_sorted(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return torch.segment_reduce(logits, reduce="max", lengths=lengths, axis=0)


def run_capture(args: argparse.Namespace) -> list[dict]:
    structures = AseDBDataset(config={"src": args.dataset_src})
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    calculator = build_calculator(args)
    configure_precision(args.device, args.precision_mode)

    captures: list[dict] = []
    original = interaction_block.fused_line_attention_or_none

    def wrapped(source_logits, target_logits, values, source_index, target_index, num_segments, *, enable_hint, atom_graph=False):
        if (
            enable_hint
            and not atom_graph
            and len(captures) < args.max_captures
            and source_index.is_cuda
            and target_index.is_cuda
        ):
            captures.append(
                {
                    "source_index": source_index.detach().cpu(),
                    "target_index": target_index.detach().cpu(),
                    "num_segments": int(num_segments),
                    "rows": int(source_index.numel()),
                }
            )
        return original(
            source_logits,
            target_logits,
            values,
            source_index,
            target_index,
            num_segments,
            enable_hint=enable_hint,
            atom_graph=atom_graph,
        )

    interaction_block.fused_line_attention_or_none = wrapped
    try:
        for graph_id in tqdm(keys, desc="capture", leave=False):
            if len(captures) >= args.max_captures:
                break
            atom = structures.get_atoms(int(graph_id))
            calculator._adjust_pbc(atom)
            structure = AseAtomsAdaptor.get_structure(atom)
            graph_cpu = calculator.model.graph_converter(structure)
            n_atoms_factor = 1 if not calculator.model.is_intensive else structure.composition.num_atoms
            sync_if_needed(args.device)
            with autocast_context(args.device, args.precision_mode):
                _ = calculator.calculate_graphs([graph_cpu], [n_atoms_factor])[0]
            sync_if_needed(args.device)
    finally:
        interaction_block.fused_line_attention_or_none = original
    return captures


def choose_cases(captures: list[dict], max_cases: int) -> list[dict]:
    by_shape: dict[tuple[int, int], dict] = {}
    for capture in captures:
        key = (int(capture["rows"]), int(capture["num_segments"]))
        by_shape.setdefault(key, capture)
    cases = list(by_shape.values())
    cases.sort(key=lambda item: item["rows"], reverse=True)
    if len(cases) <= max_cases:
        return cases
    picks = cases[: max_cases // 2]
    remaining = cases[max_cases // 2 :]
    if remaining:
        step = max(1, len(remaining) // max(1, max_cases - len(picks)))
        picks.extend(remaining[::step][: max_cases - len(picks)])
    return picks


def benchmark_case(matris_op, capture: dict, args: argparse.Namespace) -> dict:
    device = args.device
    rows = int(capture["rows"])
    num_segments = int(capture["num_segments"])
    source_index = capture["source_index"].to(device=device, dtype=torch.long, non_blocking=True)
    target_index = capture["target_index"].to(device=device, dtype=torch.long, non_blocking=True)
    source_logits = torch.randn((rows, 128), device=device, dtype=torch.float32)
    target_logits = torch.randn((rows, 128), device=device, dtype=torch.float32)

    atomic_out = matris_op.fused_line_attention_max_atomic(
        source_logits, target_logits, source_index, target_index, num_segments
    )
    scatter_out = scatter_reduce_pair(source_logits, target_logits, source_index, target_index, num_segments)
    source_atomic = matris_op.fused_line_attention_single_max_atomic(source_logits, source_index, num_segments)
    target_atomic = matris_op.fused_line_attention_single_max_atomic(target_logits, target_index, num_segments)
    target_lengths = torch.bincount(target_index, minlength=num_segments)
    target_segment = segment_reduce_sorted(target_logits, target_lengths)
    torch.cuda.synchronize()
    def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        diff = torch.nan_to_num((a - b).abs(), nan=0.0, posinf=0.0, neginf=0.0)
        return float(diff.max().item())

    source_diff = max_diff(atomic_out[0], scatter_out[0])
    target_diff = max_diff(atomic_out[1], scatter_out[1])
    single_source_diff = max_diff(source_atomic, atomic_out[0])
    single_target_diff = max_diff(target_atomic, atomic_out[1])
    target_segment_diff = max_diff(target_segment, target_atomic)

    atomic_ms = time_cuda(
        lambda: matris_op.fused_line_attention_max_atomic(
            source_logits, target_logits, source_index, target_index, num_segments
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    scatter_ms = time_cuda(
        lambda: scatter_reduce_pair(source_logits, target_logits, source_index, target_index, num_segments),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    source_atomic_ms = time_cuda(
        lambda: matris_op.fused_line_attention_single_max_atomic(source_logits, source_index, num_segments),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    target_atomic_ms = time_cuda(
        lambda: matris_op.fused_line_attention_single_max_atomic(target_logits, target_index, num_segments),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    source_scatter_ms = time_cuda(
        lambda: scatter_reduce_single(source_logits, source_index, num_segments),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    target_scatter_ms = time_cuda(
        lambda: scatter_reduce_single(target_logits, target_index, num_segments),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    target_segment_ms = time_cuda(
        lambda: segment_reduce_sorted(target_logits, target_lengths),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    stats = summarize_capture(capture["source_index"], capture["target_index"], num_segments)
    return {
        **stats,
        "atomic_ms": atomic_ms,
        "scatter_reduce_ms": scatter_ms,
        "scatter_speedup_vs_atomic": atomic_ms / scatter_ms if scatter_ms > 0 else None,
        "source_atomic_ms": source_atomic_ms,
        "target_atomic_ms": target_atomic_ms,
        "source_scatter_ms": source_scatter_ms,
        "target_scatter_ms": target_scatter_ms,
        "target_segment_ms": target_segment_ms,
        "target_segment_speedup_vs_target_atomic": target_atomic_ms / target_segment_ms if target_segment_ms > 0 else None,
        "max_abs_diff": max(source_diff, target_diff, single_source_diff, single_target_diff, target_segment_diff),
    }


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("P82B benchmark requires CUDA.")
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "fused_line_attention_max_atomic"):
        raise RuntimeError("matris_op.fused_line_attention_max_atomic is unavailable; rebuild matris_op first.")

    captures = run_capture(args)
    summaries = [
        summarize_capture(item["source_index"], item["target_index"], int(item["num_segments"]))
        for item in captures
    ]
    cases = choose_cases(captures, args.max_bench_cases)
    bench = [benchmark_case(matris_op, item, args) for item in tqdm(cases, desc="bench")]

    grouped_source = sum(1 for item in summaries if item["source"]["is_grouped"])
    grouped_target = sum(1 for item in summaries if item["target"]["is_grouped"])
    monotonic_source = sum(1 for item in summaries if item["source"]["is_monotonic"])
    monotonic_target = sum(1 for item in summaries if item["target"]["is_monotonic"])
    speedups = [item["scatter_speedup_vs_atomic"] for item in bench if item["scatter_speedup_vs_atomic"] is not None]
    target_segment_speedups = [
        item["target_segment_speedup_vs_target_atomic"]
        for item in bench
        if item["target_segment_speedup_vs_target_atomic"] is not None
    ]
    payload = {
        "metadata": vars(args),
        "num_captures": len(captures),
        "num_bench_cases": len(bench),
        "index_order_summary": {
            "source_grouped_count": grouped_source,
            "target_grouped_count": grouped_target,
            "source_monotonic_count": monotonic_source,
            "target_monotonic_count": monotonic_target,
        },
        "rows_summary": {
            "min": min((item["rows"] for item in summaries), default=0),
            "max": max((item["rows"] for item in summaries), default=0),
            "mean": statistics.mean([item["rows"] for item in summaries]) if summaries else 0.0,
        },
        "scatter_speedup_summary": {
            "min": min(speedups) if speedups else None,
            "max": max(speedups) if speedups else None,
            "mean": statistics.mean(speedups) if speedups else None,
            "median": statistics.median(speedups) if speedups else None,
        },
        "target_segment_speedup_summary": {
            "min": min(target_segment_speedups) if target_segment_speedups else None,
            "max": max(target_segment_speedups) if target_segment_speedups else None,
            "mean": statistics.mean(target_segment_speedups) if target_segment_speedups else None,
            "median": statistics.median(target_segment_speedups) if target_segment_speedups else None,
        },
        "captures": summaries,
        "bench": bench,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["index_order_summary"], indent=2))
    print(json.dumps(payload["rows_summary"], indent=2))
    print(json.dumps(payload["scatter_speedup_summary"], indent=2))
    print(json.dumps(payload["target_segment_speedup_summary"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
