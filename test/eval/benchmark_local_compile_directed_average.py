from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
OP_SRC = REPO_ROOT / "matris" / "model" / "op" / "src"
if str(OP_SRC) not in sys.path:
    sys.path.insert(0, str(OP_SRC))

from infer_salex_lmdb_quant import (  # noqa: E402
    autocast_context,
    build_calculator,
    configure_precision,
    select_group_aligned_keys,
)
import matris.model.functions as functions_mod  # noqa: E402
import matris.model.interaction_block as interaction_block  # noqa: E402


P108_LOCAL_DIRECTED_ENV = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
    "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "0",
    "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
    "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "0",
    "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "0",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_OP_BWD": "0",
}


@dataclass
class DirectedAverageCase:
    case_id: int
    rows: int
    num_segments: int
    data: torch.Tensor
    segment: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local torch.compile experiment for directed2undirected average: "
            "PyTorch index_add baseline vs local compile vs MatRIS CUDA op."
        )
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--max-rows", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--env-preset", default="p108_local_directed", choices=["p108_local_directed", "none"])
    parser.add_argument("--output-json", default="results/compile_local_directed_average_limit6_cases60_20260526.json")
    parser.add_argument("--print-case-details", action="store_true")
    return parser.parse_args()


def apply_env_preset(name: str) -> dict[str, str]:
    if name == "none":
        return {}
    if name == "p108_local_directed":
        for key, value in P108_LOCAL_DIRECTED_ENV.items():
            os.environ[key] = value
        return dict(P108_LOCAL_DIRECTED_ENV)
    raise ValueError(f"Unknown env preset: {name}")


def load_matris_op():
    import matris_op  # type: ignore

    return matris_op


def cuda_time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    mean = sum(samples) / len(samples)
    std = math.sqrt(sum((x - mean) ** 2 for x in samples) / max(1, len(samples) - 1))
    return {"mean_ms": mean, "std_ms": std, "min_ms": min(samples), "max_ms": max(samples)}


def rel_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).detach().abs()
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def torch_directed_average_template(
    data: torch.Tensor,
    segment: torch.Tensor,
    out_template: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(out_template)
    return out.index_add(0, segment, data) * 0.5


def torch_directed_average_eval_template(
    data: torch.Tensor,
    segment: torch.Tensor,
    out_template: torch.Tensor,
) -> torch.Tensor:
    return torch_directed_average_template(data, segment, out_template)


def torch_directed_average_grad_template(
    data: torch.Tensor,
    segment: torch.Tensor,
    out_template: torch.Tensor,
) -> torch.Tensor:
    return torch_directed_average_template(data, segment, out_template)


def autograd_grad(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    data_base: torch.Tensor,
    segment: torch.Tensor,
    out_template: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    data = data_base.detach().requires_grad_(True)
    out = fn(data, segment, out_template)
    (grad_data,) = torch.autograd.grad(out, (data,), grad_outputs=(grad_out,), allow_unused=False)
    return grad_data


class DirectedAverageCollector:
    def __init__(self, *, max_cases: int, max_rows: int) -> None:
        self.max_cases = max_cases
        self.max_rows = max_rows
        self.cases: list[DirectedAverageCase] = []
        self.skipped: dict[str, int] = {}
        self._original_functions = functions_mod.directed2undirected_average_or_none
        self._original_interaction = interaction_block.directed2undirected_average_or_none

    def __enter__(self):
        def wrapped(data, segment, num_segment, enable_hint):
            result = self._original_functions(data, segment, num_segment, enable_hint)
            self._maybe_record(result, data, segment, num_segment)
            return result

        functions_mod.directed2undirected_average_or_none = wrapped
        interaction_block.directed2undirected_average_or_none = wrapped
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        functions_mod.directed2undirected_average_or_none = self._original_functions
        interaction_block.directed2undirected_average_or_none = self._original_interaction

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def _maybe_record(self, result, data: torch.Tensor, segment: torch.Tensor, num_segment) -> None:
        if result is None:
            self._skip("cuda_op_returned_none")
            return
        if len(self.cases) >= self.max_cases:
            self._skip("max_cases_reached")
            return
        if data.ndim != 2 or data.shape[1] != 128 or segment.ndim != 1:
            self._skip("unsupported_shape")
            return
        rows = int(data.shape[0])
        if rows <= 0 or rows > self.max_rows:
            self._skip("rows_filtered")
            return
        resolved_num_segments = int(num_segment) if num_segment is not None else int(segment.max().item()) + 1
        if rows != resolved_num_segments * 2:
            self._skip("not_pair_directed")
            return
        self.cases.append(
            DirectedAverageCase(
                case_id=len(self.cases),
                rows=rows,
                num_segments=resolved_num_segments,
                data=data.detach().float().cpu().contiguous(),
                segment=segment.detach().cpu().long().contiguous(),
            )
        )


def make_infer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_src=args.dataset_src,
        model=args.model,
        model_path=args.model_path,
        task=args.task,
        device=args.device,
        precision_mode=args.precision_mode,
        quant_mode=args.quant_mode,
        fusion_mode=args.fusion_mode,
        torch_compile=False,
        torch_compile_mode="default",
        torch_compile_fullgraph=False,
        timing_warmup_samples=0,
        batch_size=1,
        prefetch_graphs=False,
        activation_calibration_limit=0,
        activation_calibration_seed=43,
    )


def collect_cases(args: argparse.Namespace) -> tuple[list[DirectedAverageCase], dict[str, Any]]:
    structures = AseDBDataset(config={"src": args.dataset_src})
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    calculator = build_calculator(make_infer_args(args))
    errors: list[str] = []
    with DirectedAverageCollector(max_cases=args.max_cases, max_rows=args.max_rows) as collector:
        for idx in tqdm(range(len(keys)), desc="collect directed average cases", leave=False):
            if len(collector.cases) >= args.max_cases:
                break
            graph_id = int(keys[idx])
            try:
                atom = structures.get_atoms(graph_id)
                atom.calc = calculator
                torch.cuda.synchronize()
                with autocast_context(args.device, args.precision_mode):
                    _ = atom.get_potential_energy()
                    if args.task in ("ef", "efs", "efsm"):
                        _ = atom.get_forces()
                    if args.task in ("efs", "efsm"):
                        _ = atom.get_stress()
                torch.cuda.synchronize()
            except Exception as exc:
                errors.append(f"idx={idx}, graph_id={graph_id}: {exc}")
                continue
    return collector.cases, {
        "dataset_len": len(structures),
        "selected_keys": [int(x) for x in keys[: args.limit]],
        "collection_errors": errors[:20],
        "num_collection_errors": len(errors),
        "skipped": collector.skipped,
    }


def case_to_device(case: DirectedAverageCase, device: torch.device) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "rows": case.rows,
        "num_segments": case.num_segments,
        "data": case.data.to(device),
        "segment": case.segment.to(device),
        "out_template": torch.empty((case.num_segments, 128), device=device, dtype=torch.float32),
    }


def benchmark_case(
    matris_op,
    compiled_eval: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    compiled_grad: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    case: DirectedAverageCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    item = case_to_device(case, device)
    data = item["data"]
    segment = item["segment"]
    out_template = item["out_template"]
    rows = int(item["rows"])
    num_segments = int(item["num_segments"])
    grad_out = torch.randn((num_segments, 128), device=device)

    eager_f = torch_directed_average_template(data, segment, out_template)
    compile_f = compiled_eval(data, segment, out_template)
    cuda_f = matris_op.directed2undirected_average_forward(data.contiguous(), segment.contiguous(), num_segments)
    eager_b = autograd_grad(torch_directed_average_template, data, segment, out_template, grad_out)
    compile_b = autograd_grad(compiled_grad, data, segment, out_template, grad_out)
    cuda_b = matris_op.directed2undirected_average_backward(grad_out.contiguous(), segment.contiguous(), rows)
    torch.cuda.synchronize()

    forward_timings = {
        "torch_eager": cuda_time_ms(
            lambda: torch_directed_average_template(data, segment, out_template),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "torch_compile": cuda_time_ms(
            lambda: compiled_eval(data, segment, out_template),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "cuda_fused": cuda_time_ms(
            lambda: matris_op.directed2undirected_average_forward(data.contiguous(), segment.contiguous(), num_segments),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }
    fwd_bwd_timings = {
        "torch_eager": cuda_time_ms(
            lambda: autograd_grad(torch_directed_average_template, data, segment, out_template, grad_out),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "torch_compile": cuda_time_ms(
            lambda: autograd_grad(compiled_grad, data, segment, out_template, grad_out),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "cuda_fused": cuda_time_ms(
            lambda: (
                matris_op.directed2undirected_average_forward(data.contiguous(), segment.contiguous(), num_segments),
                matris_op.directed2undirected_average_backward(grad_out.contiguous(), segment.contiguous(), rows),
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }
    return {
        "case_id": case.case_id,
        "rows": rows,
        "num_segments": num_segments,
        "forward_time": forward_timings,
        "forward_backward_time": fwd_bwd_timings,
        "speedup_vs_eager": {
            "forward_torch_compile": forward_timings["torch_eager"]["mean_ms"] / forward_timings["torch_compile"]["mean_ms"],
            "forward_cuda_fused": forward_timings["torch_eager"]["mean_ms"] / forward_timings["cuda_fused"]["mean_ms"],
            "forward_backward_torch_compile": fwd_bwd_timings["torch_eager"]["mean_ms"]
            / fwd_bwd_timings["torch_compile"]["mean_ms"],
            "forward_backward_cuda_fused": fwd_bwd_timings["torch_eager"]["mean_ms"]
            / fwd_bwd_timings["cuda_fused"]["mean_ms"],
        },
        "errors_vs_torch_eager": {
            "torch_compile_forward": rel_error(eager_f, compile_f),
            "cuda_fused_forward": rel_error(eager_f, cuda_f),
            "torch_compile_grad_data": rel_error(eager_b, compile_b),
            "cuda_fused_grad_data": rel_error(eager_b, cuda_b),
        },
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}

    def weighted_mean(section: str, mode: str) -> float:
        total_rows = sum(float(item["rows"]) for item in results)
        return sum(float(item["rows"]) * float(item[section][mode]["mean_ms"]) for item in results) / total_rows

    summary: dict[str, Any] = {
        "num_cases": len(results),
        "rows": {
            "min": min(int(item["rows"]) for item in results),
            "max": max(int(item["rows"]) for item in results),
            "mean": float(np.mean([int(item["rows"]) for item in results])),
        },
    }
    for section in ("forward_time", "forward_backward_time"):
        total_ms = {
            mode: sum(float(item[section][mode]["mean_ms"]) for item in results)
            for mode in ("torch_eager", "torch_compile", "cuda_fused")
        }
        summary[section] = {
            "weighted_mean_ms": {
                mode: weighted_mean(section, mode)
                for mode in ("torch_eager", "torch_compile", "cuda_fused")
            },
            "local_all_cases_total_ms": total_ms,
        }
        eager = summary[section]["weighted_mean_ms"]["torch_eager"]
        summary[section]["weighted_speedup_vs_eager"] = {
            mode: eager / summary[section]["weighted_mean_ms"][mode]
            for mode in ("torch_compile", "cuda_fused")
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    precision_info = configure_precision(args.device, args.precision_mode)
    env_applied = apply_env_preset(args.env_preset)
    matris_op = load_matris_op()

    compiled_eval = torch.compile(torch_directed_average_eval_template, dynamic=True, mode=args.torch_compile_mode)
    compiled_grad = torch.compile(torch_directed_average_grad_template, dynamic=True, mode=args.torch_compile_mode)

    collect_start = time.perf_counter()
    cases, collection_meta = collect_cases(args)
    collect_s = time.perf_counter() - collect_start
    if not cases:
        raise SystemExit(f"No directed average cases were collected. Collection metadata: {collection_meta}")

    results = []
    for case in tqdm(cases, desc="benchmark directed average cases"):
        result = benchmark_case(matris_op, compiled_eval, compiled_grad, case, args)
        results.append(result)
        if args.print_case_details:
            print(json.dumps(result, indent=2))

    payload = {
        "metadata": {
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "device": torch.cuda.get_device_name(0),
            "precision": precision_info,
            "torch_version": torch.__version__,
            "torch_compile": {
                "enabled": True,
                "dynamic": True,
                "mode": args.torch_compile_mode,
                "scope": "local directed2undirected average only",
                "logs_hint": 'TORCH_LOGS="graph_breaks,recompiles"',
            },
            "env_preset": args.env_preset,
            "env_applied": env_applied,
            "warmup": args.warmup,
            "iters": args.iters,
            "limit": args.limit,
            "max_cases": args.max_cases,
            "max_rows": args.max_rows,
            "collection_seconds": collect_s,
            "collection": collection_meta,
            "baseline_mapping": {
                "torch_eager": "zeros_like(out_template).index_add(0, directed2undirected, data) * 0.5",
                "torch_compile": "torch.compile(torch_eager_directed_average, dynamic=True), local function only",
                "cuda_fused": "matris_op.directed2undirected_average_forward/backward",
            },
        },
        "summary": summarize_results(results),
        "cases": results,
    }
    output = REPO_ROOT / args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
