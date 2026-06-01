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
import matris.model.interaction_block as interaction_block  # noqa: E402


P108_LOCAL_ATTENTION_ENV = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
    # Keep lower-level fused attention visible to this local experiment.
    # The current endpoint best enables this line VJP, which can bypass
    # fused_line_attention_or_none and leave no line-attention cases to record.
    "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "0",
    "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
    "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "0",
    "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "0",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_OP_BWD": "0",
}


@dataclass
class AttentionCase:
    case_id: int
    kind: str
    rows: int
    num_segments: int
    source_logits: torch.Tensor
    target_logits: torch.Tensor
    values: torch.Tensor
    source_index: torch.Tensor
    target_index: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local torch.compile experiment for the PyTorch baseline corresponding "
            "to MatRIS fused_line_attention CUDA op. This does not compile the full model."
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
    parser.add_argument("--limit", type=int, default=8, help="Number of real structures used for case collection.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=24, help="Maximum real attention cases to benchmark.")
    parser.add_argument("--case-kind", choices=["all", "line", "atom"], default="all")
    parser.add_argument("--max-rows", type=int, default=65536, help="Skip very large local cases above this row count.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--env-preset", default="p108_local_attention", choices=["p108_local_attention", "none"])
    parser.add_argument("--output-json", default="results/compile_local_attention_smoke_20260526.json")
    parser.add_argument("--print-case-details", action="store_true")
    return parser.parse_args()


def apply_env_preset(name: str) -> dict[str, str]:
    if name == "none":
        return {}
    if name == "p108_local_attention":
        for key, value in P108_LOCAL_ATTENTION_ENV.items():
            os.environ[key] = value
        return dict(P108_LOCAL_ATTENTION_ENV)
    raise ValueError(f"Unknown env preset: {name}")


def load_matris_op():
    import matris_op  # type: ignore

    return matris_op


def rel_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).detach().abs()
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
    }


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
    return {
        "mean_ms": mean,
        "std_ms": std,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def torch_attention_side_template(
    logits: torch.Tensor,
    values: torch.Tensor,
    index: torch.Tensor,
    out_template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = index.reshape(-1, 1).expand(-1, logits.shape[1])
    max_out = torch.empty_like(out_template)
    max_out.fill_(-torch.inf)
    max_out.scatter_reduce_(0, expanded, logits, reduce="amax", include_self=True)
    exp_logits = torch.exp(logits - max_out.index_select(0, index))
    sums = torch.zeros_like(out_template)
    sums.index_add_(0, index, exp_logits)
    alpha = exp_logits / sums.index_select(0, index).clamp_min(1.0e-20)
    out = torch.zeros_like(out_template)
    out.index_add_(0, index, alpha * values)
    return out, alpha


def torch_attention_forward_template(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    values: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    out_template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_out, source_alpha = torch_attention_side_template(source_logits, values, source_index, out_template)
    target_out, target_alpha = torch_attention_side_template(target_logits, values, target_index, out_template)
    return source_out, target_out, source_alpha, target_alpha


def torch_attention_forward_eval_template(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    values: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    out_template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch_attention_forward_template(
        source_logits,
        target_logits,
        values,
        source_index,
        target_index,
        out_template,
    )


def torch_attention_forward_grad_template(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    values: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    out_template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch_attention_forward_template(
        source_logits,
        target_logits,
        values,
        source_index,
        target_index,
        out_template,
    )


def torch_attention_backward_manual(
    grad_source_out: torch.Tensor,
    grad_target_out: torch.Tensor,
    values: torch.Tensor,
    source_out: torch.Tensor,
    target_out: torch.Tensor,
    source_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gs = grad_source_out.index_select(0, source_index)
    gt = grad_target_out.index_select(0, target_index)
    os_ = source_out.index_select(0, source_index)
    ot = target_out.index_select(0, target_index)
    grad_source_logits = source_alpha * gs * (values - os_)
    grad_target_logits = target_alpha * gt * (values - ot)
    grad_values = source_alpha * gs + target_alpha * gt
    return grad_source_logits, grad_target_logits, grad_values


def forward_autograd_grads(
    forward_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    source_logits_base: torch.Tensor,
    target_logits_base: torch.Tensor,
    values_base: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    out_template: torch.Tensor,
    grad_source_out: torch.Tensor,
    grad_target_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_logits = source_logits_base.detach().requires_grad_(True)
    target_logits = target_logits_base.detach().requires_grad_(True)
    values = values_base.detach().requires_grad_(True)
    source_out, target_out, _, _ = forward_fn(
        source_logits,
        target_logits,
        values,
        source_index,
        target_index,
        out_template,
    )
    grads = torch.autograd.grad(
        (source_out, target_out),
        (source_logits, target_logits, values),
        grad_outputs=(grad_source_out, grad_target_out),
        allow_unused=False,
    )
    return grads


class RealAttentionCaseCollector:
    def __init__(self, *, max_cases: int, case_kind: str, max_rows: int) -> None:
        self.max_cases = max_cases
        self.case_kind = case_kind
        self.max_rows = max_rows
        self.cases: list[AttentionCase] = []
        self.skipped: dict[str, int] = {}
        self._original = interaction_block.fused_line_attention_or_none

    def __enter__(self):
        def wrapped(
            source_logits,
            target_logits,
            values,
            source_index,
            target_index,
            num_segments,
            *,
            enable_hint,
            atom_graph=False,
            target_offsets=None,
        ):
            result = self._original(
                source_logits,
                target_logits,
                values,
                source_index,
                target_index,
                num_segments,
                enable_hint=enable_hint,
                atom_graph=atom_graph,
                target_offsets=target_offsets,
            )
            self._maybe_record(
                result,
                source_logits,
                target_logits,
                values,
                source_index,
                target_index,
                int(num_segments),
                bool(atom_graph),
            )
            return result

        interaction_block.fused_line_attention_or_none = wrapped
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        interaction_block.fused_line_attention_or_none = self._original

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def _maybe_record(
        self,
        result,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        values: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        num_segments: int,
        atom_graph: bool,
    ) -> None:
        if result is None:
            self._skip("fused_op_returned_none")
            return
        if len(self.cases) >= self.max_cases:
            self._skip("max_cases_reached")
            return
        kind = "atom" if atom_graph else "line"
        if self.case_kind != "all" and self.case_kind != kind:
            self._skip(f"kind_filtered_{kind}")
            return
        if source_logits.ndim != 2 or source_logits.shape[1] != 128:
            self._skip("unsupported_shape")
            return
        rows = int(source_logits.shape[0])
        if rows <= 0 or rows > self.max_rows:
            self._skip("rows_filtered")
            return
        case_id = len(self.cases)
        self.cases.append(
            AttentionCase(
                case_id=case_id,
                kind=kind,
                rows=rows,
                num_segments=num_segments,
                source_logits=source_logits.detach().float().cpu().contiguous(),
                target_logits=target_logits.detach().float().cpu().contiguous(),
                values=values.detach().float().cpu().contiguous(),
                source_index=source_index.detach().cpu().long().contiguous(),
                target_index=target_index.detach().cpu().long().contiguous(),
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


def collect_real_attention_cases(args: argparse.Namespace) -> tuple[list[AttentionCase], dict[str, Any]]:
    structures = AseDBDataset(config={"src": args.dataset_src})
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    infer_args = make_infer_args(args)
    calculator = build_calculator(infer_args)
    errors: list[str] = []

    with RealAttentionCaseCollector(
        max_cases=args.max_cases,
        case_kind=args.case_kind,
        max_rows=args.max_rows,
    ) as collector:
        for idx in tqdm(range(len(keys)), desc="collect real attention cases", leave=False):
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
            except Exception as exc:  # Keep collection robust across occasional bad structures.
                errors.append(f"idx={idx}, graph_id={graph_id}: {exc}")
                continue

    metadata = {
        "dataset_len": len(structures),
        "selected_keys": [int(x) for x in keys[: args.limit]],
        "collection_errors": errors[:20],
        "num_collection_errors": len(errors),
        "skipped": collector.skipped,
    }
    return collector.cases, metadata


def tensor_case_to_device(case: AttentionCase, device: torch.device) -> dict[str, torch.Tensor | int | str]:
    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "rows": case.rows,
        "num_segments": case.num_segments,
        "source_logits": case.source_logits.to(device, non_blocking=False),
        "target_logits": case.target_logits.to(device, non_blocking=False),
        "values": case.values.to(device, non_blocking=False),
        "source_index": case.source_index.to(device, non_blocking=False),
        "target_index": case.target_index.to(device, non_blocking=False),
        "out_template": torch.empty((case.num_segments, case.source_logits.shape[1]), device=device, dtype=torch.float32),
    }


def benchmark_case(
    matris_op,
    compiled_forward_eval: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    compiled_forward_grad: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    case: AttentionCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    item = tensor_case_to_device(case, device)
    source_logits = item["source_logits"]
    target_logits = item["target_logits"]
    values = item["values"]
    source_index = item["source_index"]
    target_index = item["target_index"]
    out_template = item["out_template"]
    num_segments = int(item["num_segments"])
    grad_source_out = torch.randn((num_segments, 128), device=device)
    grad_target_out = torch.randn((num_segments, 128), device=device)

    eager_f = torch_attention_forward_template(
        source_logits, target_logits, values, source_index, target_index, out_template
    )
    compile_f = compiled_forward_eval(source_logits, target_logits, values, source_index, target_index, out_template)
    cuda_f = matris_op.fused_line_attention_forward(
        source_logits.contiguous(),
        target_logits.contiguous(),
        values.contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
        num_segments,
    )
    eager_b = torch_attention_backward_manual(
        grad_source_out,
        grad_target_out,
        values,
        eager_f[0],
        eager_f[1],
        eager_f[2],
        eager_f[3],
        source_index,
        target_index,
    )
    compile_b = forward_autograd_grads(
        compiled_forward_grad,
        source_logits,
        target_logits,
        values,
        source_index,
        target_index,
        out_template,
        grad_source_out,
        grad_target_out,
    )
    cuda_b = matris_op.fused_line_attention_backward(
        grad_source_out.contiguous(),
        grad_target_out.contiguous(),
        values.contiguous(),
        cuda_f[0].contiguous(),
        cuda_f[1].contiguous(),
        cuda_f[2].contiguous(),
        cuda_f[3].contiguous(),
        source_index.contiguous(),
        target_index.contiguous(),
    )
    torch.cuda.synchronize()

    forward_timings = {
        "torch_eager": cuda_time_ms(
            lambda: torch_attention_forward_template(
                source_logits, target_logits, values, source_index, target_index, out_template
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "torch_compile": cuda_time_ms(
            lambda: compiled_forward_eval(source_logits, target_logits, values, source_index, target_index, out_template),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "cuda_fused": cuda_time_ms(
            lambda: matris_op.fused_line_attention_forward(
                source_logits.contiguous(),
                target_logits.contiguous(),
                values.contiguous(),
                source_index.contiguous(),
                target_index.contiguous(),
                num_segments,
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }
    fwd_bwd_timings = {
        "torch_eager": cuda_time_ms(
            lambda: forward_autograd_grads(
                torch_attention_forward_template,
                source_logits,
                target_logits,
                values,
                source_index,
                target_index,
                out_template,
                grad_source_out,
                grad_target_out,
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "torch_compile": cuda_time_ms(
            lambda: forward_autograd_grads(
                compiled_forward_grad,
                source_logits,
                target_logits,
                values,
                source_index,
                target_index,
                out_template,
                grad_source_out,
                grad_target_out,
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "cuda_fused": cuda_time_ms(
            lambda: (
                lambda f: matris_op.fused_line_attention_backward(
                    grad_source_out.contiguous(),
                    grad_target_out.contiguous(),
                    values.contiguous(),
                    f[0].contiguous(),
                    f[1].contiguous(),
                    f[2].contiguous(),
                    f[3].contiguous(),
                    source_index.contiguous(),
                    target_index.contiguous(),
                )
            )(
                matris_op.fused_line_attention_forward(
                    source_logits.contiguous(),
                    target_logits.contiguous(),
                    values.contiguous(),
                    source_index.contiguous(),
                    target_index.contiguous(),
                    num_segments,
                )
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }

    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "rows": case.rows,
        "num_segments": case.num_segments,
        "forward_time": forward_timings,
        "forward_backward_time": fwd_bwd_timings,
        "speedup_vs_eager": {
            "forward_torch_compile": forward_timings["torch_eager"]["mean_ms"]
            / forward_timings["torch_compile"]["mean_ms"],
            "forward_cuda_fused": forward_timings["torch_eager"]["mean_ms"]
            / forward_timings["cuda_fused"]["mean_ms"],
            "forward_backward_torch_compile": fwd_bwd_timings["torch_eager"]["mean_ms"]
            / fwd_bwd_timings["torch_compile"]["mean_ms"],
            "forward_backward_cuda_fused": fwd_bwd_timings["torch_eager"]["mean_ms"]
            / fwd_bwd_timings["cuda_fused"]["mean_ms"],
        },
        "errors_vs_torch_eager": {
            "torch_compile_source_out": rel_error(eager_f[0], compile_f[0]),
            "torch_compile_target_out": rel_error(eager_f[1], compile_f[1]),
            "cuda_fused_source_out": rel_error(eager_f[0], cuda_f[0]),
            "cuda_fused_target_out": rel_error(eager_f[1], cuda_f[1]),
            "torch_compile_grad_source_logits": rel_error(eager_b[0], compile_b[0]),
            "torch_compile_grad_target_logits": rel_error(eager_b[1], compile_b[1]),
            "torch_compile_grad_values": rel_error(eager_b[2], compile_b[2]),
            "cuda_fused_grad_source_logits": rel_error(eager_b[0], cuda_b[0]),
            "cuda_fused_grad_target_logits": rel_error(eager_b[1], cuda_b[1]),
            "cuda_fused_grad_values": rel_error(eager_b[2], cuda_b[2]),
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
        "by_kind": {},
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
    for kind in sorted({str(item["kind"]) for item in results}):
        subset = [item for item in results if item["kind"] == kind]
        summary["by_kind"][kind] = {
            "count": len(subset),
            "rows_mean": float(np.mean([int(item["rows"]) for item in subset])),
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

    compiled_forward_eval = torch.compile(
        torch_attention_forward_eval_template,
        dynamic=True,
        mode=args.torch_compile_mode,
    )
    compiled_forward_grad = torch.compile(
        torch_attention_forward_grad_template,
        dynamic=True,
        mode=args.torch_compile_mode,
    )

    collect_start = time.perf_counter()
    cases, collection_meta = collect_real_attention_cases(args)
    collect_s = time.perf_counter() - collect_start
    if not cases:
        raise SystemExit(f"No fused attention cases were collected. Collection metadata: {collection_meta}")

    results = []
    for case in tqdm(cases, desc="benchmark local attention cases"):
        result = benchmark_case(matris_op, compiled_forward_eval, compiled_forward_grad, case, args)
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
                "scope": "local fused_line_attention PyTorch baseline only; no torch.compile(model)",
                "logs_hint": 'TORCH_LOGS="graph_breaks,recompiles"',
            },
            "env_preset": args.env_preset,
            "env_applied": env_applied,
            "warmup": args.warmup,
            "iters": args.iters,
            "limit": args.limit,
            "max_cases": args.max_cases,
            "case_kind": args.case_kind,
            "max_rows": args.max_rows,
            "collection_seconds": collect_s,
            "collection": collection_meta,
            "baseline_mapping": {
                "cuda_fused": "matris_op.fused_line_attention_forward/backward",
                "torch_eager": (
                    "PyTorch equivalent of Dimwise_softmax + alpha*value + aggregate "
                    "for source and target attention sides"
                ),
                "torch_compile": "torch.compile(torch_eager_forward, dynamic=True), local function only",
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
