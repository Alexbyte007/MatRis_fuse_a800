from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from fairchem.core.datasets import AseDBDataset
from pymatgen.io.ase import AseAtomsAdaptor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from infer_salex_lmdb_quant import build_calculator, configure_precision, select_group_aligned_keys  # noqa: E402
from matris.model.processgraph import process_graphs  # noqa: E402
from profile_salex_pipeline import run_embedding_only  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feasibility check for one Interaction_Block input-grad-only manual VJP."
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--block-index", type=int, default=9)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--grad-seed", type=int, default=123)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--atol", type=float, default=5.0e-4)
    parser.add_argument("--rtol", type=float, default=5.0e-4)
    parser.add_argument(
        "--output-json",
        default="results/interaction_block_input_grad_vjp/block9.json",
    )
    return parser.parse_args()


@contextlib.contextmanager
def patched_env(updates: dict[str, str | None]):
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed_ms(device: str, fn):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def tensor_diff(a: torch.Tensor | None, b: torch.Tensor | None, atol: float, rtol: float) -> dict[str, Any]:
    if a is None or b is None:
        return {"both_none": a is None and b is None, "allclose": a is None and b is None}
    diff = (a.detach().float() - b.detach().float()).abs()
    denom = torch.maximum(a.detach().float().abs(), b.detach().float().abs()).clamp_min(1.0e-12)
    rel = diff / denom
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def clone_leaf(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    out = tensor.detach().clone()
    if torch.is_floating_point(out):
        out.requires_grad_(True)
    return out


def clone_smooth(smooth_weight: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: clone_leaf(value) for key, value in smooth_weight.items()}


def flatten_inputs(
    node_feat: torch.Tensor,
    edge_feat: torch.Tensor,
    threebody_feat: torch.Tensor | None,
    smooth_weight: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    tensors = [node_feat, edge_feat]
    if threebody_feat is not None:
        tensors.append(threebody_feat)
    tensors.extend([smooth_weight["atom graph"], smooth_weight["line graph"]])
    return tensors


def deterministic_grad_like(tensor: torch.Tensor | None, seed: int, offset: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed + offset)
    return torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)


def prepare_block_inputs(calculator, dataset, sample_index: int, block_index: int, device: str):
    atom = dataset.get_atoms(sample_index)
    structure = AseAtomsAdaptor.get_structure(atom)
    graph = calculator.model.graph_converter(structure).to(device)
    batch_graph = process_graphs([graph], compute_stress=True)
    with torch.no_grad():
        node_feat, edge_feat, threebody_feat, smooth_weight = run_embedding_only(calculator.model, batch_graph)
        for idx, block in enumerate(calculator.model.interaction_block):
            if idx >= block_index:
                break
            block.profile_prefix = f"interaction_block.{idx}"
            node_feat, edge_feat, threebody_feat = block(
                batch_graph=batch_graph,
                node_feat=node_feat,
                edge_feat=edge_feat,
                threebody_feat=threebody_feat,
                smooth_weight=smooth_weight,
            )
    return batch_graph, node_feat.detach(), edge_feat.detach(), (
        threebody_feat.detach() if isinstance(threebody_feat, torch.Tensor) else None
    ), {key: value.detach() for key, value in smooth_weight.items()}


def run_block_vjp(
    block,
    batch_graph,
    node_base: torch.Tensor,
    edge_base: torch.Tensor,
    threebody_base: torch.Tensor | None,
    smooth_base: dict[str, torch.Tensor],
    block_index: int,
    grad_seed: int,
    candidate: bool,
):
    node = clone_leaf(node_base)
    edge = clone_leaf(edge_base)
    threebody = clone_leaf(threebody_base)
    smooth = clone_smooth(smooth_base)
    block.profile_prefix = f"interaction_block.{block_index}"
    env = {
        "MATRIS_P53B_FULL_BLOCK_MANUAL_VJP": "1" if candidate else None,
        "MATRIS_P53B_FULL_BLOCK_INDEX": str(block_index) if candidate else None,
    }
    with patched_env(env):
        out_node, out_edge, out_threebody = block(
            batch_graph=batch_graph,
            node_feat=node,
            edge_feat=edge,
            threebody_feat=threebody,
            smooth_weight=smooth,
        )
    grad_node = deterministic_grad_like(out_node, grad_seed, 1)
    grad_edge = deterministic_grad_like(out_edge, grad_seed, 2)
    grad_threebody = deterministic_grad_like(out_threebody, grad_seed, 3)
    loss = (out_node * grad_node).sum() + (out_edge * grad_edge).sum()
    if out_threebody is not None and grad_threebody is not None:
        loss = loss + (out_threebody * grad_threebody).sum()
    inputs = flatten_inputs(node, edge, threebody, smooth)
    grads = torch.autograd.grad(loss, inputs, allow_unused=True)
    return {
        "outputs": {
            "node_out": out_node.detach(),
            "edge_out": out_edge.detach(),
            "threebody_out": out_threebody.detach() if out_threebody is not None else None,
        },
        "grads": dict(zip(
            [
                "grad_node",
                "grad_edge",
                *([] if threebody is None else ["grad_threebody"]),
                "grad_smooth_atom",
                "grad_smooth_line",
            ],
            [g.detach() if isinstance(g, torch.Tensor) else None for g in grads],
        )),
    }


def benchmark_mode(
    device: str,
    iters: int,
    fn,
) -> dict[str, float]:
    times = []
    for _ in range(iters):
        _, ms = timed_ms(device, fn)
        times.append(ms)
    mean = sum(times) / len(times) if times else 0.0
    return {
        "mean_ms": mean,
        "min_ms": min(times) if times else 0.0,
        "max_ms": max(times) if times else 0.0,
    }


def main() -> None:
    args = parse_args()
    configure_precision(args.device, args.precision_mode)
    dataset = AseDBDataset(config={"src": args.dataset_src})
    keys = select_group_aligned_keys(len(dataset), args.limit, args.sample_seed)
    calculator = build_calculator(args)
    model = calculator.model
    model.eval()

    records = []
    for sample_pos, sample_index in enumerate(keys):
        batch_graph, node, edge, threebody, smooth = prepare_block_inputs(
            calculator,
            dataset,
            int(sample_index),
            args.block_index,
            args.device,
        )
        block = model.interaction_block[args.block_index]
        ref = run_block_vjp(
            block,
            batch_graph,
            node,
            edge,
            threebody,
            smooth,
            args.block_index,
            args.grad_seed + sample_pos * 100,
            candidate=False,
        )
        cand = run_block_vjp(
            block,
            batch_graph,
            node,
            edge,
            threebody,
            smooth,
            args.block_index,
            args.grad_seed + sample_pos * 100,
            candidate=True,
        )
        diffs = {
            name: tensor_diff(ref["outputs"][name], cand["outputs"][name], args.atol, args.rtol)
            for name in ref["outputs"]
        }
        diffs.update(
            {
                name: tensor_diff(ref["grads"].get(name), cand["grads"].get(name), args.atol, args.rtol)
                for name in sorted(set(ref["grads"]) | set(cand["grads"]))
            }
        )
        passed = all(item.get("allclose", False) for item in diffs.values())
        records.append(
            {
                "sample_index": int(sample_index),
                "n_atoms": int(node.shape[0]),
                "edge_rows": int(edge.shape[0]),
                "threebody_rows": int(threebody.shape[0]) if threebody is not None else 0,
                "passed": passed,
                "diffs": diffs,
            }
        )
        print(
            f"[{sample_pos + 1}/{len(keys)}] sample_index={int(sample_index)} "
            f"block={args.block_index} passed={passed}"
        )

    bench_batch_graph, bench_node, bench_edge, bench_threebody, bench_smooth = prepare_block_inputs(
        calculator,
        dataset,
        int(keys[0]),
        args.block_index,
        args.device,
    )
    bench_block = model.interaction_block[args.block_index]
    for _ in range(args.warmup_iters):
        run_block_vjp(
            bench_block,
            bench_batch_graph,
            bench_node,
            bench_edge,
            bench_threebody,
            bench_smooth,
            args.block_index,
            args.grad_seed,
            candidate=False,
        )
        run_block_vjp(
            bench_block,
            bench_batch_graph,
            bench_node,
            bench_edge,
            bench_threebody,
            bench_smooth,
            args.block_index,
            args.grad_seed,
            candidate=True,
        )
    sync(args.device)
    ref_bench = benchmark_mode(
        args.device,
        args.bench_iters,
        lambda: run_block_vjp(
            bench_block,
            bench_batch_graph,
            bench_node,
            bench_edge,
            bench_threebody,
            bench_smooth,
            args.block_index,
            args.grad_seed,
            candidate=False,
        ),
    )
    cand_bench = benchmark_mode(
        args.device,
        args.bench_iters,
        lambda: run_block_vjp(
            bench_block,
            bench_batch_graph,
            bench_node,
            bench_edge,
            bench_threebody,
            bench_smooth,
            args.block_index,
            args.grad_seed,
            candidate=True,
        ),
    )
    payload = {
        "block_index": args.block_index,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "grad_seed": args.grad_seed,
        "all_passed": all(record["passed"] for record in records),
        "records": records,
        "benchmark": {
            "sample_index": int(keys[0]),
            "warmup_iters": args.warmup_iters,
            "bench_iters": args.bench_iters,
            "reference_autograd": ref_bench,
            "p53b_full_block_manual_vjp": cand_bench,
            "speedup": ref_bench["mean_ms"] / cand_bench["mean_ms"] if cand_bench["mean_ms"] else 0.0,
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(json.dumps(payload["benchmark"], ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
