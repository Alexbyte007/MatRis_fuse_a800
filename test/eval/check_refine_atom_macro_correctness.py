from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
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

from infer_salex_lmdb_quant import build_calculator, configure_precision, select_group_aligned_keys  # noqa: E402
from matris.model.processgraph import process_graphs  # noqa: E402


DEFAULT_CANDIDATE_ENVS = {
    "reference": {},
    "at_cuda1_gather_project": {
        "MATRIS_P110_REFINE_ATOM_EDGE_UPDATE": "1",
    },
    "at_cuda1_first_silu": {
        "MATRIS_P110_REFINE_ATOM_EDGE_UPDATE": "1",
        "MATRIS_P111_REFINE_ATOM_FUSED_FIRST": "1",
    },
    "at_cuda1b_p112": {
        "MATRIS_P110_REFINE_ATOM_EDGE_UPDATE": "1",
        "MATRIS_P111_REFINE_ATOM_FUSED_FIRST": "1",
        "MATRIS_P112_REFINE_ATOM_FUSED_FIRST_BWD": "1",
        "MATRIS_P112_REFINE_ATOM_FUSED_FIRST_BWD_MAX_ROWS": "2048",
    },
}

KNOWN_ENV_KEYS = (
    "MATRIS_P110_REFINE_ATOM_EDGE_UPDATE",
    "MATRIS_P111_REFINE_ATOM_FUSED_FIRST",
    "MATRIS_P112_REFINE_ATOM_FUSED_FIRST_BWD",
    "MATRIS_P112_REFINE_ATOM_FUSED_FIRST_BWD_MIN_ROWS",
    "MATRIS_P112_REFINE_ATOM_FUSED_FIRST_BWD_MAX_ROWS",
    "MATRIS_P102_REFINE_LINE_R1_FFN_PAIR_VJP",
    "MATRIS_P103_REFINE_LINE_R2_EDGE_UPDATE_EDGE_FFN_VJP",
    "MATRIS_P104_REFINE_LINE_R3_BLOCK_VJP",
    "MATRIS_P109_REFINE_LINE_R_CUDA3_FFN_RESIDUAL",
    "MATRIS_P58_REFINE_LINE_SMOOTH_REDUCE",
    "MATRIS_P58_REFINE_LINE_SMOOTH_REDUCE_SORTED",
    "MATRIS_P58_REFINE_LINE_EDGE_UPDATE",
    "MATRIS_P58_REFINE_LINE_PROJECT_SCATTER_BWD",
    "MATRIS_P60_REFINE_LINE_FUSED_FIRST",
    "MATRIS_P61_REFINE_LINE_FUSED_FIRST_ACTS",
    "MATRIS_P61_REFINE_LINE_FIRST_TAIL",
    "MATRIS_P63_REFINE_LINE_EDGE_SMOOTH_REDUCE",
    "MATRIS_P64_REFINE_LINE_FUSED_BACKWARD",
    "MATRIS_P65_REFINE_LINE_TILED_FUSED_BACKWARD",
    "MATRIS_P65B_REFINE_LINE_PACKED_TILED_BWD",
    "MATRIS_P69_REFINE_LINE_FIRST_TAIL_SMOOTH_REDUCE",
)

PURE_FUSE_ENV = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
}

P108_BEST_ENV = {
    "MATRIS_P83B_LINE_ATTENTION_TARGET_OFFSETS": "1",
    "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
    "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
    "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
    "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_OP_BWD": "1",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_THRESHOLD": "4096",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="refine_atom macro correctness harness")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--candidate", default="reference")
    parser.add_argument("--candidate-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--grad-seed", type=int, default=321)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    parser.add_argument("--output-json", default="results/refine_atom_macro_correctness/reference_block0.json")
    parser.add_argument("--output-md", default="results/refine_atom_macro_correctness/reference_block0.md")
    parser.add_argument("--no-set-pure-fuse-env", action="store_true")
    parser.add_argument(
        "--no-set-p108-best-env",
        action="store_true",
        help="Do not set the current P108 attn_line best env before checking refine_atom.",
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


def parse_candidate_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(DEFAULT_CANDIDATE_ENVS.get(args.candidate, {}))
    for item in args.candidate_env:
        if "=" not in item:
            raise ValueError(f"--candidate-env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key.strip()] = value
    return env


def tensor_diff(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    diff = (a.detach().float() - b.detach().float()).abs()
    denom = torch.maximum(a.detach().float().abs(), b.detach().float().abs()).clamp_min(1.0e-12)
    rel = diff / denom
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
        "ref_abs_mean": float(a.detach().float().abs().mean().item()) if a.numel() else 0.0,
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def deterministic_grad_like(tensor: torch.Tensor, seed: int, offset: int) -> torch.Tensor:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed) + int(offset))
    return torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)


def prepare_inputs(calculator, graph, block_index: int, task: str, disabled_env: dict[str, None]):
    batch_graph = process_graphs([graph], compute_stress="s" in task)
    if len(batch_graph["atom_graph_dict"]["atom_graph"]) == 0:
        return None

    node_feat = calculator.model.atom_embedding(batch_graph["atomic_numbers"] - 1)
    edge_feat, smooth_weight = calculator.model.edge_embedding(graphs=batch_graph)
    threebody_feat = calculator.model.three_body_embedding(graphs=batch_graph)

    with patched_env(disabled_env):
        for idx, block in enumerate(calculator.model.interaction_block):
            if idx >= block_index:
                break
            node_feat, edge_feat, threebody_feat = block(
                batch_graph=batch_graph,
                node_feat=node_feat,
                edge_feat=edge_feat,
                threebody_feat=threebody_feat,
                smooth_weight=smooth_weight,
            )
        block = calculator.model.interaction_block[block_index]
        attn_edge, attn_threebody = block.attn_block_line_graph(
            edge_feat,
            threebody_feat,
            batch_graph["line_graph_dict"],
            None,
        )
        attn_node, attn_edge = block.attn_block_atom_graph(
            node_feat,
            attn_edge,
            batch_graph["atom_graph_dict"],
            batch_graph["directed2undirected"],
        )
        if threebody_feat is not None:
            update_edge, _update_threebody = block.refine_block_line_graph(
                attn_edge,
                attn_threebody,
                smooth_weight["line graph"],
                batch_graph["line_graph_dict"],
                None,
                attn_node,
            )
        else:
            update_edge = attn_edge

    layer = calculator.model.interaction_block[block_index].refine_block_atom_graph
    layer.profile_prefix = f"interaction_block.{block_index}.refine_atom"
    return {
        "layer": layer,
        "graph": batch_graph["atom_graph_dict"],
        "directed2undirected": batch_graph["directed2undirected"],
        "node_feat": attn_node.detach(),
        "edge_feat": update_edge.detach(),
        "smooth": smooth_weight["atom graph"].detach(),
    }


def run_layer(layer, graph: dict[str, Any], directed2undirected, node, edge, smooth):
    return layer(node, edge, smooth, graph, directed2undirected, None)


def check_one(calculator, graph, graph_id: int, args: argparse.Namespace):
    candidate_env = parse_candidate_env(args)
    disabled_env = {key: None for key in (*KNOWN_ENV_KEYS, *candidate_env.keys())}
    prepared = prepare_inputs(calculator, graph, args.block_index, args.task, disabled_env)
    if prepared is None:
        return {"graph_id": int(graph_id), "skipped": True, "reason": "empty_atom_graph"}

    layer = prepared["layer"]
    atom_graph = prepared["graph"]
    directed2undirected = prepared["directed2undirected"]
    bases = [prepared["node_feat"], prepared["edge_feat"], prepared["smooth"]]

    ref_inputs = [base.detach().clone().requires_grad_(True) for base in bases]
    with patched_env(disabled_env):
        ref_node_out, ref_edge_out = run_layer(layer, atom_graph, directed2undirected, *ref_inputs)

    cand_inputs = [base.detach().clone().requires_grad_(True) for base in bases]
    with patched_env({**disabled_env, **candidate_env}):
        cand_node_out, cand_edge_out = run_layer(layer, atom_graph, directed2undirected, *cand_inputs)

    grad_node_out = deterministic_grad_like(ref_node_out, args.grad_seed, 0)
    grad_edge_out = deterministic_grad_like(ref_edge_out, args.grad_seed, 1)
    ref_grads = torch.autograd.grad(
        (ref_node_out, ref_edge_out),
        ref_inputs,
        grad_outputs=(grad_node_out, grad_edge_out),
        allow_unused=False,
    )
    cand_grads = torch.autograd.grad(
        (cand_node_out, cand_edge_out),
        cand_inputs,
        grad_outputs=(grad_node_out, grad_edge_out),
        allow_unused=False,
    )

    comparisons = {
        "node_out": tensor_diff(ref_node_out, cand_node_out, atol=args.atol, rtol=args.rtol),
        "edge_out": tensor_diff(ref_edge_out, cand_edge_out, atol=args.atol, rtol=args.rtol),
        "grad_node": tensor_diff(ref_grads[0], cand_grads[0], atol=args.atol, rtol=args.rtol),
        "grad_edge": tensor_diff(ref_grads[1], cand_grads[1], atol=args.atol, rtol=args.rtol),
        "grad_smooth": tensor_diff(ref_grads[2], cand_grads[2], atol=args.atol, rtol=args.rtol),
    }
    return {
        "graph_id": int(graph_id),
        "skipped": False,
        "passed": all(bool(item["allclose"]) for item in comparisons.values()),
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    if not args.no_set_pure_fuse_env:
        for key, value in PURE_FUSE_ENV.items():
            os.environ[key] = value
        os.environ.pop("MATRIS_W8A8_BACKEND", None)
        os.environ.pop("MATRIS_W8A8_DISABLE_FAST_WRAPPER", None)
    if not args.no_set_p108_best_env:
        os.environ.update(P108_BEST_ENV)

    configure_precision(args.device, args.precision_mode)
    calculator = build_calculator(args)
    calculator.model.eval()
    dataset = AseDBDataset(config=dict(src=args.dataset_src))
    keys = select_group_aligned_keys(len(dataset), args.limit, args.sample_seed)

    results = []
    for graph_id in tqdm(keys, desc="refine_atom_correctness", leave=False):
        atom = dataset.get_atoms(int(graph_id))
        calculator._adjust_pbc(atom)
        structure = AseAtomsAdaptor.get_structure(atom)
        graph_cpu = calculator.model.graph_converter(structure)
        if isinstance(graph_cpu, list):
            graph = graph_cpu[0].to(args.device)
        else:
            graph = graph_cpu.to(args.device)
        results.append(check_one(calculator, graph, int(graph_id), args))

    checked = [item for item in results if not item.get("skipped")]
    passed = [item for item in checked if item.get("passed")]
    worst: dict[str, dict[str, float]] = {}
    for key in ("node_out", "edge_out", "grad_node", "grad_edge", "grad_smooth"):
        values = [item["comparisons"][key] for item in checked]
        if values:
            worst[key] = {
                "max_abs": max(float(v["max_abs"]) for v in values),
                "mean_abs": max(float(v["mean_abs"]) for v in values),
                "max_rel": max(float(v["max_rel"]) for v in values),
            }

    summary = {
        "candidate": args.candidate,
        "block_index": args.block_index,
        "limit": args.limit,
        "checked": len(checked),
        "skipped": len(results) - len(checked),
        "passed": len(passed),
        "failed": len(checked) - len(passed),
        "all_passed": len(checked) == len(passed),
        "worst": worst,
        "results": results,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# refine_atom Macro Correctness",
        "",
        f"- candidate: `{summary['candidate']}`",
        f"- block_index: `{summary['block_index']}`",
        f"- checked/skipped: `{summary['checked']}/{summary['skipped']}`",
        f"- passed/failed: `{summary['passed']}/{summary['failed']}`",
        f"- P108 best env: `{not args.no_set_p108_best_env}`",
        "",
        "| tensor | max_abs | max_mean_abs | max_rel |",
        "|---|---:|---:|---:|",
    ]
    for name, item in summary["worst"].items():
        lines.append(
            f"| `{name}` | `{item['max_abs']:.12g}` | `{item['mean_abs']:.12g}` | `{item['max_rel']:.12g}` |"
        )
    lines.extend(["", "Result: `" + ("PASS" if summary["all_passed"] else "FAIL") + "`", ""])
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
