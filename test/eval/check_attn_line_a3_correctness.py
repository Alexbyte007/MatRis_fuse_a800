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
    "p99b": {"MATRIS_P99B_ATTN_LINE_MODULE_VJP": "1"},
    "p101": {"MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1"},
    "p101_no_node": {
        "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
        "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
    },
    "p105": {
        "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
        "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
        "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
    },
    "p106": {
        "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
        "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
        "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
        "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
    },
    "p107": {
        "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
        "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
        "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
        "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
        "MATRIS_P107_A_CUDA2_ATTN_ALPHA_PROJECT_FUSED": "1",
    },
    "p108": {
        "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
        "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
        "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
        "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
        "MATRIS_P108_A_CUDA3_ATTN_LINE_ALPHA_TILED_BWD": "1",
    },
    "p108_target_reduce": {
        "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
        "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
        "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
        "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
        "MATRIS_P108_A_CUDA3_ATTN_LINE_TARGET_REDUCE_BWD": "1",
    },
    "p108_dense_gemm_scatter": {
        "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
        "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
        "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
        "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
        "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_BWD": "1",
        "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_THRESHOLD": "4096",
    },
    "p108_dense_gemm_op": {
        "MATRIS_P101_A3_LITE_ATTN_LINE_VJP": "1",
        "MATRIS_P101_USE_NODE_INPUT_ATTENTION": "0",
        "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD": "1",
        "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT": "1",
        "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_OP_BWD": "1",
        "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_THRESHOLD": "4096",
    },
}

KNOWN_A3_ENV_KEYS = (
    "MATRIS_P99B_ATTN_LINE_MODULE_VJP",
    "MATRIS_P101_A3_LITE_ATTN_LINE_VJP",
    "MATRIS_P101_USE_CUDA_GATHER_CAT",
    "MATRIS_P101_USE_NODE_INPUT_ATTENTION",
    "MATRIS_P105_A_CUDA1_ATTN_LINE_EDGE_ALPHA_BWD",
    "MATRIS_P106_A_CUDA2_ATTN_LINE_BWD_EDGE_DIRECT",
    "MATRIS_P107_A_CUDA2_ATTN_ALPHA_PROJECT_FUSED",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_ALPHA_TILED_BWD",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_TARGET_REDUCE_BWD",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_BWD",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_OP_BWD",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SPLIT_EDGE",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_INPLACE_HIDDEN",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_GROUPED_ALPHA",
    "MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_SCATTER_THRESHOLD",
)

PURE_FUSE_ENV = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A3 attn_line correctness harness: compare one reference attn_line block "
            "against a candidate macro on forward outputs and input gradients."
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
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument(
        "--candidate",
        default="p99b",
        help="Built-in candidate name, or a label used with --candidate-env.",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional candidate env flag. Can be repeated for new A3 CUDA/input-grad macros.",
    )
    parser.add_argument("--grad-seed", type=int, default=123)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    parser.add_argument("--output-json", default="results/attn_line_a3_correctness/p99b_block0.json")
    parser.add_argument("--output-md", default="results/attn_line_a3_correctness/p99b_block0.md")
    parser.add_argument(
        "--no-set-pure-fuse-env",
        action="store_true",
        help="Do not set the current pure-fuse env defaults inside this process.",
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


def env_without_candidate() -> dict[str, None]:
    return {key: None for key in KNOWN_A3_ENV_KEYS}


def parse_candidate_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(DEFAULT_CANDIDATE_ENVS.get(args.candidate, {}))
    for item in args.candidate_env:
        if "=" not in item:
            raise ValueError(f"--candidate-env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--candidate-env has empty key: {item}")
        env[key] = value
    return env


def tensor_diff(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | bool | list[int]]:
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


def deterministic_grad_like(tensor: torch.Tensor, seed: int, label_offset: int) -> torch.Tensor:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed) + int(label_offset))
    return torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)


def prepare_attn_line_inputs(
    calculator,
    graph,
    block_index: int,
    task: str,
    disabled_candidate_env: dict[str, None],
):
    batch_graph = process_graphs([graph], compute_stress="s" in task)
    if len(batch_graph["line_graph_dict"]["line_graph"]) == 0:
        return None

    node_feat = calculator.model.atom_embedding(batch_graph["atomic_numbers"] - 1)
    edge_feat, smooth_weight = calculator.model.edge_embedding(graphs=batch_graph)
    threebody_feat = calculator.model.three_body_embedding(graphs=batch_graph)

    with patched_env(disabled_candidate_env):
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

    layer = calculator.model.interaction_block[block_index].attn_block_line_graph
    layer.profile_prefix = f"interaction_block.{block_index}.attn_line"
    return {
        "batch_graph": batch_graph,
        "graph": batch_graph["line_graph_dict"],
        "layer": layer,
        "node_feat": edge_feat.detach(),
        "edge_feat": threebody_feat.detach(),
    }


def run_layer(layer, graph: dict[str, Any], node_input: torch.Tensor, edge_input: torch.Tensor):
    return layer(node_input, edge_input, graph, None)


def check_one(calculator, graph, graph_id: int, args: argparse.Namespace) -> dict[str, Any] | None:
    candidate_env = parse_candidate_env(args)
    disabled_candidate_env = {
        **env_without_candidate(),
        **{key: None for key in candidate_env},
    }
    prepared = prepare_attn_line_inputs(
        calculator,
        graph,
        args.block_index,
        args.task,
        disabled_candidate_env,
    )
    if prepared is None:
        return {
            "graph_id": int(graph_id),
            "skipped": True,
            "reason": "empty_line_graph",
        }

    layer = prepared["layer"]
    line_graph = prepared["graph"]
    node_base = prepared["node_feat"]
    edge_base = prepared["edge_feat"]

    node_ref = node_base.detach().clone().requires_grad_(True)
    edge_ref = edge_base.detach().clone().requires_grad_(True)
    with patched_env(disabled_candidate_env):
        ref_node_out, ref_edge_out = run_layer(layer, line_graph, node_ref, edge_ref)

    node_candidate = node_base.detach().clone().requires_grad_(True)
    edge_candidate = edge_base.detach().clone().requires_grad_(True)
    with patched_env({**disabled_candidate_env, **candidate_env}):
        candidate_node_out, candidate_edge_out = run_layer(
            layer,
            line_graph,
            node_candidate,
            edge_candidate,
        )

    grad_node_out = deterministic_grad_like(ref_node_out, args.grad_seed, 0)
    grad_edge_out = deterministic_grad_like(ref_edge_out, args.grad_seed, 1)
    ref_grad_node, ref_grad_edge = torch.autograd.grad(
        (ref_node_out, ref_edge_out),
        (node_ref, edge_ref),
        grad_outputs=(grad_node_out, grad_edge_out),
        retain_graph=False,
        allow_unused=False,
    )
    cand_grad_node, cand_grad_edge = torch.autograd.grad(
        (candidate_node_out, candidate_edge_out),
        (node_candidate, edge_candidate),
        grad_outputs=(grad_node_out, grad_edge_out),
        retain_graph=False,
        allow_unused=False,
    )

    comparisons = {
        "node_out": tensor_diff(ref_node_out, candidate_node_out, atol=args.atol, rtol=args.rtol),
        "edge_out": tensor_diff(ref_edge_out, candidate_edge_out, atol=args.atol, rtol=args.rtol),
        "grad_node": tensor_diff(ref_grad_node, cand_grad_node, atol=args.atol, rtol=args.rtol),
        "grad_edge": tensor_diff(ref_grad_edge, cand_grad_edge, atol=args.atol, rtol=args.rtol),
    }
    passed = all(bool(item["allclose"]) for item in comparisons.values())
    return {
        "graph_id": int(graph_id),
        "skipped": False,
        "passed": passed,
        "block_index": int(args.block_index),
        "rows": {
            "line_edges": int(line_graph["target_index"].numel()),
            "line_nodes": int(node_base.shape[0]),
        },
        "comparisons": comparisons,
    }


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    checked = [record for record in records if not record.get("skipped")]
    passed = [record for record in checked if record.get("passed")]
    worst: dict[str, dict[str, float]] = {}
    for name in ("node_out", "edge_out", "grad_node", "grad_edge"):
        items = [record["comparisons"][name] for record in checked]
        worst[name] = {
            "max_abs": max((float(item["max_abs"]) for item in items), default=0.0),
            "mean_abs": max((float(item["mean_abs"]) for item in items), default=0.0),
            "max_rel": max((float(item["max_rel"]) for item in items), default=0.0),
        }
    return {
        "candidate": args.candidate,
        "block_index": int(args.block_index),
        "limit": int(args.limit),
        "sample_seed": int(args.sample_seed),
        "grad_seed": int(args.grad_seed),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "checked": len(checked),
        "skipped": len(records) - len(checked),
        "passed": len(passed),
        "failed": len(checked) - len(passed),
        "all_passed": len(checked) > 0 and len(passed) == len(checked),
        "worst": worst,
    }


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = payload["summary"]
    lines = [
        "# attn_line A3 Correctness",
        "",
        f"- candidate: `{summary['candidate']}`",
        f"- block_index: `{summary['block_index']}`",
        f"- checked/skipped: `{summary['checked']}/{summary['skipped']}`",
        f"- passed/failed: `{summary['passed']}/{summary['failed']}`",
        f"- atol/rtol: `{summary['atol']}` / `{summary['rtol']}`",
        "",
        "| tensor | max_abs | max_mean_abs | max_rel |",
        "|---|---:|---:|---:|",
    ]
    for name, item in summary["worst"].items():
        lines.append(
            f"| `{name}` | `{item['max_abs']:.12g}` | `{item['mean_abs']:.12g}` | `{item['max_rel']:.12g}` |"
        )
    lines.append("")
    lines.append("Result: `" + ("PASS" if summary["all_passed"] else "FAIL") + "`")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    records: list[dict[str, Any]] = []
    for graph_id in tqdm(keys, desc="attn_line_a3_correctness", leave=False):
        atom = structures.get_atoms(int(graph_id))
        calculator._adjust_pbc(atom)
        structure = AseAtomsAdaptor.get_structure(atom)
        graph_cpu = calculator.model.graph_converter(structure)
        if isinstance(graph_cpu, list):
            graph = graph_cpu[0].to(args.device)
        else:
            graph = graph_cpu.to(args.device)
        records.append(check_one(calculator, graph, int(graph_id), args))

    payload = {
        "summary": summarize(records, args),
        "records": records,
        "env": {
            "pure_fuse": {key: os.environ.get(key) for key in PURE_FUSE_ENV},
            "candidate": parse_candidate_env(args),
        },
    }
    write_outputs(payload, args)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if payload["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
