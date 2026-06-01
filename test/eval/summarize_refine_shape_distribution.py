from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_salex_lmdb_quant import build_calculator, select_group_aligned_keys  # noqa: E402
from matris.graph import RadiusGraph  # noqa: E402
from matris.model.processgraph import process_graphs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize real MatRIS refine graph shape distributions.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-stress", action="store_true")
    parser.add_argument("--output-json", default="results/refine_shape_distribution_salex_limit5000_seed42_20260526.json")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--quant-mode", default="none")
    parser.add_argument("--fusion-mode", default="none")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    return parser.parse_args()


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def threshold_counts(values: list[int], thresholds: list[int]) -> dict[str, Any]:
    total = len(values)
    arr = np.asarray(values, dtype=np.int64) if values else np.asarray([], dtype=np.int64)
    out = {}
    for threshold in thresholds:
        count = int((arr >= threshold).sum()) if total else 0
        out[f">={threshold}"] = {
            "count": count,
            "pct": float(count / total * 100.0) if total else 0.0,
        }
    return out


def target_degree_summary(index: torch.Tensor, rows: int) -> dict[str, Any]:
    if index.numel() == 0 or rows <= 0:
        return {"mean": 0.0, "max": 0}
    counts = torch.bincount(index.detach().cpu(), minlength=rows).to(torch.float64)
    nonzero = counts[counts > 0]
    return {
        "mean_all": float(counts.mean().item()),
        "mean_nonzero": float(nonzero.mean().item()) if nonzero.numel() else 0.0,
        "p90_nonzero": float(torch.quantile(nonzero, 0.90).item()) if nonzero.numel() else 0.0,
        "max": int(counts.max().item()) if counts.numel() else 0,
        "nonzero_segments": int(nonzero.numel()),
    }


def graph_to_device(graph, device: str):
    for name in (
        "atomic_number",
        "atom_frac_coord",
        "lattice",
        "neighbor_image",
        "atom_graph",
        "line_graph",
        "undirected2directed",
        "directed2undirected",
    ):
        value = getattr(graph, name, None)
        if isinstance(value, torch.Tensor):
            setattr(graph, name, value.to(device))
    return graph


def main() -> None:
    args = parse_args()
    os.environ["MATRIS_P83B_LINE_ATTENTION_TARGET_OFFSETS"] = "1"
    structures = AseDBDataset(config=dict(src=args.dataset_src))
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    calculator = build_calculator(args)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for graph_id in tqdm(keys, total=len(keys)):
        try:
            atoms = structures.get_atoms(int(graph_id))
            structure = AseAtomsAdaptor.get_structure(atoms)
            graph_cpu = calculator.model.graph_converter(structure)
            graphs = [graph_cpu] if isinstance(graph_cpu, RadiusGraph) else graph_cpu
            device_graphs = [graph_to_device(graph, args.device) for graph in graphs]
            batched = process_graphs(device_graphs, compute_stress=bool(args.compute_stress))
            atom_graph = batched["atom_graph_dict"]
            line_graph = batched["line_graph_dict"]
            n_atoms = int(batched["atomic_numbers"].shape[0])
            atom_direct_edges = int(atom_graph["target_index"].numel())
            atom_undirected_edges = int(batched["undirected2directed"].numel())
            line_rows = int(line_graph["target_index"].numel()) if line_graph.get("target_index") is not None else 0
            line_nodes = int(line_graph.get("num_segment", 0)) if line_rows else atom_undirected_edges
            atom_degree = target_degree_summary(atom_graph["target_index"], n_atoms)
            line_degree = target_degree_summary(line_graph["target_index"], line_nodes) if line_rows else {"mean_all": 0.0, "max": 0}
            records.append(
                {
                    "graph_id": int(graph_id),
                    "n_atoms": n_atoms,
                    "atom_refine_rows": atom_direct_edges,
                    "atom_node_rows": n_atoms,
                    "atom_undirected_edges": atom_undirected_edges,
                    "line_refine_rows": line_rows,
                    "line_node_rows": line_nodes,
                    "line_rows_per_node": float(line_rows / line_nodes) if line_nodes else 0.0,
                    "atom_rows_per_node": float(atom_direct_edges / n_atoms) if n_atoms else 0.0,
                    "atom_target_degree": atom_degree,
                    "line_target_degree": line_degree,
                }
            )
        except Exception as exc:
            failures.append({"graph_id": int(graph_id), "error": str(exc)})

    thresholds = [512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576]
    fields = [
        "n_atoms",
        "atom_refine_rows",
        "atom_node_rows",
        "atom_undirected_edges",
        "line_refine_rows",
        "line_node_rows",
        "line_rows_per_node",
        "atom_rows_per_node",
    ]
    summary = {
        field: numeric_summary([float(row[field]) for row in records])
        for field in fields
    }
    summary["thresholds"] = {
        "line_refine_rows": threshold_counts([int(row["line_refine_rows"]) for row in records], thresholds),
        "atom_refine_rows": threshold_counts([int(row["atom_refine_rows"]) for row in records], thresholds),
    }
    nonempty_line = [row for row in records if int(row["line_refine_rows"]) > 0]
    summary["num_records"] = len(records)
    summary["num_failures"] = len(failures)
    summary["num_nonempty_line_graph"] = len(nonempty_line)
    summary["nonempty_line_pct"] = float(len(nonempty_line) / len(records) * 100.0) if records else 0.0
    summary["top_line_rows"] = sorted(
        (
            {
                "graph_id": row["graph_id"],
                "line_refine_rows": row["line_refine_rows"],
                "line_node_rows": row["line_node_rows"],
                "line_rows_per_node": row["line_rows_per_node"],
                "atom_refine_rows": row["atom_refine_rows"],
                "n_atoms": row["n_atoms"],
            }
            for row in records
        ),
        key=lambda item: item["line_refine_rows"],
        reverse=True,
    )[:30]
    summary["top_atom_rows"] = sorted(
        (
            {
                "graph_id": row["graph_id"],
                "atom_refine_rows": row["atom_refine_rows"],
                "atom_node_rows": row["atom_node_rows"],
                "atom_rows_per_node": row["atom_rows_per_node"],
                "line_refine_rows": row["line_refine_rows"],
            }
            for row in records
        ),
        key=lambda item: item["atom_refine_rows"],
        reverse=True,
    )[:30]

    payload = {
        "metadata": {
            "dataset_src": args.dataset_src,
            "dataset_size": len(structures),
            "limit": args.limit,
            "sample_seed": args.sample_seed,
            "device": args.device,
            "compute_stress": bool(args.compute_stress),
            "thresholds": thresholds,
        },
        "summary": summary,
        "records": records,
        "failures": failures,
    }
    output = REPO_ROOT / args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "failures": failures[:5]}, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
