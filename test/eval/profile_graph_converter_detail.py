import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset
from pymatgen.io.ase import AseAtomsAdaptor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.graph.converter import GraphConverter
from matris.graph.converter import line_graph_adjacency_list_fast
from matris.graph.radiusgraph import RadiusGraph


DETAIL_KEYS = [
    "atoms_to_structure_ms",
    "metadata_tensors_ms",
    "neighbor_list_ms",
    "create_graph_ms",
    "adjacency_list_ms",
    "undirected2directed_ms",
    "line_graph_ms",
    "isolated_check_ms",
    "tensor_conversion_ms",
    "total_converter_detail_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split GraphConverter CPU time into neighbor/list/line-graph/tensor stages."
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-selection",
        default="teacher-aligned",
        choices=("teacher-aligned", "sorted-random"),
    )
    parser.add_argument("--atom-graph-cutoff", type=float, default=6.0)
    parser.add_argument("--line-graph-cutoff", type=float, default=4.0)
    parser.add_argument("--neighbor-backend", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000.0


def select_teacher_aligned_indices(dataset_len: int, limit: int, seed: int) -> list[int]:
    indices = list(range(dataset_len))
    state = random.getstate()
    random.seed(seed)
    random.shuffle(indices)
    random.setstate(state)
    return indices[:limit] if limit > 0 else indices


def select_sorted_random_indices(dataset_len: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= dataset_len:
        return list(range(dataset_len))
    rng = random.Random(seed)
    indices = rng.sample(range(dataset_len), limit)
    indices.sort()
    return indices


def select_indices(dataset_len: int, limit: int, seed: int, sample_selection: str) -> list[int]:
    if sample_selection == "teacher-aligned":
        return select_teacher_aligned_indices(dataset_len, limit, seed)
    if sample_selection == "sorted-random":
        return select_sorted_random_indices(dataset_len, limit, seed)
    raise ValueError(f"unsupported sample_selection={sample_selection!r}")


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = round((len(sorted_values) - 1) * q)
    return sorted_values[int(idx)]


def numeric_summary(values: list[float]) -> dict:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    ordered = sorted(float(v) for v in values)
    return {
        "mean": statistics.mean(ordered),
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "median": statistics.median(ordered),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def profile_converter_detail(
    converter: GraphConverter,
    structure,
    *,
    graph_id: int,
    mp_id=None,
) -> tuple[RadiusGraph, dict]:
    record = {key: 0.0 for key in DETAIL_KEYS}
    n_atoms = len(structure)

    def make_metadata():
        atomic_number = torch.tensor([site.specie.Z for site in structure], dtype=torch.int32)
        atom_frac_coord = torch.tensor(structure.frac_coords, dtype=torch.float32)
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
        return atomic_number, atom_frac_coord, lattice

    (atomic_number, atom_frac_coord, lattice), record["metadata_tensors_ms"] = timed(make_metadata)

    (center_index, neighbor_index, image, distance), record["neighbor_list_ms"] = timed(
        lambda: converter._get_neighbor_list(structure)
    )

    graph, record["create_graph_ms"] = timed(
        lambda: converter.create_graph(n_atoms, center_index, neighbor_index, image, distance)
    )

    (atom_graph, directed2undirected), record["adjacency_list_ms"] = timed(graph.adjacency_list)
    undirected2directed, record["undirected2directed_ms"] = timed(graph.undirected2directed)

    use_fast_line_graph = (
        line_graph_adjacency_list_fast is not None
        and os.environ.get("MATRIS_USE_FAST_LINE_GRAPH", "0") == "1"
    )
    disable_gc_for_line_graph = (
        os.environ.get("MATRIS_DISABLE_GC_DURING_LINE_GRAPH", "1") != "0"
    )
    gc_was_enabled = gc.isenabled()
    if disable_gc_for_line_graph and gc_was_enabled:
        gc.disable()
    try:
        try:
            if use_fast_line_graph:
                line_graph, record["line_graph_ms"] = timed(
                    lambda: line_graph_adjacency_list_fast(
                        graph.nodes,
                        graph.undirected_edges_list,
                        converter.line_graph_cutoff,
                    )
                )
            else:
                line_graph, record["line_graph_ms"] = timed(
                    lambda: graph.line_graph_adjacency_list(
                        cutoff=converter.line_graph_cutoff
                    )
                )
        except Exception:
            structure.to(filename="error_graph.cif")
            raise
    finally:
        if disable_gc_for_line_graph and gc_was_enabled:
            gc.enable()

    def check_isolated():
        n_isolated_atoms = len({*range(n_atoms)} - {*center_index})
        if n_isolated_atoms:
            raise ValueError(
                f"Error: Detected {n_isolated_atoms} isolated atom. Calculation stopped"
            )

    _, record["isolated_check_ms"] = timed(check_isolated)

    def make_radius_graph():
        return RadiusGraph(
            atomic_number=atomic_number,
            atom_frac_coord=atom_frac_coord,
            atom_graph=torch.tensor(atom_graph, dtype=torch.int32),
            neighbor_image=torch.tensor(image, dtype=torch.float32),
            directed2undirected=torch.tensor(directed2undirected, dtype=torch.int32),
            undirected2directed=torch.tensor(undirected2directed, dtype=torch.int32),
            line_graph=torch.tensor(line_graph, dtype=torch.int32),
            lattice=lattice,
            graph_id=graph_id,
            mp_id=mp_id,
            composition=structure.composition.formula,
            atom_graph_cutoff=converter.atom_graph_cutoff,
            line_graph_cutoff=converter.line_graph_cutoff,
        )

    radius_graph, record["tensor_conversion_ms"] = timed(make_radius_graph)

    record["total_converter_detail_ms"] = sum(
        record[key]
        for key in DETAIL_KEYS
        if key not in ("atoms_to_structure_ms", "total_converter_detail_ms")
    )

    neighbor_count = int(len(center_index))
    degree = np.bincount(np.asarray(center_index, dtype=np.int64), minlength=n_atoms)
    record.update(
        {
            "n_neighbor_pairs": neighbor_count,
            "n_atom_graph_edges": int(len(atom_graph)),
            "n_line_graph_edges": int(len(line_graph)),
            "degree_mean": float(degree.mean()) if degree.size else 0.0,
            "degree_max": int(degree.max()) if degree.size else 0,
            "cell_volume": float(structure.lattice.volume),
            "cell_a": float(structure.lattice.a),
            "cell_b": float(structure.lattice.b),
            "cell_c": float(structure.lattice.c),
            "cell_alpha": float(structure.lattice.alpha),
            "cell_beta": float(structure.lattice.beta),
            "cell_gamma": float(structure.lattice.gamma),
        }
    )
    return radius_graph, record


def summarize(records: list[dict], top_k: int) -> dict:
    summary = {
        "num_success": len(records),
        "stages": {key: numeric_summary([float(r[key]) for r in records]) for key in DETAIL_KEYS},
        "sizes": {
            key: numeric_summary([float(r[key]) for r in records])
            for key in (
                "n_atoms",
                "n_neighbor_pairs",
                "n_atom_graph_edges",
                "n_line_graph_edges",
                "degree_mean",
                "degree_max",
                "cell_volume",
            )
        },
    }
    means = {key: summary["stages"][key]["mean"] for key in DETAIL_KEYS}
    total = means["total_converter_detail_ms"]
    summary["stage_ranking"] = [
        {
            "name": key,
            "mean_ms": value,
            "pct_of_converter": value / total * 100.0 if total else 0.0,
        }
        for key, value in sorted(means.items(), key=lambda item: item[1], reverse=True)
        if key != "total_converter_detail_ms"
    ]
    summary["top_total_converter"] = sorted(
        records, key=lambda r: float(r["total_converter_detail_ms"]), reverse=True
    )[:top_k]
    summary["top_neighbor_list"] = sorted(
        records, key=lambda r: float(r["neighbor_list_ms"]), reverse=True
    )[:top_k]
    summary["top_line_graph"] = sorted(
        records, key=lambda r: float(r["line_graph_ms"]), reverse=True
    )[:top_k]
    return summary


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    dataset = AseDBDataset(config={"src": args.dataset_src})
    indices = select_indices(len(dataset), args.limit, args.sample_seed, args.sample_selection)
    converter = GraphConverter(
        atom_graph_cutoff=args.atom_graph_cutoff,
        line_graph_cutoff=args.line_graph_cutoff,
        neighbor_backend=args.neighbor_backend,
    )

    records = []
    failures = []
    for ordinal, sample_index in enumerate(indices, start=1):
        try:
            atoms = dataset.get_atoms(int(sample_index))
            structure, atoms_to_structure_ms = timed(lambda: AseAtomsAdaptor.get_structure(atoms))
            _, record = profile_converter_detail(
                converter,
                structure,
                graph_id=int(sample_index),
            )
            record["atoms_to_structure_ms"] = atoms_to_structure_ms
            record.update(
                {
                    "sample_index": int(sample_index),
                    "graph_id": int(sample_index),
                    "formula": atoms.get_chemical_formula(),
                    "n_atoms": int(len(atoms)),
                }
            )
            records.append(record)
            print(
                f"[{ordinal}/{len(indices)}] graph_id={sample_index} "
                f"atoms={len(atoms)} total={record['total_converter_detail_ms']:.3f} ms "
                f"neighbor={record['neighbor_list_ms']:.3f} ms "
                f"line={record['line_graph_ms']:.3f} ms "
                f"edges={record['n_atom_graph_edges']} line_edges={record['n_line_graph_edges']}"
            )
        except Exception as exc:
            failures.append({"sample_index": int(sample_index), "error": repr(exc)})
            print(f"[{ordinal}/{len(indices)}] graph_id={sample_index} failed: {exc}")

    payload = {
        "metadata": {
            "dataset_src": str(Path(args.dataset_src).resolve()),
            "dataset_size": len(dataset),
            "limit": args.limit,
            "sample_seed": args.sample_seed,
            "sample_selection": args.sample_selection,
            "atom_graph_cutoff": args.atom_graph_cutoff,
            "line_graph_cutoff": args.line_graph_cutoff,
            "neighbor_backend": converter.neighbor_backend,
            "converter_algorithm": converter.algorithm,
            "use_fast_line_graph": (
                line_graph_adjacency_list_fast is not None
                and os.environ.get("MATRIS_USE_FAST_LINE_GRAPH", "0") == "1"
            ),
            "disable_gc_during_line_graph": (
                os.environ.get("MATRIS_DISABLE_GC_DURING_LINE_GRAPH", "1") != "0"
            ),
        },
        "summary": summarize(records, args.top_k),
        "records": records,
        "failures": failures,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"saved {output_json}")


if __name__ == "__main__":
    main()
