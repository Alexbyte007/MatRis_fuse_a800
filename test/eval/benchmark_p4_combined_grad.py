import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from fairchem.core.datasets import AseDBDataset
from pymatgen.io.ase import AseAtomsAdaptor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from matris.model.model import MatRIS
from matris.model.processgraph import process_graphs
from profile_salex_pipeline import run_embedding_only, run_interaction_only, run_readout_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare separate vs combined force/stress autograd.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--output-dir", default="results/p4_combined_grad_benchmark")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=5)
    return parser.parse_args()


def select_indices(dataset_len: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= dataset_len:
        return list(range(dataset_len))
    rng = random.Random(seed)
    indices = rng.sample(range(dataset_len), limit)
    indices.sort()
    return indices


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed(device: str, fn):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def make_energy(dataset: AseDBDataset, model: MatRIS, sample_index: int, device: str):
    item = dataset[sample_index]
    atoms = dataset.get_atoms(sample_index)
    structure = AseAtomsAdaptor.get_structure(atoms)
    graph = model.graph_converter(structure).to(device)
    batch_graph = process_graphs([graph], compute_stress=True)
    node_feat, edge_feat, threebody_feat, smooth_weight = run_embedding_only(model, batch_graph)
    node_feat, edge_feat, threebody_feat = run_interaction_only(
        model, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight
    )
    total_energy = run_readout_only(model, batch_graph, node_feat)
    metadata = {
        "sample_index": sample_index,
        "sid": item["sid"] if "sid" in item else "",
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "num_directed_edges": int(batch_graph["atom_graph_dict"]["atom_graph"].shape[0]),
        "num_line_graph_angles": int(batch_graph["line_graph_dict"]["line_graph"].shape[0]),
    }
    return batch_graph, total_energy, metadata


def profile_sample(dataset: AseDBDataset, model: MatRIS, sample_index: int, device: str) -> dict:
    batch_graph, total_energy, metadata = make_energy(dataset, model, sample_index, device)
    _, force_ms = timed(
        device,
        lambda: torch.autograd.grad(
            total_energy.sum(),
            [batch_graph["batch_cart_coords"]],
            create_graph=False,
            retain_graph=True,
        )[0],
    )
    _, stress_ms = timed(
        device,
        lambda: torch.autograd.grad(
            total_energy.sum(),
            [batch_graph["batch_strains"]],
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0],
    )

    batch_graph, total_energy, _ = make_energy(dataset, model, sample_index, device)
    _, combined_ms = timed(
        device,
        lambda: torch.autograd.grad(
            total_energy.sum(),
            [batch_graph["batch_cart_coords"], batch_graph["batch_strains"]],
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        ),
    )
    return {
        **metadata,
        "force_separate_ms": force_ms,
        "stress_separate_ms": stress_ms,
        "separate_total_ms": force_ms + stress_ms,
        "combined_force_stress_ms": combined_ms,
        "combined_vs_separate_speedup": (force_ms + stress_ms) / combined_ms if combined_ms else 0.0,
    }


def numeric(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def summarize(records: list[dict]) -> dict:
    keys = [
        "force_separate_ms",
        "stress_separate_ms",
        "separate_total_ms",
        "combined_force_stress_ms",
        "combined_vs_separate_speedup",
    ]
    return {key: numeric([float(record[key]) for record in records]) for key in keys}


def write_markdown(output_dir: Path, payload: dict) -> None:
    s = payload["summary"]
    lines = [
        "# P4 Combined Force/Stress Grad Benchmark",
        "",
        f"- device: `{payload['device']}`",
        f"- samples: `{payload['sample_indices']}`",
        "",
        "| metric | mean ms | std | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in (
        "force_separate_ms",
        "stress_separate_ms",
        "separate_total_ms",
        "combined_force_stress_ms",
        "combined_vs_separate_speedup",
    ):
        unit = "" if key.endswith("speedup") else " ms"
        lines.append(
            f"| {key} | {s[key]['mean']:.6f}{unit} | {s[key]['std']:.6f} | "
            f"{s[key]['min']:.6f} | {s[key]['max']:.6f} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AseDBDataset(config={"src": args.dataset_src})
    indices = select_indices(len(dataset), args.limit, args.sample_seed)
    model = MatRIS.load(model_name=args.model, device=args.device)
    model.eval()

    for idx in indices[: min(args.warmup_steps, len(indices))]:
        batch_graph, total_energy, _ = make_energy(dataset, model, idx, args.device)
        _ = torch.autograd.grad(
            total_energy.sum(),
            [batch_graph["batch_cart_coords"], batch_graph["batch_strains"]],
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        sync(args.device)

    records = []
    for pos, sample_index in enumerate(indices, start=1):
        record = profile_sample(dataset, model, sample_index, args.device)
        records.append(record)
        print(
            f"[{pos}/{len(indices)}] sample_index={sample_index} "
            f"separate={record['separate_total_ms']:.3f} ms "
            f"combined={record['combined_force_stress_ms']:.3f} ms"
        )

    payload = {
        "phase": "P4",
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "model": args.model,
        "sample_indices": indices,
        "warmup_steps": args.warmup_steps,
        "records": records,
        "summary": summarize(records),
    }
    with output_dir.joinpath("summary.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    write_markdown(output_dir, payload)
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
