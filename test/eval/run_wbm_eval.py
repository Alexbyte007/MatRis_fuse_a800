import argparse
import contextlib
import csv
import gzip
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from pymatgen.core import Structure


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications.relax import StructOptimizer


SUPPORTED_PRECISIONS = {"fp32", "tf32", "bf16", "fp16"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MatRIS relaxation on a small WBM subset and save relaxed energy predictions."
    )
    parser.add_argument(
        "--wbm-init",
        default="/home/lht/lab/wbm/wbm_2022-10-19-wbm-init-structs.jsonl.gz",
        help="Path to WBM initial structures jsonl.gz file.",
    )
    parser.add_argument(
        "--wbm-summary",
        default="/home/lht/lab/wbm/2023-12-13-wbm-summary.csv.gz",
        help="Path to WBM summary csv.gz file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save run_config.json, per_structure_predictions.jsonl and summary_overall.json",
    )
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="efs", choices=("e", "ef", "efs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--optimizer", default="FIRE")
    parser.add_argument("--relax-fmax", type=float, default=0.05)
    parser.add_argument("--relax-steps", type=int, default=100)
    parser.add_argument("--ase-filter", default="FrechetCellFilter")
    parser.add_argument(
        "--no-relax-cell",
        action="store_true",
        help="Disable cell relaxation and only relax atomic positions.",
    )
    parser.add_argument(
        "--save-final-structure",
        action="store_true",
        help="Store the relaxed final_structure dict in the output jsonl.",
    )
    parser.add_argument(
        "--precision-mode",
        default="fp32",
        choices=sorted(SUPPORTED_PRECISIONS),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the loaded model with torch.compile when available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of WBM samples to run for the smoke pass.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260424,
        help="Random seed for proportional_stability sampling.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.0,
        help="Energy-above-hull threshold in eV/atom for stable classification when sampling.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Number of initial structures to use as warmup before measured runs.",
    )
    return parser.parse_args()


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def configure_precision(device: str, precision_mode: str) -> dict:
    tf32_enabled = precision_mode == "tf32"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled

    autocast_dtype = None
    if precision_mode == "bf16":
        autocast_dtype = torch.bfloat16
    elif precision_mode == "fp16":
        autocast_dtype = torch.float16

    return {
        "tf32_enabled": tf32_enabled,
        "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
    }


def autocast_context(device: str, precision_mode: str):
    if device != "cuda":
        return contextlib.nullcontext()
    if precision_mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision_mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def build_relaxer(args: argparse.Namespace) -> StructOptimizer:
    relaxer = StructOptimizer(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        device=args.device,
    )
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in the current PyTorch.")
        relaxer.calculator.model = torch.compile(relaxer.calculator.model)
    return relaxer


def iter_wbm_init_records(path: str) -> list[dict]:
    records: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        for line in fp:
            records.append(json.loads(line))
    return records


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_proportional_material_ids(
    summary_path: str,
    limit: int,
    stability_threshold: float,
    sample_seed: int,
) -> list[str]:
    stable_ids: list[str] = []
    unstable_ids: list[str] = []

    with gzip.open(summary_path, "rt", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if not parse_bool(row.get("unique_prototype", "")):
                continue
            mat_id = row["material_id"]
            each = float(row["e_above_hull_mp2020_corrected_ppd_mp"])
            if each <= stability_threshold:
                stable_ids.append(mat_id)
            else:
                unstable_ids.append(mat_id)

    total = len(stable_ids) + len(unstable_ids)
    if total == 0:
        raise RuntimeError("No unique_prototype WBM rows found in summary file.")

    target_stable = round(limit * len(stable_ids) / total)
    target_stable = min(max(target_stable, 1), len(stable_ids), limit)
    target_unstable = limit - target_stable
    target_unstable = min(target_unstable, len(unstable_ids))

    # Fill any shortfall caused by class limits. This should not happen for WBM,
    # but keeps the sampler well-defined for tiny custom subsets.
    if target_stable + target_unstable < limit:
        target_stable = min(len(stable_ids), limit - target_unstable)

    rng = random.Random(sample_seed)
    selected = (
        rng.sample(stable_ids, target_stable)
        + rng.sample(unstable_ids, target_unstable)
    )
    rng.shuffle(selected)
    return selected


def select_init_records(args: argparse.Namespace) -> list[dict]:
    all_records = iter_wbm_init_records(args.wbm_init)
    if args.limit <= 0:
        return all_records

    selected_ids = set(
        load_proportional_material_ids(
            args.wbm_summary,
            args.limit,
            args.stability_threshold,
            args.sample_seed,
        )
    )

    selected_records = [
        record for record in all_records if record["material_id"] in selected_ids
    ]
    return selected_records[: args.limit]


def structure_dict_to_structure(struct_dct: dict) -> Structure:
    return Structure.from_dict(struct_dct)


def run_prediction(record: dict, relaxer: StructOptimizer, args: argparse.Namespace) -> dict:
    structure = structure_dict_to_structure(record["initial_structure"])
    sync_if_needed(args.device)
    start = time.perf_counter()
    with autocast_context(args.device, args.precision_mode):
        relax_result = relaxer.relax(
            atoms=structure,
            fmax=args.relax_fmax,
            steps=args.relax_steps,
            relax_cell=not args.no_relax_cell,
            ase_filter=args.ase_filter,
            verbose=False,
        )
    sync_if_needed(args.device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    trajectory = relax_result["trajectory"]
    final_structure = relax_result["final_structure"]
    final_energy = float(trajectory.energies[-1])

    result = {
        "material_id": record["material_id"],
        "formula_from_cse": record.get("formula_from_cse", ""),
        "n_atoms": len(structure),
        "pred_relaxed_energy_eV": final_energy,
        "num_relax_frames": len(trajectory),
        "latency_ms": latency_ms,
    }
    if args.save_final_structure:
        result["final_structure"] = final_structure.as_dict()
    return result


def run_warmup(records: list[dict], relaxer: StructOptimizer, args: argparse.Namespace) -> int:
    warmup_records = records[: min(args.warmup_steps, len(records))]
    for idx, record in enumerate(warmup_records, start=1):
        _ = run_prediction(record, relaxer, args)
        print(f"[warmup {idx}/{len(warmup_records)}] {record['material_id']}")
    return len(warmup_records)


def summarize(records: list[dict]) -> dict:
    latencies = [r["latency_ms"] for r in records]
    relax_frames = [r["num_relax_frames"] for r in records]
    return {
        "num_structures": len(records),
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "latency_ms_std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_ms_min": min(latencies) if latencies else 0.0,
        "latency_ms_max": max(latencies) if latencies else 0.0,
        "relax_frames_mean": statistics.mean(relax_frames) if relax_frames else 0.0,
        "relax_frames_max": max(relax_frames) if relax_frames else 0,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_info = configure_precision(args.device, args.precision_mode)
    relaxer = build_relaxer(args)
    init_records = select_init_records(args)

    run_config = {
        "wbm_init": str(Path(args.wbm_init).resolve()),
        "wbm_summary": str(Path(args.wbm_summary).resolve()),
        "output_dir": str(output_dir.resolve()),
        "model": args.model,
        "task": args.task,
        "device": args.device,
        "optimizer": args.optimizer,
        "relax_fmax": args.relax_fmax,
        "relax_steps": args.relax_steps,
        "ase_filter": args.ase_filter,
        "relax_cell": not args.no_relax_cell,
        "save_final_structure": args.save_final_structure,
        "precision_mode": args.precision_mode,
        "compile": args.compile,
        "limit": args.limit,
        "sample_mode": "proportional_stability",
        "sample_seed": args.sample_seed,
        "stability_threshold": args.stability_threshold,
        "warmup_steps": args.warmup_steps,
        "git_commit": get_git_commit(),
        **precision_info,
    }

    warmup_used = run_warmup(init_records, relaxer, args) if args.warmup_steps > 0 else 0

    results = []
    jsonl_path = output_dir / "per_structure_predictions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as out_fp:
        for idx, record in enumerate(init_records, start=1):
            result = run_prediction(record, relaxer, args)
            results.append(result)
            out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{idx}/{len(init_records)}] {result['material_id']} "
                f"n_atoms={result['n_atoms']} "
                f"relax_frames={result['num_relax_frames']} "
                f"energy={result['pred_relaxed_energy_eV']:.6f} eV"
            )

    summary = summarize(results)
    summary["warmup_steps_used"] = warmup_used

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(run_config, fp, ensure_ascii=False, indent=2)

    with open(output_dir / "summary_overall.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
