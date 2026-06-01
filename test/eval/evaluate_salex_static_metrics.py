import argparse
import contextlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from fairchem.core.datasets import AseDBDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications.base import MatRISCalculator
from quant.config import get_quant_config
from quant.fusion import apply_gated_mlp_fusion
from quant.injector import apply_quant_config
from quant.stats import collect_quant_stats


SUPPORTED_PRECISIONS = {"fp32", "tf32", "bf16", "fp16"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MatRIS static E/F/S evaluation on an sAlex ASE LMDB subset."
    )
    parser.add_argument(
        "--dataset-src",
        default="/home/lht/lab/sAlex/val",
        help="Path to sAlex split directory containing *.aselmdb shards.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save run_config.json, sample_indices.json, predictions and summary.",
    )
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="efs", choices=("e", "ef", "efs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision-mode",
        default="fp32",
        choices=sorted(SUPPORTED_PRECISIONS),
    )
    parser.add_argument("--quant-mode", default="none")
    parser.add_argument("--fusion-mode", default="none")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the loaded model with torch.compile when available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of sAlex validation samples to evaluate. Use 0 for the whole dataset.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used to choose sample indices.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Number of selected samples to run as warmup before measured evaluation.",
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


def build_calculator(args: argparse.Namespace) -> MatRISCalculator:
    calc = MatRISCalculator(
        model=args.model,
        task=args.task,
        device=args.device,
    )
    quant_config = get_quant_config(args.quant_mode)
    replaced = apply_quant_config(calc.model, quant_config)
    fused = apply_gated_mlp_fusion(calc.model, args.fusion_mode)
    calc.quant_config = quant_config
    calc.quant_replaced_modules = replaced
    calc.fusion_mode = args.fusion_mode
    calc.fused_modules = fused
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in the current PyTorch.")
        calc.model = torch.compile(calc.model)
    return calc


def tensor_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def tensor_to_nested_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def optional_item_value(item, key: str, default=""):
    if key not in item:
        return default
    value = item[key]
    if isinstance(value, torch.Tensor):
        return tensor_to_nested_list(value)
    return value


def flatten(values) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if isinstance(values, (int, float)):
        return [float(values)]
    out: list[float] = []
    for value in values:
        if isinstance(value, list):
            out.extend(flatten(value))
        else:
            out.append(float(value))
    return out


def tensor_stress_to_ase_voigt(stress) -> list[float]:
    if isinstance(stress, torch.Tensor):
        stress = stress.detach().cpu()
        if stress.ndim == 3:
            stress = stress[0]
        stress = stress.tolist()
    return [
        float(stress[0][0]),
        float(stress[1][1]),
        float(stress[2][2]),
        float(stress[1][2]),
        float(stress[0][2]),
        float(stress[0][1]),
    ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rmse_from_diffs(diffs: list[float]) -> float:
    return math.sqrt(mean([value * value for value in diffs]))


def mae_from_diffs(diffs: list[float]) -> float:
    return mean([abs(value) for value in diffs])


def select_indices(dataset_len: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= dataset_len:
        return list(range(dataset_len))
    rng = random.Random(seed)
    indices = rng.sample(range(dataset_len), limit)
    indices.sort()
    return indices


def get_sample_refs(item) -> dict:
    return {
        "ref_energy_eV": tensor_to_float(item["energy"]),
        "ref_forces_eVA": tensor_to_nested_list(item["forces"]),
        "ref_stress_eVA3": tensor_stress_to_ase_voigt(item["stress"]),
    }


def calc_component_metrics(pred, ref) -> tuple[float, float]:
    pred_flat = flatten(pred)
    ref_flat = flatten(ref)
    diffs = [p - r for p, r in zip(pred_flat, ref_flat)]
    return mae_from_diffs(diffs), rmse_from_diffs(diffs)


def run_prediction(
    sample_index: int,
    dataset: AseDBDataset,
    calc: MatRISCalculator,
    args: argparse.Namespace,
) -> dict:
    item = dataset[sample_index]
    refs = get_sample_refs(item)
    atoms = dataset.get_atoms(sample_index)
    atoms.calc = calc

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sync_if_needed(args.device)
    start = time.perf_counter()
    with autocast_context(args.device, args.precision_mode):
        pred_energy = float(atoms.get_potential_energy())
        pred_forces = atoms.get_forces() if "f" in args.task else None
        pred_stress = atoms.get_stress() if "s" in args.task else None
    sync_if_needed(args.device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    peak_mem_mb = 0.0
    if args.device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    n_atoms = len(atoms)
    energy_abs_error_eV = abs(pred_energy - refs["ref_energy_eV"])
    result = {
        "sample_index": sample_index,
        "sid": optional_item_value(item, "sid", ""),
        "formula": atoms.get_chemical_formula(),
        "n_atoms": n_atoms,
        "pred_energy_eV": pred_energy,
        "ref_energy_eV": refs["ref_energy_eV"],
        "energy_abs_error_eV": energy_abs_error_eV,
        "energy_abs_error_per_atom_eV": energy_abs_error_eV / n_atoms,
        "latency_ms": latency_ms,
        "peak_mem_mb": peak_mem_mb,
    }

    if pred_forces is not None:
        force_mae, force_rmse = calc_component_metrics(
            pred_forces.tolist(), refs["ref_forces_eVA"]
        )
        result["force_mae_eVA"] = force_mae
        result["force_rmse_eVA"] = force_rmse

    if pred_stress is not None:
        stress_mae, stress_rmse = calc_component_metrics(
            pred_stress.tolist(), refs["ref_stress_eVA3"]
        )
        result["stress_mae_eVA3"] = stress_mae
        result["stress_rmse_eVA3"] = stress_rmse

    return result


def run_warmup(
    indices: list[int],
    dataset: AseDBDataset,
    calc: MatRISCalculator,
    args: argparse.Namespace,
) -> int:
    warmup_indices = indices[: min(args.warmup_steps, len(indices))]
    for idx, sample_index in enumerate(warmup_indices, start=1):
        _ = run_prediction(sample_index, dataset, calc, args)
        print(f"[warmup {idx}/{len(warmup_indices)}] sample_index={sample_index}")
    return len(warmup_indices)


def summarize(records: list[dict]) -> dict:
    latencies = [r["latency_ms"] for r in records]
    peaks = [r["peak_mem_mb"] for r in records]
    n_atoms = [r["n_atoms"] for r in records]
    total_latency_s = sum(latencies) / 1000.0

    summary = {
        "num_structures": len(records),
        "num_atoms_mean": mean(n_atoms),
        "num_atoms_min": min(n_atoms) if n_atoms else 0,
        "num_atoms_max": max(n_atoms) if n_atoms else 0,
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "latency_ms_std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_ms_min": min(latencies) if latencies else 0.0,
        "latency_ms_max": max(latencies) if latencies else 0.0,
        "throughput_structures_per_s": len(records) / total_latency_s
        if total_latency_s > 0
        else 0.0,
        "throughput_atoms_per_s": sum(n_atoms) / total_latency_s
        if total_latency_s > 0
        else 0.0,
        "energy_mae_eV": mean([r["energy_abs_error_eV"] for r in records]),
        "energy_mae_per_atom": mean(
            [r["energy_abs_error_per_atom_eV"] for r in records]
        ),
        "peak_mem_mb_max": max(peaks) if peaks else 0.0,
    }

    if records and "force_mae_eVA" in records[0]:
        summary["force_mae_eVA"] = mean([r["force_mae_eVA"] for r in records])
        summary["force_rmse_eVA"] = mean([r["force_rmse_eVA"] for r in records])
    if records and "stress_mae_eVA3" in records[0]:
        summary["stress_mae_eVA3"] = mean([r["stress_mae_eVA3"] for r in records])
        summary["stress_rmse_eVA3"] = mean([r["stress_rmse_eVA3"] for r in records])

    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_info = configure_precision(args.device, args.precision_mode)
    dataset = AseDBDataset(config={"src": args.dataset_src})
    indices = select_indices(len(dataset), args.limit, args.sample_seed)
    calc = build_calculator(args)

    run_config = {
        "dataset_name": "sAlex",
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "dataset_size": len(dataset),
        "output_dir": str(output_dir.resolve()),
        "model": args.model,
        "task": args.task,
        "device": args.device,
        "precision_mode": args.precision_mode,
        "quant_mode": args.quant_mode,
        "fusion_mode": args.fusion_mode,
        "quant_config": getattr(calc, "quant_config", None),
        "quant_replaced_modules": getattr(calc, "quant_replaced_modules", []),
        "fused_modules": getattr(calc, "fused_modules", []),
        "compile": args.compile,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "warmup_steps": args.warmup_steps,
        "git_commit": get_git_commit(),
        **precision_info,
    }

    with open(output_dir / "sample_indices.json", "w", encoding="utf-8") as fp:
        json.dump(indices, fp, ensure_ascii=False, indent=2)

    warmup_used = run_warmup(indices, dataset, calc, args) if args.warmup_steps > 0 else 0

    records = []
    jsonl_path = output_dir / "per_structure_predictions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as out_fp:
        for idx, sample_index in enumerate(indices, start=1):
            record = run_prediction(sample_index, dataset, calc, args)
            records.append(record)
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{idx}/{len(indices)}] sample_index={sample_index} "
                f"formula={record['formula']} n_atoms={record['n_atoms']} "
                f"latency={record['latency_ms']:.3f} ms"
            )

    summary = summarize(records)
    summary["warmup_steps_used"] = warmup_used

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(run_config, fp, ensure_ascii=False, indent=2)

    with open(output_dir / "summary_overall.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    with open(output_dir / "quant_stats.json", "w", encoding="utf-8") as fp:
        json.dump(collect_quant_stats(calc.model), fp, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
