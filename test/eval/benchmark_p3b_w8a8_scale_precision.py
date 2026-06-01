import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications import MatRISCalculator
from quant.config import get_quant_config
from quant.fusion import apply_gated_mlp_fusion
from quant.injector import apply_quant_config
from quant.layers import ActivationWeightFakeQuantLinear
from quant.stats import collect_quant_stats


FUSION_MODE = "line_all_candidate_gated_mlp_second_fused_fp32"
DEFAULT_MODES = (
    "none",
    "a8_line_edge_core_gate",
    "a8_line_edge_core_gate_static_a8",
    "a8_line_edge_core_gate_tile_m32_a8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P3b-3 activation-scale precision screen for W8A8 line edge candidates."
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--output-dir", default="results/p3b_w8a8_scale_precision")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=("fp32", "tf32", "bf16", "fp16"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--calibration-limit", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--calibration-seed", type=int, default=43)
    parser.add_argument("--max-ele-num", type=int, default=120)
    parser.add_argument("--measure-time", action="store_true")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    return parser.parse_args()


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


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


def select_group_aligned_keys(dataset_len: int, limit: int, seed: int) -> np.ndarray:
    keys = np.arange(dataset_len)
    state = random.getstate()
    random.seed(seed)
    random.shuffle(keys)
    random.setstate(state)
    if limit > 0:
        keys = keys[:limit]
    return keys


def summarize_group_metrics(
    energy_err: list[float],
    force_err: list[np.ndarray],
    stress_err: list[np.ndarray],
    atom_num: list[np.ndarray],
) -> dict:
    atom_num_array = np.array(atom_num)
    energy_err_array = np.array(energy_err)
    energy_err_per_atom = energy_err_array / atom_num_array.sum(-1)
    return {
        "energy_mae": [float(np.mean(np.abs(np.stack(energy_err_array))))],
        "energy_rmse": [float(np.sqrt(np.mean(np.square(energy_err_array))))],
        "energy_mae_natoms": [float(np.mean(np.abs(np.stack(energy_err_per_atom))))],
        "energy_rmse_natoms": [float(np.sqrt(np.mean(np.square(energy_err_per_atom))))],
        "force_mae": [float(np.mean(np.abs(np.concatenate(force_err))))],
        "force_rmse": [float(np.sqrt(np.mean(np.square(np.concatenate(force_err)))))],
        "stress_mae": [float(np.mean(np.abs(np.concatenate(stress_err))))],
        "stress_rmse": [float(np.sqrt(np.mean(np.square(np.concatenate(stress_err)))))],
    }


def summarize_timing(latencies_ms: list[float], n_atoms: list[int]) -> dict:
    total_s = sum(latencies_ms) / 1000.0
    return {
        "latency_ms_mean": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "latency_ms_std": float(np.std(latencies_ms, ddof=1)) if len(latencies_ms) > 1 else 0.0,
        "latency_ms_min": float(np.min(latencies_ms)) if latencies_ms else 0.0,
        "latency_ms_max": float(np.max(latencies_ms)) if latencies_ms else 0.0,
        "throughput_structures_per_s": len(latencies_ms) / total_s if total_s > 0 else 0.0,
        "throughput_atoms_per_s": sum(n_atoms) / total_s if total_s > 0 else 0.0,
    }


def build_calculator(args: argparse.Namespace, quant_mode: str) -> MatRISCalculator:
    calculator = MatRISCalculator(
        model=args.model,
        task=args.task,
        device=args.device,
    )
    quant_config = get_quant_config(quant_mode)
    replaced = apply_quant_config(calculator.model, quant_config)
    fused = apply_gated_mlp_fusion(calculator.model, FUSION_MODE)
    calculator.quant_config = quant_config
    calculator.quant_replaced_modules = replaced
    calculator.fusion_mode = FUSION_MODE
    calculator.fused_modules = fused
    return calculator


def activation_quant_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    required = (
        "reset_activation_calibration",
        "finalize_activation_calibration",
        "calibrating_activation",
        "activation_scale_granularity",
    )
    return [
        module
        for module in model.modules()
        if isinstance(module, ActivationWeightFakeQuantLinear)
        or all(hasattr(module, attr) for attr in required)
    ]


def run_calibration(
    calculator: MatRISCalculator,
    structures: AseDBDataset,
    keys: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    modules = activation_quant_modules(calculator.model)
    if not modules:
        return {"enabled": False, "module_count": 0, "sample_count": 0}
    static_modules = [m for m in modules if m.activation_scale_granularity == "static_per_tensor"]
    if not static_modules:
        return {"enabled": False, "module_count": len(modules), "sample_count": 0}

    for module in static_modules:
        module.reset_activation_calibration()
        module.calibrating_activation = True

    for graph_id in tqdm(keys, desc="calibration", leave=False):
        atom = structures.get_atoms(int(graph_id))
        atom.calc = calculator
        with autocast_context(args.device, args.precision_mode):
            _ = atom.get_potential_energy()
            _ = atom.get_forces()
            _ = atom.get_stress()
        sync_if_needed(args.device)

    for module in static_modules:
        module.calibrating_activation = False
        module.finalize_activation_calibration()

    scales = torch.stack([m.activation_static_scale.detach().float().cpu() for m in static_modules])
    return {
        "enabled": True,
        "module_count": len(static_modules),
        "sample_count": int(len(keys)),
        "scale_min": float(scales.min()),
        "scale_mean": float(scales.mean()),
        "scale_max": float(scales.max()),
    }


def evaluate_mode(
    quant_mode: str,
    structures: AseDBDataset,
    eval_keys: np.ndarray,
    calibration_keys: np.ndarray,
    args: argparse.Namespace,
    precision_info: dict,
) -> dict:
    calculator = build_calculator(args, quant_mode)
    calibration = run_calibration(calculator, structures, calibration_keys, args)

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    energy_err = []
    force_err = []
    stress_err = []
    atom_num = []
    n_atoms = []
    latencies_ms = []

    for idx, graph_id in enumerate(tqdm(eval_keys, desc=quant_mode, leave=False)):
        atom = structures.get_atoms(int(graph_id))
        energy_label = atom.get_potential_energy()
        force_label = atom.get_forces()
        stress_label = atom.get_stress()
        atomic_numbers = atom.get_atomic_numbers()
        atom.calc = calculator

        sync_if_needed(args.device)
        start = time.perf_counter()
        with autocast_context(args.device, args.precision_mode):
            energy_pred = atom.get_potential_energy()
            force_pred = atom.get_forces()
            stress_pred = atom.get_stress()
        sync_if_needed(args.device)
        if args.measure_time:
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

        atom_num.append(np.bincount(atomic_numbers, minlength=args.max_ele_num))
        n_atoms.append(len(atom))
        energy_err.append(energy_label - energy_pred)
        force_err.append(force_label - force_pred)
        stress_err.append(stress_label - stress_pred)

    payload = {
        "res": summarize_group_metrics(energy_err, force_err, stress_err, atom_num),
        "metadata": {
            "quant_mode": quant_mode,
            "fusion_mode": FUSION_MODE,
            "sample_seed": args.sample_seed,
            "limit": args.limit,
            "sample_count": len(eval_keys),
            "calibration": calibration,
            "quant_config": calculator.quant_config,
            "quant_replaced_modules": calculator.quant_replaced_modules,
            "fused_modules": calculator.fused_modules,
            **precision_info,
        },
        "quant_stats": collect_quant_stats(calculator.model),
    }
    if args.measure_time:
        payload["timing"] = summarize_timing(latencies_ms, n_atoms)
    if args.device == "cuda":
        payload["metadata"]["peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    return payload


def metric_scalar(payload: dict, key: str) -> float:
    return float(payload["res"][key][0])


def compare_to_baseline(results: dict[str, dict], baseline_name: str = "none") -> dict:
    baseline = results[baseline_name]
    metrics = (
        "energy_mae",
        "energy_mae_natoms",
        "force_mae",
        "force_rmse",
        "stress_mae",
        "stress_rmse",
    )
    comparison = {}
    for mode, payload in results.items():
        row = {}
        for metric in metrics:
            base_value = metric_scalar(baseline, metric)
            value = metric_scalar(payload, metric)
            row[metric] = value
            row[f"{metric}_ratio"] = value / base_value if base_value != 0.0 else 0.0
        comparison[mode] = row
    return comparison


def write_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    lines = [
        "# P3b-3 W8A8 Activation Scale Precision Screen",
        "",
        f"- dataset: `{payload['dataset_src']}`",
        f"- limit: `{payload['limit']}`",
        f"- calibration_limit: `{payload['calibration_limit']}`",
        f"- fusion_mode: `{payload['fusion_mode']}`",
        "",
        "| mode | energy_mae_natoms | energy ratio | force_mae | force ratio | force_rmse ratio | stress_mae ratio | stress_rmse ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, row in payload["comparison"].items():
        lines.append(
            "| {mode} | {energy_mae_natoms:.10f} | {energy_mae_natoms_ratio:.6f} | "
            "{force_mae:.10f} | {force_mae_ratio:.6f} | {force_rmse_ratio:.6f} | "
            "{stress_mae_ratio:.6f} | {stress_rmse_ratio:.6f} |".format(mode=mode, **row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    precision_info = configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config={"src": args.dataset_src})
    eval_keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    calibration_keys = select_group_aligned_keys(len(structures), args.calibration_limit, args.calibration_seed)

    results = {}
    for mode in args.modes:
        results[mode] = evaluate_mode(
            mode,
            structures,
            eval_keys,
            calibration_keys,
            args,
            precision_info,
        )

    payload = {
        "phase": "P3b-3-scale-precision",
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "model": args.model,
        "task": args.task,
        "device": args.device,
        "precision_mode": args.precision_mode,
        "fusion_mode": FUSION_MODE,
        "limit": args.limit,
        "calibration_limit": args.calibration_limit,
        "sample_seed": args.sample_seed,
        "calibration_seed": args.calibration_seed,
        "modes": list(args.modes),
        "results": results,
        "comparison": compare_to_baseline(results),
    }
    write_outputs(Path(args.output_dir), payload)
    print(f"Wrote {Path(args.output_dir) / 'summary.json'}")
    print(f"Wrote {Path(args.output_dir) / 'summary.md'}")


if __name__ == "__main__":
    main()
