import argparse
import contextlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from ase.stress import full_3x3_to_voigt_6_stress
from fairchem.core.datasets import AseDBDataset
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications import MatRISCalculator
from matris.model.model import MatRIS
from quant.config import get_quant_config
from quant.fusion import apply_gated_mlp_fusion
from quant.injector import apply_quant_config
from quant.runtime import freeze_model_params_for_efs
from quant.stats import collect_quant_stats


SUPPORTED_PRECISIONS = {"fp32", "tf32", "bf16", "fp16"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sAlex LMDB inference using the group-aligned 500-sample E/F/S metric protocol."
    )
    parser.add_argument(
        "--dataset-src",
        default="/home/lht/lab/sAlex/val",
        help="Path to sAlex val split directory containing *.aselmdb shards.",
    )
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument(
        "--model-path",
        default="",
        help=(
            "Optional checkpoint path for compatibility with the original infer_lmdb script. "
            "Relative names are resolved against cwd and ~/.cache/matris."
        ),
    )
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=sorted(SUPPORTED_PRECISIONS))
    parser.add_argument("--quant-mode", default="none")
    parser.add_argument("--fusion-mode", default="none")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-ele-num", type=int, default=120)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--save-predictions", default="")
    parser.add_argument("--measure-time", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "P59 scheduling experiment: evaluate this many structures in one "
            "MatRISCalculator.calculate_many call. Default 1 preserves the ASE "
            "single-structure path."
        ),
    )
    parser.add_argument(
        "--max-batch-atoms",
        type=int,
        default=0,
        help=(
            "Optional safety cap for batched inference. When >0, batch-size is "
            "treated as the maximum number of structures and batches are split "
            "early before the total atom count would exceed this value."
        ),
    )
    parser.add_argument(
        "--prefetch-graphs",
        action="store_true",
        help=(
            "P59 scheduling experiment: build the next CPU graph in a one-worker "
            "background thread while the current graph runs on GPU. Only applies "
            "to batch-size 1."
        ),
    )
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
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


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(REPO_ROOT / path)
        candidates.append(Path.home() / ".cache" / "matris" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find model checkpoint: {model_path}")


def load_model_from_path(model_path: str, device: str) -> MatRIS:
    ckpt_path = resolve_model_path(model_path)
    ckpt_state = torch.load(
        ckpt_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model = MatRIS.from_dict(ckpt_state)
    model = model.to(device)
    model.eval()
    print(f"Loading checkpoint from {ckpt_path}, running on {device}.")
    return model


def build_calculator(args: argparse.Namespace) -> MatRISCalculator:
    calculator = MatRISCalculator(
        model=args.model,
        task=args.task,
        device=args.device,
    )
    if args.model_path:
        calculator.model = load_model_from_path(args.model_path, args.device)

    quant_config = get_quant_config(args.quant_mode)
    replaced = apply_quant_config(calculator.model, quant_config)
    fused = apply_gated_mlp_fusion(calculator.model, args.fusion_mode)
    calculator.model.eval()
    freeze_info = freeze_model_params_for_efs(calculator.model)
    calculator.quant_config = quant_config
    calculator.quant_replaced_modules = replaced
    calculator.fusion_mode = args.fusion_mode
    calculator.fused_modules = fused
    calculator.freeze_model_params_for_efs = freeze_info
    return calculator


def activation_calibration_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules = []
    for module in model.modules():
        if all(
            hasattr(module, name)
            for name in (
                "reset_activation_calibration",
                "finalize_activation_calibration",
                "calibrating_activation",
            )
        ):
            modules.append(module)
    return modules


def run_activation_calibration(
    structures: AseDBDataset,
    calculator: MatRISCalculator,
    args: argparse.Namespace,
) -> dict:
    modules = activation_calibration_modules(calculator.model)
    if args.activation_calibration_limit <= 0 or not modules:
        return {
            "enabled": False,
            "num_modules": len(modules),
            "num_samples": 0,
        }

    keys = select_group_aligned_keys(
        len(structures),
        args.activation_calibration_limit,
        args.activation_calibration_seed,
    )
    for module in modules:
        module.reset_activation_calibration()
        module.calibrating_activation = True

    for graph_id in tqdm(keys, desc="activation calibration", leave=False):
        atom = structures.get_atoms(int(graph_id))
        atom.calc = calculator
        with autocast_context(args.device, args.precision_mode):
            _ = atom.get_potential_energy()
            if args.task in ("ef", "efs", "efsm"):
                _ = atom.get_forces()
            if args.task in ("efs", "efsm"):
                _ = atom.get_stress()

    for module in modules:
        module.calibrating_activation = False
        module.finalize_activation_calibration()

    scales = [
        module.activation_static_scale.detach().float().cpu()
        for module in modules
        if hasattr(module, "activation_static_scale")
    ]
    if scales:
        scale_tensor = torch.stack(scales)
        scale_summary = {
            "scale_mean": float(scale_tensor.mean().item()),
            "scale_min": float(scale_tensor.min().item()),
            "scale_max": float(scale_tensor.max().item()),
        }
    else:
        scale_summary = {
            "scale_mean": 0.0,
            "scale_min": 0.0,
            "scale_max": 0.0,
        }
    return {
        "enabled": True,
        "num_modules": len(modules),
        "num_samples": len(keys),
        "sample_seed": args.activation_calibration_seed,
        **scale_summary,
    }


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_group_metrics(
    energy_err: list[float],
    force_err: list[np.ndarray],
    stress_err: list[np.ndarray],
    atom_num: list[np.ndarray],
) -> dict:
    atom_num_array = np.array(atom_num)
    energy_err_array = np.array(energy_err)
    energy_err_per_atom = energy_err_array / atom_num_array.sum(-1)

    res = {
        "energy_mae": [np.mean(np.abs(np.stack(energy_err_array)))],
        "energy_rmse": [np.sqrt(np.mean(np.square(energy_err_array)))],
        "energy_mae_natoms": [np.mean(np.abs(np.stack(energy_err_per_atom)))],
        "energy_rmse_natoms": [
            np.sqrt(np.mean(np.square(energy_err_per_atom)))
        ],
    }
    res.update(
        {
            "force_mae": [np.mean(np.abs(np.concatenate(force_err)))],
            "force_rmse": [
                np.sqrt(np.mean(np.square(np.concatenate(force_err))))
            ],
        }
    )
    res.update(
        {
            "stress_mae": [np.mean(np.abs(np.concatenate(stress_err)))],
            "stress_rmse": [
                np.sqrt(np.mean(np.square(np.concatenate(stress_err))))
            ],
        }
    )
    return res


def summarize_timing(latencies_ms: list[float], n_atoms: list[int]) -> dict:
    total_s = sum(latencies_ms) / 1000.0
    return {
        "num_timed_samples": len(latencies_ms),
        "latency_ms_total": float(np.sum(latencies_ms)) if latencies_ms else 0.0,
        "latency_s_total": float(total_s),
        "latency_ms_mean": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "latency_ms_std": float(np.std(latencies_ms, ddof=1)) if len(latencies_ms) > 1 else 0.0,
        "latency_ms_min": float(np.min(latencies_ms)) if latencies_ms else 0.0,
        "latency_ms_max": float(np.max(latencies_ms)) if latencies_ms else 0.0,
        "throughput_structures_per_s": len(latencies_ms) / total_s if total_s > 0 else 0.0,
        "throughput_atoms_per_s": sum(n_atoms) / total_s if total_s > 0 else 0.0,
    }


def main() -> None:
    args = parse_args()
    precision_info = configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config={"src": args.dataset_src})
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    calculator = build_calculator(args)
    activation_calibration = run_activation_calibration(structures, calculator, args)

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    energy_err = []
    force_err = []
    stress_err = []
    atom_num = []
    n_atoms = []
    predictions = []
    latencies_ms = []
    batch_stats = {
        "num_batches": 0,
        "max_structures_per_batch": 0,
        "max_atoms_per_batch": 0,
    }

    pred_fp = None
    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_fp = pred_path.open("w", encoding="utf-8")

    def append_record(
        idx: int,
        graph_id: int,
        atom,
        energy_label: float,
        pred_energy: float,
        label_force,
        pred_force,
        label_stress,
        pred_stress,
        per_structure_latency_ms: float | None,
    ) -> None:
        atomic_numbers = atom.get_atomic_numbers()
        atom_num.append(np.bincount(atomic_numbers, minlength=args.max_ele_num))
        n_atoms.append(len(atom))
        energy_err.append(energy_label - pred_energy)
        force_err.append(label_force - pred_force)
        stress_err.append(label_stress - pred_stress)
        if per_structure_latency_ms is not None:
            latencies_ms.append(per_structure_latency_ms)

        if args.save_predictions:
            record = {
                "sample_order": idx,
                "graph_id": graph_id,
                "formula": atom.get_chemical_formula(),
                "n_atoms": len(atom),
                "energy_label": float(energy_label),
                "energy_pred": float(pred_energy),
                "energy_err": float(energy_err[-1]),
                "force_mae": float(np.mean(np.abs(force_err[-1]))),
                "stress_mae": float(np.mean(np.abs(stress_err[-1]))),
            }
            if per_structure_latency_ms is not None:
                record["latency_ms"] = per_structure_latency_ms
            predictions.append(record)
            pred_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_atoms_batch(batch_atoms: list, batch_meta: list[dict]) -> None:
        if not batch_atoms:
            return

        batch_stats["num_batches"] += 1
        batch_stats["max_structures_per_batch"] = max(
            batch_stats["max_structures_per_batch"],
            len(batch_atoms),
        )
        batch_stats["max_atoms_per_batch"] = max(
            batch_stats["max_atoms_per_batch"],
            sum(len(atom) for atom in batch_atoms),
        )

        try:
            sync_if_needed(args.device)
            start = time.perf_counter()
            with autocast_context(args.device, args.precision_mode):
                batch_results = calculator.calculate_many(batch_atoms)
            sync_if_needed(args.device)
            per_latency = None
            if args.measure_time:
                per_latency = (time.perf_counter() - start) * 1000.0 / max(1, len(batch_results))

            for atom, meta, result in zip(batch_atoms, batch_meta, batch_results):
                append_record(
                    meta["idx"],
                    meta["graph_id"],
                    atom,
                    meta["energy_label"],
                    float(result["energy"]),
                    meta["label_force"],
                    result["forces"],
                    meta["label_stress"],
                    (
                        full_3x3_to_voigt_6_stress(result["stress"])
                        if result["stress"] is not None and np.asarray(result["stress"]).shape == (3, 3)
                        else result["stress"]
                    ),
                    per_latency,
                )
        except Exception as exc:
            graph_id_text = ",".join(str(meta["graph_id"]) for meta in batch_meta)
            print(f"处理 batch (graph_id: {graph_id_text}) 时出错: {exc}")

    try:
        if args.prefetch_graphs and args.batch_size <= 1:
            def prepare_one(idx: int) -> dict:
                graph_id = int(keys[idx])
                atom = structures.get_atoms(graph_id)
                energy_label = atom.get_potential_energy()
                label_force = atom.get_forces()
                label_stress = atom.get_stress()
                calculator._adjust_pbc(atom)
                structure = AseAtomsAdaptor.get_structure(atom)
                graph_cpu = calculator.model.graph_converter(structure)
                n_atoms_factor = 1 if not calculator.model.is_intensive else structure.composition.num_atoms
                return {
                    "idx": idx,
                    "graph_id": graph_id,
                    "atom": atom,
                    "energy_label": energy_label,
                    "label_force": label_force,
                    "label_stress": label_stress,
                    "graph_cpu": graph_cpu,
                    "n_atoms_factor": n_atoms_factor,
                }

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(prepare_one, 0) if len(keys) else None
                for idx in tqdm(range(len(keys))):
                    try:
                        start = time.perf_counter()
                        prepared = future.result()
                        next_idx = idx + 1
                        future = (
                            executor.submit(prepare_one, next_idx)
                            if next_idx < len(keys)
                            else None
                        )

                        sync_if_needed(args.device)
                        with autocast_context(args.device, args.precision_mode):
                            result = calculator.calculate_graphs(
                                [prepared["graph_cpu"]],
                                [prepared["n_atoms_factor"]],
                            )[0]
                        sync_if_needed(args.device)
                        per_latency = (time.perf_counter() - start) * 1000.0 if args.measure_time else None
                        pred_stress = (
                            full_3x3_to_voigt_6_stress(result["stress"])
                            if result["stress"] is not None and np.asarray(result["stress"]).shape == (3, 3)
                            else result["stress"]
                        )
                        append_record(
                            prepared["idx"],
                            prepared["graph_id"],
                            prepared["atom"],
                            prepared["energy_label"],
                            float(result["energy"]),
                            prepared["label_force"],
                            result["forces"],
                            prepared["label_stress"],
                            pred_stress,
                            per_latency,
                        )
                    except Exception as exc:
                        graph_id_text = locals().get("prepared", {}).get("graph_id", "unknown")
                        print(f"处理索引 {idx} (graph_id: {graph_id_text}) 时出错: {exc}")
                        continue
        elif args.batch_size <= 1:
            for idx in tqdm(range(len(keys))):
                try:
                    graph_id = int(keys[idx])

                    atom = structures.get_atoms(graph_id)
                    energy_label = atom.get_potential_energy()
                    label_force = atom.get_forces()
                    label_stress = atom.get_stress()

                    atom.calc = calculator

                    sync_if_needed(args.device)
                    start = time.perf_counter()
                    with autocast_context(args.device, args.precision_mode):
                        pred_energy = atom.get_potential_energy()
                        pred_force = atom.get_forces()
                        pred_stress = atom.get_stress()
                    sync_if_needed(args.device)
                    per_latency = (time.perf_counter() - start) * 1000.0 if args.measure_time else None

                    append_record(
                        idx,
                        graph_id,
                        atom,
                        energy_label,
                        pred_energy,
                        label_force,
                        pred_force,
                        label_stress,
                        pred_stress,
                        per_latency,
                    )
                except Exception as exc:
                    graph_id_text = locals().get("graph_id", "unknown")
                    print(f"处理索引 {idx} (graph_id: {graph_id_text}) 时出错: {exc}")
                    continue
        else:
            pending_atoms = []
            pending_meta = []
            pending_natoms = 0

            for idx in tqdm(range(len(keys))):
                try:
                    graph_id = int(keys[idx])
                    atom = structures.get_atoms(graph_id)
                    atom_n = len(atom)

                    if (
                        pending_atoms
                        and args.max_batch_atoms > 0
                        and pending_natoms + atom_n > args.max_batch_atoms
                    ):
                        run_atoms_batch(pending_atoms, pending_meta)
                        pending_atoms = []
                        pending_meta = []
                        pending_natoms = 0

                    pending_atoms.append(atom)
                    pending_meta.append(
                        {
                            "idx": idx,
                            "graph_id": graph_id,
                            "energy_label": atom.get_potential_energy(),
                            "label_force": atom.get_forces(),
                            "label_stress": atom.get_stress(),
                        }
                    )
                    pending_natoms += atom_n

                    if len(pending_atoms) >= args.batch_size:
                        run_atoms_batch(pending_atoms, pending_meta)
                        pending_atoms = []
                        pending_meta = []
                        pending_natoms = 0
                except Exception as exc:
                    graph_id_text = locals().get("graph_id", "unknown")
                    print(f"处理索引 {idx} (graph_id: {graph_id_text}) 时出错: {exc}")
                    continue

            run_atoms_batch(pending_atoms, pending_meta)
    finally:
        if pred_fp is not None:
            pred_fp.close()

    res = summarize_group_metrics(energy_err, force_err, stress_err, atom_num)
    print(res)

    metadata = {
        "dataset_name": "sAlex",
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "dataset_size": len(structures),
        "sample_seed": args.sample_seed,
        "limit": args.limit,
        "sample_count": len(keys),
        "sample_selection": "np.arange(dataset_len), random.seed(sample_seed), random.shuffle(keys), keys[:limit]",
        "num_success": len(energy_err),
        "model": args.model,
        "model_path": args.model_path,
        "task": args.task,
        "device": args.device,
        "precision_mode": args.precision_mode,
        "quant_mode": args.quant_mode,
        "fusion_mode": args.fusion_mode,
        "batch_size": args.batch_size,
        "max_batch_atoms": args.max_batch_atoms,
        "batch_stats": batch_stats,
        "prefetch_graphs": args.prefetch_graphs,
        "quant_config": getattr(calculator, "quant_config", None),
        "quant_replaced_modules": getattr(calculator, "quant_replaced_modules", []),
        "fused_modules": getattr(calculator, "fused_modules", []),
        "freeze_model_params_for_efs": getattr(calculator, "freeze_model_params_for_efs", None),
        "activation_calibration": activation_calibration,
        **precision_info,
    }
    if args.device == "cuda":
        metadata["peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)

    payload = {
        "res": res,
        "metadata": metadata,
    }
    if args.measure_time:
        payload["timing"] = summarize_timing(latencies_ms, n_atoms)
    if args.save_predictions:
        payload["predictions_path"] = str(Path(args.save_predictions).resolve())
    quant_stats = collect_quant_stats(calculator.model)
    if quant_stats:
        payload["quant_stats"] = quant_stats

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2, default=to_jsonable)


if __name__ == "__main__":
    main()
