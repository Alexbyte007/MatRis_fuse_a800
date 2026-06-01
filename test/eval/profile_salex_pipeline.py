import argparse
import contextlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn
from fairchem.core.datasets import AseDBDataset
from pymatgen.io.ase import AseAtomsAdaptor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.model.model import MatRIS
from matris.model.processgraph import process_graphs
from quant.config import get_quant_config
from quant.fusion import apply_gated_mlp_fusion
from quant.injector import apply_quant_config
from quant.runtime import freeze_model_params_for_efs
from quant.stats import collect_quant_stats


SUPPORTED_PRECISIONS = {"fp32", "tf32", "bf16", "fp16"}
PROFILE_KEYS = [
    "dataset_get_item_ms",
    "dataset_get_atoms_ms",
    "atoms_to_structure_ms",
    "graph_converter_ms",
    "graph_to_device_ms",
    "process_graphs_ms",
    "embedding_ms",
    "interaction_blocks_ms",
    "energy_head_ms",
    "force_autograd_ms",
    "stress_autograd_ms",
    "combined_force_stress_autograd_ms",
    "cpu_output_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile MatRIS pipeline stages on fixed sAlex samples."
    )
    parser.add_argument(
        "--dataset-src",
        default="/home/lht/lab/sAlex/val",
        help="Path to sAlex split directory containing *.aselmdb shards.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save run_config.json, sample_indices.json, profile records and summary.",
    )
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="efs", choices=("e", "ef", "efs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=sorted(SUPPORTED_PRECISIONS))
    parser.add_argument("--quant-mode", default="none")
    parser.add_argument("--fusion-mode", default="none")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-selection",
        default="teacher-aligned",
        choices=("teacher-aligned", "sorted-random"),
        help=(
            "Sample selection protocol. teacher-aligned matches "
            "infer_salex_lmdb_quant.py / MatRISCalculator runs: "
            "range(dataset_len), random.shuffle, then keys[:limit]. "
            "sorted-random preserves the older profile_salex_pipeline behavior."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip samples that fail graph conversion/profile and continue until limit successful samples are recorded.",
    )
    parser.add_argument(
        "--profile-max-attempts",
        type=int,
        default=0,
        help="Maximum candidate samples to try when --skip-errors is set. Defaults to a small oversample.",
    )
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument(
        "--combined-force-stress-autograd",
        action="store_true",
        help=(
            "For task=efs, compute force and stress in one torch.autograd.grad call. "
            "The default keeps the historical separate force/stress timing."
        ),
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


def timed_call(device: str, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    return result, (time.perf_counter() - start) * 1000.0


def select_sorted_random_indices(dataset_len: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= dataset_len:
        return list(range(dataset_len))
    rng = random.Random(seed)
    indices = rng.sample(range(dataset_len), limit)
    indices.sort()
    return indices


def select_teacher_aligned_indices(dataset_len: int, limit: int, seed: int) -> list[int]:
    indices = list(range(dataset_len))
    state = random.getstate()
    random.seed(seed)
    random.shuffle(indices)
    random.setstate(state)
    if limit > 0:
        indices = indices[:limit]
    return indices


def select_indices(dataset_len: int, limit: int, seed: int, sample_selection: str) -> list[int]:
    if sample_selection == "teacher-aligned":
        return select_teacher_aligned_indices(dataset_len, limit, seed)
    if sample_selection == "sorted-random":
        return select_sorted_random_indices(dataset_len, limit, seed)
    raise ValueError(f"Unsupported sample selection: {sample_selection}")


def select_profile_candidates(
    dataset_len: int,
    limit: int,
    seed: int,
    *,
    skip_errors: bool,
    max_attempts: int,
    sample_selection: str,
) -> list[int]:
    if not skip_errors:
        return select_indices(dataset_len, limit, seed, sample_selection)
    if limit <= 0 or limit >= dataset_len:
        return select_indices(dataset_len, limit, seed, sample_selection)
    candidate_count = max_attempts if max_attempts > 0 else max(limit + 1000, int(limit * 1.2))
    candidate_count = min(dataset_len, max(limit, candidate_count))
    return select_indices(dataset_len, candidate_count, seed, sample_selection)


def activation_calibration_modules(model: nn.Module) -> list[nn.Module]:
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


def build_model(args: argparse.Namespace) -> MatRIS:
    model = MatRIS.load(model_name=args.model, device=args.device)
    quant_config = get_quant_config(args.quant_mode)
    replaced = apply_quant_config(model, quant_config)
    fused = apply_gated_mlp_fusion(model, args.fusion_mode)
    freeze_info = freeze_model_params_for_efs(model)
    model.quant_config = quant_config
    model.quant_replaced_modules = replaced
    model.fusion_mode = args.fusion_mode
    model.fused_modules = fused
    model.freeze_model_params_for_efs = freeze_info
    model.eval()
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in the current PyTorch.")
        model = torch.compile(model)
    return model


def run_embedding_only(model: MatRIS, batch_graph):
    node_feat = model.atom_embedding(batch_graph["atomic_numbers"] - 1)
    edge_feat, smooth_weight = model.edge_embedding(graphs=batch_graph)

    threebody_feat = None
    if len(batch_graph["line_graph_dict"]["line_graph"]) != 0:
        threebody_feat = model.three_body_embedding(graphs=batch_graph)
    return node_feat, edge_feat, threebody_feat, smooth_weight


def run_interaction_only(model: MatRIS, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight):
    for mp_layer in model.interaction_block:
        node_feat, edge_feat, threebody_feat = mp_layer(
            batch_graph=batch_graph,
            node_feat=node_feat,
            edge_feat=edge_feat,
            threebody_feat=threebody_feat,
            smooth_weight=smooth_weight,
        )
    return node_feat, edge_feat, threebody_feat


def run_readout_only(model: MatRIS, batch_graph, node_feat):
    node_feat = model.readout_norm(node_feat)
    return model.energy_head(batch_graph=batch_graph, node_feat=node_feat)


def export_to_cpu(prediction: dict) -> dict:
    exported = {}
    for key, value in prediction.items():
        cpu_value = value.detach().cpu()
        if cpu_value.dtype in (torch.bfloat16, torch.float16):
            cpu_value = cpu_value.float()
        exported[key] = cpu_value.numpy()
    return exported


def profile_model_path(
    model: MatRIS,
    graph,
    task: str,
    device: str,
    precision_mode: str,
    combined_force_stress_autograd: bool = False,
) -> dict:
    result = {key: 0.0 for key in PROFILE_KEYS}
    graphs = [graph]

    with autocast_context(device, precision_mode):
        compute_stress = "s" in task
        batch_graph, result["process_graphs_ms"] = timed_call(
            device, lambda: process_graphs(graphs, compute_stress=compute_stress)
        )
        (node_feat, edge_feat, threebody_feat, smooth_weight), result["embedding_ms"] = timed_call(
            device, lambda: run_embedding_only(model, batch_graph)
        )
        (node_feat, edge_feat, threebody_feat), result["interaction_blocks_ms"] = timed_call(
            device,
            lambda: run_interaction_only(
                model, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight
            ),
        )
        total_energy, result["energy_head_ms"] = timed_call(
            device, lambda: run_readout_only(model, batch_graph, node_feat)
        )

        prediction_for_cpu = {"e": total_energy}

        if combined_force_stress_autograd and "f" in task and "s" in task:
            (force_tensor, stress_tensor), result["combined_force_stress_autograd_ms"] = timed_call(
                device,
                lambda: torch.autograd.grad(
                    total_energy.sum(),
                    [batch_graph["batch_cart_coords"], batch_graph["batch_strains"]],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                ),
            )
            prediction_for_cpu["f"] = force_tensor
            prediction_for_cpu["s"] = stress_tensor
        elif "f" in task:
            force_tensor, result["force_autograd_ms"] = timed_call(
                device,
                lambda: torch.autograd.grad(
                    total_energy.sum(),
                    [batch_graph["batch_cart_coords"]],
                    create_graph=False,
                    retain_graph="s" in task,
                )[0],
            )
            prediction_for_cpu["f"] = force_tensor

        if "s" in task and not (combined_force_stress_autograd and "f" in task):
            stress_tensor, result["stress_autograd_ms"] = timed_call(
                device,
                lambda: torch.autograd.grad(
                    total_energy.sum(),
                    [batch_graph["batch_strains"]],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )[0],
            )
            prediction_for_cpu["s"] = stress_tensor

    _, result["cpu_output_ms"] = timed_call(device, lambda: export_to_cpu(prediction_for_cpu))
    return result


def profile_once(sample_index: int, dataset: AseDBDataset, model: MatRIS, args: argparse.Namespace) -> dict:
    record = {key: 0.0 for key in PROFILE_KEYS}

    item, record["dataset_get_item_ms"] = timed_call(args.device, lambda: dataset[sample_index])
    atoms, record["dataset_get_atoms_ms"] = timed_call(args.device, lambda: dataset.get_atoms(sample_index))
    structure, record["atoms_to_structure_ms"] = timed_call(
        args.device, lambda: AseAtomsAdaptor.get_structure(atoms)
    )
    graph_cpu, record["graph_converter_ms"] = timed_call(
        args.device, lambda: model.graph_converter(structure)
    )
    graph, record["graph_to_device_ms"] = timed_call(args.device, lambda: graph_cpu.to(args.device))

    model_record = profile_model_path(
        model,
        graph,
        args.task,
        args.device,
        args.precision_mode,
        args.combined_force_stress_autograd,
    )
    for key, value in model_record.items():
        record[key] += value

    record.update(
        {
            "sample_index": sample_index,
            "sid": item["sid"] if "sid" in item else "",
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
        }
    )
    record["profile_total_ms"] = sum(record[key] for key in PROFILE_KEYS)
    return record


def run_activation_calibration(
    dataset: AseDBDataset,
    model: MatRIS,
    args: argparse.Namespace,
) -> dict:
    modules = activation_calibration_modules(model)
    if args.activation_calibration_limit <= 0 or not modules:
        return {
            "enabled": False,
            "num_modules": len(modules),
            "num_samples": 0,
        }

    indices = select_indices(
        len(dataset),
        args.activation_calibration_limit,
        args.activation_calibration_seed,
        args.sample_selection,
    )
    for module in modules:
        module.reset_activation_calibration()
        module.calibrating_activation = True

    for idx, sample_index in enumerate(indices, start=1):
        _ = profile_once(sample_index, dataset, model, args)
        print(f"[activation calibration {idx}/{len(indices)}] sample_index={sample_index}")

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
        "num_samples": len(indices),
        "sample_seed": args.activation_calibration_seed,
        "sample_selection": args.sample_selection,
        **scale_summary,
    }


def numeric_summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def summarize(records: list[dict]) -> dict:
    summary = {
        "num_structures": len(records),
        "num_atoms": numeric_summary([float(r["n_atoms"]) for r in records]),
    }
    for key in PROFILE_KEYS + ["profile_total_ms"]:
        summary[key] = numeric_summary([float(r[key]) for r in records])

    means = {key: summary[key]["mean"] for key in PROFILE_KEYS}
    overhead_keys = [
        "dataset_get_item_ms",
        "dataset_get_atoms_ms",
        "atoms_to_structure_ms",
        "graph_converter_ms",
        "graph_to_device_ms",
        "process_graphs_ms",
        "cpu_output_ms",
    ]
    core_keys = [
        "embedding_ms",
        "interaction_blocks_ms",
        "energy_head_ms",
        "force_autograd_ms",
        "stress_autograd_ms",
        "combined_force_stress_autograd_ms",
    ]
    overhead_total = sum(means[key] for key in overhead_keys)
    core_total = sum(means[key] for key in core_keys)
    total = overhead_total + core_total
    summary["category_summary"] = {
        "overhead_total_ms": overhead_total,
        "model_core_total_ms": core_total,
        "overhead_pct": overhead_total / total * 100.0 if total else 0.0,
        "model_core_pct": core_total / total * 100.0 if total else 0.0,
    }
    summary["time_ranking"] = [
        {
            "name": key,
            "mean_ms": value,
            "pct": value / total * 100.0 if total else 0.0,
            "category": "overhead" if key in overhead_keys else "model_core",
        }
        for key, value in sorted(means.items(), key=lambda item: item[1], reverse=True)
    ]
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_info = configure_precision(args.device, args.precision_mode)
    dataset = AseDBDataset(config={"src": args.dataset_src})
    indices = select_profile_candidates(
        len(dataset),
        args.limit,
        args.sample_seed,
        skip_errors=args.skip_errors,
        max_attempts=args.profile_max_attempts,
        sample_selection=args.sample_selection,
    )
    model = build_model(args)
    activation_calibration = run_activation_calibration(dataset, model, args)

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
        "quant_config": getattr(model, "quant_config", None),
        "quant_replaced_modules": getattr(model, "quant_replaced_modules", []),
        "fused_modules": getattr(model, "fused_modules", []),
        "freeze_model_params_for_efs": getattr(model, "freeze_model_params_for_efs", None),
        "compile": args.compile,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "sample_selection": args.sample_selection,
        "sample_selection_description": (
            "range(dataset_len), random.seed(sample_seed), random.shuffle(keys), keys[:limit]"
            if args.sample_selection == "teacher-aligned"
            else "random.sample(range(dataset_len), limit), sorted(indices)"
        ),
        "warmup_steps": args.warmup_steps,
        "skip_errors": args.skip_errors,
        "profile_max_attempts": args.profile_max_attempts,
        "candidate_sample_count": len(indices),
        "combined_force_stress_autograd": args.combined_force_stress_autograd,
        "activation_calibration_limit": args.activation_calibration_limit,
        "activation_calibration_seed": args.activation_calibration_seed,
        "activation_calibration": activation_calibration,
        "git_commit": get_git_commit(),
        **precision_info,
    }

    with open(output_dir / "sample_indices.json", "w", encoding="utf-8") as fp:
        json.dump(indices, fp, ensure_ascii=False, indent=2)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(run_config, fp, ensure_ascii=False, indent=2)

    failed_records = []
    warmup_target = min(args.warmup_steps, len(indices))
    warmup_used = 0
    for sample_index in indices:
        if warmup_used >= warmup_target:
            break
        try:
            _ = profile_once(sample_index, dataset, model, args)
        except Exception as exc:
            if not args.skip_errors:
                raise
            failed_records.append(
                {
                    "stage": "warmup",
                    "sample_index": int(sample_index),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[warmup skip] sample_index={sample_index} error={type(exc).__name__}: {exc}")
            continue
        warmup_used += 1
        print(f"[warmup {warmup_used}/{warmup_target}] sample_index={sample_index}")

    records = []
    target_records = args.limit if args.skip_errors and args.limit > 0 else len(indices)
    records_path = output_dir / "pipeline_profile_records.jsonl"
    with open(records_path, "w", encoding="utf-8") as out_fp:
        attempts = 0
        for sample_index in indices:
            if len(records) >= target_records:
                break
            attempts += 1
            try:
                record = profile_once(sample_index, dataset, model, args)
            except Exception as exc:
                if not args.skip_errors:
                    raise
                failed_records.append(
                    {
                        "stage": "profile",
                        "sample_index": int(sample_index),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"[profile skip] sample_index={sample_index} error={type(exc).__name__}: {exc}")
                continue
            records.append(record)
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{len(records)}/{target_records}] sample_index={sample_index} "
                f"formula={record['formula']} n_atoms={record['n_atoms']} "
                f"total={record['profile_total_ms']:.3f} ms "
                f"graph={record['graph_converter_ms']:.3f} ms "
                f"interaction={record['interaction_blocks_ms']:.3f} ms "
                f"force={record['force_autograd_ms']:.3f} ms "
                f"stress={record['stress_autograd_ms']:.3f} ms"
            )

    if len(records) < target_records:
        raise RuntimeError(
            f"Only collected {len(records)} successful records, target was {target_records}. "
            f"Increase --profile-max-attempts or inspect profile_failed_records.jsonl."
        )

    with open(output_dir / "successful_sample_indices.json", "w", encoding="utf-8") as fp:
        json.dump([record["sample_index"] for record in records], fp, ensure_ascii=False, indent=2)
    with open(output_dir / "profile_failed_records.jsonl", "w", encoding="utf-8") as fp:
        for failed in failed_records:
            fp.write(json.dumps(failed, ensure_ascii=False) + "\n")

    summary = summarize(records)
    summary["warmup_steps_used"] = warmup_used
    summary["target_records"] = target_records
    summary["profile_attempts"] = attempts
    summary["profile_successes"] = len(records)
    summary["failed_records"] = len(failed_records)
    with open(output_dir / "pipeline_profile_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    with open(output_dir / "quant_stats.json", "w", encoding="utf-8") as fp:
        json.dump(collect_quant_stats(model), fp, ensure_ascii=False, indent=2)

    print("\n=== Category Summary ===")
    for key, value in summary["category_summary"].items():
        print(f"{key}: {value:.6f}")

    print("\n=== Time Ranking ===")
    for idx, row in enumerate(summary["time_ranking"], start=1):
        print(
            f"{idx}. {row['name']}: {row['mean_ms']:.6f} ms "
            f"({row['pct']:.2f}%, {row['category']})"
        )

if __name__ == "__main__":
    main()
