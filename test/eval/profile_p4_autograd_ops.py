import argparse
import json
import random
import sys
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
from profile_salex_pipeline import (
    run_embedding_only,
    run_interaction_only,
    run_readout_only,
)
from quant.config import get_quant_config
from quant.fusion import apply_gated_mlp_fusion
from quant.injector import apply_quant_config
from quant.runtime import freeze_model_params_for_efs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile MatRIS force/stress autograd operators on sAlex samples."
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--output-dir", default="results/p4_autograd_ops_profile")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--quant-mode", default="none")
    parser.add_argument("--fusion-mode", default="none")
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument(
        "--combined-force-stress-autograd",
        action="store_true",
        help="Profile force and stress in one autograd.grad call to match the combined efs evaluation path.",
    )
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


def build_batch_graph(dataset: AseDBDataset, model: MatRIS, sample_index: int, device: str) -> tuple[dict, dict]:
    item = dataset[sample_index]
    atoms = dataset.get_atoms(sample_index)
    structure = AseAtomsAdaptor.get_structure(atoms)
    graph_cpu = model.graph_converter(structure)
    graph = graph_cpu.to(device)
    batch_graph = process_graphs([graph], compute_stress=True)
    metadata = {
        "sample_index": sample_index,
        "sid": item["sid"] if "sid" in item else "",
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "num_directed_edges": int(batch_graph["atom_graph_dict"]["atom_graph"].shape[0]),
        "num_undirected_edges": int(batch_graph["undirected2directed"].shape[0]),
        "num_line_graph_angles": int(batch_graph["line_graph_dict"]["line_graph"].shape[0]),
    }
    return batch_graph, metadata


def forward_energy(model: MatRIS, batch_graph: dict) -> torch.Tensor:
    node_feat, edge_feat, threebody_feat, smooth_weight = run_embedding_only(model, batch_graph)
    node_feat, edge_feat, threebody_feat = run_interaction_only(
        model,
        batch_graph,
        node_feat,
        edge_feat,
        threebody_feat,
        smooth_weight,
    )
    return run_readout_only(model, batch_graph, node_feat)


def profile_grad(name: str, fn, device: str):
    sync(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.profiler.record_function(name):
            result = fn()
    sync(device)
    return result, prof


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


def run_activation_calibration(dataset: AseDBDataset, model: MatRIS, args: argparse.Namespace) -> dict:
    modules = activation_calibration_modules(model)
    if args.activation_calibration_limit <= 0 or not modules:
        return {"enabled": False, "num_modules": len(modules), "num_samples": 0}

    indices = select_indices(len(dataset), args.activation_calibration_limit, args.activation_calibration_seed)
    for module in modules:
        module.reset_activation_calibration()
        module.calibrating_activation = True

    for idx, sample_index in enumerate(indices, start=1):
        batch_graph, _ = build_batch_graph(dataset, model, sample_index, args.device)
        _ = forward_energy(model, batch_graph)
        sync(args.device)
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
        scale_summary = {"scale_mean": 0.0, "scale_min": 0.0, "scale_max": 0.0}
    return {
        "enabled": True,
        "num_modules": len(modules),
        "num_samples": len(indices),
        "sample_seed": args.activation_calibration_seed,
        **scale_summary,
    }


def event_to_dict(event) -> dict:
    device_time_total = float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0)))
    self_device_time_total = float(
        getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
    )
    return {
        "key": event.key,
        "count": int(event.count),
        "cpu_time_total_us": float(event.cpu_time_total),
        "self_cpu_time_total_us": float(event.self_cpu_time_total),
        "cuda_time_total_us": device_time_total,
        "self_cuda_time_total_us": self_device_time_total,
        "cpu_memory_usage": int(getattr(event, "cpu_memory_usage", 0)),
        "cuda_memory_usage": int(getattr(event, "cuda_memory_usage", 0)),
        "input_shapes": str(getattr(event, "input_shapes", "")),
    }


def summarize_profiler(prof, row_limit: int) -> dict:
    events = list(prof.key_averages(group_by_input_shape=False))
    shape_events = list(prof.key_averages(group_by_input_shape=True))
    by_cuda_total = sorted(
        events,
        key=lambda event: float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))),
        reverse=True,
    )[:row_limit]
    by_self_cuda = sorted(
        events,
        key=lambda event: float(
            getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
        ),
        reverse=True,
    )[:row_limit]
    by_cpu_total = sorted(events, key=lambda event: float(event.cpu_time_total), reverse=True)[:row_limit]
    by_cuda_total_shape = sorted(
        shape_events,
        key=lambda event: float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))),
        reverse=True,
    )[: row_limit * 4]
    by_self_cuda_shape = sorted(
        shape_events,
        key=lambda event: float(
            getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
        ),
        reverse=True,
    )[: row_limit * 4]
    return {
        "top_cuda_total": [event_to_dict(event) for event in by_cuda_total],
        "top_self_cuda": [event_to_dict(event) for event in by_self_cuda],
        "top_cpu_total": [event_to_dict(event) for event in by_cpu_total],
        "top_cuda_total_by_shape": [event_to_dict(event) for event in by_cuda_total_shape],
        "top_self_cuda_by_shape": [event_to_dict(event) for event in by_self_cuda_shape],
    }


def profile_sample(dataset: AseDBDataset, model: MatRIS, sample_index: int, args: argparse.Namespace) -> dict:
    batch_graph, metadata = build_batch_graph(dataset, model, sample_index, args.device)
    total_energy = forward_energy(model, batch_graph)

    if args.combined_force_stress_autograd:
        (force, stress), combined_prof = profile_grad(
            "combined_force_stress_autograd",
            lambda: torch.autograd.grad(
                total_energy.sum(),
                [batch_graph["batch_cart_coords"], batch_graph["batch_strains"]],
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            ),
            args.device,
        )
        force_norm = float(force.detach().norm().cpu())
        stress_norm = float(stress.detach().norm().cpu())
        return {
            **metadata,
            "force_norm": force_norm,
            "stress_norm": stress_norm,
            "combined_force_stress_autograd": summarize_profiler(combined_prof, args.row_limit),
        }

    force, force_prof = profile_grad(
        "force_autograd",
        lambda: torch.autograd.grad(
            total_energy.sum(),
            [batch_graph["batch_cart_coords"]],
            create_graph=False,
            retain_graph=True,
        )[0],
        args.device,
    )
    stress, stress_prof = profile_grad(
        "stress_autograd",
        lambda: torch.autograd.grad(
            total_energy.sum(),
            [batch_graph["batch_strains"]],
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0],
        args.device,
    )

    force_norm = float(force.detach().norm().cpu())
    stress_norm = float(stress.detach().norm().cpu())
    return {
        **metadata,
        "force_norm": force_norm,
        "stress_norm": stress_norm,
        "force_autograd": summarize_profiler(force_prof, args.row_limit),
        "stress_autograd": summarize_profiler(stress_prof, args.row_limit),
    }


def aggregate_events(records: list[dict], grad_key: str, table_key: str, row_limit: int) -> list[dict]:
    merged: dict[str, dict] = {}
    for record in records:
        for event in record[grad_key][table_key]:
            merge_key = event["key"]
            if table_key.endswith("_by_shape"):
                merge_key = f"{event['key']}|{event.get('input_shapes', '')}"
            item = merged.setdefault(
                merge_key,
                {
                    "key": event["key"],
                    "count": 0,
                    "cpu_time_total_us": 0.0,
                    "self_cpu_time_total_us": 0.0,
                    "cuda_time_total_us": 0.0,
                    "self_cuda_time_total_us": 0.0,
                    "cpu_memory_usage": 0,
                    "cuda_memory_usage": 0,
                    "input_shapes": event.get("input_shapes", ""),
                },
            )
            item["count"] += event["count"]
            item["cpu_time_total_us"] += event["cpu_time_total_us"]
            item["self_cpu_time_total_us"] += event["self_cpu_time_total_us"]
            item["cuda_time_total_us"] += event["cuda_time_total_us"]
            item["self_cuda_time_total_us"] += event["self_cuda_time_total_us"]
            item["cpu_memory_usage"] += event["cpu_memory_usage"]
            item["cuda_memory_usage"] += event["cuda_memory_usage"]
    sort_key = "self_cuda_time_total_us" if table_key == "top_self_cuda" else "cuda_time_total_us"
    return sorted(merged.values(), key=lambda item: item[sort_key], reverse=True)[:row_limit]


def write_markdown(output_dir: Path, payload: dict) -> None:
    lines = [
        "# P4 Force/Stress Autograd Operator Profile",
        "",
        f"- device: `{payload['device']}`",
        f"- torch: `{payload['torch_version']}`",
        f"- samples: `{payload['sample_indices']}`",
        "",
    ]
    grad_keys = ["combined_force_stress_autograd"] if payload.get("combined_force_stress_autograd") else [
        "force_autograd",
        "stress_autograd",
    ]
    for grad_key in grad_keys:
        lines.extend([f"## {grad_key}", ""])
        for table_key, title in (
            ("top_self_cuda", "Top Self CUDA"),
            ("top_cuda_total", "Top CUDA Total"),
            ("top_self_cuda_by_shape", "Top Self CUDA By Shape"),
        ):
            lines.extend(
                [
                    f"### {title}",
                    "",
                    "| op | input shapes | count | self cuda ms | cuda total ms | self cpu ms | cpu total ms |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for event in payload["aggregate"][grad_key][table_key]:
                lines.append(
                    "| {key} | `{shapes}` | {count} | {self_cuda:.3f} | {cuda:.3f} | {self_cpu:.3f} | {cpu:.3f} |".format(
                        key=event["key"],
                        shapes=event.get("input_shapes", ""),
                        count=event["count"],
                        self_cuda=event["self_cuda_time_total_us"] / 1000.0,
                        cuda=event["cuda_time_total_us"] / 1000.0,
                        self_cpu=event["self_cpu_time_total_us"] / 1000.0,
                        cpu=event["cpu_time_total_us"] / 1000.0,
                    )
                )
            lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("This profiler is intended for CUDA hotspot analysis.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AseDBDataset(config={"src": args.dataset_src})
    indices = select_indices(len(dataset), args.limit, args.sample_seed)
    model = MatRIS.load(model_name=args.model, device=args.device)
    quant_config = get_quant_config(args.quant_mode)
    replaced = apply_quant_config(model, quant_config)
    fused = apply_gated_mlp_fusion(model, args.fusion_mode)
    freeze_info = freeze_model_params_for_efs(model)
    model.eval()
    activation_calibration = run_activation_calibration(dataset, model, args)

    warmup_indices = indices[: min(args.warmup_steps, len(indices))]
    for idx in warmup_indices:
        batch_graph, _ = build_batch_graph(dataset, model, idx, args.device)
        total_energy = forward_energy(model, batch_graph)
        _ = torch.autograd.grad(
            total_energy.sum(),
            [batch_graph["batch_cart_coords"], batch_graph["batch_strains"]],
            create_graph=False,
            retain_graph=False,
        )
        sync(args.device)

    records = []
    for pos, sample_index in enumerate(indices, start=1):
        print(f"[{pos}/{len(indices)}] profiling sample_index={sample_index}")
        records.append(profile_sample(dataset, model, sample_index, args))

    payload = {
        "phase": "P4",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "model": args.model,
        "quant_mode": args.quant_mode,
        "fusion_mode": args.fusion_mode,
        "quant_config": quant_config,
        "quant_replaced_modules": replaced,
        "fused_modules": fused,
        "freeze_model_params_for_efs": freeze_info,
        "activation_calibration": activation_calibration,
        "sample_indices": indices,
        "warmup_steps": args.warmup_steps,
        "combined_force_stress_autograd": args.combined_force_stress_autograd,
        "row_limit": args.row_limit,
        "records": records,
        "aggregate":
            {
                "combined_force_stress_autograd": {
                    "top_self_cuda": aggregate_events(
                        records,
                        "combined_force_stress_autograd",
                        "top_self_cuda",
                        args.row_limit,
                    ),
                    "top_cuda_total": aggregate_events(
                        records,
                        "combined_force_stress_autograd",
                        "top_cuda_total",
                        args.row_limit,
                    ),
                    "top_cpu_total": aggregate_events(
                        records,
                        "combined_force_stress_autograd",
                        "top_cpu_total",
                        args.row_limit,
                    ),
                    "top_cuda_total_by_shape": aggregate_events(
                        records,
                        "combined_force_stress_autograd",
                        "top_cuda_total_by_shape",
                        args.row_limit,
                    ),
                    "top_self_cuda_by_shape": aggregate_events(
                        records,
                        "combined_force_stress_autograd",
                        "top_self_cuda_by_shape",
                        args.row_limit,
                    ),
                }
            }
            if args.combined_force_stress_autograd
            else {
                "force_autograd": {
                "top_self_cuda": aggregate_events(records, "force_autograd", "top_self_cuda", args.row_limit),
                "top_cuda_total": aggregate_events(records, "force_autograd", "top_cuda_total", args.row_limit),
                "top_cpu_total": aggregate_events(records, "force_autograd", "top_cpu_total", args.row_limit),
                "top_cuda_total_by_shape": aggregate_events(records, "force_autograd", "top_cuda_total_by_shape", args.row_limit),
                "top_self_cuda_by_shape": aggregate_events(records, "force_autograd", "top_self_cuda_by_shape", args.row_limit),
                },
                "stress_autograd": {
                "top_self_cuda": aggregate_events(records, "stress_autograd", "top_self_cuda", args.row_limit),
                "top_cuda_total": aggregate_events(records, "stress_autograd", "top_cuda_total", args.row_limit),
                "top_cpu_total": aggregate_events(records, "stress_autograd", "top_cpu_total", args.row_limit),
                "top_cuda_total_by_shape": aggregate_events(records, "stress_autograd", "top_cuda_total_by_shape", args.row_limit),
                "top_self_cuda_by_shape": aggregate_events(records, "stress_autograd", "top_self_cuda_by_shape", args.row_limit),
                },
            },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    write_markdown(output_dir, payload)
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
