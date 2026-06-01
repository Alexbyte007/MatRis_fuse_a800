import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
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
from quant.config import get_quant_config
from quant.fusion import apply_gated_mlp_fusion
from quant.injector import apply_quant_config
from quant.runtime import freeze_model_params_for_efs


TARGET_SUFFIX_TO_ROLE = {
    "attn_block_line_graph": "line_attention",
    "attn_block_atom_graph": "atom_attention",
    "refine_block_line_graph": "line_refinement",
    "refine_block_atom_graph": "atom_refinement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile MatRIS interaction block module hotspots.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--output-dir", default="results/p4_interaction_module_profile")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--quant-mode", default="none")
    parser.add_argument("--fusion-mode", default="none")
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
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
        _ = forward_to_energy(model, batch_graph)
        sync(args.device)
        print(f"[activation calibration {idx}/{len(indices)}] sample_index={sample_index}")

    for module in modules:
        module.calibrating_activation = False
        module.finalize_activation_calibration()

    return {
        "enabled": True,
        "num_modules": len(modules),
        "num_samples": len(indices),
        "sample_seed": args.activation_calibration_seed,
    }


def timed(device: str, fn):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def role_for_module(name: str) -> str | None:
    for suffix, role in TARGET_SUFFIX_TO_ROLE.items():
        if name.endswith(suffix):
            return role
    return None


class ModuleTimer:
    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.device = device
        self.handles = []
        self.forward_records = []
        self.backward_records = []
        self._forward_starts = {}
        self._backward_starts = {}
        for name, module in model.named_modules():
            role = role_for_module(name)
            if role is None:
                continue
            self.handles.append(module.register_forward_pre_hook(self._make_forward_pre(name, role)))
            self.handles.append(module.register_forward_hook(self._make_forward_post(name, role)))
            self.handles.append(module.register_full_backward_pre_hook(self._make_backward_pre(name, role)))
            self.handles.append(module.register_full_backward_hook(self._make_backward_post(name, role)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.forward_records.clear()
        self.backward_records.clear()
        self._forward_starts.clear()
        self._backward_starts.clear()

    def finalize_records(self) -> tuple[list[dict], list[dict]]:
        sync(self.device)

        def finalize(records: list[dict]) -> list[dict]:
            out = []
            for record in records:
                item = {key: value for key, value in record.items() if key not in ("start_event", "end_event")}
                item["ms"] = float(record["start_event"].elapsed_time(record["end_event"]))
                out.append(item)
            return out

        return finalize(self.forward_records), finalize(self.backward_records)

    def _make_forward_pre(self, name: str, role: str):
        def hook(module, inputs):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._forward_starts[id(module)] = start

        return hook

    def _make_forward_post(self, name: str, role: str):
        def hook(module, inputs, output):
            start = self._forward_starts.pop(id(module), None)
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.forward_records.append(
                {
                    "module": name,
                    "role": role,
                    "start_event": start,
                    "end_event": end,
                }
            )

        return hook

    def _make_backward_pre(self, name: str, role: str):
        def hook(module, grad_output):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._backward_starts[id(module)] = start

        return hook

    def _make_backward_post(self, name: str, role: str):
        def hook(module, grad_input, grad_output):
            start = self._backward_starts.pop(id(module), None)
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.backward_records.append(
                {
                    "module": name,
                    "role": role,
                    "start_event": start,
                    "end_event": end,
                }
            )

        return hook


def build_batch_graph(dataset: AseDBDataset, model: MatRIS, sample_index: int, device: str):
    item = dataset[sample_index]
    atoms = dataset.get_atoms(sample_index)
    structure = AseAtomsAdaptor.get_structure(atoms)
    graph = model.graph_converter(structure).to(device)
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


def forward_to_energy(model: MatRIS, batch_graph: dict):
    node_feat, edge_feat, threebody_feat, smooth_weight = run_embedding_only(model, batch_graph)
    (node_feat, edge_feat, threebody_feat), interaction_ms = timed(
        "cuda",
        lambda: run_interaction_only(
            model, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight
        ),
    )
    total_energy = run_readout_only(model, batch_graph, node_feat)
    return total_energy, interaction_ms


def profile_interaction_ops(model: MatRIS, batch_graph: dict, row_limit: int) -> dict:
    node_feat, edge_feat, threebody_feat, smooth_weight = run_embedding_only(model, batch_graph)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    sync("cuda")
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with torch.profiler.record_function("interaction_forward"):
            _ = run_interaction_only(model, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight)
    sync("cuda")
    events = list(prof.key_averages(group_by_input_shape=False))
    events = [event for event in events if event.key != "interaction_forward"]
    events = sorted(
        events,
        key=lambda event: float(getattr(event, "self_device_time_total", 0.0)),
        reverse=True,
    )[:row_limit]
    return [event_to_dict(event) for event in events]


def event_to_dict(event) -> dict:
    return {
        "key": event.key,
        "count": int(event.count),
        "self_cuda_time_total_us": float(getattr(event, "self_device_time_total", 0.0)),
        "cuda_time_total_us": float(getattr(event, "device_time_total", 0.0)),
        "self_cpu_time_total_us": float(event.self_cpu_time_total),
        "cpu_time_total_us": float(event.cpu_time_total),
    }


def profile_sample(dataset: AseDBDataset, model: MatRIS, timer: ModuleTimer, sample_index: int, args):
    batch_graph, metadata = build_batch_graph(dataset, model, sample_index, args.device)
    timer.clear()
    total_energy, interaction_ms = forward_to_energy(model, batch_graph)
    _, combined_grad_ms = timed(
        args.device,
        lambda: torch.autograd.grad(
            total_energy.sum(),
            [batch_graph["batch_cart_coords"], batch_graph["batch_strains"]],
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        ),
    )
    forward_records, backward_records = timer.finalize_records()
    op_profile = profile_interaction_ops(model, batch_graph, args.row_limit)
    return {
        **metadata,
        "interaction_forward_ms": interaction_ms,
        "combined_grad_ms": combined_grad_ms,
        "module_forward_records": forward_records,
        "module_backward_records": backward_records,
        "interaction_forward_top_self_cuda": op_profile,
    }


def aggregate_module_records(records: list[dict], key: str) -> dict:
    by_role = defaultdict(list)
    by_module = defaultdict(list)
    for record in records:
        for item in record[key]:
            by_role[item["role"]].append(float(item["ms"]))
            by_module[item["module"]].append(float(item["ms"]))
    return {
        "by_role": {
            role: numeric(values)
            for role, values in sorted(by_role.items(), key=lambda item: sum(item[1]), reverse=True)
        },
        "by_module": {
            module: numeric(values)
            for module, values in sorted(by_module.items(), key=lambda item: sum(item[1]), reverse=True)
        },
    }


def aggregate_ops(records: list[dict], row_limit: int) -> list[dict]:
    merged = {}
    for record in records:
        for event in record["interaction_forward_top_self_cuda"]:
            item = merged.setdefault(
                event["key"],
                {
                    "key": event["key"],
                    "count": 0,
                    "self_cuda_time_total_us": 0.0,
                    "cuda_time_total_us": 0.0,
                    "self_cpu_time_total_us": 0.0,
                    "cpu_time_total_us": 0.0,
                },
            )
            item["count"] += event["count"]
            item["self_cuda_time_total_us"] += event["self_cuda_time_total_us"]
            item["cuda_time_total_us"] += event["cuda_time_total_us"]
            item["self_cpu_time_total_us"] += event["self_cpu_time_total_us"]
            item["cpu_time_total_us"] += event["cpu_time_total_us"]
    return sorted(merged.values(), key=lambda item: item["self_cuda_time_total_us"], reverse=True)[:row_limit]


def numeric(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "sum": sum(values),
        "count": len(values),
    }


def summarize(records: list[dict], row_limit: int) -> dict:
    return {
        "interaction_forward_ms": numeric([float(r["interaction_forward_ms"]) for r in records]),
        "combined_grad_ms": numeric([float(r["combined_grad_ms"]) for r in records]),
        "module_forward": aggregate_module_records(records, "module_forward_records"),
        "module_backward": aggregate_module_records(records, "module_backward_records"),
        "interaction_forward_top_self_cuda": aggregate_ops(records, row_limit),
    }


def write_markdown(output_dir: Path, payload: dict) -> None:
    summary = payload["summary"]
    lines = [
        "# P4 Interaction Module Profile",
        "",
        f"- device: `{payload['device']}`",
        f"- samples: `{payload['sample_indices']}`",
        "",
        "## Totals",
        "",
        "| metric | mean ms | std | min | max |",
        "|---|---:|---:|---:|---:|",
        "| interaction_forward_ms | {mean:.6f} | {std:.6f} | {min:.6f} | {max:.6f} |".format(
            **summary["interaction_forward_ms"]
        ),
        "| combined_grad_ms | {mean:.6f} | {std:.6f} | {min:.6f} | {max:.6f} |".format(
            **summary["combined_grad_ms"]
        ),
        "",
    ]
    for title, table in (
        ("Module Forward By Role", summary["module_forward"]["by_role"]),
        ("Module Backward By Role", summary["module_backward"]["by_role"]),
    ):
        lines.extend([f"## {title}", "", "| role | mean ms/call | calls | summed ms |", "|---|---:|---:|---:|"])
        for role, row in table.items():
            lines.append(f"| {role} | {row['mean']:.6f} | {row['count']} | {row['sum']:.6f} |")
        lines.append("")
    lines.extend(
        [
            "## Interaction Forward Top Self CUDA Ops",
            "",
            "| op | count | self cuda ms | cuda total ms | self cpu ms | cpu total ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for event in summary["interaction_forward_top_self_cuda"]:
        lines.append(
            "| {key} | {count} | {self_cuda:.3f} | {cuda:.3f} | {self_cpu:.3f} | {cpu:.3f} |".format(
                key=event["key"],
                count=event["count"],
                self_cuda=event["self_cuda_time_total_us"] / 1000.0,
                cuda=event["cuda_time_total_us"] / 1000.0,
                self_cpu=event["self_cpu_time_total_us"] / 1000.0,
                cpu=event["cpu_time_total_us"] / 1000.0,
            )
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    timer = ModuleTimer(model, args.device)

    try:
        for idx in indices[: min(args.warmup_steps, len(indices))]:
            batch_graph, _ = build_batch_graph(dataset, model, idx, args.device)
            total_energy, _ = forward_to_energy(model, batch_graph)
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
            record = profile_sample(dataset, model, timer, sample_index, args)
            records.append(record)
            print(
                f"[{pos}/{len(indices)}] sample_index={sample_index} "
                f"interaction={record['interaction_forward_ms']:.3f} ms "
                f"combined_grad={record['combined_grad_ms']:.3f} ms"
            )
    finally:
        timer.close()

    payload = {
        "phase": "P4a",
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
        "row_limit": args.row_limit,
        "records": records,
        "summary": summarize(records, args.row_limit),
    }
    with output_dir.joinpath("summary.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    write_markdown(output_dir, payload)
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
