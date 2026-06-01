import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_salex_lmdb_quant import (  # noqa: E402
    autocast_context,
    build_calculator,
    configure_precision,
    run_activation_calibration,
    select_group_aligned_keys,
    sync_if_needed,
    to_jsonable,
)


TARGET_SUFFIX_TO_ROLE = {
    "attn_block_line_graph": "attn_line",
    "attn_block_atom_graph": "attn_atom",
    "refine_block_line_graph": "refine_line",
    "refine_block_atom_graph": "refine_atom",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-aligned MatRISCalculator stage profiler. "
            "This keeps the ASE get_potential_energy/get_forces/get_stress path, "
            "but records calculator/model stage timings."
        )
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="none")
    parser.add_argument("--fusion-mode", default="none")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument(
        "--module-profile",
        action="store_true",
        help="Attach CUDA event hooks to top-level interaction submodules during the E/F/S path.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def timed_call(device: str, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    return result, (time.perf_counter() - start) * 1000.0


def numeric_summary(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def role_for_module(name: str) -> str | None:
    for suffix, role in TARGET_SUFFIX_TO_ROLE.items():
        if name.endswith(suffix):
            return role
    return None


class ModuleTimer:
    def __init__(self, model: torch.nn.Module, device: str) -> None:
        if device != "cuda":
            raise ValueError("--module-profile requires --device cuda")
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

    def finalize_totals(self) -> tuple[dict[str, float], dict[str, float]]:
        sync_if_needed(self.device)

        def finalize(records: list[dict]) -> dict[str, float]:
            totals: dict[str, float] = defaultdict(float)
            for record in records:
                ms = float(record["start_event"].elapsed_time(record["end_event"]))
                totals[record["role"]] += ms
            return dict(totals)

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


def profile_one(
    structures,
    graph_id: int,
    calculator,
    args: argparse.Namespace,
    module_timer: ModuleTimer | None = None,
) -> dict:
    atom = structures.get_atoms(graph_id)
    atom.calc = calculator

    record = {
        "graph_id": int(graph_id),
        "n_atoms": int(len(atom)),
    }

    with autocast_context(args.device, args.precision_mode):
        if module_timer is not None:
            module_timer.clear()
        _, record["ase_get_potential_energy_ms"] = timed_call(
            args.device,
            lambda: atom.get_potential_energy(),
        )
        if "f" in args.task:
            _, record["ase_get_forces_ms"] = timed_call(
                args.device,
                lambda: atom.get_forces(),
            )
        else:
            record["ase_get_forces_ms"] = 0.0
        if "s" in args.task:
            _, record["ase_get_stress_ms"] = timed_call(
                args.device,
                lambda: atom.get_stress(),
            )
        else:
            record["ase_get_stress_ms"] = 0.0
        if module_timer is not None:
            forward_totals, backward_totals = module_timer.finalize_totals()
            for role, ms in forward_totals.items():
                record[f"module_forward.{role}_ms"] = ms
            for role, ms in backward_totals.items():
                record[f"module_backward.{role}_ms"] = ms

    record["latency_ms"] = (
        record["ase_get_potential_energy_ms"]
        + record["ase_get_forces_ms"]
        + record["ase_get_stress_ms"]
    )
    for key, value in getattr(calculator, "last_profile", {}).items():
        if isinstance(value, (int, float)):
            record[key] = float(value)
    return record


def summarize(records: list[dict]) -> dict:
    numeric_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if isinstance(value, (int, float)) and key not in {"graph_id"}
        }
    )
    summary = {
        "num_success": len(records),
        "num_atoms": numeric_summary([float(record["n_atoms"]) for record in records]),
        "stages": {
            key: numeric_summary([float(record.get(key, 0.0)) for record in records])
            for key in numeric_keys
        },
    }
    latency_mean = summary["stages"].get("latency_ms", {}).get("mean", 0.0)
    ranking = []
    for key, stats in summary["stages"].items():
        if not key.endswith("_ms") or key == "latency_ms":
            continue
        mean = stats["mean"]
        ranking.append(
            {
                "name": key,
                "mean_ms": mean,
                "pct_of_latency": mean / latency_mean * 100.0 if latency_mean else 0.0,
            }
        )
    summary["stage_ranking"] = sorted(ranking, key=lambda item: item["mean_ms"], reverse=True)
    summary["derived"] = {
        "model_forward_pct": (
            summary["stages"].get("model_forward_total_ms", {}).get("mean", 0.0) / latency_mean * 100.0
            if latency_mean
            else 0.0
        ),
        "graph_pipeline_ms": sum(
            summary["stages"].get(key, {}).get("mean", 0.0)
            for key in (
                "ase_atoms_to_structure_ms",
                "graph_converter_ms",
                "graph_to_device_ms",
                "model.process_graphs_ms",
            )
        ),
        "force_stress_autograd_ms": summary["stages"].get(
            "model.force_stress.autograd_grad_ms", {}
        ).get("mean", 0.0),
        "magmom_head_ms": summary["stages"].get("model.magmom_head_ms", {}).get("mean", 0.0),
        "tensor_cpu_numpy_export_ms": summary["stages"].get(
            "tensor_cpu_numpy_export_ms", {}
        ).get("mean", 0.0),
    }
    if latency_mean:
        summary["derived"]["graph_pipeline_pct"] = summary["derived"]["graph_pipeline_ms"] / latency_mean * 100.0
        summary["derived"]["force_stress_autograd_pct"] = (
            summary["derived"]["force_stress_autograd_ms"] / latency_mean * 100.0
        )
        summary["derived"]["magmom_head_pct"] = summary["derived"]["magmom_head_ms"] / latency_mean * 100.0
        summary["derived"]["tensor_cpu_numpy_export_pct"] = (
            summary["derived"]["tensor_cpu_numpy_export_ms"] / latency_mean * 100.0
        )
    return summary


def main() -> None:
    args = parse_args()
    import os

    os.environ["MATRIS_CALCULATOR_STAGE_PROFILE"] = "1"
    precision_info = configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config=dict(src=args.dataset_src))
    calculator = build_calculator(args)
    module_timer = ModuleTimer(calculator.model, args.device) if args.module_profile else None
    activation_calibration = run_activation_calibration(structures, calculator, args)
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)

    warmup_keys = keys[: max(0, args.warmup_steps)]
    for graph_id in tqdm(warmup_keys, desc="warmup", leave=False):
        try:
            _ = profile_one(structures, int(graph_id), calculator, args, module_timer)
        except Exception:
            continue

    records = []
    failures = []
    for graph_id in tqdm(keys, desc="stage profile"):
        try:
            records.append(profile_one(structures, int(graph_id), calculator, args, module_timer))
        except Exception as exc:
            failures.append({"graph_id": int(graph_id), "error": str(exc)})

    payload = {
        "metadata": {
            "dataset_src": str(Path(args.dataset_src).resolve()),
            "dataset_size": len(structures),
            "limit": args.limit,
            "warmup_steps": args.warmup_steps,
            "sample_seed": args.sample_seed,
            "task": args.task,
            "device": args.device,
            "precision_mode": args.precision_mode,
            "quant_mode": args.quant_mode,
            "fusion_mode": args.fusion_mode,
            "quant_config": getattr(calculator, "quant_config", None),
            "quant_replaced_modules": getattr(calculator, "quant_replaced_modules", []),
            "fused_modules": getattr(calculator, "fused_modules", []),
            "activation_calibration": activation_calibration,
            **precision_info,
        },
        "summary": summarize(records),
        "failures": failures,
        "records": records,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=to_jsonable)
    if module_timer is not None:
        module_timer.close()

    print(json.dumps(payload["summary"]["derived"], ensure_ascii=False, indent=2))
    for item in payload["summary"]["stage_ranking"][:15]:
        print(f"{item['name']}: {item['mean_ms']:.6f} ms ({item['pct_of_latency']:.2f}%)")


if __name__ == "__main__":
    main()
