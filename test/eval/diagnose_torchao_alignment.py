import argparse
import fnmatch
import json
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

from matris.applications.base import MatRISCalculator
from matris.graph import RadiusGraph
from matris.model.model import MatRIS
from quant.config import get_quant_config
from quant.injector import apply_quant_config
from quant.layers import FakeQuantLinear, TorchAOInt8WeightOnlyLinear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether torchao int8 weight-only matches MatRIS fake W8A32 "
            "semantics and whether failures come from scale/layout, shape/kernel, "
            "autograd, or dtype/autocast interaction."
        )
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fake-quant-mode", default="p0_line_gate_stable_w8a32")
    parser.add_argument("--torchao-quant-mode", default="p0_line_gate_stable_w8a32_torchao")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["e", "ef"],
        choices=["e", "ef", "efs"],
        help="Small end-to-end task probes. Keep short; ef/efs test autograd sensitivity.",
    )
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-captures-per-module", type=int, default=1)
    parser.add_argument("--max-modules", type=int, default=48)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument(
        "--output-json",
        default="/home/lht/lab/MatRIS/results/torchao_alignment/diagnosis.json",
    )
    return parser.parse_args()


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def select_indices(dataset_len: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= dataset_len:
        return list(range(dataset_len))
    rng = random.Random(seed)
    indices = rng.sample(range(dataset_len), limit)
    indices.sort()
    return indices


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def expand_target_linear_names(model: torch.nn.Module, patterns: list[str]) -> list[str]:
    selected: list[str] = []
    selected_set: set[str] = set()
    modules = dict(model.named_modules())
    for module_name, module in modules.items():
        if not matches_any(module_name, patterns):
            continue
        if isinstance(module, torch.nn.Linear):
            if module_name not in selected_set:
                selected.append(module_name)
                selected_set.add(module_name)
            continue
        prefix = f"{module_name}."
        for child_name, child in modules.items():
            if child_name.startswith(prefix) and isinstance(child, torch.nn.Linear):
                if child_name not in selected_set:
                    selected.append(child_name)
                    selected_set.add(child_name)
    return selected


def tensor_summary(tensor: torch.Tensor) -> dict:
    t = tensor.detach()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "abs_max": float(t.float().abs().max().cpu()) if t.numel() else 0.0,
        "mean": float(t.float().mean().cpu()) if t.numel() else 0.0,
        "std": float(t.float().std(unbiased=False).cpu()) if t.numel() else 0.0,
    }


def compare_tensors(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    cand = candidate.detach().float()
    ref = reference.detach().float()
    diff = cand - ref
    ref_abs = ref.abs()
    return {
        "mae": float(diff.abs().mean().cpu()) if diff.numel() else 0.0,
        "max_abs": float(diff.abs().max().cpu()) if diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).cpu()) if diff.numel() else 0.0,
        "relative_mae": float((diff.abs().mean() / ref_abs.mean().clamp_min(1e-12)).cpu())
        if diff.numel()
        else 0.0,
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                cand.reshape(1, -1),
                ref.reshape(1, -1),
                dim=1,
            ).cpu()[0]
        )
        if diff.numel()
        else 1.0,
    }


def timed_forward(module: torch.nn.Module, x: torch.Tensor, device: str, repeats: int) -> dict:
    if repeats <= 0:
        return {}
    with torch.no_grad():
        for _ in range(min(3, repeats)):
            _ = module(x)
        sync_if_needed(device)
        start = time.perf_counter()
        for _ in range(repeats):
            _ = module(x)
        sync_if_needed(device)
    return {"latency_ms_mean": (time.perf_counter() - start) * 1000.0 / repeats}


def make_torchao_wrapper(original: torch.nn.Linear, module_name: str):
    try:
        return TorchAOInt8WeightOnlyLinear(original, module_name=module_name).to(original.weight.device)
    except Exception as exc:
        return exc


def collect_target_inputs(
    model: MatRIS,
    dataset: AseDBDataset,
    indices: list[int],
    target_names: list[str],
    device: str,
    max_captures_per_module: int,
) -> dict[str, list[torch.Tensor]]:
    captures: dict[str, list[torch.Tensor]] = {name: [] for name in target_names}
    target_set = set(target_names)
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if len(captures[name]) >= max_captures_per_module:
                return
            if not inputs:
                return
            captures[name].append(inputs[0].detach().to(device))

        return hook

    for name, module in model.named_modules():
        if name in target_set:
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.enable_grad():
        for sample_index in indices:
            atoms = dataset.get_atoms(sample_index)
            structure = AseAtomsAdaptor.get_structure(atoms)
            graph = model.graph_converter(structure).to(device)
            graphs = [graph] if isinstance(graph, RadiusGraph) else graph
            _ = model(graphs, task="e", is_training=False)
            if all(len(values) >= max_captures_per_module for values in captures.values()):
                break

    for handle in handles:
        handle.remove()
    return {name: values for name, values in captures.items() if values}


def diagnose_module_alignment(
    model: MatRIS,
    captures: dict[str, list[torch.Tensor]],
    device: str,
    timing_repeats: int,
) -> list[dict]:
    modules = dict(model.named_modules())
    records = []
    for module_name, inputs in captures.items():
        original = modules[module_name]
        if not isinstance(original, torch.nn.Linear):
            continue
        fake = FakeQuantLinear(original, module_name=module_name).to(device).eval()
        torchao = make_torchao_wrapper(original, module_name)

        torchao_error = None
        torchao_info = {}
        if isinstance(torchao, Exception):
            torchao_error = f"{type(torchao).__name__}: {torchao}"
        else:
            torchao.eval()
            torchao_info = {
                "wrapped_linear_type": type(torchao.linear).__name__,
                "weight_type": type(torchao.linear.weight).__name__,
                "weight_dtype": str(getattr(torchao.linear.weight, "dtype", "unknown")),
                "state_dict_keys": list(torchao.state_dict().keys())[:12],
            }

        for capture_idx, x in enumerate(inputs):
            with torch.no_grad():
                ref = original(x.float())
                fake_out = fake(x)
                record = {
                    "module_name": module_name,
                    "capture_idx": capture_idx,
                    "input": tensor_summary(x),
                    "weight": tensor_summary(original.weight),
                    "bias": None if original.bias is None else tensor_summary(original.bias),
                    "fp32_output": tensor_summary(ref),
                    "fake_output": tensor_summary(fake_out),
                    "fake_vs_fp32": compare_tensors(fake_out, ref),
                    "timing": {
                        "fp32": timed_forward(original, x.float(), device, timing_repeats),
                        "fake": timed_forward(fake, x, device, timing_repeats),
                    },
                    "torchao": {
                        "available": torchao_error is None,
                        "error": torchao_error,
                        **torchao_info,
                    },
                }
                if torchao_error is None:
                    torchao_out = torchao(x)
                    record["torchao_output"] = tensor_summary(torchao_out)
                    record["torchao_vs_fp32"] = compare_tensors(torchao_out, ref)
                    record["torchao_vs_fake"] = compare_tensors(torchao_out, fake_out)
                    record["timing"]["torchao"] = timed_forward(torchao, x, device, timing_repeats)

                    if x.is_floating_point():
                        bf16_x = x.to(torch.bfloat16)
                        bf16_out = torchao(bf16_x)
                        record["torchao_bf16_input_probe"] = {
                            "input_dtype": str(bf16_x.dtype),
                            "output_dtype": str(bf16_out.dtype),
                            "vs_fp32_input_output": compare_tensors(bf16_out, torchao_out),
                            "note": (
                                "TorchAOInt8WeightOnlyLinear.forward currently casts x.float(), "
                                "so autocast/bf16 inputs are expected to be promoted to fp32 before the wrapped linear."
                            ),
                        }
                records.append(record)
    return records


def metric_mean(records: list[dict], key: str) -> float:
    values = []
    for record in records:
        current = record
        ok = True
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                ok = False
                break
            current = current[part]
        if ok and isinstance(current, (int, float)):
            values.append(float(current))
    return statistics.mean(values) if values else 0.0


def calc_component_metrics(pred, ref) -> dict:
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref_arr = np.asarray(ref, dtype=np.float64).reshape(-1)
    diff = pred_arr - ref_arr
    return {
        "mae": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }


def run_prediction(dataset: AseDBDataset, sample_index: int, calc: MatRISCalculator, task: str, device: str) -> dict:
    atoms = dataset.get_atoms(sample_index)
    atoms.calc = calc
    sync_if_needed(device)
    start = time.perf_counter()
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces() if "f" in task else None
    stress = atoms.get_stress() if "s" in task else None
    sync_if_needed(device)
    return {
        "sample_index": sample_index,
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "energy": energy,
        "forces": None if forces is None else forces.tolist(),
        "stress": None if stress is None else stress.tolist(),
    }


def build_calc(model_name: str, task: str, device: str, quant_mode: str) -> MatRISCalculator:
    calc = MatRISCalculator(model=model_name, task=task, device=device)
    quant_config = get_quant_config(quant_mode)
    replaced = apply_quant_config(calc.model, quant_config)
    calc.quant_config = quant_config
    calc.quant_replaced_modules = replaced
    return calc


def diagnose_end_to_end(
    dataset: AseDBDataset,
    indices: list[int],
    model_name: str,
    device: str,
    fake_mode: str,
    torchao_mode: str,
    tasks: list[str],
) -> dict:
    result = {}
    for task in tasks:
        task_records = {"fp32": [], "fake": [], "torchao": [], "comparisons": []}
        calcs = {
            "fp32": build_calc(model_name, task, device, "none"),
            "fake": build_calc(model_name, task, device, fake_mode),
        }
        try:
            calcs["torchao"] = build_calc(model_name, task, device, torchao_mode)
            torchao_error = None
        except Exception as exc:
            calcs["torchao"] = None
            torchao_error = f"{type(exc).__name__}: {exc}"

        for sample_index in indices:
            fp32 = run_prediction(dataset, sample_index, calcs["fp32"], task, device)
            fake = run_prediction(dataset, sample_index, calcs["fake"], task, device)
            task_records["fp32"].append(fp32)
            task_records["fake"].append(fake)
            comparison = {
                "sample_index": sample_index,
                "fake_vs_fp32": {
                    "energy": calc_component_metrics([fake["energy"]], [fp32["energy"]]),
                },
            }
            if "f" in task:
                comparison["fake_vs_fp32"]["forces"] = calc_component_metrics(fake["forces"], fp32["forces"])
            if "s" in task:
                comparison["fake_vs_fp32"]["stress"] = calc_component_metrics(fake["stress"], fp32["stress"])

            if calcs["torchao"] is not None:
                torchao = run_prediction(dataset, sample_index, calcs["torchao"], task, device)
                task_records["torchao"].append(torchao)
                comparison["torchao_vs_fp32"] = {
                    "energy": calc_component_metrics([torchao["energy"]], [fp32["energy"]]),
                }
                comparison["torchao_vs_fake"] = {
                    "energy": calc_component_metrics([torchao["energy"]], [fake["energy"]]),
                }
                if "f" in task:
                    comparison["torchao_vs_fp32"]["forces"] = calc_component_metrics(torchao["forces"], fp32["forces"])
                    comparison["torchao_vs_fake"]["forces"] = calc_component_metrics(torchao["forces"], fake["forces"])
                if "s" in task:
                    comparison["torchao_vs_fp32"]["stress"] = calc_component_metrics(torchao["stress"], fp32["stress"])
                    comparison["torchao_vs_fake"]["stress"] = calc_component_metrics(torchao["stress"], fake["stress"])
            task_records["comparisons"].append(comparison)

        task_records["torchao_build_error"] = torchao_error
        result[task] = task_records
    return result


def summarize_end_to_end(end_to_end: dict) -> dict:
    summary = {}
    for task, task_records in end_to_end.items():
        rows = task_records["comparisons"]
        task_summary = {}
        for pair in ("fake_vs_fp32", "torchao_vs_fp32", "torchao_vs_fake"):
            pair_summary = {}
            for quantity in ("energy", "forces", "stress"):
                values = [
                    row[pair][quantity]["mae"]
                    for row in rows
                    if pair in row and quantity in row[pair]
                ]
                if values:
                    pair_summary[f"{quantity}_mae_mean"] = float(statistics.mean(values))
            if pair_summary:
                task_summary[pair] = pair_summary
        for backend in ("fp32", "fake", "torchao"):
            latencies = [row["latency_ms"] for row in task_records.get(backend, [])]
            if latencies:
                task_summary[f"{backend}_latency_ms_mean"] = float(statistics.mean(latencies))
        summary[task] = task_summary
    return summary


def make_findings(module_records: list[dict], end_to_end_summary: dict, torchao_mode: str) -> list[str]:
    findings = []
    torchao_unavailable = [r for r in module_records if not r["torchao"]["available"]]
    if torchao_unavailable:
        findings.append(
            "torchao wrapper failed for at least one target Linear; inspect module_records[*].torchao.error."
        )

    torchao_vs_fake = metric_mean(module_records, "torchao_vs_fake.mae")
    fake_vs_fp32 = metric_mean(module_records, "fake_vs_fp32.mae")
    torchao_vs_fp32 = metric_mean(module_records, "torchao_vs_fp32.mae")
    if torchao_vs_fake > max(fake_vs_fp32, 1e-12) * 2.0:
        findings.append(
            "torchao offline outputs differ from fake W8A32 much more than fake differs from FP32; "
            "scale/layout/kernel semantics are likely not aligned."
        )
    elif torchao_vs_fp32 > 0:
        findings.append(
            "torchao offline outputs are close to fake W8A32 at the sampled Linear inputs; "
            "large ef/efs degradation would then point more toward autograd/backward or graph-level effects."
        )

    slow_modules = []
    for record in module_records:
        fp32_ms = record.get("timing", {}).get("fp32", {}).get("latency_ms_mean")
        torchao_ms = record.get("timing", {}).get("torchao", {}).get("latency_ms_mean")
        if fp32_ms and torchao_ms and torchao_ms > fp32_ms * 1.1:
            slow_modules.append(record["module_name"])
    if slow_modules:
        findings.append(
            "torchao is slower than FP32 on sampled Linear shapes; small/irregular MatRIS feature matrices "
            "or fallback/dequant overhead may dominate."
        )

    bf16_promoted = [
        r
        for r in module_records
        if r.get("torchao_bf16_input_probe", {}).get("output_dtype") == r.get("torchao_output", {}).get("dtype")
    ]
    if bf16_promoted:
        findings.append(
            "bf16 input probes do not create a distinct low-precision torchao path in this wrapper; "
            "the current TorchAOInt8WeightOnlyLinear.forward casts inputs with x.float()."
        )

    e_force_gap = None
    if "e" in end_to_end_summary and "ef" in end_to_end_summary:
        e_err = (
            end_to_end_summary["e"]
            .get("torchao_vs_fp32", {})
            .get("energy_mae_mean", 0.0)
        )
        ef_force = (
            end_to_end_summary["ef"]
            .get("torchao_vs_fp32", {})
            .get("forces_mae_mean", 0.0)
        )
        e_force_gap = ef_force > max(e_err, 1e-12) * 100.0
    if e_force_gap:
        findings.append(
            "energy-only drift is much smaller than ef force drift; torchao may be acceptable for forward "
            "values but problematic for force/stress autograd."
        )

    if not findings:
        findings.append(
            f"No obvious torchao mismatch was detected for {torchao_mode} on this small probe; "
            "increase --limit/--max-captures-per-module or inspect kernel dispatch with a profiler."
        )
    return findings


def main() -> None:
    args = parse_args()
    dataset = AseDBDataset(config={"src": args.dataset_src})
    indices = select_indices(len(dataset), args.limit, args.sample_seed)

    fake_config = get_quant_config(args.fake_quant_mode)
    torchao_config = get_quant_config(args.torchao_quant_mode)
    if fake_config is None or torchao_config is None:
        raise ValueError("fake and torchao quant modes must both be non-none.")
    if fake_config.get("targets") != torchao_config.get("targets"):
        print(
            "[warn] fake and torchao target lists differ; module alignment will use their intersection-like "
            "expanded names from fake targets, and end-to-end will use each full mode."
        )

    model = MatRIS.load(model_name=args.model, device=args.device)
    model.eval()
    target_names = expand_target_linear_names(model, fake_config["targets"])[: args.max_modules]
    captures = collect_target_inputs(
        model,
        dataset,
        indices,
        target_names,
        args.device,
        args.max_captures_per_module,
    )
    module_records = diagnose_module_alignment(
        model,
        captures,
        args.device,
        args.timing_repeats,
    )

    end_to_end = diagnose_end_to_end(
        dataset,
        indices,
        args.model,
        args.device,
        args.fake_quant_mode,
        args.torchao_quant_mode,
        args.tasks,
    )
    end_to_end_summary = summarize_end_to_end(end_to_end)

    payload = {
        "config": {
            "dataset_src": str(Path(args.dataset_src).resolve()),
            "dataset_size": len(dataset),
            "indices": indices,
            "model": args.model,
            "device": args.device,
            "fake_quant_mode": args.fake_quant_mode,
            "torchao_quant_mode": args.torchao_quant_mode,
            "tasks": args.tasks,
            "limit": args.limit,
            "max_captures_per_module": args.max_captures_per_module,
            "max_modules": args.max_modules,
            "timing_repeats": args.timing_repeats,
        },
        "target_linear_count": len(target_names),
        "captured_module_count": len(captures),
        "module_records": module_records,
        "module_summary": {
            "fake_vs_fp32_mae_mean": metric_mean(module_records, "fake_vs_fp32.mae"),
            "torchao_vs_fp32_mae_mean": metric_mean(module_records, "torchao_vs_fp32.mae"),
            "torchao_vs_fake_mae_mean": metric_mean(module_records, "torchao_vs_fake.mae"),
            "fp32_latency_ms_mean": metric_mean(module_records, "timing.fp32.latency_ms_mean"),
            "fake_latency_ms_mean": metric_mean(module_records, "timing.fake.latency_ms_mean"),
            "torchao_latency_ms_mean": metric_mean(module_records, "timing.torchao.latency_ms_mean"),
        },
        "end_to_end_summary": end_to_end_summary,
        "end_to_end_records": end_to_end,
    }
    payload["findings"] = make_findings(module_records, end_to_end_summary, args.torchao_quant_mode)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print("\n=== TorchAO Alignment Diagnosis ===")
    print(json.dumps(payload["module_summary"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["end_to_end_summary"], ensure_ascii=False, indent=2))
    print("\nFindings:")
    for finding in payload["findings"]:
        print(f"- {finding}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
