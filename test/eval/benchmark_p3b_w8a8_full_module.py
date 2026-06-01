import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from fairchem.core.datasets import AseDBDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications.base import MatRISCalculator
from quant.fusion import apply_gated_mlp_fusion


FUSION_MODE = "line_all_candidate_gated_mlp_second_fused_fp32"

CANDIDATE_BRANCHES = {
    "a8_line_nonlinear_core_gate": (
        "attn_block_line_graph.edge_nonlinear_update",
        "refine_block_line_graph.edge_nonlinear_update",
        "attn_block_line_graph.node_nonlinear_update",
    ),
    "a8_line_edge_core_gate": (
        "attn_block_line_graph.edge_nonlinear_update",
        "refine_block_line_graph.edge_nonlinear_update",
    ),
    "a8_line_node_core_gate": (
        "attn_block_line_graph.node_nonlinear_update",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "P3b-2 full W8A8 single-module microbenchmark for MatRIS fused line "
            "GatedMLP shapes."
        )
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--output-dir", default="results/p3b_w8a8_full_module_microbench")
    parser.add_argument(
        "--shape-gate-source",
        default="results/p3b_w8a8_gemm_microbench/summary.json",
        help="P3b-1 summary.json. Shapes with pass_1p3x=true are allowed to use W8A8.",
    )
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="e", choices=("e", "ef", "efs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--speedup-threshold", type=float, default=1.3)
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


def branch_name(module_name: str) -> str | None:
    for branches in CANDIDATE_BRANCHES.values():
        for branch in branches:
            if branch in module_name:
                return branch
    return None


def point_name(module_name: str) -> str | None:
    if module_name.endswith(".fused_first"):
        return "fused_first"
    if module_name.endswith(".fused_second"):
        return "fused_second"
    return None


def shape_key(row: dict) -> tuple:
    return (
        row["candidate"],
        row["branch"],
        row["fused_point"],
        int(row["M"]),
        int(row["K"]),
        int(row["N"]),
    )


def load_shape_gate(path: str) -> set[tuple]:
    source = Path(path)
    if not source.exists():
        return set()
    with source.open() as fp:
        data = json.load(fp)
    return {shape_key(row) for row in data.get("results", []) if row.get("pass_1p3x")}


def collect_shapes(calc: MatRISCalculator, dataset: AseDBDataset, indices: list[int], device: str) -> list[dict]:
    captured: dict[tuple[str, str, str, int, int, int], dict] = {}
    hooks = []

    def make_hook(module_name: str):
        def hook(module, inputs):
            point = point_name(module_name)
            branch = branch_name(module_name)
            if point is None or branch is None:
                return
            x = inputs[0]
            if x.ndim < 2:
                return
            m = int(x.reshape(-1, x.shape[-1]).shape[0])
            k = int(x.shape[-1])
            n = int(module.weight.shape[0])
            for candidate, branches in CANDIDATE_BRANCHES.items():
                if branch not in branches:
                    continue
                key = (candidate, branch, point, m, k, n)
                captured[key] = {
                    "candidate": candidate,
                    "branch": branch,
                    "fused_point": point,
                    "M": m,
                    "K": k,
                    "N": n,
                    "source_module": module_name,
                }

        return hook

    for name, module in calc.model.named_modules():
        if point_name(name) is not None and branch_name(name) is not None:
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        for sample_index in indices:
            atoms = dataset.get_atoms(sample_index)
            atoms.calc = calc
            _ = atoms.get_potential_energy()
            if "f" in calc.task:
                _ = atoms.get_forces()
            if "s" in calc.task:
                _ = atoms.get_stress()
            sync(device)
    finally:
        for hook in hooks:
            hook.remove()

    return sorted(
        captured.values(),
        key=lambda x: (x["candidate"], x["branch"], x["fused_point"], x["M"], x["K"], x["N"]),
    )


def measure_ms(fn, device: str, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    sync(device)

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def quantize_weight_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 127
    weight_fp32 = weight.float().contiguous()
    scale = weight_fp32.abs().amax(dim=1).clamp_min(1e-12) / qmax
    q_weight = torch.round(weight_fp32 / scale.reshape(-1, 1)).clamp(-qmax, qmax).to(torch.int8)
    return q_weight.t().contiguous(), scale.contiguous()


def dynamic_quantize_activation_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 127
    x_fp32 = x.float()
    scale = x_fp32.abs().amax().clamp_min(1e-12) / qmax
    q_x = torch.round(x_fp32 / scale).clamp(-qmax, qmax).to(torch.int8)
    return q_x.contiguous(), scale


def w8a8_full_linear(
    x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    q_x, activation_scale = dynamic_quantize_activation_per_tensor(x)
    acc = torch._int_mm(q_x, q_weight_t)
    combined_scale = activation_scale * weight_scale.reshape(1, -1)
    out = acc.float() * combined_scale
    if bias is not None:
        out = out + bias
    return out


def reference_error(
    x: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[float, float]:
    q_x, activation_scale = dynamic_quantize_activation_per_tensor(x)
    w8a8_out = torch._int_mm(q_x, q_weight_t).float()
    w8a8_out = w8a8_out * (activation_scale * weight_scale.reshape(1, -1))
    if bias is not None:
        w8a8_out = w8a8_out + bias

    dq_x = q_x.float() * activation_scale
    dq_weight = q_weight_t.t().float() * weight_scale.reshape(-1, 1)
    ref_out = F.linear(dq_x, dq_weight, bias)
    diff = (w8a8_out - ref_out).abs()
    return float(diff.max().detach().cpu()), float(diff.mean().detach().cpu())


def benchmark_shape(
    shape: dict,
    modules: dict[str, torch.nn.Module],
    gate_pass_shapes: set[tuple],
    args: argparse.Namespace,
) -> dict:
    if args.device != "cuda":
        raise RuntimeError("P3b-2 W8A8 full-module benchmark requires CUDA.")
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("torch._int_mm is unavailable in this PyTorch build.")

    module = modules[shape["source_module"]]
    m = int(shape["M"])
    k = int(shape["K"])
    weight = module.weight.detach().float().contiguous()
    bias = module.bias.detach().float().contiguous() if module.bias is not None else None
    q_weight_t, weight_scale = quantize_weight_per_channel(weight)
    x = torch.randn((m, k), device=args.device, dtype=torch.float32)
    gate_enabled = shape_key(shape) in gate_pass_shapes
    max_abs_diff, mean_abs_diff = reference_error(x, q_weight_t, weight_scale, bias)

    fp32_mean, fp32_std = measure_ms(
        lambda: F.linear(x, weight, bias),
        args.device,
        args.warmup,
        args.repeat,
    )
    w8a8_mean, w8a8_std = measure_ms(
        lambda: w8a8_full_linear(x, q_weight_t, weight_scale, bias),
        args.device,
        args.warmup,
        args.repeat,
    )
    gated_mean, gated_std = measure_ms(
        lambda: (
            w8a8_full_linear(x, q_weight_t, weight_scale, bias)
            if gate_enabled
            else F.linear(x, weight, bias)
        ),
        args.device,
        args.warmup,
        args.repeat,
    )
    full_speedup = fp32_mean / w8a8_mean if w8a8_mean > 0 else 0.0
    gated_speedup = fp32_mean / gated_mean if gated_mean > 0 else 0.0

    return {
        **shape,
        "bias": bias is not None,
        "activation_quant": "dynamic_per_tensor_symmetric_int8",
        "weight_quant": "per_output_channel_symmetric_int8",
        "accumulation": "int32",
        "output_dtype": "fp32",
        "max_abs_diff_vs_fake_quant_ref": max_abs_diff,
        "mean_abs_diff_vs_fake_quant_ref": mean_abs_diff,
        "shape_gate_enabled": gate_enabled,
        "shape_gate_source": args.shape_gate_source,
        "fp32_fused_latency_ms_mean": fp32_mean,
        "fp32_fused_latency_ms_std": fp32_std,
        "w8a8_full_latency_ms_mean": w8a8_mean,
        "w8a8_full_latency_ms_std": w8a8_std,
        "w8a8_full_speedup": full_speedup,
        "w8a8_full_pass_1p3x": full_speedup >= args.speedup_threshold,
        "shape_gated_latency_ms_mean": gated_mean,
        "shape_gated_latency_ms_std": gated_std,
        "shape_gated_speedup": gated_speedup,
        "shape_gated_pass_1p3x": gated_speedup >= args.speedup_threshold,
    }


def summarize_by_candidate(results: list[dict]) -> dict:
    summary = {}
    for row in results:
        candidate = row["candidate"]
        item = summary.setdefault(
            candidate,
            {
                "total": 0,
                "shape_gate_enabled": 0,
                "w8a8_full_pass_1p3x": 0,
                "shape_gated_pass_1p3x": 0,
                "max_w8a8_full_speedup": 0.0,
                "max_shape_gated_speedup": 0.0,
            },
        )
        item["total"] += 1
        item["shape_gate_enabled"] += int(row["shape_gate_enabled"])
        item["w8a8_full_pass_1p3x"] += int(row["w8a8_full_pass_1p3x"])
        item["shape_gated_pass_1p3x"] += int(row["shape_gated_pass_1p3x"])
        item["max_w8a8_full_speedup"] = max(item["max_w8a8_full_speedup"], row["w8a8_full_speedup"])
        item["max_shape_gated_speedup"] = max(item["max_shape_gated_speedup"], row["shape_gated_speedup"])
    return summary


def write_summary(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as fp:
        json.dump(payload, fp, indent=2)

    lines = [
        "# P3b-2 W8A8 Full Module Microbenchmark",
        "",
        f"- device: `{payload['device']}`",
        f"- torch: `{payload['torch_version']}`",
        f"- fusion_mode: `{payload['fusion_mode']}`",
        f"- shape_gate_source: `{payload['shape_gate_source']}`",
        f"- speedup threshold: `{payload['speedup_threshold']}`",
        "",
        "## Candidate Summary",
        "",
        "| candidate | total | gate enabled | full pass | gated pass | max full speedup | max gated speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, row in payload["candidate_summary"].items():
        lines.append(
            "| {candidate} | {total} | {shape_gate_enabled} | {w8a8_full_pass_1p3x} | "
            "{shape_gated_pass_1p3x} | {max_w8a8_full_speedup:.3f} | "
            "{max_shape_gated_speedup:.3f} |".format(candidate=candidate, **row)
        )
    lines.extend(
        [
            "",
            "## Shape Results",
            "",
            "| candidate | branch | point | M | K | N | gate | fp32 ms | full w8a8 ms | full speedup | gated ms | gated speedup |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["results"]:
        lines.append(
            "| {candidate} | {branch} | {fused_point} | {M} | {K} | {N} | "
            "{shape_gate_enabled} | {fp32_fused_latency_ms_mean:.6f} | "
            "{w8a8_full_latency_ms_mean:.6f} | {w8a8_full_speedup:.3f} | "
            "{shape_gated_latency_ms_mean:.6f} | {shape_gated_speedup:.3f} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset = AseDBDataset(config={"src": args.dataset_src})
    indices = select_indices(len(dataset), args.limit, args.sample_seed)
    calc = MatRISCalculator(model=args.model, task=args.task, device=args.device)
    fused_modules = apply_gated_mlp_fusion(calc.model, FUSION_MODE)
    modules = dict(calc.model.named_modules())
    gate_pass_shapes = load_shape_gate(args.shape_gate_source)

    shapes = collect_shapes(calc, dataset, indices, args.device)
    results = [benchmark_shape(shape, modules, gate_pass_shapes, args) for shape in shapes]
    payload = {
        "phase": "P3b-2",
        "note": (
            "Full single-module benchmark with dynamic activation quantize, per-channel "
            "weight scale, scale combine, optional bias, FP32 output, and shape gate. "
            "Weight quantization is precomputed, matching inference-time packed weights."
        ),
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "sample_indices": indices,
        "model": args.model,
        "task": args.task,
        "fusion_mode": FUSION_MODE,
        "fused_module_count": len(fused_modules),
        "shape_gate_source": args.shape_gate_source,
        "shape_gate_pass_count": len(gate_pass_shapes),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "speedup_threshold": args.speedup_threshold,
        "results": results,
        "candidate_summary": summarize_by_candidate(results),
        "max_w8a8_full_speedup": max((row["w8a8_full_speedup"] for row in results), default=0.0),
        "max_shape_gated_speedup": max((row["shape_gated_speedup"] for row in results), default=0.0),
        "any_full_pass_1p3x": any(row["w8a8_full_pass_1p3x"] for row in results),
        "any_gated_pass_1p3x": any(row["shape_gated_pass_1p3x"] for row in results),
    }
    write_summary(Path(args.output_dir), payload)
    print(f"Wrote {Path(args.output_dir) / 'summary.json'}")
    print(f"Wrote {Path(args.output_dir) / 'summary.md'}")


if __name__ == "__main__":
    main()
