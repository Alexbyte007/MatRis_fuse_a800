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
        description="P3b-1 W8A8 GEMM-only microbenchmark for MatRIS fused line GatedMLP shapes."
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--output-dir", default="results/p3b_w8a8_gemm_microbench")
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

    return sorted(captured.values(), key=lambda x: (x["candidate"], x["branch"], x["fused_point"], x["M"], x["K"], x["N"]))


def measure_ms(fn, device: str, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    sync(device)

    times = []
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(repeat):
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
    else:
        import time

        for _ in range(repeat):
            begin = time.perf_counter()
            fn()
            times.append((time.perf_counter() - begin) * 1000.0)

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def benchmark_shape(shape: dict, args: argparse.Namespace) -> dict:
    if args.device != "cuda":
        raise RuntimeError("P3b-1 W8A8 library-path benchmark requires CUDA.")
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("torch._int_mm is unavailable in this PyTorch build.")

    m, k, n = shape["M"], shape["K"], shape["N"]
    x_fp32 = torch.randn((m, k), device=args.device, dtype=torch.float32)
    w_fp32 = torch.randn((n, k), device=args.device, dtype=torch.float32)
    x_i8 = torch.randint(-127, 128, (m, k), device=args.device, dtype=torch.int8)
    w_i8_t = torch.randint(-127, 128, (k, n), device=args.device, dtype=torch.int8)

    fp32_mean, fp32_std = measure_ms(
        lambda: F.linear(x_fp32, w_fp32),
        args.device,
        args.warmup,
        args.repeat,
    )
    int8_mean, int8_std = measure_ms(
        lambda: torch._int_mm(x_i8, w_i8_t),
        args.device,
        args.warmup,
        args.repeat,
    )
    speedup = fp32_mean / int8_mean if int8_mean > 0 else 0.0
    return {
        **shape,
        "fp32_fused_latency_ms_mean": fp32_mean,
        "fp32_fused_latency_ms_std": fp32_std,
        "w8a8_library_path": "torch._int_mm",
        "w8a8_library_latency_ms_mean": int8_mean,
        "w8a8_library_latency_ms_std": int8_std,
        "speedup": speedup,
        "pass_1p3x": speedup >= args.speedup_threshold,
    }


def write_summary(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as fp:
        json.dump(payload, fp, indent=2)

    lines = [
        "# P3b-1 W8A8 GEMM Microbenchmark",
        "",
        f"- device: `{payload['device']}`",
        f"- torch: `{payload['torch_version']}`",
        f"- fusion_mode: `{payload['fusion_mode']}`",
        f"- library_path: `torch._int_mm`",
        f"- speedup threshold: `{payload['speedup_threshold']}`",
        "",
        "| candidate | branch | point | M | K | N | fp32 ms | int8 ms | speedup | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        lines.append(
            "| {candidate} | {branch} | {fused_point} | {M} | {K} | {N} | "
            "{fp32_fused_latency_ms_mean:.6f} | {w8a8_library_latency_ms_mean:.6f} | "
            "{speedup:.3f} | {pass_1p3x} |".format(**row)
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

    shapes = collect_shapes(calc, dataset, indices, args.device)
    results = [benchmark_shape(shape, args) for shape in shapes]
    payload = {
        "phase": "P3b-1",
        "note": "GEMM-only upper-bound benchmark. Dynamic activation quantization, scale combine, bias, output cast, and shape gate are intentionally excluded.",
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "sample_indices": indices,
        "model": args.model,
        "task": args.task,
        "fusion_mode": FUSION_MODE,
        "fused_module_count": len(fused_modules),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "speedup_threshold": args.speedup_threshold,
        "results": results,
        "max_speedup": max((row["speedup"] for row in results), default=0.0),
        "any_pass_1p3x": any(row["pass_1p3x"] for row in results),
    }
    write_summary(Path(args.output_dir), payload)
    print(f"Wrote {Path(args.output_dir) / 'summary.json'}")
    print(f"Wrote {Path(args.output_dir) / 'summary.md'}")


if __name__ == "__main__":
    main()
