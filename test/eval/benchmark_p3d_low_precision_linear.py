import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SHAPES = [
    ("attn_edge_first", 2086, 384, 256),
    ("attn_edge_first", 2760, 384, 256),
    ("attn_edge_first", 3672, 384, 256),
    ("attn_edge_first", 5088, 384, 256),
    ("attn_edge_first", 7064, 384, 256),
    ("attn_edge_second", 2086, 256, 256),
    ("attn_edge_second", 2760, 256, 256),
    ("attn_edge_second", 3672, 256, 256),
    ("attn_edge_second", 5088, 256, 256),
    ("attn_edge_second", 7064, 256, 256),
    ("refine_edge_first", 2086, 512, 256),
    ("refine_edge_first", 2760, 512, 256),
    ("refine_edge_first", 3672, 512, 256),
    ("refine_edge_first", 5088, 512, 256),
    ("refine_edge_first", 7064, 512, 256),
    ("refine_edge_second", 2086, 256, 256),
    ("refine_edge_second", 2760, 256, 256),
    ("refine_edge_second", 3672, 256, 256),
    ("refine_edge_second", 5088, 256, 256),
    ("refine_edge_second", 7064, 256, 256),
]


def _bench(fn, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low precision linear cast-cost microbenchmark.")
    parser.add_argument("--output-dir", default="results/p3d_low_precision_linear_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260503)
    return parser.parse_args()


def run_shape(
    branch: str,
    m: int,
    k: int,
    n: int,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> dict:
    x = torch.randn((m, k), device=device, dtype=torch.float32)
    weight = torch.randn((n, k), device=device, dtype=torch.float32) / (k**0.5)
    bias = torch.randn((n,), device=device, dtype=torch.float32)
    x_low = x.to(dtype)
    weight_low = weight.to(dtype).contiguous()
    bias_low = bias.to(dtype).contiguous()

    fp32_ms, fp32_std = _bench(lambda: F.linear(x, weight, bias), warmup, repeat)
    cast_only_ms, cast_only_std = _bench(lambda: x.to(dtype), warmup, repeat)
    low_precast_ms, low_precast_std = _bench(
        lambda: F.linear(x_low, weight_low, bias_low).float(),
        warmup,
        repeat,
    )
    low_with_cast_ms, low_with_cast_std = _bench(
        lambda: F.linear(x.to(dtype), weight_low, bias_low).float(),
        warmup,
        repeat,
    )

    ref = F.linear(x, weight, bias)
    low = F.linear(x_low, weight_low, bias_low).float()
    return {
        "branch": branch,
        "M": m,
        "K": k,
        "N": n,
        "dtype": str(dtype),
        "fp32_latency_ms_mean": fp32_ms,
        "fp32_latency_ms_std": fp32_std,
        "cast_only_ms_mean": cast_only_ms,
        "cast_only_ms_std": cast_only_std,
        "low_precast_latency_ms_mean": low_precast_ms,
        "low_precast_latency_ms_std": low_precast_std,
        "low_with_cast_latency_ms_mean": low_with_cast_ms,
        "low_with_cast_latency_ms_std": low_with_cast_std,
        "speedup_precast_vs_fp32": fp32_ms / low_precast_ms,
        "speedup_with_cast_vs_fp32": fp32_ms / low_with_cast_ms,
        "cast_pct_of_with_cast": cast_only_ms / low_with_cast_ms * 100.0,
        "max_abs_diff_vs_fp32": (low - ref).abs().max().item(),
        "mean_abs_diff_vs_fp32": (low - ref).abs().mean().item(),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    results = []
    for dtype in (torch.bfloat16, torch.float16):
        for branch, m, k, n in SHAPES:
            item = run_shape(branch, m, k, n, args.device, dtype, args.warmup, args.repeat)
            results.append(item)
            print(
                f"{branch} M={m} K={k} N={n} {str(dtype).replace('torch.', '')}: "
                f"fp32={item['fp32_latency_ms_mean']:.4f}ms "
                f"precast={item['low_precast_latency_ms_mean']:.4f}ms "
                f"with_cast={item['low_with_cast_latency_ms_mean']:.4f}ms "
                f"speedup_with_cast={item['speedup_with_cast_vs_fp32']:.3f}x"
            )

    summary = {
        "phase": "p3d_low_precision_linear_cast_cost",
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else args.device,
        "torch": torch.__version__,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# P3d Low Precision Linear Cast-Cost Benchmark",
        "",
        f"- device: `{summary['device']}`",
        f"- torch: `{torch.__version__}`",
        "",
        "| dtype | branch | M | K | N | fp32 ms | precast low ms | with-cast low ms | precast speedup | with-cast speedup | cast % | mean abs diff |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        dtype = r["dtype"].replace("torch.", "")
        lines.append(
            f"| {dtype} | {r['branch']} | {r['M']} | {r['K']} | {r['N']} | "
            f"{r['fp32_latency_ms_mean']:.6f} | {r['low_precast_latency_ms_mean']:.6f} | "
            f"{r['low_with_cast_latency_ms_mean']:.6f} | {r['speedup_precast_vs_fp32']:.3f} | "
            f"{r['speedup_with_cast_vs_fp32']:.3f} | {r['cast_pct_of_with_cast']:.1f} | "
            f"{r['mean_abs_diff_vs_fp32']:.6e} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    faster_with_cast = [r for r in results if r["speedup_with_cast_vs_fp32"] > 1.0]
    best = max(results, key=lambda r: r["speedup_with_cast_vs_fp32"])
    analysis = [
        "# P3d Low Precision Linear Analysis",
        "",
        f"- With explicit input cast, low precision is faster than FP32 on `{len(faster_with_cast)}/{len(results)}` shapes.",
        f"- Best with-cast speedup: `{best['speedup_with_cast_vs_fp32']:.3f}x` "
        f"for `{best['dtype']}` `{best['branch']}` M={best['M']} K={best['K']} N={best['N']}.",
        "- If precast speedup is high but with-cast speedup is weak, the model needs a wider low-precision dataflow rather than isolated low-precision Linear replacements.",
    ]
    (out_dir / "analysis.md").write_text("\n".join(analysis) + "\n")


if __name__ == "__main__":
    main()
