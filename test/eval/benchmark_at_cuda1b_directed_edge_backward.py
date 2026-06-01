from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OP_SRC = REPO_ROOT / "matris" / "model" / "op" / "src"
for path in (REPO_ROOT, OP_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matris_op  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microbench AT-CUDA1B directed-edge fused backward.")
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 4096, 8192, 16384])
    parser.add_argument("--node-rows", type=int, default=2048)
    parser.add_argument("--edge-rows", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="results/at_cuda1b_directed_edge_backward_microbench_20260525.json")
    return parser.parse_args()


def silu_grad(x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def time_call(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def bench_rows(rows: int, node_rows: int, edge_rows: int, warmup: int, iters: int, seed: int) -> dict:
    torch.manual_seed(seed + rows)
    device = torch.device("cuda")
    grad_core = torch.randn(rows, 128, device=device, dtype=torch.float32)
    grad_gate = torch.randn(rows, 128, device=device, dtype=torch.float32)
    core_raw = torch.randn(rows, 128, device=device, dtype=torch.float32)
    gate_raw = torch.randn(rows, 128, device=device, dtype=torch.float32)
    weight = torch.randn(256, 384, device=device, dtype=torch.float32) * 0.02
    edge_index = torch.randint(0, edge_rows, (rows,), device=device, dtype=torch.int64)
    source_index = torch.randint(0, node_rows, (rows,), device=device, dtype=torch.int64)
    target_index = torch.randint(0, node_rows, (rows,), device=device, dtype=torch.int64)

    def torch_path() -> tuple[torch.Tensor, torch.Tensor]:
        grad_projected = torch.cat(
            [
                grad_core * silu_grad(core_raw),
                grad_gate * silu_grad(gate_raw),
            ],
            dim=-1,
        )
        grad_cat = grad_projected.contiguous().matmul(weight)
        grad_node, grad_edge = matris_op.directed_edge_cat_grad_scatter_backward(
            grad_cat.contiguous(),
            edge_index,
            source_index,
            target_index,
            node_rows,
            edge_rows,
        )
        return grad_node, grad_edge

    def fused_path() -> tuple[torch.Tensor, torch.Tensor]:
        return matris_op.directed_edge_silu_project_grad_scatter_backward_tile32(
            grad_core,
            grad_gate,
            core_raw,
            gate_raw,
            weight,
            edge_index,
            source_index,
            target_index,
            node_rows,
            edge_rows,
        )

    ref_node, ref_edge = torch_path()
    got_node, got_edge = fused_path()
    torch.cuda.synchronize()
    node_diff = (ref_node - got_node).abs()
    edge_diff = (ref_edge - got_edge).abs()
    torch_ms = time_call(torch_path, warmup, iters)
    fused_ms = time_call(fused_path, warmup, iters)
    return {
        "rows": rows,
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "torch_ms": torch_ms,
        "fused_ms": fused_ms,
        "speedup": torch_ms / fused_ms if fused_ms > 0 else None,
        "delta_ms": fused_ms - torch_ms,
        "grad_node_max_abs": float(node_diff.max().item()),
        "grad_node_mean_abs": float(node_diff.mean().item()),
        "grad_edge_max_abs": float(edge_diff.max().item()),
        "grad_edge_mean_abs": float(edge_diff.mean().item()),
    }


def main() -> None:
    args = parse_args()
    torch.cuda.init()
    results = [
        bench_rows(rows, args.node_rows, args.edge_rows, args.warmup, args.iters, args.seed)
        for rows in args.rows
    ]
    payload = {
        "metadata": {
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "device": torch.cuda.get_device_name(0),
        },
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
