from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from diagnose_p95b_group_replay import (
    EXPECTED_ORDER,
    diff_summary,
    run_capture,
    tensor_to_device,
)
from quant.layers import _load_matris_op


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P95B-B C++ group4 driver sanity test.")
    parser.add_argument("--output-dir", default="results/p95b_group4_driver")
    parser.add_argument("--capture-path", default="")
    parser.add_argument("--target-group", default="combo_8_9")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--bench-iters", type=int, default=200)
    parser.add_argument("--use-tail-bwd-v2", action="store_true")
    parser.add_argument("--use-cublas-pair", action="store_true")
    return parser.parse_args()


def load_records(args: argparse.Namespace, output_dir: Path) -> tuple[Path, list[dict[str, Any]], str]:
    if args.capture_path:
        capture_path = Path(args.capture_path)
    else:
        capture_path = output_dir / "p95b_group_replay_capture.pt"
        run_capture(args, output_dir, capture_path)
    if not capture_path.exists():
        raise FileNotFoundError(f"Capture not found: {capture_path}")
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    return capture_path, payload["records"], str(payload.get("format", ""))


def prepare_batch(records: list[dict[str, Any]], device: str) -> dict[str, Any]:
    if len(records) != 4:
        raise ValueError(f"P95B-B expects exactly 4 records, got {len(records)}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    order = [(r["block_id"], r["graph_group"]) for r in records]
    rows = [int(r["rows"]) for r in records]
    group_valid = {
        "same_rows": len(set(rows)) == 1,
        "expected_order": order == EXPECTED_ORDER,
        "all_hidden_dim_128": all(int(r["hidden_dim"]) == 128 for r in records),
        "all_paths_default": all(
            r["tail_path"] == "tail_backward" and r["input_grad_path"] == "torch_linear"
            for r in records
        ),
    }

    def tensors(name: str) -> list[torch.Tensor]:
        return [tensor_to_device(record, name, device) for record in records]

    return {
        "records": records,
        "rows": rows,
        "order": order,
        "group_valid": group_valid,
        "grad_outs": [t.float().reshape(-1, 128).contiguous() for t in tensors("grad_out")],
        "core_pres": tensors("core_pre"),
        "gate_pres": tensors("gate_pre"),
        "core_dq_weight_ts": tensors("core_dq_weight_t"),
        "gate_dq_weight_ts": tensors("gate_dq_weight_t"),
        "core_norm_weights": tensors("core_norm_weight"),
        "core_norm_biases": tensors("core_norm_bias"),
        "gate_norm_weights": tensors("gate_norm_weight"),
        "gate_norm_biases": tensors("gate_norm_bias"),
        "eps_values": [float(record["eps"]) for record in records],
        "core_shapes": [tuple(record["core_shape"]) for record in records],
        "gate_shapes": [tuple(record["gate_shape"]) for record in records],
        "grad_core_refs": tensors("grad_core_ref"),
        "grad_gate_refs": tensors("grad_gate_ref"),
    }


def python_loop_outputs(matris_op: Any, batch: dict[str, Any], use_tail_bwd_v2: bool) -> list[torch.Tensor]:
    tail_fn_name = (
        "input_grad_only_gated_tail_backward_n128_v2"
        if use_tail_bwd_v2
        else "input_grad_only_gated_tail_backward"
    )
    tail_fn = getattr(matris_op, tail_fn_name)
    outputs: list[torch.Tensor] = []
    for idx in range(4):
        grad_core_pre, grad_gate_pre = tail_fn(
            batch["grad_outs"][idx],
            batch["core_pres"][idx],
            batch["gate_pres"][idx],
            batch["core_norm_weights"][idx],
            batch["core_norm_biases"][idx],
            batch["gate_norm_weights"][idx],
            batch["gate_norm_biases"][idx],
            batch["eps_values"][idx],
        )
        grad_core = F.linear(grad_core_pre, batch["core_dq_weight_ts"][idx]).reshape(batch["core_shapes"][idx])
        grad_gate = F.linear(grad_gate_pre, batch["gate_dq_weight_ts"][idx]).reshape(batch["gate_shapes"][idx])
        outputs.extend([grad_core, grad_gate])
    return outputs


def group_driver_outputs(
    matris_op: Any,
    batch: dict[str, Any],
    use_tail_bwd_v2: bool,
    use_cublas_pair: bool,
) -> list[torch.Tensor]:
    outputs = matris_op.w8a8_saved_pre_backward_group4_driver(
        batch["grad_outs"],
        batch["core_pres"],
        batch["gate_pres"],
        batch["core_dq_weight_ts"],
        batch["gate_dq_weight_ts"],
        batch["core_norm_weights"],
        batch["core_norm_biases"],
        batch["gate_norm_weights"],
        batch["gate_norm_biases"],
        batch["eps_values"],
        bool(use_tail_bwd_v2),
        bool(use_cublas_pair),
    )
    reshaped: list[torch.Tensor] = []
    for idx in range(4):
        reshaped.append(outputs[2 * idx].reshape(batch["core_shapes"][idx]))
        reshaped.append(outputs[2 * idx + 1].reshape(batch["gate_shapes"][idx]))
    return reshaped


def compare_to_refs(outputs: list[torch.Tensor], batch: dict[str, Any]) -> dict[str, Any]:
    rows = []
    max_core = 0.0
    max_gate = 0.0
    for idx, record in enumerate(batch["records"]):
        core_diff = diff_summary(outputs[2 * idx], batch["grad_core_refs"][idx])
        gate_diff = diff_summary(outputs[2 * idx + 1], batch["grad_gate_refs"][idx])
        max_core = max(max_core, core_diff["max_abs"])
        max_gate = max(max_gate, gate_diff["max_abs"])
        rows.append(
            {
                "idx": idx,
                "module_name": record["module_name"],
                "rows": int(record["rows"]),
                "grad_core": core_diff,
                "grad_gate": gate_diff,
            }
        )
    return {
        "max_grad_core_abs": max_core,
        "max_grad_gate_abs": max_gate,
        "records": rows,
    }


def bench(name: str, fn: Callable[[], list[torch.Tensor]], warmup_iters: int, bench_iters: int) -> dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_cpu = time.perf_counter()
    for _ in range(bench_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start_cpu) * 1000.0
    return {
        "name": name,
        "iters": float(bench_iters),
        "total_ms": total_ms,
        "mean_ms": total_ms / max(bench_iters, 1),
    }


def write_summary(output_dir: Path, capture_path: Path, result: dict[str, Any]) -> None:
    summary_json = output_dir / "p95b_group4_driver_summary.json"
    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# P95B-B Group4 Driver",
        "",
        f"- capture: `{capture_path}`",
        f"- summary json: `{summary_json}`",
        "",
        "## Group Validity",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    for key, value in result["group_valid"].items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    lines += [
        "",
        f"- rows: `{result['rows']}`",
        f"- order: `{result['order']}`",
        f"- group4 max grad_core abs: `{result['group4_diff']['max_grad_core_abs']:.6e}`",
        f"- group4 max grad_gate abs: `{result['group4_diff']['max_grad_gate_abs']:.6e}`",
        f"- PASS: `{str(result['pass']).lower()}`",
        "",
        "## Microbenchmark",
        "",
        "| path | mean ms / group4 |",
        "|---|---:|",
        f"| python loop | `{result['bench']['python_loop']['mean_ms']:.6f}` |",
        f"| c++ group4 driver | `{result['bench']['group4_driver']['mean_ms']:.6f}` |",
        f"| speedup | `{result['bench']['speedup']:.6f}x` |",
        "",
        "## Per-Call Diff",
        "",
        "| idx | rows | module | grad_core max_abs | grad_gate max_abs |",
        "|---:|---:|---|---:|---:|",
    ]
    for row in result["group4_diff"]["records"]:
        lines.append(
            f"| {row['idx']} | {row['rows']} | `{row['module_name']}` | "
            f"`{row['grad_core']['max_abs']:.6e}` | `{row['grad_gate']['max_abs']:.6e}` |"
        )
    lines += [
        "",
        "结论：",
        "- 这个测试只验证 4-call C++ driver 的局部可行性，还没有减少主模型里的 autograd Function 数量。",
        "- 如果 group4 driver 不能明显快过 Python loop，就不值得继续做更重的主路径接入。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_path, records, capture_format = load_records(args, output_dir)
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "w8a8_saved_pre_backward_group4_driver"):
        raise RuntimeError("matris_op.w8a8_saved_pre_backward_group4_driver is unavailable")
    batch = prepare_batch(records, args.device)

    python_outputs = python_loop_outputs(matris_op, batch, args.use_tail_bwd_v2)
    group_outputs = group_driver_outputs(matris_op, batch, args.use_tail_bwd_v2, args.use_cublas_pair)
    if args.device == "cuda":
        torch.cuda.synchronize()

    python_diff = compare_to_refs(python_outputs, batch)
    group_diff = compare_to_refs(group_outputs, batch)

    py_bench = bench(
        "python_loop",
        lambda: python_loop_outputs(matris_op, batch, args.use_tail_bwd_v2),
        args.warmup_iters,
        args.bench_iters,
    )
    group_bench = bench(
        "group4_driver",
        lambda: group_driver_outputs(matris_op, batch, args.use_tail_bwd_v2, args.use_cublas_pair),
        args.warmup_iters,
        args.bench_iters,
    )
    speedup = py_bench["mean_ms"] / group_bench["mean_ms"] if group_bench["mean_ms"] > 0 else 0.0

    result = {
        "capture_path": str(capture_path),
        "format": capture_format,
        "device": args.device,
        "use_tail_bwd_v2": bool(args.use_tail_bwd_v2),
        "use_cublas_pair": bool(args.use_cublas_pair),
        "rows": batch["rows"],
        "order": batch["order"],
        "group_valid": batch["group_valid"],
        "python_diff": python_diff,
        "group4_diff": group_diff,
        "pass": all(batch["group_valid"].values())
        and group_diff["max_grad_core_abs"] == 0.0
        and group_diff["max_grad_gate_abs"] == 0.0,
        "bench": {
            "python_loop": py_bench,
            "group4_driver": group_bench,
            "speedup": speedup,
        },
    }
    write_summary(output_dir, capture_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
