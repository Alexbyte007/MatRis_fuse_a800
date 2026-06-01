from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from diagnose_p95b_group_replay import EXPECTED_ORDER, diff_summary, tensor_to_device  # noqa: E402
from quant.layers import _load_matris_op  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P97C-A group8 saved-pre backward chunk replay benchmark.")
    parser.add_argument(
        "--capture-path",
        default="results/p95b_group_replay_20260513_230748/p95b_group_replay_capture.pt",
    )
    parser.add_argument("--output-dir", default="results/p97c_group8_backward_chunk")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--bench-iters", type=int, default=500)
    return parser.parse_args()


def load_records(capture_path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    return payload["records"], str(payload.get("format", ""))


def prepare_batch(records: list[dict[str, Any]], device: str) -> dict[str, Any]:
    if len(records) != 4:
        raise ValueError(f"P97C-A expects exactly 4 records, got {len(records)}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    order = [(record["block_id"], record["graph_group"]) for record in records]
    rows = [int(record["rows"]) for record in records]

    def tensors(name: str) -> list[torch.Tensor]:
        return [tensor_to_device(record, name, device) for record in records]

    return {
        "records": records,
        "rows": rows,
        "order": order,
        "group_valid": {
            "same_rows": len(set(rows)) == 1,
            "expected_order": order == EXPECTED_ORDER,
            "all_hidden_dim_128": all(int(record["hidden_dim"]) == 128 for record in records),
            "all_paths_default": all(
                record["tail_path"] == "tail_backward" and record["input_grad_path"] == "torch_linear"
                for record in records
            ),
        },
        "grad_outs": [tensor.float().reshape(-1, 128).contiguous() for tensor in tensors("grad_out")],
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


def compute_tail_backward(matris_op: Any, batch: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    grad_core_pres = []
    grad_gate_pres = []
    for idx in range(4):
        grad_core_pre, grad_gate_pre = matris_op.input_grad_only_gated_tail_backward(
            batch["grad_outs"][idx],
            batch["core_pres"][idx],
            batch["gate_pres"][idx],
            batch["core_norm_weights"][idx],
            batch["core_norm_biases"][idx],
            batch["gate_norm_weights"][idx],
            batch["gate_norm_biases"][idx],
            batch["eps_values"][idx],
        )
        grad_core_pres.append(grad_core_pre.contiguous())
        grad_gate_pres.append(grad_gate_pre.contiguous())
    return grad_core_pres, grad_gate_pres


def outputs_default(matris_op: Any, batch: dict[str, Any]) -> list[torch.Tensor]:
    grad_core_pres, grad_gate_pres = compute_tail_backward(matris_op, batch)
    outputs: list[torch.Tensor] = []
    for idx in range(4):
        outputs.append(F.linear(grad_core_pres[idx], batch["core_dq_weight_ts"][idx]).reshape(batch["core_shapes"][idx]))
        outputs.append(F.linear(grad_gate_pres[idx], batch["gate_dq_weight_ts"][idx]).reshape(batch["gate_shapes"][idx]))
    return outputs


def outputs_group8(matris_op: Any, batch: dict[str, Any]) -> list[torch.Tensor]:
    grad_core_pres, grad_gate_pres = compute_tail_backward(matris_op, batch)
    grad_pres_8: list[torch.Tensor] = []
    weights_8: list[torch.Tensor] = []
    shapes_8: list[tuple[int, ...]] = []
    for idx in range(4):
        grad_pres_8.extend([grad_core_pres[idx], grad_gate_pres[idx]])
        weights_8.extend([batch["core_dq_weight_ts"][idx], batch["gate_dq_weight_ts"][idx]])
        shapes_8.extend([batch["core_shapes"][idx], batch["gate_shapes"][idx]])
    outputs = matris_op.w8a8_group8_input_grad_matmul_n128_cublas_grouped(grad_pres_8, weights_8)
    return [outputs[idx].reshape(shapes_8[idx]) for idx in range(8)]


def refs_8(batch: dict[str, Any]) -> list[torch.Tensor]:
    refs: list[torch.Tensor] = []
    for idx in range(4):
        refs.extend([batch["grad_core_refs"][idx], batch["grad_gate_refs"][idx]])
    return refs


def labels_8() -> list[str]:
    return [f"{idx}.{branch}" for idx in range(4) for branch in ("core", "gate")]


def compare_outputs(outputs: list[torch.Tensor], refs: list[torch.Tensor]) -> dict[str, Any]:
    rows = []
    max_abs = 0.0
    for label, output, ref in zip(labels_8(), outputs, refs):
        diff = diff_summary(output, ref)
        max_abs = max(max_abs, diff["max_abs"])
        rows.append({"label": label, **diff})
    return {"max_abs": max_abs, "rows": rows}


def bench(fn: Callable[[], list[torch.Tensor]], warmup_iters: int, bench_iters: int) -> dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(bench_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    return {"iters": float(bench_iters), "total_ms": total_ms, "mean_ms": total_ms / max(bench_iters, 1)}


def write_summary(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "p97c_group8_backward_chunk_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    base = result["bench"]["default_tail_plus_8x_flinear"]["mean_ms"]
    lines = [
        "# P97C-A Group8 Backward Chunk Replay",
        "",
        f"- capture: `{result['capture_path']}`",
        f"- rows: `{result['rows']}`",
        f"- order: `{result['order']}`",
        "",
        "## Benchmark",
        "",
        "| path | mean ms / 4-call chunk | speedup vs default | max abs diff |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["bench"].items():
        speedup = base / row["mean_ms"] if row["mean_ms"] > 0 else 0.0
        max_abs = result["diffs"][name]["max_abs"]
        lines.append(f"| `{name}` | `{row['mean_ms']:.6f}` | `{speedup:.6f}x` | `{max_abs:.6e}` |")
    lines.extend(
        [
            "",
            "说明：",
            "- 这里测的是真实四连组 replay 的 backward chunk：4 个 tail backward + input-grad。",
            "- `group8_input_grad` 仍然不是主模型路径；它代表未来如果做成 group-level autograd boundary，这个 chunk 的局部上限。",
            "- 直接在单个 saved-pre backward 里打开 group8 不成立，因为每个 autograd Function 必须立即返回自己的梯度。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, capture_format = load_records(Path(args.capture_path))
    matris_op = _load_matris_op()
    if matris_op is None:
        raise RuntimeError("matris_op is unavailable")
    if not hasattr(matris_op, "w8a8_group8_input_grad_matmul_n128_cublas_grouped"):
        raise RuntimeError("P97B group8 op is unavailable")
    batch = prepare_batch(records, args.device)
    paths: dict[str, Callable[[], list[torch.Tensor]]] = {
        "default_tail_plus_8x_flinear": lambda: outputs_default(matris_op, batch),
        "tail_plus_group8_input_grad": lambda: outputs_group8(matris_op, batch),
    }
    refs = refs_8(batch)
    diffs = {}
    bench_rows = {}
    for name, fn in paths.items():
        outputs = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        diffs[name] = compare_outputs(outputs, refs)
        bench_rows[name] = bench(fn, args.warmup_iters, args.bench_iters)
    result = {
        "capture_path": args.capture_path,
        "capture_format": capture_format,
        "device": args.device,
        "rows": batch["rows"],
        "order": batch["order"],
        "group_valid": batch["group_valid"],
        "bench": bench_rows,
        "diffs": diffs,
    }
    write_summary(output_dir, result)
    print(json.dumps({"bench": bench_rows, "max_abs": {k: v["max_abs"] for k, v in diffs.items()}}, indent=2))


if __name__ == "__main__":
    main()
