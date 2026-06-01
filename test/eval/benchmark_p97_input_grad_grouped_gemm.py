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
    parser = argparse.ArgumentParser(description="P97 input-grad grouped GEMM feasibility microbenchmark.")
    parser.add_argument(
        "--capture-path",
        default="results/p95b_group_replay_20260513_230748/p95b_group_replay_capture.pt",
    )
    parser.add_argument("--output-dir", default="results/p97_input_grad_grouped_gemm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--bench-iters", type=int, default=500)
    return parser.parse_args()


def load_records(capture_path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    return payload["records"], str(payload.get("format", ""))


def prepare_batch(records: list[dict[str, Any]], device: str) -> dict[str, Any]:
    if len(records) != 4:
        raise ValueError(f"P97 expects exactly 4 records, got {len(records)}")
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


def compute_grad_pre(matris_op: Any, batch: dict[str, Any]) -> dict[str, Any]:
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    grad_pres_8: list[torch.Tensor] = []
    weights_8: list[torch.Tensor] = []
    shapes_8: list[tuple[int, ...]] = []
    refs_8: list[torch.Tensor] = []
    labels_8: list[str] = []
    for idx in range(4):
        grad_pres_8.extend([grad_core_pres[idx], grad_gate_pres[idx]])
        weights_8.extend([batch["core_dq_weight_ts"][idx], batch["gate_dq_weight_ts"][idx]])
        shapes_8.extend([batch["core_shapes"][idx], batch["gate_shapes"][idx]])
        refs_8.extend([batch["grad_core_refs"][idx], batch["grad_gate_refs"][idx]])
        labels_8.extend([f"{idx}.core", f"{idx}.gate"])

    pre_stack = torch.stack(grad_pres_8, dim=0).contiguous()
    weight_stack = torch.stack(weights_8, dim=0).contiguous()
    weight_stack_t = weight_stack.transpose(1, 2).contiguous()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "grad_core_pres": grad_core_pres,
        "grad_gate_pres": grad_gate_pres,
        "grad_pres_8": grad_pres_8,
        "weights_8": weights_8,
        "shapes_8": shapes_8,
        "refs_8": refs_8,
        "labels_8": labels_8,
        "pre_stack": pre_stack,
        "weight_stack": weight_stack,
        "weight_stack_t": weight_stack_t,
    }


def outputs_flinear(data: dict[str, Any]) -> list[torch.Tensor]:
    return [
        F.linear(grad_pre, weight).reshape(shape)
        for grad_pre, weight, shape in zip(data["grad_pres_8"], data["weights_8"], data["shapes_8"])
    ]


def outputs_stack_bmm(data: dict[str, Any]) -> list[torch.Tensor]:
    pre_stack = torch.stack(data["grad_pres_8"], dim=0).contiguous()
    weight_stack_t = torch.stack(data["weights_8"], dim=0).transpose(1, 2).contiguous()
    out_stack = torch.bmm(pre_stack, weight_stack_t)
    return [out_stack[idx].reshape(data["shapes_8"][idx]) for idx in range(8)]


def outputs_bmm_prestacked(data: dict[str, Any]) -> list[torch.Tensor]:
    out_stack = torch.bmm(data["pre_stack"], data["weight_stack_t"])
    return [out_stack[idx].reshape(data["shapes_8"][idx]) for idx in range(8)]


def outputs_cublas_pair(matris_op: Any, data: dict[str, Any]) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for idx in range(4):
        grad_core, grad_gate = matris_op.w8a8_dual_input_grad_matmul_n128_cublas_pair(
            data["grad_core_pres"][idx],
            data["grad_gate_pres"][idx],
            data["weights_8"][2 * idx],
            data["weights_8"][2 * idx + 1],
        )
        outputs.extend(
            [
                grad_core.reshape(data["shapes_8"][2 * idx]),
                grad_gate.reshape(data["shapes_8"][2 * idx + 1]),
            ]
        )
    return outputs


def outputs_cublas_grouped_pair(matris_op: Any, data: dict[str, Any]) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for idx in range(4):
        grad_core, grad_gate = matris_op.w8a8_dual_input_grad_matmul_n128_cublas_grouped(
            data["grad_core_pres"][idx],
            data["grad_gate_pres"][idx],
            data["weights_8"][2 * idx],
            data["weights_8"][2 * idx + 1],
        )
        outputs.extend(
            [
                grad_core.reshape(data["shapes_8"][2 * idx]),
                grad_gate.reshape(data["shapes_8"][2 * idx + 1]),
            ]
        )
    return outputs


def outputs_cublas_group8(matris_op: Any, data: dict[str, Any]) -> list[torch.Tensor]:
    outputs = matris_op.w8a8_group8_input_grad_matmul_n128_cublas_grouped(
        data["grad_pres_8"],
        data["weights_8"],
    )
    return [outputs[idx].reshape(data["shapes_8"][idx]) for idx in range(8)]


def outputs_tiled(matris_op: Any, data: dict[str, Any], variant: str) -> list[torch.Tensor]:
    fn_name = (
        "w8a8_dual_input_grad_matmul_n128_tiled_m32n8"
        if variant == "m32n8"
        else "w8a8_dual_input_grad_matmul_n128_tiled"
    )
    fn = getattr(matris_op, fn_name)
    outputs: list[torch.Tensor] = []
    for idx in range(4):
        grad_core, grad_gate = fn(
            data["grad_core_pres"][idx],
            data["grad_gate_pres"][idx],
            # Tiled kernels consume int8 weights + scale; unavailable in this benchmark.
            data["weights_8"][2 * idx],
            data["weights_8"][2 * idx + 1],
            data["weights_8"][2 * idx].new_ones(128),
            data["weights_8"][2 * idx + 1].new_ones(128),
        )
        outputs.extend(
            [
                grad_core.reshape(data["shapes_8"][2 * idx]),
                grad_gate.reshape(data["shapes_8"][2 * idx + 1]),
            ]
        )
    return outputs


def compare_outputs(outputs: list[torch.Tensor], refs: list[torch.Tensor], labels: list[str]) -> dict[str, Any]:
    rows = []
    max_abs = 0.0
    for label, output, ref in zip(labels, outputs, refs):
        diff = diff_summary(output, ref)
        max_abs = max(max_abs, diff["max_abs"])
        rows.append({"label": label, **diff})
    return {"max_abs": max_abs, "rows": rows}


def bench(name: str, fn: Callable[[], list[torch.Tensor]], warmup_iters: int, bench_iters: int) -> dict[str, float]:
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
    return {
        "name": name,
        "iters": float(bench_iters),
        "total_ms": total_ms,
        "mean_ms": total_ms / max(bench_iters, 1),
    }


def write_summary(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "p97_input_grad_grouped_gemm_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# P97 Input-Grad Grouped GEMM Feasibility",
        "",
        f"- capture: `{result['capture_path']}`",
        f"- rows: `{result['rows']}`",
        f"- order: `{result['order']}`",
        "",
        "## Group Validity",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    for key, value in result["group_valid"].items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    lines.extend(["", "## Benchmark", "", "| path | mean ms / 8 GEMMs | speedup vs F.linear | max abs diff |", "|---|---:|---:|---:|"])
    base = result["bench"]["flinear_8x"]["mean_ms"]
    for name, row in result["bench"].items():
        diff = result["diffs"][name]["max_abs"]
        speedup = base / row["mean_ms"] if row["mean_ms"] > 0 else 0.0
        lines.append(f"| `{name}` | `{row['mean_ms']:.6f}` | `{speedup:.6f}x` | `{diff:.6e}` |")
    lines.extend(
        [
            "",
            "说明：",
            "- `stack_bmm_including_stack` 包含每次 stack/copy 开销。",
            "- `bmm_prestacked` 使用预先 stack 好的 tensor，只看 bmm 本体上限；主模型若要用它，还需要处理 stack/copy 或更底层 grouped GEMM。",
            "- `cublas_grouped_pair_4x` 是现有 C++ op，每次只 group 一个 module 的 core/gate 两个 GEMM，所以这里仍需调用 4 次。",
            "- `cublas_group8` 是 P97B 新增 sanity op，一次 group 4 个 module 的 core/gate 共 8 个 GEMM。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = Path(args.capture_path)
    records, capture_format = load_records(capture_path)
    matris_op = _load_matris_op()
    if matris_op is None:
        raise RuntimeError("matris_op is unavailable")
    batch = prepare_batch(records, args.device)
    data = compute_grad_pre(matris_op, batch)
    paths: dict[str, Callable[[], list[torch.Tensor]]] = {
        "flinear_8x": lambda: outputs_flinear(data),
        "stack_bmm_including_stack": lambda: outputs_stack_bmm(data),
        "bmm_prestacked": lambda: outputs_bmm_prestacked(data),
    }
    if hasattr(matris_op, "w8a8_dual_input_grad_matmul_n128_cublas_pair"):
        paths["cublas_pair_4x"] = lambda: outputs_cublas_pair(matris_op, data)
    if hasattr(matris_op, "w8a8_dual_input_grad_matmul_n128_cublas_grouped"):
        paths["cublas_grouped_pair_4x"] = lambda: outputs_cublas_grouped_pair(matris_op, data)
    if hasattr(matris_op, "w8a8_group8_input_grad_matmul_n128_cublas_grouped"):
        paths["cublas_group8"] = lambda: outputs_cublas_group8(matris_op, data)

    diffs: dict[str, Any] = {}
    bench_rows: dict[str, Any] = {}
    for name, fn in paths.items():
        outputs = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        diffs[name] = compare_outputs(outputs, data["refs_8"], data["labels_8"])
        bench_rows[name] = bench(name, fn, args.warmup_iters, args.bench_iters)

    result = {
        "capture_path": str(capture_path),
        "capture_format": capture_format,
        "device": args.device,
        "rows": batch["rows"],
        "order": batch["order"],
        "group_valid": batch["group_valid"],
        "bench": bench_rows,
        "diffs": diffs,
        "notes": {
            "stack_bmm_including_stack": "Includes stack/copy overhead.",
            "bmm_prestacked": "Upper-bound bmm timing with pre-stacked tensors.",
            "cublas_grouped_pair_4x": "Existing C++ op groups core/gate within one module only, still called 4 times.",
            "cublas_group8": "P97B C++ op groups 4 modules x core/gate into one cublasSgemmGroupedBatched call.",
        },
    }
    write_summary(output_dir, result)
    print(json.dumps({"bench": bench_rows, "max_abs": {k: v["max_abs"] for k, v in diffs.items()}}, indent=2))


if __name__ == "__main__":
    main()
