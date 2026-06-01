from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant.layers import _load_matris_op  # noqa: E402


EXPECTED_ORDER = [
    (9, "refine_line"),
    (9, "attn_line"),
    (8, "refine_line"),
    (8, "attn_line"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P95B-A group replay feasibility test.")
    parser.add_argument("--output-dir", default="results/p95b_group_replay")
    parser.add_argument("--target-group", default="combo_8_9")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-capture", action="store_true")
    return parser.parse_args()


def run_capture(args: argparse.Namespace, output_dir: Path, capture_path: Path) -> None:
    if args.skip_capture and capture_path.exists():
        return
    profile_dir = output_dir / "capture_profile"
    env = os.environ.copy()
    env.update(
        {
            "MATRIS_P95_GROUP_REPLAY_CAPTURE_PATH": str(capture_path),
            "MATRIS_P95_GROUP_REPLAY_CAPTURE_LIMIT": "4",
            "MATRIS_P94_PRECOMPUTE_DQ_WEIGHT_T": "0",
            "MATRIS_P94_SAVED_PRE_SLIM": "0",
            "MATRIS_P91_BACKWARD_DRIVER": "0",
            "MATRIS_P91_CUBLAS_PAIR": "0",
            "MATRIS_P89_TAIL_BWD_V2": "0",
        }
    )
    if capture_path.exists():
        capture_path.unlink()
    cmd = [
        sys.executable,
        str(EVAL_DIR / "profile_p90_w8a8_backward_chain_breakdown.py"),
        "--target-group",
        args.target_group,
        "--limit",
        str(args.limit),
        "--warmup-steps",
        str(args.warmup_steps),
        "--activation-calibration-limit",
        str(args.activation_calibration_limit),
        "--activation-calibration-seed",
        str(args.activation_calibration_seed),
        "--sample-seed",
        str(args.sample_seed),
        "--output-dir",
        str(profile_dir),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    if not capture_path.exists():
        raise FileNotFoundError(f"P95B-A capture was not produced: {capture_path}")


def tensor_to_device(record: dict[str, Any], name: str, device: str) -> torch.Tensor:
    return record["tensors"][name].to(device=device, non_blocking=False)


def diff_summary(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean((a - b).float() ** 2)).item()) if diff.numel() else 0.0,
    }


def replay_group(records: list[dict[str, Any]], device: str) -> dict[str, Any]:
    if len(records) != 4:
        raise ValueError(f"P95B-A expects exactly 4 records, got {len(records)}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "input_grad_only_gated_tail_backward"):
        raise RuntimeError("matris_op.input_grad_only_gated_tail_backward is unavailable")

    rows = [int(r["rows"]) for r in records]
    order = [(r["block_id"], r["graph_group"]) for r in records]
    group_valid = {
        "same_rows": len(set(rows)) == 1,
        "expected_order": order == EXPECTED_ORDER,
        "all_hidden_dim_128": all(int(r["hidden_dim"]) == 128 for r in records),
        "all_paths_default": all(
            r["tail_path"] == "tail_backward" and r["input_grad_path"] == "torch_linear"
            for r in records
        ),
    }

    replay_rows = []
    for idx, record in enumerate(records):
        tensors = {
            name: tensor_to_device(record, name, device)
            for name in (
                "grad_out",
                "core_pre",
                "gate_pre",
                "core_dq_weight_t",
                "gate_dq_weight_t",
                "core_norm_weight",
                "core_norm_bias",
                "gate_norm_weight",
                "gate_norm_bias",
                "grad_core_ref",
                "grad_gate_ref",
            )
        }
        grad_core_pre, grad_gate_pre = matris_op.input_grad_only_gated_tail_backward(
            tensors["grad_out"].float().reshape(-1, 128).contiguous(),
            tensors["core_pre"],
            tensors["gate_pre"],
            tensors["core_norm_weight"],
            tensors["core_norm_bias"],
            tensors["gate_norm_weight"],
            tensors["gate_norm_bias"],
            float(record["eps"]),
        )
        grad_core = F.linear(grad_core_pre, tensors["core_dq_weight_t"]).reshape(record["core_shape"])
        grad_gate = F.linear(grad_gate_pre, tensors["gate_dq_weight_t"]).reshape(record["gate_shape"])
        if device == "cuda":
            torch.cuda.synchronize()
        replay_rows.append(
            {
                "idx": idx,
                "module_name": record["module_name"],
                "rows": int(record["rows"]),
                "grad_core": diff_summary(grad_core, tensors["grad_core_ref"]),
                "grad_gate": diff_summary(grad_gate, tensors["grad_gate_ref"]),
            }
        )
    max_core = max(row["grad_core"]["max_abs"] for row in replay_rows)
    max_gate = max(row["grad_gate"]["max_abs"] for row in replay_rows)
    return {
        "group_valid": group_valid,
        "rows": rows,
        "order": order,
        "max_grad_core_abs": max_core,
        "max_grad_gate_abs": max_gate,
        "pass": all(group_valid.values()) and max_core == 0.0 and max_gate == 0.0,
        "records": replay_rows,
    }


def write_summary(output_dir: Path, capture_path: Path, result: dict[str, Any]) -> None:
    summary_json = output_dir / "p95b_group_replay_summary.json"
    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# P95B-A Group Replay Feasibility",
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
        f"- max grad_core abs: `{result['max_grad_core_abs']:.6e}`",
        f"- max grad_gate abs: `{result['max_grad_gate_abs']:.6e}`",
        f"- PASS: `{str(result['pass']).lower()}`",
        "",
        "## Replay Diffs",
        "",
        "| idx | rows | module | grad_core max_abs | grad_gate max_abs |",
        "|---:|---:|---|---:|---:|",
    ]
    for row in result["records"]:
        lines.append(
            f"| {row['idx']} | {row['rows']} | `{row['module_name']}` | "
            f"`{row['grad_core']['max_abs']:.6e}` | `{row['grad_gate']['max_abs']:.6e}` |"
        )
    lines += [
        "",
        "结论：",
        "- 这个测试只验证 group replay 的数值可行性，不代表已经减少 autograd Function 数量。",
        "- 若 PASS，下一步可以尝试 C++ group4 driver 或更大的 group-level autograd boundary。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = output_dir / "p95b_group_replay_capture.pt"
    run_capture(args, output_dir, capture_path)
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    result = replay_group(records, args.device)
    result["capture_path"] = str(capture_path)
    result["format"] = payload.get("format", "")
    write_summary(output_dir, capture_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
