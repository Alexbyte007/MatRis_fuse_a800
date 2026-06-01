from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"

BASE_QUANT_MODE = "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8"
BASE_FUSION_MODE = "p28_p26_all_ffn_mlp_input_grad_only"


@dataclass(frozen=True)
class Experiment:
    idx: int
    label: str
    quant_mode: str
    added_targets: tuple[str, ...]


EXPERIMENTS = [
    Experiment(
        0,
        "stable baseline",
        BASE_QUANT_MODE,
        (),
    ),
    Experiment(
        1,
        "line edge second blocks 4-9",
        "p38f_line_edge_second_blocks_4_9_w8a8",
        (
            "interaction_block.4-9.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.4-9.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ),
    ),
    Experiment(
        2,
        "line edge second all",
        "p38a_line_edge_second_all_w8a8",
        (
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ),
    ),
    Experiment(
        3,
        "atom attention source/target linear",
        "w8a8_expand_atom_attn_linear",
        (
            "interaction_block.*.attn_block_atom_graph.source_weight_linear",
            "interaction_block.*.attn_block_atom_graph.target_weight_linear",
        ),
    ),
    Experiment(
        4,
        "atom refine edge_FFN",
        "w8a8_expand_atom_refine_edge_ffn",
        ("interaction_block.*.refine_block_atom_graph.edge_FFN",),
    ),
    Experiment(
        5,
        "atom attn edge core",
        "w8a8_expand_atom_attn_edge_core",
        ("interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_core",),
    ),
    Experiment(
        6,
        "atom refine edge core",
        "w8a8_expand_atom_refine_edge_core",
        ("interaction_block.*.refine_block_atom_graph.edge_nonlinear_update.mlp_core",),
    ),
    Experiment(
        7,
        "atom attn edge gate",
        "w8a8_expand_atom_attn_edge_gate",
        ("interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_gate",),
    ),
    Experiment(
        8,
        "atom refine edge gate",
        "w8a8_expand_atom_refine_edge_gate",
        ("interaction_block.*.refine_block_atom_graph.edge_nonlinear_update.mlp_gate",),
    ),
    Experiment(
        9,
        "atom edge core+gate combined",
        "w8a8_expand_atom_edge_core_gate",
        (
            "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_core",
            "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_gate",
            "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update.mlp_core",
            "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update.mlp_gate",
        ),
    ),
    Experiment(
        10,
        "atom attn node core",
        "w8a8_expand_atom_attn_node_core",
        ("interaction_block.*.attn_block_atom_graph.node_nonlinear_update.mlp_core",),
    ),
    Experiment(
        11,
        "atom attn node gate",
        "w8a8_expand_atom_attn_node_gate",
        ("interaction_block.*.attn_block_atom_graph.node_nonlinear_update.mlp_gate",),
    ),
    Experiment(
        12,
        "atom refine node_FFN",
        "w8a8_expand_atom_refine_node_ffn",
        ("interaction_block.*.refine_block_atom_graph.node_FFN",),
    ),
    Experiment(
        13,
        "atom_p0 small combo",
        "w8a8_expand_atom_p0",
        (
            "interaction_block.*.refine_block_atom_graph.node_FFN",
            "interaction_block.*.refine_block_atom_graph.edge_FFN",
            "interaction_block.*.attn_block_atom_graph.source_weight_linear",
            "interaction_block.*.attn_block_atom_graph.target_weight_linear",
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the W8A8 expansion sweep.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--profile-limit", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--output-root", default=str(RESULTS_ROOT / "w8a8_expansion_sweep"))
    parser.add_argument(
        "--summary-path",
        default=str(RESULTS_ROOT / "w8a8_expansion_sweep_summary.md"),
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated experiment indices to run. Empty runs baseline plus all experiments.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def scalar(payload: dict[str, Any] | None, path: tuple[str, ...]) -> float | None:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, list):
        cur = cur[0] if cur else None
    if isinstance(cur, dict) and "mean" in cur:
        cur = cur["mean"]
    try:
        return None if cur is None else float(cur)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
            "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
            "MATRIS_P29_MLP_BWD_KERNEL": "1",
            "MATRIS_W8A8_BACKEND": "cuda_wmma_tail_n128_parallel",
            "MATRIS_W8A8_DISABLE_FAST_WRAPPER": "1",
            "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
            "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
            "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
        }
    )
    return env


def run_experiment(exp: Experiment, run_name: str, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.output_root) / run_name
    eval_dir = run_dir / "eval"
    profile_dir = run_dir / "profile"
    eval_summary = eval_dir / "summary.json"
    profile_summary = profile_dir / "pipeline_profile_summary.json"
    record: dict[str, Any] = {
        "idx": exp.idx,
        "label": exp.label,
        "run_name": run_name,
        "quant_mode": exp.quant_mode,
        "fusion_mode": BASE_FUSION_MODE,
        "added_targets": list(exp.added_targets),
        "eval_summary": str(eval_summary),
        "profile_summary": str(profile_summary),
        "eval_status": "pending",
        "profile_status": "pending",
    }

    env = base_env()

    if not (args.skip_existing and eval_summary.exists()):
        eval_cmd = [
            sys.executable,
            "test/eval/infer_salex_lmdb_quant.py",
            "--dataset-src",
            args.dataset_src,
            "--model",
            args.model,
            "--task",
            "efsm",
            "--device",
            args.device,
            "--precision-mode",
            "fp32",
            "--quant-mode",
            exp.quant_mode,
            "--fusion-mode",
            BASE_FUSION_MODE,
            "--limit",
            str(args.eval_limit),
            "--sample-seed",
            str(args.sample_seed),
            "--activation-calibration-limit",
            str(args.activation_calibration_limit),
            "--activation-calibration-seed",
            str(args.activation_calibration_seed),
            "--measure-time",
            "--output-json",
            str(eval_summary),
            "--save-predictions",
            str(eval_dir / "predictions.jsonl"),
        ]
        rc = run_command(eval_cmd, run_dir / "eval.log", env)
        record["eval_status"] = "ok" if rc == 0 else f"failed:{rc}"
    else:
        record["eval_status"] = "existing"

    if not (args.skip_existing and profile_summary.exists()):
        profile_cmd = [
            sys.executable,
            "test/eval/profile_salex_pipeline.py",
            "--dataset-src",
            args.dataset_src,
            "--output-dir",
            str(profile_dir),
            "--model",
            args.model,
            "--task",
            "efs",
            "--device",
            args.device,
            "--precision-mode",
            "fp32",
            "--quant-mode",
            exp.quant_mode,
            "--fusion-mode",
            BASE_FUSION_MODE,
            "--warmup-steps",
            str(args.warmup_steps),
            "--limit",
            str(args.profile_limit),
            "--sample-seed",
            str(args.sample_seed),
            "--activation-calibration-limit",
            str(args.activation_calibration_limit),
            "--activation-calibration-seed",
            str(args.activation_calibration_seed),
            "--combined-force-stress-autograd",
        ]
        rc = run_command(profile_cmd, run_dir / "profile.log", env)
        record["profile_status"] = "ok" if rc == 0 else f"failed:{rc}"
    else:
        record["profile_status"] = "existing"

    eval_payload = load_json(eval_summary)
    profile_payload = load_json(profile_summary)
    record.update(extract_metrics(eval_payload, profile_payload))
    return record


def extract_metrics(eval_payload: dict[str, Any] | None, profile_payload: dict[str, Any] | None) -> dict[str, Any]:
    meta = eval_payload.get("metadata", {}) if eval_payload else {}
    return {
        "replaced_linear_count": len(meta.get("quant_replaced_modules") or []),
        "energy_mae_natoms": scalar(eval_payload, ("res", "energy_mae_natoms")),
        "force_mae": scalar(eval_payload, ("res", "force_mae")),
        "force_rmse": scalar(eval_payload, ("res", "force_rmse")),
        "stress_mae": scalar(eval_payload, ("res", "stress_mae")),
        "stress_rmse": scalar(eval_payload, ("res", "stress_rmse")),
        "latency_ms_mean": scalar(eval_payload, ("timing", "latency_ms_mean")),
        "profile_total_ms": scalar(profile_payload, ("profile_total_ms",)),
        "model_core_total_ms": scalar(profile_payload, ("category_summary", "model_core_total_ms")),
        "interaction_blocks_ms": scalar(profile_payload, ("interaction_blocks_ms",)),
        "combined_force_stress_autograd_ms": scalar(
            profile_payload,
            ("combined_force_stress_autograd_ms",),
        ),
    }


def ratio(value: float | None, base: float | None) -> float | None:
    if value is None or base in (None, 0):
        return None
    return value / base


def classify(record: dict[str, Any], baseline: dict[str, Any]) -> str:
    if record["eval_status"].startswith("failed") or record["profile_status"].startswith("failed"):
        return "FAIL"
    required = ["energy_mae_natoms", "force_mae", "force_rmse", "stress_mae", "stress_rmse", "profile_total_ms"]
    if any(record.get(key) is None or not math.isfinite(float(record[key])) for key in required):
        return "FAIL"

    energy_r = ratio(record["energy_mae_natoms"], baseline["energy_mae_natoms"])
    force_r = ratio(record["force_mae"], baseline["force_mae"])
    stress_r = ratio(record["stress_mae"], baseline["stress_mae"])
    speedup = ratio(baseline["profile_total_ms"], record["profile_total_ms"])

    if all(x is not None for x in (energy_r, force_r, stress_r, speedup)):
        if energy_r <= 1.008 and force_r <= 1.008 and stress_r <= 1.008 and speedup >= 0.99:
            return "PASS"
        if energy_r <= 1.02 and force_r <= 1.02 and stress_r <= 1.02 and speedup >= 0.95:
            return "BORDERLINE"
    return "FAIL"


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(records: list[dict[str, Any]], summary_path: Path) -> None:
    baseline = records[0]
    for record in records:
        record["energy_ratio"] = ratio(record.get("energy_mae_natoms"), baseline.get("energy_mae_natoms"))
        record["force_ratio"] = ratio(record.get("force_mae"), baseline.get("force_mae"))
        record["force_rmse_ratio"] = ratio(record.get("force_rmse"), baseline.get("force_rmse"))
        record["stress_ratio"] = ratio(record.get("stress_mae"), baseline.get("stress_mae"))
        record["profile_speedup_vs_baseline"] = ratio(
            baseline.get("profile_total_ms"),
            record.get("profile_total_ms"),
        )
        record["status"] = "BASELINE" if record["idx"] == 0 else classify(record, baseline)

    pass_records = [r for r in records if r.get("status") == "PASS"]
    borderline_records = [r for r in records if r.get("status") == "BORDERLINE"]
    fail_records = [r for r in records if r.get("status") == "FAIL"]

    lines = [
        "# W8A8 Expansion Sweep Summary",
        "",
        f"- baseline quant mode: `{BASE_QUANT_MODE}`",
        f"- fusion mode: `{BASE_FUSION_MODE}`",
        f"- generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Result Table",
        "",
        "| idx | status | label | quant_mode | replaced | E MAE/atom | E ratio | F MAE | F ratio | F RMSE ratio | S MAE | S ratio | profile ms | speedup | model core ms | interaction ms | combined grad ms |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["idx"]),
                    r.get("status", "n/a"),
                    r["label"],
                    f"`{r['quant_mode']}`",
                    str(r.get("replaced_linear_count", "n/a")),
                    fmt(r.get("energy_mae_natoms")),
                    fmt(r.get("energy_ratio")),
                    fmt(r.get("force_mae")),
                    fmt(r.get("force_ratio")),
                    fmt(r.get("force_rmse_ratio")),
                    fmt(r.get("stress_mae")),
                    fmt(r.get("stress_ratio")),
                    fmt(r.get("profile_total_ms")),
                    fmt(r.get("profile_speedup_vs_baseline")),
                    fmt(r.get("model_core_total_ms")),
                    fmt(r.get("interaction_blocks_ms")),
                    fmt(r.get("combined_force_stress_autograd_ms")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Added Targets", ""])
    for r in records[1:]:
        target_text = "<br>".join(f"`{t}`" for t in r["added_targets"]) or "n/a"
        lines.append(f"- **{r['idx']}. {r['label']}** `{r['quant_mode']}`: {target_text}")

    lines.extend(["", "## Recommendation", ""])
    if pass_records:
        lines.append("Recommended to keep testing:")
        lines.extend(f"- {r['label']} (`{r['quant_mode']}`)" for r in pass_records)
    else:
        lines.append("No non-baseline experiment met the PASS gate.")

    if borderline_records:
        lines.extend(["", "Borderline candidates, keep only if speed/accuracy tolerance allows:"])
        lines.extend(f"- {r['label']} (`{r['quant_mode']}`)" for r in borderline_records)

    if fail_records:
        lines.extend(["", "Do not expand first:"])
        lines.extend(f"- {r['label']} (`{r['quant_mode']}`)" for r in fail_records)

    lines.extend(["", "## W4A8 Next Step", ""])
    if pass_records:
        lines.append(
            "Try W4A8 only on PASS groups first, keeping the same baseline and replacing one passed W8 target group at a time."
        )
    else:
        lines.append(
            "Do not start W4A8 from these expansion groups yet; first repair W8A8 accuracy or speed regressions."
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_experiments(only: str) -> list[Experiment]:
    if not only.strip():
        return EXPERIMENTS
    wanted = {int(item.strip()) for item in only.split(",") if item.strip()}
    if 0 not in wanted:
        wanted.add(0)
    return [exp for exp in EXPERIMENTS if exp.idx in wanted]


def main() -> None:
    args = parse_args()
    run_prefix = args.run_prefix or time.strftime("w8a8_expand_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for exp in select_experiments(args.only):
        run_name = f"{run_prefix}_{exp.idx:02d}_{exp.quant_mode}"
        print(f"\n=== [{exp.idx}] {exp.label}: {exp.quant_mode} ===", flush=True)
        record = run_experiment(exp, run_name, args)
        records.append(record)
        if records and records[0].get("profile_total_ms") is not None:
            write_summary(records, Path(args.summary_path))
        (output_root / f"{run_prefix}_partial_records.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    write_summary(records, Path(args.summary_path))
    print(f"\nSummary written to {args.summary_path}", flush=True)


if __name__ == "__main__":
    main()
