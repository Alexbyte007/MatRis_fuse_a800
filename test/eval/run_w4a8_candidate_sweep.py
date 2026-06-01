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

BASE_FUSION_MODE = "p28_p26_all_ffn_mlp_input_grad_only"
STABLE_W8A8_MODE = "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8"
PASS_W8A8_MODE = "p38f_line_edge_second_blocks_4_9_w8a8"
BORDERLINE_W8A8_MODE = "p38a_line_edge_second_all_w8a8"


@dataclass(frozen=True)
class Experiment:
    idx: int
    label: str
    quant_mode: str
    fusion_mode: str
    kind: str
    compare_to: str
    notes: str = ""


EXPERIMENTS = [
    Experiment(0, "fp32 none/no fusion", "none", "none", "anchor", "", "reference only"),
    Experiment(1, "stable W8A8 baseline", STABLE_W8A8_MODE, BASE_FUSION_MODE, "anchor", ""),
    Experiment(2, "expanded W8A8 PASS", PASS_W8A8_MODE, BASE_FUSION_MODE, "anchor", "stable W8A8 baseline"),
    Experiment(3, "expanded W8A8 BORDERLINE", BORDERLINE_W8A8_MODE, BASE_FUSION_MODE, "anchor", "stable W8A8 baseline"),
    Experiment(
        4,
        "W4A8 fake extra attn line second blocks 4-7 only",
        "w4a8_fake_attn_line_second_blocks_4_7",
        BASE_FUSION_MODE,
        "w4a8_fake",
        "stable W8A8 baseline",
        "isolates the extra blocks added by W8A8 PASS, without stable W8A8 targets",
    ),
    Experiment(
        5,
        "W4A8 fake attn line second blocks 4-9 only",
        "w4a8_fake_attn_line_second_blocks_4_9",
        BASE_FUSION_MODE,
        "w4a8_fake",
        "stable W8A8 baseline",
    ),
    Experiment(
        6,
        "mixed stable W8A8 + W4A8 extra blocks 4-7",
        "w4a8_mixed_stable_plus_attn_line_second_blocks_4_7",
        BASE_FUSION_MODE,
        "w4a8_mixed",
        "stable W8A8 baseline",
        "main W4A8 candidate: keep stable W8A8 and test W4 on the extra PASS expansion group",
    ),
    Experiment(
        7,
        "mixed refine W8A8 + W4A8 attn blocks 8-9",
        "w4a8_mixed_refine_w8a8_attn_line_second_blocks_8_9",
        BASE_FUSION_MODE,
        "w4a8_mixed",
        "stable W8A8 baseline",
    ),
    Experiment(
        8,
        "mixed refine W8A8 + W4A8 attn blocks 4-9",
        "w4a8_mixed_refine_w8a8_attn_line_second_blocks_4_9",
        BASE_FUSION_MODE,
        "w4a8_mixed",
        "stable W8A8 baseline",
    ),
    Experiment(
        9,
        "mixed W4A8 refine line second + W8A8 attn89",
        "w4a8_mixed_refine_line_second_attn89_w8a8",
        BASE_FUSION_MODE,
        "w4a8_mixed",
        "stable W8A8 baseline",
    ),
    Experiment(
        10,
        "W4A8 fake line edge second blocks 4-9",
        "w4a8_fake_line_edge_second_blocks_4_9",
        BASE_FUSION_MODE,
        "w4a8_fake",
        "stable W8A8 baseline",
    ),
    Experiment(
        11,
        "W4A8 fake line edge second all",
        "w4a8_fake_line_edge_second_all",
        BASE_FUSION_MODE,
        "w4a8_fake",
        "stable W8A8 baseline",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W4A8 candidate sensitivity sweep.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-mode", default="fp32")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--output-root", default=str(RESULTS_ROOT / "w4a8_candidate_sweep"))
    parser.add_argument("--summary-path", default=str(RESULTS_ROOT / "w4a8_candidate_sweep_summary.md"))
    parser.add_argument("--only", default="", help="Comma-separated experiment indices. Empty runs all.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--pass-ratio", type=float, default=1.01)
    parser.add_argument("--borderline-ratio", type=float, default=1.02)
    return parser.parse_args()


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


def select_experiments(only: str) -> list[Experiment]:
    if not only.strip():
        return EXPERIMENTS
    wanted = {int(item.strip()) for item in only.split(",") if item.strip()}
    return [exp for exp in EXPERIMENTS if exp.idx in wanted]


def slug(text: str) -> str:
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "/", "+", "-", "_"}:
            chars.append("_")
    return "_".join("".join(chars).split("_"))[:90]


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


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def scalar(payload: dict[str, Any] | None, path: tuple[str, ...]) -> float | None:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, list):
        cur = cur[0] if cur else None
    try:
        return None if cur is None else float(cur)
    except (TypeError, ValueError):
        return None


def ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return num / den


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def run_experiment(exp: Experiment, run_name: str, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.output_root) / run_name
    eval_dir = run_dir / "eval"
    summary_path = eval_dir / "summary.json"
    record: dict[str, Any] = {
        "idx": exp.idx,
        "label": exp.label,
        "kind": exp.kind,
        "run_name": run_name,
        "quant_mode": exp.quant_mode,
        "fusion_mode": exp.fusion_mode,
        "compare_to": exp.compare_to,
        "notes": exp.notes,
        "eval_summary": str(summary_path),
        "eval_status": "pending",
    }
    if args.skip_existing and summary_path.exists():
        record["eval_status"] = "ok"
    else:
        cmd = [
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
            args.precision_mode,
            "--quant-mode",
            exp.quant_mode,
            "--fusion-mode",
            exp.fusion_mode,
            "--limit",
            str(args.limit),
            "--sample-seed",
            str(args.sample_seed),
            "--activation-calibration-limit",
            str(args.activation_calibration_limit),
            "--activation-calibration-seed",
            str(args.activation_calibration_seed),
            "--measure-time",
            "--output-json",
            str(summary_path),
            "--save-predictions",
            str(eval_dir / "predictions.jsonl"),
        ]
        code = run_command(cmd, run_dir / "eval.log", base_env())
        record["eval_status"] = "ok" if code == 0 and summary_path.exists() else f"failed:{code}"

    payload = load_json(summary_path)
    metadata = payload.get("metadata", {}) if payload else {}
    qstats = metadata.get("quant_stats", []) if isinstance(metadata, dict) else []
    record["replaced_linear_count"] = len(metadata.get("quant_replaced_modules", [])) if metadata else None
    record["fused_module_count"] = len(metadata.get("fused_modules", [])) if metadata else None
    record["latency_ms_mean"] = scalar(payload, ("timing", "latency_ms_mean"))
    record["throughput_structures_per_s"] = scalar(payload, ("timing", "throughput_structures_per_s"))
    record["energy_mae_natoms"] = scalar(payload, ("res", "energy_mae_natoms"))
    record["force_mae"] = scalar(payload, ("res", "force_mae"))
    record["force_rmse"] = scalar(payload, ("res", "force_rmse"))
    record["stress_mae"] = scalar(payload, ("res", "stress_mae"))
    record["stress_rmse"] = scalar(payload, ("res", "stress_rmse"))
    if qstats:
        record["weight_quant_error_mae_mean"] = sum(s.get("weight_quant_error_mae", 0.0) for s in qstats) / len(qstats)
        record["saturation_ratio_mean"] = sum(s.get("saturation_ratio", 0.0) for s in qstats) / len(qstats)
    else:
        record["weight_quant_error_mae_mean"] = None
        record["saturation_ratio_mean"] = None
    return record


def add_comparisons(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_label = {record["label"]: record for record in records}
    stable = by_label.get("stable W8A8 baseline")
    fp32 = by_label.get("fp32 none/no fusion")
    for record in records:
        parent = by_label.get(record.get("compare_to", ""))
        for metric in ("energy_mae_natoms", "force_mae", "force_rmse", "stress_mae", "stress_rmse"):
            record[f"{metric}_ratio_vs_stable"] = ratio(record.get(metric), stable.get(metric) if stable else None)
            record[f"{metric}_ratio_vs_fp32"] = ratio(record.get(metric), fp32.get(metric) if fp32 else None)
            record[f"{metric}_ratio_vs_parent"] = ratio(record.get(metric), parent.get(metric) if parent else None)
        record["latency_speedup_vs_stable"] = ratio(stable.get("latency_ms_mean") if stable else None, record.get("latency_ms_mean"))
        record["latency_speedup_vs_fp32"] = ratio(fp32.get("latency_ms_mean") if fp32 else None, record.get("latency_ms_mean"))
        if record.get("eval_status") != "ok":
            record["accuracy_status"] = "FAILED"
            continue
        if record["kind"] == "anchor":
            record["accuracy_status"] = "ANCHOR"
            continue
        ratios = [
            record.get("energy_mae_natoms_ratio_vs_stable"),
            record.get("force_mae_ratio_vs_stable"),
            record.get("stress_mae_ratio_vs_stable"),
        ]
        if all(x is not None and x <= args.pass_ratio for x in ratios):
            record["accuracy_status"] = "PASS"
        elif all(x is not None and x <= args.borderline_ratio for x in ratios):
            record["accuracy_status"] = "BORDERLINE"
        else:
            record["accuracy_status"] = "FAIL"


def write_summary(records: list[dict[str, Any]], summary_path: Path, args: argparse.Namespace) -> None:
    add_comparisons(records, args)
    pass_records = [r for r in records if r.get("accuracy_status") == "PASS"]
    borderline_records = [r for r in records if r.get("accuracy_status") == "BORDERLINE"]
    fail_records = [r for r in records if r.get("accuracy_status") == "FAIL"]
    lines = [
        "# W4A8 Candidate Sweep Summary",
        "",
        f"- generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- eval limit: `{args.limit}`",
        f"- sample seed: `{args.sample_seed}`",
        f"- activation calibration limit: `{args.activation_calibration_limit}`",
        "- W4A8 modes in this file are fake/mixed sensitivity modes, not true int4 CUDA kernels.",
        "- Accuracy gate compares non-anchor rows against `stable W8A8 baseline`.",
        "",
        "## Result Table",
        "",
        "| idx | status | kind | label | quant_mode | replaced | E ratio | F ratio | S ratio | force_mae | stress_mae | latency ms | speedup vs stable | wqerr mean | sat mean |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["idx"]),
                    record.get("accuracy_status", "n/a"),
                    record["kind"],
                    record["label"],
                    f"`{record['quant_mode']}`",
                    fmt(record.get("replaced_linear_count"), 0),
                    fmt(record.get("energy_mae_natoms_ratio_vs_stable")),
                    fmt(record.get("force_mae_ratio_vs_stable")),
                    fmt(record.get("stress_mae_ratio_vs_stable")),
                    fmt(record.get("force_mae")),
                    fmt(record.get("stress_mae")),
                    fmt(record.get("latency_ms_mean")),
                    fmt(record.get("latency_speedup_vs_stable")),
                    fmt(record.get("weight_quant_error_mae_mean")),
                    fmt(record.get("saturation_ratio_mean")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Recommendation", ""])
    if pass_records:
        lines.append("W4A8 candidates worth considering for true int4 kernel work:")
        lines.extend(f"- {r['label']} (`{r['quant_mode']}`)" for r in pass_records)
    else:
        lines.append("No W4A8 candidate met the PASS gate yet.")
    if borderline_records:
        lines.append("")
        lines.append("Borderline W4A8 candidates:")
        lines.extend(f"- {r['label']} (`{r['quant_mode']}`)" for r in borderline_records)
    if fail_records:
        lines.append("")
        lines.append("Do not implement true int4 kernel first for:")
        lines.extend(f"- {r['label']} (`{r['quant_mode']}`)" for r in fail_records)

    lines.extend(["", "## Notes", ""])
    for record in records:
        if record.get("notes"):
            lines.append(f"- **{record['label']}**: {record['notes']}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_prefix = args.run_prefix or time.strftime("w4a8_candidate_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    partial_path = output_root / f"{run_prefix}_partial_records.json"
    for exp in select_experiments(args.only):
        run_name = f"{run_prefix}_{exp.idx:02d}_{slug(exp.label)}"
        print(f"\n=== [{exp.idx}] {exp.label}: {exp.quant_mode} ===", flush=True)
        record = run_experiment(exp, run_name, args)
        records.append(record)
        write_summary(records, Path(args.summary_path), args)
        partial_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(records, Path(args.summary_path), args)
    print(f"\nSummary written to {args.summary_path}", flush=True)
    print(f"Partial records written to {partial_path}", flush=True)


if __name__ == "__main__":
    main()
