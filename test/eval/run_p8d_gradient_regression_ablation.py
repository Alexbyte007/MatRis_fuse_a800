from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"

FP32_FUSED_QUANT_MODE = "p71_latency_pruned_fusion_only"
P8D_QUANT_MODE = "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8"
FUSION_MODE = "p28_p26_all_ffn_mlp_input_grad_only"


KNOWN_MATRIS_ENV_KEYS = (
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY",
    "MATRIS_P29_MLP_BWD_KERNEL",
    "MATRIS_P78_FP32_GATED_TAIL_FORWARD",
    "MATRIS_P79_ATTN_LINE_GATHER_CAT",
    "MATRIS_P83C_LINE_ATTENTION_NODE_INPUT",
    "MATRIS_W8A8_BACKEND",
    "MATRIS_W8A8_DISABLE_FAST_WRAPPER",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE",
    "MATRIS_QUANT_INCLUDE_GLOBS",
    "MATRIS_QUANT_EXCLUDE_GLOBS",
)

BASE_ENV_FLAGS = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_P78_FP32_GATED_TAIL_FORWARD": "1",
    "MATRIS_P79_ATTN_LINE_GATHER_CAT": "1",
    "MATRIS_P83C_LINE_ATTENTION_NODE_INPUT": "1",
    "MATRIS_W8A8_BACKEND": "cuda_wmma_tail_n128_parallel",
    "MATRIS_W8A8_DISABLE_FAST_WRAPPER": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
}


def refine_core(block: int | str = "*") -> str:
    return f"interaction_block.{block}.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"


def refine_gate(block: int | str = "*") -> str:
    return f"interaction_block.{block}.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"


def attn_core(block: int) -> str:
    return f"interaction_block.{block}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"


def attn_gate(block: int) -> str:
    return f"interaction_block.{block}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"


REFINE_ALL = (refine_core(), refine_gate())
REFINE_CORE_ALL = (refine_core(),)
REFINE_GATE_ALL = (refine_gate(),)
ATTN_89 = tuple(
    pattern
    for block in (8, 9)
    for pattern in (attn_core(block), attn_gate(block))
)


def refine_blocks(start: int, stop_inclusive: int) -> tuple[str, ...]:
    return tuple(
        pattern
        for block in range(start, stop_inclusive + 1)
        for pattern in (refine_core(block), refine_gate(block))
    )


def env_flags(**overrides: str | None) -> dict[str, str]:
    flags = dict(BASE_ENV_FLAGS)
    for key, value in overrides.items():
        if value is None:
            flags.pop(key, None)
        else:
            flags[key] = value
    return flags


@dataclass(frozen=True)
class Experiment:
    idx: int
    label: str
    quant_mode: str
    fusion_mode: str = FUSION_MODE
    include_globs: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=env_flags)
    group: str = "target"
    notes: str = ""


EXPERIMENTS: list[Experiment] = [
    Experiment(
        0,
        "fp32_fused_p84b_baseline",
        FP32_FUSED_QUANT_MODE,
        group="baseline",
        notes="official evalfix FP32 fused baseline",
    ),
    Experiment(
        1,
        "p8d_full",
        P8D_QUANT_MODE,
        group="anchor",
        notes="current failing p8d target set: refine-line all blocks plus attn-line blocks 8-9",
    ),
    Experiment(
        2,
        "refine_line_second_all_only",
        P8D_QUANT_MODE,
        include_globs=REFINE_ALL,
        group="bottom_up",
    ),
    Experiment(
        3,
        "attn_line_blocks_8_9_second_only",
        P8D_QUANT_MODE,
        include_globs=ATTN_89,
        group="bottom_up",
    ),
    Experiment(
        4,
        "refine_line_core_only",
        P8D_QUANT_MODE,
        include_globs=REFINE_CORE_ALL,
        group="branch",
    ),
    Experiment(
        5,
        "refine_line_gate_only",
        P8D_QUANT_MODE,
        include_globs=REFINE_GATE_ALL,
        group="branch",
    ),
    Experiment(
        6,
        "refine_line_blocks_0_3",
        P8D_QUANT_MODE,
        include_globs=refine_blocks(0, 3),
        group="block_group",
    ),
    Experiment(
        7,
        "refine_line_blocks_4_7",
        P8D_QUANT_MODE,
        include_globs=refine_blocks(4, 7),
        group="block_group",
    ),
    Experiment(
        8,
        "refine_line_blocks_8_9",
        P8D_QUANT_MODE,
        include_globs=refine_blocks(8, 9),
        group="block_group",
    ),
    *[
        Experiment(
            9 + block,
            f"refine_line_block_{block}",
            P8D_QUANT_MODE,
            include_globs=refine_blocks(block, block),
            group="per_block",
        )
        for block in range(10)
    ],
    Experiment(
        19,
        "refine_line_second_all_only_fusion_none",
        P8D_QUANT_MODE,
        fusion_mode="none",
        include_globs=REFINE_ALL,
        group="fallback",
        notes="same refine target without GatedMLP fusion; checks whether fused path/input-grad is the culprit",
    ),
    Experiment(
        20,
        "refine_line_second_all_only_backend_default",
        P8D_QUANT_MODE,
        include_globs=REFINE_ALL,
        env=env_flags(MATRIS_W8A8_BACKEND=None),
        group="fallback",
        notes="same refine target with default W8A8 backend instead of cuda_wmma_tail_n128_parallel",
    ),
    Experiment(
        21,
        "refine_line_second_all_only_no_p29_bwd",
        P8D_QUANT_MODE,
        include_globs=REFINE_ALL,
        env=env_flags(MATRIS_P29_MLP_BWD_KERNEL=None),
        group="fallback",
        notes="same refine target with P29 MLP backward kernel disabled",
    ),
    Experiment(
        22,
        "refine_line_second_all_only_no_input_grad_fast_flags",
        P8D_QUANT_MODE,
        include_globs=REFINE_ALL,
        env=env_flags(
            MATRIS_P26_TAIL_INPUT_GRAD_ONLY=None,
            MATRIS_P28_MLP_INPUT_GRAD_ONLY=None,
            MATRIS_P29_MLP_BWD_KERNEL=None,
        ),
        group="fallback",
        notes="same refine target with the main input-grad fast flags disabled",
    ),
    Experiment(
        23,
        "p8d_full_fusion_none",
        P8D_QUANT_MODE,
        fusion_mode="none",
        group="fallback",
        notes="full p8d target set without GatedMLP fusion",
    ),
    Experiment(
        24,
        "p8d_full_backend_default",
        P8D_QUANT_MODE,
        env=env_flags(MATRIS_W8A8_BACKEND=None),
        group="fallback",
        notes="full p8d target set with default W8A8 backend",
    ),
    Experiment(
        25,
        "p8d_full_no_p29_bwd",
        P8D_QUANT_MODE,
        env=env_flags(MATRIS_P29_MLP_BWD_KERNEL=None),
        group="fallback",
        notes="full p8d target set with P29 MLP backward kernel disabled",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate the p8d W8A8 force/stress regression under evalfix.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-mode", default="fp32")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--output-root", default=str(RESULTS_ROOT / "p8d_gradient_regression_ablation"))
    parser.add_argument(
        "--summary-path",
        default=str(RESULTS_ROOT / "p8d_gradient_regression_ablation_summary.md"),
    )
    parser.add_argument("--only", default="", help="Comma-separated experiment indices. Empty runs all.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-pass-ratio", type=float, default=1.02)
    parser.add_argument("--force-borderline-ratio", type=float, default=1.10)
    parser.add_argument("--stress-pass-ratio", type=float, default=1.02)
    parser.add_argument("--stress-borderline-ratio", type=float, default=1.10)
    return parser.parse_args()


def selected_experiments(only: str) -> list[Experiment]:
    if not only.strip():
        return EXPERIMENTS
    wanted = {int(item.strip()) for item in only.split(",") if item.strip()}
    return [exp for exp in EXPERIMENTS if exp.idx in wanted]


def slug(text: str) -> str:
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "/", "+", "-", "_"}:
            out.append("_")
    return "_".join("".join(out).split("_"))[:96]


def clean_env(extra: dict[str, str], include_globs: tuple[str, ...]) -> dict[str, str]:
    env = os.environ.copy()
    for key in KNOWN_MATRIS_ENV_KEYS:
        env.pop(key, None)
    env.update(extra)
    if include_globs:
        env["MATRIS_QUANT_INCLUDE_GLOBS"] = ",".join(include_globs)
    return env


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
        "group": exp.group,
        "quant_mode": exp.quant_mode,
        "fusion_mode": exp.fusion_mode,
        "include_globs": list(exp.include_globs),
        "env": exp.env,
        "notes": exp.notes,
        "run_name": run_name,
        "eval_summary": str(summary_path),
        "eval_log": str(run_dir / "eval.log"),
        "eval_status": "pending",
    }

    if args.skip_existing and summary_path.exists():
        record["eval_status"] = "ok"
    else:
        env = clean_env(exp.env, exp.include_globs)
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
        ]
        code = run_command(cmd, run_dir / "eval.log", env)
        record["eval_status"] = "ok" if code == 0 and summary_path.exists() else f"failed:{code}"

    payload = load_json(summary_path)
    metadata = payload.get("metadata", {}) if payload else {}
    record["replaced_linear_count"] = len(metadata.get("quant_replaced_modules", []))
    record["fused_module_count"] = len(metadata.get("fused_modules", []))
    record["peak_mem_mb"] = metadata.get("peak_mem_mb")
    record["latency_ms_mean"] = scalar(payload, ("timing", "latency_ms_mean"))
    record["latency_ms_std"] = scalar(payload, ("timing", "latency_ms_std"))
    record["throughput_structures_per_s"] = scalar(payload, ("timing", "throughput_structures_per_s"))
    record["energy_mae_natoms"] = scalar(payload, ("res", "energy_mae_natoms"))
    record["force_mae"] = scalar(payload, ("res", "force_mae"))
    record["force_rmse"] = scalar(payload, ("res", "force_rmse"))
    record["stress_mae"] = scalar(payload, ("res", "stress_mae"))
    record["stress_rmse"] = scalar(payload, ("res", "stress_rmse"))
    return record


def add_ratios_and_status(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    baseline = next((r for r in records if r["idx"] == 0 and r.get("eval_status") == "ok"), None)
    full = next((r for r in records if r["idx"] == 1 and r.get("eval_status") == "ok"), None)
    base_force = baseline.get("force_mae") if baseline else None
    base_stress = baseline.get("stress_mae") if baseline else None
    base_latency = baseline.get("latency_ms_mean") if baseline else None
    full_latency = full.get("latency_ms_mean") if full else None

    for record in records:
        latency = record.get("latency_ms_mean")
        force = record.get("force_mae")
        stress = record.get("stress_mae")
        record["speedup_vs_fp32_fused"] = (
            base_latency / latency if base_latency and latency else None
        )
        record["speedup_vs_p8d_full"] = (
            full_latency / latency if full_latency and latency else None
        )
        record["force_ratio_vs_fp32_fused"] = force / base_force if force and base_force else None
        record["stress_ratio_vs_fp32_fused"] = stress / base_stress if stress and base_stress else None

        force_ratio = record.get("force_ratio_vs_fp32_fused")
        stress_ratio = record.get("stress_ratio_vs_fp32_fused")
        if record.get("eval_status") != "ok":
            status = "FAILED"
        elif force_ratio is None or stress_ratio is None:
            status = "UNKNOWN"
        elif force_ratio <= args.force_pass_ratio and stress_ratio <= args.stress_pass_ratio:
            status = "PASS"
        elif force_ratio <= args.force_borderline_ratio and stress_ratio <= args.stress_borderline_ratio:
            status = "BORDERLINE"
        else:
            status = "FAIL"
        record["accuracy_status"] = status


def write_summary(records: list[dict[str, Any]], summary_path: Path, args: argparse.Namespace) -> None:
    add_ratios_and_status(records, args)
    lines = [
        "# P8D Gradient Regression Ablation",
        "",
        f"- generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- protocol: `limit={args.limit}`, `sample_seed={args.sample_seed}`, `activation_calibration_limit={args.activation_calibration_limit}`",
        f"- quant target under test: `{P8D_QUANT_MODE}`",
        f"- fusion mode: `{FUSION_MODE}` unless noted",
        "",
        "## Result Table",
        "",
        "| idx | status | group | label | replaced | fused | latency ms | speedup vs FP32 fused | speedup vs p8d full | force_mae | force ratio | stress_mae | stress ratio | peak mem MB |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(records, key=lambda item: item["idx"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["idx"]),
                    r.get("accuracy_status", "n/a"),
                    r["group"],
                    r["label"],
                    fmt(r.get("replaced_linear_count"), 0),
                    fmt(r.get("fused_module_count"), 0),
                    fmt(r.get("latency_ms_mean")),
                    fmt(r.get("speedup_vs_fp32_fused")),
                    fmt(r.get("speedup_vs_p8d_full")),
                    fmt(r.get("force_mae")),
                    fmt(r.get("force_ratio_vs_fp32_fused")),
                    fmt(r.get("stress_mae")),
                    fmt(r.get("stress_ratio_vs_fp32_fused")),
                    fmt(r.get("peak_mem_mb"), 3),
                ]
            )
            + " |"
        )

    failing = [r for r in records if r.get("accuracy_status") == "FAIL"]
    passing = [r for r in records if r.get("accuracy_status") == "PASS" and r["idx"] not in (0,)]
    lines.extend(["", "## Automatic Readout", ""])
    if passing:
        lines.append("- PASS groups: " + ", ".join(f"`{r['label']}`" for r in passing))
    if failing:
        lines.append("- FAIL groups: " + ", ".join(f"`{r['label']}`" for r in failing))

    refine_all = next((r for r in records if r["label"] == "refine_line_second_all_only"), None)
    attn_89 = next((r for r in records if r["label"] == "attn_line_blocks_8_9_second_only"), None)
    if refine_all and attn_89:
        lines.append(
            "- First split: "
            f"`refine_line_second_all_only` is {refine_all.get('accuracy_status')}, "
            f"`attn_line_blocks_8_9_second_only` is {attn_89.get('accuracy_status')}."
        )

    lines.extend(["", "## Details", ""])
    for r in sorted(records, key=lambda item: item["idx"]):
        lines.extend(
            [
                f"- **{r['idx']}. {r['label']}**",
                f"  - quant_mode: `{r['quant_mode']}`",
                f"  - fusion_mode: `{r['fusion_mode']}`",
                f"  - include_globs: `{', '.join(r['include_globs']) or 'n/a'}`",
                f"  - env: `{json.dumps(r['env'], sort_keys=True)}`",
                f"  - summary: `{r['eval_summary']}`",
                f"  - log: `{r['eval_log']}`",
            ]
        )
        if r.get("notes"):
            lines.append(f"  - notes: {r['notes']}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_prefix = args.run_prefix or time.strftime("p8d_grad_regression_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    partial_path = output_root / f"{run_prefix}_partial_records.json"

    records: list[dict[str, Any]] = []
    for exp in selected_experiments(args.only):
        run_name = f"{run_prefix}_{exp.idx:02d}_{slug(exp.label)}"
        print(f"\n=== [{exp.idx}] {exp.label} ===", flush=True)
        record = run_experiment(exp, run_name, args)
        records.append(record)
        partial_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_summary(records, Path(args.summary_path), args)

    write_summary(records, Path(args.summary_path), args)
    print(f"\nSummary written to {args.summary_path}", flush=True)
    print(f"Partial records written to {partial_path}", flush=True)


if __name__ == "__main__":
    main()
