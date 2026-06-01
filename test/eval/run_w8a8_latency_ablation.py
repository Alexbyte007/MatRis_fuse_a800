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

BASE_QUANT_MODE = "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8"
EXPAND_PASS_MODE = "p38f_line_edge_second_blocks_4_9_w8a8"
EXPAND_BORDERLINE_MODE = "p38a_line_edge_second_all_w8a8"
BASE_FUSION_MODE = "p28_p26_all_ffn_mlp_input_grad_only"

REFINE_LINE_EDGE_SECOND = (
    "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
    "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
)
ATTN_89_CORE = (
    "interaction_block.8.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
    "interaction_block.9.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
)
ATTN_89_GATE = (
    "interaction_block.8.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
    "interaction_block.9.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
)
ATTN_89_CORE_GATE = ATTN_89_CORE + ATTN_89_GATE
ATTN_47_CORE_GATE = tuple(
    f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.{branch}.layers.3"
    for idx in range(4, 8)
    for branch in ("mlp_core", "mlp_gate")
)

KNOWN_MATRIS_ENV_KEYS = (
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY",
    "MATRIS_P29_MLP_BWD_KERNEL",
    "MATRIS_W8A8_BACKEND",
    "MATRIS_W8A8_DISABLE_FAST_WRAPPER",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE",
    "MATRIS_QUANT_INCLUDE_GLOBS",
    "MATRIS_QUANT_EXCLUDE_GLOBS",
)

CURRENT_ENV_FLAGS = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_W8A8_BACKEND": "cuda_wmma_tail_n128_parallel",
    "MATRIS_W8A8_DISABLE_FAST_WRAPPER": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
}


@dataclass(frozen=True)
class Experiment:
    idx: int
    label: str
    quant_mode: str
    fusion_mode: str
    group: str
    compare_to: str = ""
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    env_flags: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def current_env(**overrides: str | None) -> dict[str, str]:
    flags = dict(CURRENT_ENV_FLAGS)
    for key, value in overrides.items():
        if value is None:
            flags.pop(key, None)
        else:
            flags[key] = value
    return flags


EXPERIMENTS: list[Experiment] = [
    Experiment(0, "fp32 none/no fusion", "none", "none", "anchor", notes="plain FP32 baseline"),
    Experiment(
        1,
        "fp32 none/no fusion + current env",
        "none",
        "none",
        "env_only",
        compare_to="fp32 none/no fusion",
        env_flags=current_env(),
        notes="checks whether global env flags affect latency without quant/fusion",
    ),
    Experiment(
        2,
        "fusion all only + current env",
        "none",
        BASE_FUSION_MODE,
        "fusion_bottom_up",
        compare_to="fp32 none/no fusion + current env",
        env_flags=current_env(),
    ),
    Experiment(
        3,
        "fusion line only + current env",
        "none",
        "p28_p26_line_ffn_mlp_input_grad_only",
        "fusion_bottom_up",
        compare_to="fp32 none/no fusion + current env",
        env_flags=current_env(),
    ),
    Experiment(
        4,
        "fusion atom only + current env",
        "none",
        "p28_p26_atom_ffn_mlp_input_grad_only",
        "fusion_bottom_up",
        compare_to="fp32 none/no fusion + current env",
        env_flags=current_env(),
    ),
    Experiment(
        5,
        "stable W8A8 full",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "anchor",
        compare_to="fp32 none/no fusion",
        env_flags=current_env(),
    ),
    Experiment(
        6,
        "expanded PASS full",
        EXPAND_PASS_MODE,
        BASE_FUSION_MODE,
        "anchor",
        compare_to="stable W8A8 full",
        env_flags=current_env(),
    ),
    Experiment(
        7,
        "expanded BORDERLINE full",
        EXPAND_BORDERLINE_MODE,
        BASE_FUSION_MODE,
        "anchor",
        compare_to="stable W8A8 full",
        env_flags=current_env(),
    ),
    Experiment(
        8,
        "stable quant only no fusion",
        BASE_QUANT_MODE,
        "none",
        "fusion_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(),
    ),
    Experiment(
        9,
        "stable quant + line fusion only",
        BASE_QUANT_MODE,
        "p28_p26_line_ffn_mlp_input_grad_only",
        "fusion_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(),
    ),
    Experiment(
        10,
        "stable quant + atom fusion only",
        BASE_QUANT_MODE,
        "p28_p26_atom_ffn_mlp_input_grad_only",
        "fusion_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(),
    ),
    Experiment(
        11,
        "refine line edge second quant only",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "quant_bottom_up",
        compare_to="fusion all only + current env",
        include_globs=REFINE_LINE_EDGE_SECOND,
        env_flags=current_env(),
    ),
    Experiment(
        12,
        "attn blocks 8-9 core quant only",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "quant_bottom_up",
        compare_to="fusion all only + current env",
        include_globs=ATTN_89_CORE,
        env_flags=current_env(),
    ),
    Experiment(
        13,
        "attn blocks 8-9 gate quant only",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "quant_bottom_up",
        compare_to="fusion all only + current env",
        include_globs=ATTN_89_GATE,
        env_flags=current_env(),
    ),
    Experiment(
        14,
        "attn blocks 8-9 core+gate quant only",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "quant_bottom_up",
        compare_to="fusion all only + current env",
        include_globs=ATTN_89_CORE_GATE,
        env_flags=current_env(),
    ),
    Experiment(
        15,
        "extra attn blocks 4-7 quant only",
        EXPAND_PASS_MODE,
        BASE_FUSION_MODE,
        "quant_bottom_up",
        compare_to="fusion all only + current env",
        include_globs=ATTN_47_CORE_GATE,
        env_flags=current_env(),
        notes="isolates the extra target group added by expanded PASS",
    ),
    Experiment(
        16,
        "stable minus refine line edge second",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "quant_top_down",
        compare_to="stable W8A8 full",
        exclude_globs=REFINE_LINE_EDGE_SECOND,
        env_flags=current_env(),
    ),
    Experiment(
        17,
        "stable minus attn blocks 8-9 core",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "quant_top_down",
        compare_to="stable W8A8 full",
        exclude_globs=ATTN_89_CORE,
        env_flags=current_env(),
    ),
    Experiment(
        18,
        "stable minus attn blocks 8-9 gate",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "quant_top_down",
        compare_to="stable W8A8 full",
        exclude_globs=ATTN_89_GATE,
        env_flags=current_env(),
    ),
    Experiment(
        19,
        "stable minus attn blocks 8-9 core+gate",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "quant_top_down",
        compare_to="stable W8A8 full",
        exclude_globs=ATTN_89_CORE_GATE,
        env_flags=current_env(),
    ),
    Experiment(
        20,
        "expanded PASS minus extra blocks 4-7",
        EXPAND_PASS_MODE,
        BASE_FUSION_MODE,
        "quant_top_down",
        compare_to="expanded PASS full",
        exclude_globs=ATTN_47_CORE_GATE,
        env_flags=current_env(),
    ),
    Experiment(
        21,
        "stable full no P26 tail input grad",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "env_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(MATRIS_P26_TAIL_INPUT_GRAD_ONLY=None),
    ),
    Experiment(
        22,
        "stable full no P28 mlp input grad",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "env_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(MATRIS_P28_MLP_INPUT_GRAD_ONLY=None),
    ),
    Experiment(
        23,
        "stable full no P29 mlp bwd kernel",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "env_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(MATRIS_P29_MLP_BWD_KERNEL=None),
    ),
    Experiment(
        24,
        "stable full backend default",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "env_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(MATRIS_W8A8_BACKEND=None),
    ),
    Experiment(
        25,
        "stable full fast wrapper enabled",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "env_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(MATRIS_W8A8_DISABLE_FAST_WRAPPER=None),
    ),
    Experiment(
        26,
        "stable full no fused line attention",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "env_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(MATRIS_USE_CUDA_FUSED_LINE_ATTENTION=None),
    ),
    Experiment(
        27,
        "stable full no fused atom attention",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "env_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION=None),
    ),
    Experiment(
        28,
        "stable full no directed2undirected average",
        BASE_QUANT_MODE,
        BASE_FUSION_MODE,
        "env_top_down",
        compare_to="stable W8A8 full",
        env_flags=current_env(MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE=None),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run W8A8 component ablations and locate eval-latency slowdowns."
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-mode", default="fp32")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument(
        "--output-root",
        default=str(RESULTS_ROOT / "w8a8_latency_ablation"),
    )
    parser.add_argument(
        "--summary-path",
        default=str(RESULTS_ROOT / "w8a8_latency_ablation_summary.md"),
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated experiment indices. Empty runs all.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--slowdown-threshold",
        type=float,
        default=1.01,
        help="Ratio above parent latency treated as a slowdown.",
    )
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        default=0.99,
        help="Ratio below parent latency treated as a speedup.",
    )
    return parser.parse_args()


def select_experiments(only: str) -> list[Experiment]:
    if not only.strip():
        return EXPERIMENTS
    wanted = {int(item.strip()) for item in only.split(",") if item.strip()}
    return [exp for exp in EXPERIMENTS if exp.idx in wanted]


def slug(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "/", "+", "-", "_"}:
            out.append("_")
    return "_".join("".join(out).split("_"))[:90]


def clean_env(extra: dict[str, str], include_globs: tuple[str, ...], exclude_globs: tuple[str, ...]) -> dict[str, str]:
    env = os.environ.copy()
    for key in KNOWN_MATRIS_ENV_KEYS:
        env.pop(key, None)
    env.update(extra)
    if include_globs:
        env["MATRIS_QUANT_INCLUDE_GLOBS"] = ",".join(include_globs)
    if exclude_globs:
        env["MATRIS_QUANT_EXCLUDE_GLOBS"] = ",".join(exclude_globs)
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
        "group": exp.group,
        "compare_to": exp.compare_to,
        "run_name": run_name,
        "quant_mode": exp.quant_mode,
        "fusion_mode": exp.fusion_mode,
        "include_globs": list(exp.include_globs),
        "exclude_globs": list(exp.exclude_globs),
        "env_flags": exp.env_flags,
        "notes": exp.notes,
        "eval_summary": str(summary_path),
        "eval_status": "pending",
    }

    if args.skip_existing and summary_path.exists():
        record["eval_status"] = "ok"
    else:
        env = clean_env(exp.env_flags, exp.include_globs, exp.exclude_globs)
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
        code = run_command(cmd, run_dir / "eval.log", env)
        record["eval_status"] = "ok" if code == 0 and summary_path.exists() else f"failed:{code}"

    payload = load_json(summary_path)
    record["replaced_linear_count"] = len(payload.get("metadata", {}).get("quant_replaced_modules", [])) if payload else None
    record["fused_module_count"] = len(payload.get("metadata", {}).get("fused_modules", [])) if payload else None
    record["latency_ms_mean"] = scalar(payload, ("timing", "latency_ms_mean"))
    record["latency_ms_std"] = scalar(payload, ("timing", "latency_ms_std"))
    record["latency_ms_min"] = scalar(payload, ("timing", "latency_ms_min"))
    record["latency_ms_max"] = scalar(payload, ("timing", "latency_ms_max"))
    record["throughput_structures_per_s"] = scalar(payload, ("timing", "throughput_structures_per_s"))
    record["throughput_atoms_per_s"] = scalar(payload, ("timing", "throughput_atoms_per_s"))
    record["energy_mae_natoms"] = scalar(payload, ("res", "energy_mae_natoms"))
    record["force_mae"] = scalar(payload, ("res", "force_mae"))
    record["force_rmse"] = scalar(payload, ("res", "force_rmse"))
    record["stress_mae"] = scalar(payload, ("res", "stress_mae"))
    record["stress_rmse"] = scalar(payload, ("res", "stress_rmse"))
    return record


def add_comparisons(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_label = {r["label"]: r for r in records}
    fp32 = by_label.get("fp32 none/no fusion")
    stable = by_label.get("stable W8A8 full")
    for record in records:
        parent = by_label.get(record.get("compare_to", ""))
        record["latency_ratio_vs_fp32"] = ratio(record.get("latency_ms_mean"), fp32.get("latency_ms_mean") if fp32 else None)
        record["speedup_vs_fp32"] = ratio(fp32.get("latency_ms_mean") if fp32 else None, record.get("latency_ms_mean"))
        record["latency_ratio_vs_stable"] = ratio(record.get("latency_ms_mean"), stable.get("latency_ms_mean") if stable else None)
        record["speedup_vs_stable"] = ratio(stable.get("latency_ms_mean") if stable else None, record.get("latency_ms_mean"))
        record["latency_ratio_vs_parent"] = ratio(
            record.get("latency_ms_mean"),
            parent.get("latency_ms_mean") if parent else None,
        )
        record["speedup_vs_parent"] = ratio(
            parent.get("latency_ms_mean") if parent else None,
            record.get("latency_ms_mean"),
        )
        parent_ratio = record.get("latency_ratio_vs_parent")
        if record.get("eval_status") != "ok":
            record["latency_judgement"] = "FAILED"
        elif parent_ratio is None:
            record["latency_judgement"] = "ANCHOR"
        elif parent_ratio >= args.slowdown_threshold:
            record["latency_judgement"] = "SLOWER_THAN_PARENT"
        elif parent_ratio <= args.speedup_threshold:
            record["latency_judgement"] = "FASTER_THAN_PARENT"
        else:
            record["latency_judgement"] = "NEUTRAL"


def infer_slowdown_findings(records: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    by_label = {r["label"]: r for r in records}
    lines: list[str] = []
    for record in records:
        if record.get("eval_status") != "ok":
            lines.append(f"- FAILED: {record['label']} did not complete: {record.get('eval_status')}")
            continue
        ratio_parent = record.get("latency_ratio_vs_parent")
        if ratio_parent is None:
            continue
        group = record["group"]
        if group.endswith("bottom_up") and ratio_parent >= args.slowdown_threshold:
            lines.append(
                f"- Potential slowdown when adding `{record['label']}` over `{record['compare_to']}`: "
                f"latency ratio {fmt(ratio_parent)}."
            )
        if group.endswith("top_down") and ratio_parent <= args.speedup_threshold:
            lines.append(
                f"- Potential harmful component in `{record['compare_to']}`: removing `{record['label']}` "
                f"made latency ratio {fmt(ratio_parent)}."
            )
    if not lines:
        lines.append("- No component crossed the configured slowdown/speedup thresholds.")

    stable = by_label.get("stable W8A8 full")
    expanded = by_label.get("expanded PASS full")
    if stable and expanded and stable.get("latency_ms_mean") and expanded.get("latency_ms_mean"):
        ratio_expanded = expanded["latency_ms_mean"] / stable["latency_ms_mean"]
        if ratio_expanded >= args.slowdown_threshold:
            lines.append(
                f"- Expanded PASS is slower than stable W8A8 in eval latency: ratio {fmt(ratio_expanded)}. "
                "Use `expanded PASS minus extra blocks 4-7` and `extra attn blocks 4-7 quant only` to confirm whether the added blocks are responsible."
            )
    return lines


def write_summary(records: list[dict[str, Any]], summary_path: Path, args: argparse.Namespace) -> None:
    add_comparisons(records, args)
    lines = [
        "# W8A8 Latency Ablation Summary",
        "",
        f"- generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- eval limit: `{args.limit}`",
        f"- sample seed: `{args.sample_seed}`",
        f"- calibration limit: `{args.activation_calibration_limit}`",
        "- primary metric: `latency_ms_mean` from `infer_salex_lmdb_quant.py --measure-time`",
        "",
        "## Result Table",
        "",
        "| idx | judgement | group | label | quant_mode | fusion_mode | replaced | fused | latency ms | vs FP32 speedup | vs stable speedup | vs parent ratio | force_mae | stress_mae |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["idx"]),
                    r.get("latency_judgement", "n/a"),
                    r["group"],
                    r["label"],
                    f"`{r['quant_mode']}`",
                    f"`{r['fusion_mode']}`",
                    fmt(r.get("replaced_linear_count"), 0),
                    fmt(r.get("fused_module_count"), 0),
                    fmt(r.get("latency_ms_mean")),
                    fmt(r.get("speedup_vs_fp32")),
                    fmt(r.get("speedup_vs_stable")),
                    fmt(r.get("latency_ratio_vs_parent")),
                    fmt(r.get("force_mae")),
                    fmt(r.get("stress_mae")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Slowdown Findings", ""])
    lines.extend(infer_slowdown_findings(records, args))

    lines.extend(["", "## Experiment Details", ""])
    for r in records:
        detail = [
            f"- **{r['idx']}. {r['label']}**",
            f"  - compare_to: `{r.get('compare_to') or 'n/a'}`",
            f"  - include_globs: `{', '.join(r.get('include_globs', [])) or 'n/a'}`",
            f"  - exclude_globs: `{', '.join(r.get('exclude_globs', [])) or 'n/a'}`",
            f"  - env_flags: `{json.dumps(r.get('env_flags', {}), sort_keys=True)}`",
            f"  - summary: `{r.get('eval_summary')}`",
        ]
        if r.get("notes"):
            detail.append(f"  - notes: {r['notes']}")
        lines.extend(detail)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_prefix = args.run_prefix or time.strftime("w8a8_latency_ablation_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = select_experiments(args.only)

    records: list[dict[str, Any]] = []
    partial_path = output_root / f"{run_prefix}_partial_records.json"
    for exp in selected:
        run_name = f"{run_prefix}_{exp.idx:02d}_{slug(exp.label)}"
        print(f"\n=== [{exp.idx}] {exp.label} ===", flush=True)
        record = run_experiment(exp, run_name, args)
        records.append(record)
        write_summary(records, Path(args.summary_path), args)
        partial_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_summary(records, Path(args.summary_path), args)
    print(f"\nSummary written to {args.summary_path}", flush=True)
    print(f"Partial records written to {partial_path}", flush=True)


if __name__ == "__main__":
    main()
