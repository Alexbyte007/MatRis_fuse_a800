from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_MODES = [
    "line_graph_linear_bf16",
    "line_edge_gate_linear_bf16",
    "line_node_gate_linear_bf16",
    "p0_line_gate_paper_stable_linear_bf16",
    "line_graph_linear_fp16",
    "p0_line_gate_paper_stable_linear_fp16",
    "line_graph_linear_cached_bf16",
    "line_edge_gate_linear_cached_bf16",
    "line_node_gate_linear_cached_bf16",
    "p0_line_gate_paper_stable_linear_cached_bf16",
    "line_graph_linear_cached_fp16",
    "p0_line_gate_paper_stable_linear_cached_fp16",
]
DEFAULT_TASKS = ["e", "ef", "efs"]
PROFILE_KEYS = [
    "profile_total_ms",
    "interaction_blocks_ms",
    "force_autograd_ms",
    "stress_autograd_ms",
]
PRECISION_KEYS = [
    "energy_mae_natoms",
    "force_mae",
    "force_rmse",
    "stress_mae",
    "stress_rmse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize P2 Linear-only BF16/FP16 profile and static-eval results."
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-json", default="results/p2_linear_precision_summary/summary.json")
    parser.add_argument("--baseline-name", default="fp32_baseline")
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def scalar(value: Any) -> float | None:
    if isinstance(value, list) and len(value) == 1:
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def profile_mean(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, dict):
        return scalar(value.get("mean"))
    return None


def collect_profile(
    profile_root: Path,
    mode: str,
    task: str,
    keys: list[str],
) -> dict[str, Any]:
    prefix = f"p2_{mode}_{task}_r"
    runs = sorted(profile_root.glob(f"{prefix}*/pipeline_profile_summary.json"))
    values: dict[str, list[float]] = {key: [] for key in keys}

    for run in runs:
        payload = load_json(run)
        if payload is None:
            continue
        for key in keys:
            value = profile_mean(payload, key)
            if value is not None:
                values[key].append(value)

    medians = {
        key: statistics.median(items)
        for key, items in values.items()
        if items
    }
    return {
        "runs_found": len(runs),
        "median": medians,
        "values": values,
    }


def collect_static(summary_root: Path, comparison_root: Path, baseline_name: str, mode: str) -> dict[str, Any]:
    summary = load_json(summary_root / mode / "summary.json")
    comparison = load_json(comparison_root / f"{baseline_name}_vs_{mode}" / "comparison.json")
    if summary is None:
        return {"available": False}

    precision = {}
    res = summary.get("res", {})
    for key in PRECISION_KEYS:
        value = scalar(res.get(key))
        if value is not None:
            precision[key] = value

    timing = {}
    for key, value in summary.get("timing", {}).items():
        scalar_value = scalar(value)
        if scalar_value is not None:
            timing[key] = scalar_value

    ratios = {}
    if comparison is not None:
        for section in ("res", "timing"):
            for key, row in comparison.get(section, {}).items():
                ratio = row.get("ratio") if isinstance(row, dict) else None
                if ratio is not None:
                    ratios[key] = float(ratio)

    return {
        "available": True,
        "precision": precision,
        "timing": timing,
        "ratios": ratios,
        "comparison_available": comparison is not None,
    }


def classify(mode_result: dict[str, Any]) -> str:
    profile = mode_result.get("profile", {})
    static = mode_result.get("static", {})
    e_profile = profile.get("e", {})
    speed = e_profile.get("ratio_to_baseline", {})
    interaction_ratio = speed.get("interaction_blocks_ms")
    total_ratio = speed.get("profile_total_ms")

    has_speed = (
        (interaction_ratio is not None and interaction_ratio <= 0.97)
        or (total_ratio is not None and total_ratio <= 0.98)
    )

    ratios = static.get("ratios", {})
    has_precision = (
        ratios.get("energy_mae_natoms", float("inf")) <= 1.002
        and ratios.get("force_mae", float("inf")) <= 1.005
        and ratios.get("force_rmse", float("inf")) <= 1.005
    )

    if has_speed and has_precision:
        return "PASS_SPEED_AND_PRECISION"
    if has_speed and not static.get("available"):
        return "PASS_SPEED_NEEDS_STATIC"
    if has_speed:
        return "MAYBE_SPEED_PRECISION_CHECK"
    if interaction_ratio is None and total_ratio is None:
        return "INCOMPLETE_PROFILE"
    return "SLOW"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    profile_root = results_root / "pipeline_profile_salex"
    summary_root = results_root / "salex_lmdb_quant"
    comparison_root = results_root / "comparisons_salex_lmdb_quant"

    baseline_profile = {
        task: collect_profile(profile_root, "none", task, PROFILE_KEYS)
        for task in args.tasks
    }

    modes = {}
    for mode in args.modes:
        profile = {}
        for task in args.tasks:
            current = collect_profile(profile_root, mode, task, PROFILE_KEYS)
            baseline_median = baseline_profile.get(task, {}).get("median", {})
            ratios = {}
            for key, value in current.get("median", {}).items():
                baseline_value = baseline_median.get(key)
                if baseline_value:
                    ratios[key] = value / baseline_value
            current["ratio_to_baseline"] = ratios
            profile[task] = current

        mode_result = {
            "profile": profile,
            "static": collect_static(summary_root, comparison_root, args.baseline_name, mode),
        }
        mode_result["classification"] = classify(mode_result)
        modes[mode] = mode_result

    payload = {
        "baseline_name": args.baseline_name,
        "profile_keys": PROFILE_KEYS,
        "precision_keys": PRECISION_KEYS,
        "speed_rule": "PASS_SPEED if e-task interaction_blocks_ms <= 0.97x or profile_total_ms <= 0.98x baseline median",
        "precision_rule": "PASS_PRECISION if energy_mae_natoms <= 1.002x, force_mae <= 1.005x, force_rmse <= 1.005x",
        "baseline_profile": baseline_profile,
        "modes": modes,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    for mode, result in modes.items():
        e_ratio = result["profile"].get("e", {}).get("ratio_to_baseline", {})
        static_ratios = result["static"].get("ratios", {})
        print(
            mode,
            result["classification"],
            "e.interaction_ratio=",
            e_ratio.get("interaction_blocks_ms"),
            "e.total_ratio=",
            e_ratio.get("profile_total_ms"),
            "energy_mae_natoms_ratio=",
            static_ratios.get("energy_mae_natoms"),
            "force_mae_ratio=",
            static_ratios.get("force_mae"),
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
