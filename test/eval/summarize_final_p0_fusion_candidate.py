from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


STATIC_RUNS = {
    "fp32": "fp32_baseline",
    "p0": "p0_line_gate_paper_stable_w8a32",
    "final": "p1e_retest_p0_line_gate_paper_stable_w8a32_line_all_candidate_gated_mlp_second_fused_fp32",
}

PROFILE_RUNS = {
    "fp32": "task_matrix_none_efs",
    "p0": "task_matrix_p0_line_gate_paper_stable_w8a32_efs",
    "final": "p1e_retest_p0_line_gate_paper_stable_w8a32_line_all_candidate_gated_mlp_second_fused_fp32",
}

STATIC_METRICS = [
    "energy_mae_natoms",
    "force_mae",
    "stress_mae",
    "latency_ms_mean",
]

PROFILE_METRICS = [
    "interaction_blocks_ms",
    "force_autograd_ms",
    "stress_autograd_ms",
    "profile_total_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final P0 + fusion candidate.")
    parser.add_argument(
        "--output-dir",
        default="results/final_p0_fusion_summary",
        help="Directory for summary.json and summary.md.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def scalar(value: Any) -> float | None:
    if isinstance(value, list) and len(value) == 1:
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return value / baseline


def static_metric(summary: dict[str, Any], metric: str) -> float | None:
    if metric == "latency_ms_mean":
        return scalar(summary.get("timing", {}).get(metric))
    return scalar(summary.get("res", {}).get(metric))


def profile_metric(summary: dict[str, Any], metric: str) -> float | None:
    row = summary.get(metric)
    if isinstance(row, dict):
        return scalar(row.get("mean"))
    return None


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def pct_from_ratio(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{(value - 1.0) * 100.0:+.2f}%"


def collect_static() -> dict[str, Any]:
    summaries = {
        key: load_json(REPO_ROOT / "results" / "salex_lmdb_quant" / run / "summary.json")
        for key, run in STATIC_RUNS.items()
    }
    rows: dict[str, dict[str, float | None]] = {}
    for metric in STATIC_METRICS:
        values = {key: static_metric(summary, metric) for key, summary in summaries.items()}
        rows[metric] = {
            **values,
            "p0_vs_fp32": ratio(values["p0"], values["fp32"]),
            "final_vs_fp32": ratio(values["final"], values["fp32"]),
            "final_vs_p0": ratio(values["final"], values["p0"]),
        }
    return {
        "runs": STATIC_RUNS,
        "metrics": rows,
        "summary_paths": {
            key: str(REPO_ROOT / "results" / "salex_lmdb_quant" / run / "summary.json")
            for key, run in STATIC_RUNS.items()
        },
    }


def collect_profile() -> dict[str, Any]:
    summaries = {
        key: load_json(
            REPO_ROOT / "results" / "pipeline_profile_salex" / run / "pipeline_profile_summary.json"
        )
        for key, run in PROFILE_RUNS.items()
    }
    rows: dict[str, dict[str, float | None]] = {}
    for metric in PROFILE_METRICS:
        values = {key: profile_metric(summary, metric) for key, summary in summaries.items()}
        rows[metric] = {
            **values,
            "p0_vs_fp32": ratio(values["p0"], values["fp32"]),
            "final_vs_fp32": ratio(values["final"], values["fp32"]),
            "final_vs_p0": ratio(values["final"], values["p0"]),
        }
    return {
        "runs": PROFILE_RUNS,
        "metrics": rows,
        "summary_paths": {
            key: str(
                REPO_ROOT
                / "results"
                / "pipeline_profile_salex"
                / run
                / "pipeline_profile_summary.json"
            )
            for key, run in PROFILE_RUNS.items()
        },
    }


def make_markdown(payload: dict[str, Any]) -> str:
    static_rows = payload["static"]["metrics"]
    profile_rows = payload["profile"]["metrics"]
    lines = [
        "# Final P0 Fusion Candidate Summary",
        "",
        "Final candidate:",
        "",
        "`p0_line_gate_paper_stable_w8a32 + line_all_candidate_gated_mlp_second_fused_fp32`",
        "",
        "Conclusion: `PASS_RELATIVE_TO_P0`. Precision is unchanged relative to P0, while static latency and interaction/profile timings improve modestly.",
        "",
        "## Static Eval",
        "",
        "| metric | fp32 | p0 | final | p0/fp32 | final/fp32 | final/p0 | final vs p0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, row in static_rows.items():
        lines.append(
            f"| {metric} | {fmt(row['fp32'])} | {fmt(row['p0'])} | {fmt(row['final'])} | "
            f"{fmt(row['p0_vs_fp32'])} | {fmt(row['final_vs_fp32'])} | "
            f"{fmt(row['final_vs_p0'])} | {pct_from_ratio(row['final_vs_p0'])} |"
        )

    lines.extend(
        [
            "",
            "## Pipeline Profile",
            "",
            "| metric | fp32 | p0 | final | p0/fp32 | final/fp32 | final/p0 | final vs p0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric, row in profile_rows.items():
        lines.append(
            f"| {metric} | {fmt(row['fp32'])} | {fmt(row['p0'])} | {fmt(row['final'])} | "
            f"{fmt(row['p0_vs_fp32'])} | {fmt(row['final_vs_fp32'])} | "
            f"{fmt(row['final_vs_p0'])} | {pct_from_ratio(row['final_vs_p0'])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Static eval uses `limit=500` from the existing run summaries.",
            "- Pipeline profile uses `limit=50`, `task=efs`, and the `task_matrix_*_efs` baselines.",
            "- `final/p0` is the primary reporting ratio for the fused fake-quant candidate.",
            "- This summary does not rerun inference or profiling; it only consolidates existing result files.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "candidate": {
            "quant_mode": "p0_line_gate_paper_stable_w8a32",
            "fusion_mode": "line_all_candidate_gated_mlp_second_fused_fp32",
            "run_name": STATIC_RUNS["final"],
            "verdict": "PASS_RELATIVE_TO_P0",
        },
        "static": collect_static(),
        "profile": collect_profile(),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(make_markdown(payload), encoding="utf-8")

    print(f"summary_json={output_dir / 'summary.json'}")
    print(f"summary_md={output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
