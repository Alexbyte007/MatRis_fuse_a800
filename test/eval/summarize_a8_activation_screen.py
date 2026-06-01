import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODES = [
    "a8_refine_atom_node_ffn",
    "a8_refine_atom_edge_ffn",
    "a8_attn_atom_source_weight_linear",
    "a8_attn_atom_target_weight_linear",
    "a8_attn_atom_edge_core",
    "a8_attn_line_source_weight_linear",
    "a8_attn_line_target_weight_linear",
    "a8_refine_line_node_ffn",
    "a8_refine_line_edge_ffn",
    "a8_attn_line_edge_core",
    "a8_attn_line_edge_gate",
    "a8_refine_line_edge_core",
    "a8_refine_line_edge_gate",
    "a8_attn_line_node_core",
    "a8_attn_line_node_gate",
    "a8_atom_p0",
    "a8_line_graph",
    "a8_attn_line_edge_core_gate",
    "a8_refine_line_edge_core_gate",
    "a8_attn_line_node_core_gate",
    "a8_line_edge_core_gate",
    "a8_line_node_core_gate",
    "a8_line_nonlinear_core_gate",
    "a8_p0_line_gate_paper_stable",
    "a8_all_single_pass",
]

METRICS = [
    "energy_mae_natoms",
    "force_mae",
    "force_rmse",
    "stress_mae",
    "stress_rmse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P3a W8A8 activation fake quant screen.")
    parser.add_argument("--baseline-name", default="p3a_fp32_fused_baseline")
    parser.add_argument("--run-prefix", default="p3a")
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--energy-threshold", type=float, default=1.003)
    parser.add_argument("--force-threshold", type=float, default=1.006)
    parser.add_argument("--force-rmse-threshold", type=float, default=1.006)
    parser.add_argument("--stress-threshold", type=float, default=1.010)
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "results" / "p3a_a8_activation_screen" / "summary.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "results" / "p3a_a8_activation_screen" / "summary.md"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def scalar(value):
    if isinstance(value, list) and len(value) == 1:
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def metric(summary: dict | None, key: str) -> float | None:
    if summary is None:
        return None
    return scalar(summary.get("res", {}).get(key))


def timing(summary: dict | None, key: str) -> float | None:
    if summary is None:
        return None
    return scalar(summary.get("timing", {}).get(key))


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return candidate / baseline


def verdict(ratios: dict[str, float | None], args: argparse.Namespace) -> str:
    required = {
        "energy_mae_natoms": args.energy_threshold,
        "force_mae": args.force_threshold,
        "force_rmse": args.force_rmse_threshold,
    }
    for key, threshold in required.items():
        value = ratios.get(key)
        if value is None or value > threshold:
            return "PRECISION_FAIL"
    for key in ("stress_mae", "stress_rmse"):
        value = ratios.get(key)
        if value is not None and value > args.stress_threshold:
            return "PRECISION_FAIL_STRESS"
    return "PRECISION_PASS"


def main() -> None:
    args = parse_args()
    baseline_path = REPO_ROOT / "results" / "salex_lmdb_quant" / args.baseline_name / "summary.json"
    baseline = load_json(baseline_path)

    rows = []
    for mode in args.modes:
        run_name = f"{args.run_prefix}_{mode}"
        candidate_path = REPO_ROOT / "results" / "salex_lmdb_quant" / run_name / "summary.json"
        candidate = load_json(candidate_path)
        ratios = {
            key: ratio(metric(candidate, key), metric(baseline, key))
            for key in METRICS
        }
        replaced = []
        if candidate is not None:
            replaced = candidate.get("metadata", {}).get("quant_replaced_modules", []) or []
        rows.append(
            {
                "mode": mode,
                "run_name": run_name,
                "summary_json": str(candidate_path),
                "found": candidate is not None,
                "replaced_linear_count": len(replaced),
                "ratios": ratios,
                "latency_ms_mean_ratio": ratio(
                    timing(candidate, "latency_ms_mean"),
                    timing(baseline, "latency_ms_mean"),
                ),
                "verdict": verdict(ratios, args) if candidate is not None and baseline is not None else "MISSING",
            }
        )

    payload = {
        "screen": "p3a_w8a8_activation_fake_quant",
        "baseline_name": args.baseline_name,
        "baseline_json": str(baseline_path),
        "run_prefix": args.run_prefix,
        "thresholds": {
            "energy_mae_natoms": args.energy_threshold,
            "force_mae": args.force_threshold,
            "force_rmse": args.force_rmse_threshold,
            "stress_mae": args.stress_threshold,
            "stress_rmse": args.stress_threshold,
        },
        "rows": rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_md.open("w", encoding="utf-8") as fp:
        fp.write("# P3a W8A8 Activation Fake Quant Screen\n\n")
        fp.write(f"Baseline: `{args.baseline_name}`\n\n")
        fp.write(
            "| mode | verdict | replaced | energy | force_mae | force_rmse | "
            "stress_mae | stress_rmse | latency |\n"
        )
        fp.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            ratios = row["ratios"]
            fp.write(
                f"| `{row['mode']}` | {row['verdict']} | {row['replaced_linear_count']} | "
                f"{fmt(ratios.get('energy_mae_natoms'))} | "
                f"{fmt(ratios.get('force_mae'))} | "
                f"{fmt(ratios.get('force_rmse'))} | "
                f"{fmt(ratios.get('stress_mae'))} | "
                f"{fmt(ratios.get('stress_rmse'))} | "
                f"{fmt(row.get('latency_ms_mean_ratio'))} |\n"
            )

    print(f"summary_json={output_json}")
    print(f"summary_md={output_md}")
    for row in rows:
        print(
            f"{row['mode']}: verdict={row['verdict']} "
            f"energy={fmt(row['ratios'].get('energy_mae_natoms'))} "
            f"force={fmt(row['ratios'].get('force_mae'))} "
            f"replaced={row['replaced_linear_count']}"
        )


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
