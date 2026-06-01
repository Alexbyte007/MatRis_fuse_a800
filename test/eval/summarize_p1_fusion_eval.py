import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P1 GatedMLP fusion eval results.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--baseline-name", default="fp32_baseline")
    parser.add_argument("--profile-baseline-name", default="task_matrix_none_efs")
    parser.add_argument("--stage", default="p1d")
    parser.add_argument("--energy-ratio-threshold", type=float, default=None)
    parser.add_argument("--force-ratio-threshold", type=float, default=None)
    parser.add_argument("--speed-improvement-threshold", type=float, default=None)
    parser.add_argument("--autograd-increase-threshold", type=float, default=None)
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path. Defaults to results/p1_fusion_summary/<run-name>.json.",
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


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return candidate / baseline


def metric(summary: dict | None, key: str) -> float | None:
    if summary is None:
        return None
    return scalar(summary.get("res", {}).get(key))


def timing(summary: dict | None, key: str) -> float | None:
    if summary is None:
        return None
    return scalar(summary.get("timing", {}).get(key))


def profile_mean(summary: dict | None, key: str) -> float | None:
    if summary is None:
        return None
    row = summary.get(key, {})
    if not isinstance(row, dict):
        return None
    value = row.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def is_p1e_stage(stage: str) -> bool:
    return stage == "p1e" or stage.startswith("p1e_")


def main() -> None:
    args = parse_args()
    energy_ratio_threshold = args.energy_ratio_threshold
    force_ratio_threshold = args.force_ratio_threshold
    speed_improvement_threshold = args.speed_improvement_threshold
    autograd_increase_threshold = args.autograd_increase_threshold
    if energy_ratio_threshold is None:
        energy_ratio_threshold = 1.0005 if is_p1e_stage(args.stage) else 1.001
    if force_ratio_threshold is None:
        force_ratio_threshold = 1.001 if is_p1e_stage(args.stage) or args.stage == "p1g" else 1.003
    if speed_improvement_threshold is None:
        speed_improvement_threshold = 0.01 if is_p1e_stage(args.stage) or args.stage in ("p1f", "p1g") else 0.0
    if autograd_increase_threshold is None:
        autograd_increase_threshold = 0.01

    static_base_path = REPO_ROOT / "results" / "salex_lmdb_quant" / args.baseline_name / "summary.json"
    static_candidate_path = REPO_ROOT / "results" / "salex_lmdb_quant" / args.run_name / "summary.json"
    profile_base_path = (
        REPO_ROOT
        / "results"
        / "pipeline_profile_salex"
        / args.profile_baseline_name
        / "pipeline_profile_summary.json"
    )
    profile_candidate_path = (
        REPO_ROOT
        / "results"
        / "pipeline_profile_salex"
        / args.run_name
        / "pipeline_profile_summary.json"
    )

    static_base = load_json(static_base_path)
    static_candidate = load_json(static_candidate_path)
    profile_base = load_json(profile_base_path)
    profile_candidate = load_json(profile_candidate_path)

    energy_base = metric(static_base, "energy_mae_natoms")
    energy_candidate = metric(static_candidate, "energy_mae_natoms")
    force_base = metric(static_base, "force_mae")
    force_candidate = metric(static_candidate, "force_mae")
    latency_base = timing(static_base, "latency_ms_mean")
    latency_candidate = timing(static_candidate, "latency_ms_mean")
    interaction_base = profile_mean(profile_base, "interaction_blocks_ms")
    interaction_candidate = profile_mean(profile_candidate, "interaction_blocks_ms")
    profile_total_base = profile_mean(profile_base, "profile_total_ms")
    profile_total_candidate = profile_mean(profile_candidate, "profile_total_ms")
    force_autograd_base = profile_mean(profile_base, "force_autograd_ms")
    force_autograd_candidate = profile_mean(profile_candidate, "force_autograd_ms")
    stress_autograd_base = profile_mean(profile_base, "stress_autograd_ms")
    stress_autograd_candidate = profile_mean(profile_candidate, "stress_autograd_ms")

    energy_ratio = ratio(energy_candidate, energy_base)
    force_ratio = ratio(force_candidate, force_base)
    latency_ratio = ratio(latency_candidate, latency_base)
    interaction_ratio = ratio(interaction_candidate, interaction_base)
    profile_total_ratio = ratio(profile_total_candidate, profile_total_base)
    force_autograd_ratio = ratio(force_autograd_candidate, force_autograd_base)
    stress_autograd_ratio = ratio(stress_autograd_candidate, stress_autograd_base)

    precision_ok = (
        energy_ratio is not None
        and force_ratio is not None
        and energy_ratio <= energy_ratio_threshold
        and force_ratio <= force_ratio_threshold
    )
    speed_ratio_threshold = 1.0 - speed_improvement_threshold
    speed_ok = (
        (interaction_ratio is not None and interaction_ratio <= speed_ratio_threshold)
        or (profile_total_ratio is not None and profile_total_ratio <= speed_ratio_threshold)
        or (latency_ratio is not None and latency_ratio <= speed_ratio_threshold)
    )
    autograd_ratio_threshold = 1.0 + autograd_increase_threshold
    autograd_ok = args.stage != "p1f" or (
        (force_autograd_ratio is None or force_autograd_ratio <= autograd_ratio_threshold)
        and (stress_autograd_ratio is None or stress_autograd_ratio <= autograd_ratio_threshold)
    )

    verdict = "PASS" if precision_ok and speed_ok and autograd_ok else "FAIL"
    if precision_ok and speed_ok and not autograd_ok:
        verdict = "PRECISION_SPEED_PASS_AUTOGRAD_FAIL"
    elif precision_ok and not speed_ok:
        verdict = "PRECISION_PASS_SPEED_FAIL"
    elif speed_ok and not precision_ok:
        verdict = "SPEED_PASS_PRECISION_FAIL"

    payload = {
        "stage": args.stage,
        "run_name": args.run_name,
        "baseline_name": args.baseline_name,
        "profile_baseline_name": args.profile_baseline_name,
        "thresholds": {
            "energy_mae_natoms_ratio": energy_ratio_threshold,
            "force_mae_ratio": force_ratio_threshold,
            "speed_improvement": speed_improvement_threshold,
            "autograd_increase": autograd_increase_threshold,
        },
        "static": {
            "baseline_json": str(static_base_path),
            "candidate_json": str(static_candidate_path),
            "energy_mae_natoms": {
                "baseline": energy_base,
                "candidate": energy_candidate,
                "ratio": energy_ratio,
            },
            "force_mae": {
                "baseline": force_base,
                "candidate": force_candidate,
                "ratio": force_ratio,
            },
            "latency_ms_mean": {
                "baseline": latency_base,
                "candidate": latency_candidate,
                "ratio": latency_ratio,
            },
        },
        "profile": {
            "baseline_json": str(profile_base_path),
            "candidate_json": str(profile_candidate_path),
            "interaction_blocks_ms": {
                "baseline": interaction_base,
                "candidate": interaction_candidate,
                "ratio": interaction_ratio,
            },
            "profile_total_ms": {
                "baseline": profile_total_base,
                "candidate": profile_total_candidate,
                "ratio": profile_total_ratio,
            },
            "force_autograd_ms": {
                "baseline": force_autograd_base,
                "candidate": force_autograd_candidate,
                "ratio": force_autograd_ratio,
            },
            "stress_autograd_ms": {
                "baseline": stress_autograd_base,
                "candidate": stress_autograd_candidate,
                "ratio": stress_autograd_ratio,
            },
        },
        "precision_ok": precision_ok,
        "speed_ok": speed_ok,
        "autograd_ok": autograd_ok,
        "verdict": verdict,
    }

    output_path = Path(args.output_json) if args.output_json else (
        REPO_ROOT / "results" / "p1_fusion_summary" / f"{args.run_name}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(
        f"{args.run_name}: verdict={verdict}, "
        f"energy_ratio={energy_ratio}, force_ratio={force_ratio}, "
        f"latency_ratio={latency_ratio}, interaction_ratio={interaction_ratio}, "
        f"force_autograd_ratio={force_autograd_ratio}, stress_autograd_ratio={stress_autograd_ratio}"
    )
    print(f"summary_json={output_path}")


if __name__ == "__main__":
    main()
