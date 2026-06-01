import argparse
import json
from pathlib import Path


PAPER_STATIC_METRICS = (
    "energy_mae",
    "energy_rmse",
    "energy_mae_natoms",
    "energy_rmse_natoms",
    "force_mae",
    "force_rmse",
    "stress_mae",
    "stress_rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check fake-quant static metrics using the paper-style E/F/S error metrics only."
    )
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.005,
        help="Maximum candidate/baseline ratio allowed for each paper static metric.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(PAPER_STATIC_METRICS),
        help="Metrics to check from the comparison JSON.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    args = parse_args()
    comparison = load_json(args.comparison_json)
    rows = comparison.get("res", {})

    checked = {}
    failures = {}
    for metric in args.metrics:
        if metric not in rows:
            failures[metric] = {"reason": "missing metric"}
            continue
        ratio = rows[metric].get("ratio")
        checked[metric] = ratio
        if ratio is None or ratio > args.max_ratio:
            failures[metric] = {
                "ratio": ratio,
                "max_ratio": args.max_ratio,
            }

    result = {
        "comparison_json": str(Path(args.comparison_json).resolve()),
        "max_ratio": args.max_ratio,
        "checked_metrics": checked,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "note": "paper-style static E/F/S metrics only.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
