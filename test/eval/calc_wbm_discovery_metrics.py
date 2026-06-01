import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate small-sample WBM energy and discovery metrics from joined csv."
    )
    parser.add_argument(
        "--joined-csv",
        required=True,
        help="Path to predictions_joined_with_wbm_summary.csv",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save summary metrics json",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.0,
        help="Energy-above-hull threshold in eV/atom for stable classification.",
    )
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(mean([(pred - true) ** 2 for true, pred in zip(y_true, y_pred)]))


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return mean([abs(pred - true) for true, pred in zip(y_true, y_pred)])


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) < 2:
        return float("nan")
    y_bar = mean(y_true)
    ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
    ss_tot = sum((true - y_bar) ** 2 for true in y_true)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def calc_classification_metrics(
    each_true: list[float],
    each_pred: list[float],
    stability_threshold: float,
) -> dict:
    actual_pos = [x <= stability_threshold for x in each_true]
    pred_pos = [x <= stability_threshold for x in each_pred]

    tp = sum(a and p for a, p in zip(actual_pos, pred_pos))
    fn = sum(a and (not p) for a, p in zip(actual_pos, pred_pos))
    fp = sum((not a) and p for a, p in zip(actual_pos, pred_pos))
    tn = sum((not a) and (not p) for a, p in zip(actual_pos, pred_pos))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    accuracy = (tp + tn) / len(each_true) if each_true else float("nan")
    stable_prevalence = (tp + fn) / len(each_true) if each_true else float("nan")
    daf = (
        precision / stable_prevalence
        if stable_prevalence > 0 and not math.isnan(precision)
        else float("nan")
    )
    if precision + recall == 0 or math.isnan(precision) or math.isnan(recall):
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "TPR": recall,
        "TNR": tnr,
        "Accuracy": accuracy,
        "F1": f1,
        "DAF": daf,
        "StablePrevalence": stable_prevalence,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.joined_csv)

    e_form_true = [float(row["e_form_per_atom_mp2020_corrected"]) for row in rows]
    e_form_pred = [
        float(row["e_form_per_atom_matris_from_relaxed_energy"]) for row in rows
    ]
    each_true = [float(row["e_above_hull_mp2020_corrected_ppd_mp"]) for row in rows]
    each_pred = [
        float(row["e_above_hull_mp2020_corrected_ppd_mp"])
        + float(row["energy_delta_per_atom_eV"])
        for row in rows
    ]

    energy_metrics = {
        "num_structures": len(rows),
        "formation_energy_mae": mae(e_form_true, e_form_pred),
        "formation_energy_rmse": rmse(e_form_true, e_form_pred),
        "formation_energy_r2": r2_score(e_form_true, e_form_pred),
        "each_mae": mae(each_true, each_pred),
        "each_rmse": rmse(each_true, each_pred),
        "each_r2": r2_score(each_true, each_pred),
    }
    discovery_metrics = calc_classification_metrics(
        each_true, each_pred, args.stability_threshold
    )

    summary = {
        "joined_csv": str(Path(args.joined_csv).resolve()),
        "stability_threshold": args.stability_threshold,
        "energy_metrics": energy_metrics,
        "discovery_metrics": discovery_metrics,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("=== Energy Metrics ===")
    for key, value in energy_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\n=== Discovery Metrics ===")
    for key, value in discovery_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSummary JSON: {output_path}")


if __name__ == "__main__":
    main()
