import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare group-aligned sAlex LMDB quant summaries."
    )
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def scalar_from_list(value):
    if isinstance(value, list) and len(value) == 1:
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def compare_values(baseline: dict, candidate: dict) -> dict:
    rows = {}
    for key in sorted(set(baseline) | set(candidate)):
        base_value = scalar_from_list(baseline.get(key))
        cand_value = scalar_from_list(candidate.get(key))
        if base_value is None or cand_value is None:
            continue
        rows[key] = {
            "baseline": base_value,
            "candidate": cand_value,
            "delta": cand_value - base_value,
            "ratio": cand_value / base_value if base_value != 0 else None,
        }
    return rows


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline_json)
    candidate = load_json(args.candidate_json)

    comparison = {
        "baseline_json": str(Path(args.baseline_json).resolve()),
        "candidate_json": str(Path(args.candidate_json).resolve()),
        "baseline_metadata": baseline.get("metadata", {}),
        "candidate_metadata": candidate.get("metadata", {}),
        "res": compare_values(baseline.get("res", {}), candidate.get("res", {})),
        "timing": compare_values(baseline.get("timing", {}), candidate.get("timing", {})),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(comparison, fp, ensure_ascii=False, indent=2)

    print(json.dumps(comparison["res"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
