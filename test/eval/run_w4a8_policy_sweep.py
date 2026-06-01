from __future__ import annotations

import sys
import time

import run_w4a8_candidate_sweep as sweep


POLICY_MODES = [
    (
        10,
        "W4A8 policy block5 core+gate",
        "w4a8_policy_b5_core_gate",
        "Most conservative block-level PASS from the fine sweep.",
    ),
    (
        11,
        "W4A8 policy gates b5+b8+b9",
        "w4a8_policy_gate_b5_b8_b9",
        "Gate-only combination using the suggested stable gate candidates.",
    ),
    (
        12,
        "W4A8 policy all PASS gates",
        "w4a8_policy_all_pass_gates",
        "All gate branches that passed the fine sweep: b4, b5, b7, b8, b9.",
    ),
    (
        13,
        "W4A8 policy conservative mix",
        "w4a8_policy_conservative_mix",
        "Block5 core+gate plus b8 gate and b9 gate.",
    ),
    (
        14,
        "W4A8 policy all PASS singles",
        "w4a8_policy_all_pass_singles",
        "All single-branch PASS targets from the fine sweep combined.",
    ),
]


def build_experiments() -> list[sweep.Experiment]:
    experiments = [
        sweep.Experiment(0, "fp32 none/no fusion", "none", "none", "anchor", "", "reference only"),
        sweep.Experiment(1, "stable W8A8 baseline", sweep.STABLE_W8A8_MODE, sweep.BASE_FUSION_MODE, "anchor", ""),
    ]
    for idx, label, mode, notes in POLICY_MODES:
        experiments.append(
            sweep.Experiment(
                idx,
                label,
                mode,
                sweep.BASE_FUSION_MODE,
                "w4a8_policy",
                "stable W8A8 baseline",
                notes,
            )
        )
    return experiments


def add_default_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def main() -> None:
    sweep.EXPERIMENTS = build_experiments()
    default_prefix = time.strftime("w4a8_policy_%Y%m%d_%H%M%S")
    add_default_arg("--run-prefix", default_prefix)
    add_default_arg("--output-root", str(sweep.RESULTS_ROOT / "w4a8_policy_sweep"))
    add_default_arg("--summary-path", str(sweep.RESULTS_ROOT / f"{default_prefix}_summary.md"))
    sweep.main()


if __name__ == "__main__":
    main()
