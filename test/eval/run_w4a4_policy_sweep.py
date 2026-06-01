from __future__ import annotations

import sys
import time

import run_w4a8_candidate_sweep as sweep


def build_experiments() -> list[sweep.Experiment]:
    return [
        sweep.Experiment(0, "fp32 none/no fusion", "none", "none", "anchor", "", "reference only"),
        sweep.Experiment(1, "stable W8A8 baseline", sweep.STABLE_W8A8_MODE, sweep.BASE_FUSION_MODE, "anchor", ""),
        sweep.Experiment(
            2,
            "W4A8 policy block5 core+gate",
            "w4a8_policy_b5_core_gate",
            sweep.BASE_FUSION_MODE,
            "anchor",
            "stable W8A8 baseline",
            "Reference policy that passed the W4A8 policy sweep.",
        ),
        sweep.Experiment(
            10,
            "W4A4 policy block5 core+gate",
            "w4a4_policy_b5_core_gate",
            sweep.BASE_FUSION_MODE,
            "w4a4_policy",
            "stable W8A8 baseline",
            "Same two block5 line-attn second targets as the W4A8 policy, but both weight and activation are 4-bit fake quant.",
        ),
    ]


def add_default_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def main() -> None:
    sweep.EXPERIMENTS = build_experiments()
    default_prefix = time.strftime("w4a4_policy_%Y%m%d_%H%M%S")
    add_default_arg("--run-prefix", default_prefix)
    add_default_arg("--output-root", str(sweep.RESULTS_ROOT / "w4a4_policy_sweep"))
    add_default_arg("--summary-path", str(sweep.RESULTS_ROOT / f"{default_prefix}_summary.md"))
    sweep.main()


if __name__ == "__main__":
    main()
