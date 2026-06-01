from __future__ import annotations

import sys
import time

import run_w4a8_candidate_sweep as sweep


BRANCHES = (
    ("core", "mlp_core"),
    ("gate", "mlp_gate"),
)


def build_experiments() -> list[sweep.Experiment]:
    experiments = [
        sweep.Experiment(0, "fp32 none/no fusion", "none", "none", "anchor", "", "reference only"),
        sweep.Experiment(1, "stable W8A8 baseline", sweep.STABLE_W8A8_MODE, sweep.BASE_FUSION_MODE, "anchor", ""),
    ]
    idx = 10
    for block in range(4, 10):
        for suffix, branch in BRANCHES:
            experiments.append(
                sweep.Experiment(
                    idx,
                    f"W4A8 mixed stable + attn line second block {block} {suffix}",
                    f"w4a8_mixed_stable_single_attn_line_second_b{block}_{suffix}",
                    sweep.BASE_FUSION_MODE,
                    "w4a8_fine_single",
                    "stable W8A8 baseline",
                    (
                        "Stable W8A8 targets stay W8A8; this one "
                        f"attn line second {branch}.layers.3 target is W4A8 fake."
                    ),
                )
            )
            idx += 1
    idx = 30
    for block in range(4, 10):
        experiments.append(
            sweep.Experiment(
                idx,
                f"W4A8 mixed stable + attn line second block {block} core+gate",
                f"w4a8_mixed_stable_single_attn_line_second_b{block}_core_gate",
                sweep.BASE_FUSION_MODE,
                "w4a8_fine_block",
                "stable W8A8 baseline",
                "Stable W8A8 targets stay W8A8; this block's core and gate second line-attn targets are W4A8 fake.",
            )
        )
        idx += 1
    return experiments


def add_default_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def main() -> None:
    sweep.EXPERIMENTS = build_experiments()
    default_prefix = time.strftime("w4a8_fine_%Y%m%d_%H%M%S")
    add_default_arg("--run-prefix", default_prefix)
    add_default_arg("--output-root", str(sweep.RESULTS_ROOT / "w4a8_fine_sweep"))
    add_default_arg("--summary-path", str(sweep.RESULTS_ROOT / f"{default_prefix}_summary.md"))
    sweep.main()


if __name__ == "__main__":
    main()
