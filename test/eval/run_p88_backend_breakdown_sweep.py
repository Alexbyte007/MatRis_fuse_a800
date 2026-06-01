from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
PYTHON = sys.executable


P88_SCRIPT = REPO_ROOT / "test" / "eval" / "profile_p88_w8a8_second_tail_breakdown.py"


VARIANT_ENV_KEYS = (
    "MATRIS_P30_SAVED_PRE_FUSED_INPUT_GRAD",
    "MATRIS_P30_SAVED_PRE_FUSED_INPUT_GRAD_MAX_ROWS",
    "MATRIS_P40_SAVED_PRE_DQ_FUSED_INPUT_GRAD",
    "MATRIS_P40_SAVED_PRE_DQ_FUSED_INPUT_GRAD_MIN_ROWS",
    "MATRIS_P40_SAVED_PRE_DQ_FUSED_INPUT_GRAD_MAX_ROWS",
    "MATRIS_P34_BMM_INPUT_GRAD",
    "MATRIS_P34_BMM_INPUT_GRAD_MIN_ROWS",
    "MATRIS_P34_BMM_INPUT_GRAD_MAX_ROWS",
    "MATRIS_P41_CUBLAS_PAIR_INPUT_GRAD",
    "MATRIS_P41_CUBLAS_PAIR_INPUT_GRAD_MIN_ROWS",
    "MATRIS_P41_CUBLAS_PAIR_INPUT_GRAD_MAX_ROWS",
    "MATRIS_P31_TILED_INPUT_GRAD_MATMUL",
    "MATRIS_P31_TILED_INPUT_GRAD_MATMUL_VARIANT",
    "MATRIS_P31_TILED_INPUT_GRAD_MATMUL_MIN_ROWS",
    "MATRIS_P31_TILED_INPUT_GRAD_MATMUL_MAX_ROWS",
)


@dataclass(frozen=True)
class Variant:
    label: str
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""


VARIANTS: tuple[Variant, ...] = (
    Variant("baseline_torch_linear", notes="Current saved-pre fallback: tail backward + torch F.linear input-grad."),
    Variant(
        "p40_saved_pre_dq_fused_input_grad",
        {
            "MATRIS_P40_SAVED_PRE_DQ_FUSED_INPUT_GRAD": "1",
            "MATRIS_P40_SAVED_PRE_DQ_FUSED_INPUT_GRAD_MIN_ROWS": "0",
            "MATRIS_P40_SAVED_PRE_DQ_FUSED_INPUT_GRAD_MAX_ROWS": "1000000000",
        },
        "Fused saved-pre tail backward + input-grad using dequantized weights.",
    ),
    Variant(
        "p30_saved_pre_int8_fused_input_grad",
        {
            "MATRIS_P30_SAVED_PRE_FUSED_INPUT_GRAD": "1",
            "MATRIS_P30_SAVED_PRE_FUSED_INPUT_GRAD_MAX_ROWS": "1000000000",
        },
        "Fused saved-pre tail backward + input-grad using int8 weights and scales.",
    ),
    Variant(
        "p41_cublas_pair_input_grad",
        {
            "MATRIS_P41_CUBLAS_PAIR_INPUT_GRAD": "1",
            "MATRIS_P41_CUBLAS_PAIR_INPUT_GRAD_MIN_ROWS": "0",
            "MATRIS_P41_CUBLAS_PAIR_INPUT_GRAD_MAX_ROWS": "1000000000",
        },
        "Keep tail backward separate, use cublas pair op for dual input-grad.",
    ),
    Variant(
        "p34_bmm_input_grad",
        {
            "MATRIS_P34_BMM_INPUT_GRAD": "1",
            "MATRIS_P34_BMM_INPUT_GRAD_MIN_ROWS": "0",
            "MATRIS_P34_BMM_INPUT_GRAD_MAX_ROWS": "1000000000",
        },
        "Keep tail backward separate, stack dual input-grad into torch.bmm.",
    ),
    Variant(
        "p31_tiled_input_grad_m16n16",
        {
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL": "1",
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL_VARIANT": "m16n16",
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL_MIN_ROWS": "0",
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL_MAX_ROWS": "1000000000",
        },
        "Custom tiled input-grad matmul, m16n16 variant.",
    ),
    Variant(
        "p31_tiled_input_grad_m32n8",
        {
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL": "1",
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL_VARIANT": "m32n8",
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL_MIN_ROWS": "0",
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL_MAX_ROWS": "1000000000",
        },
        "Custom tiled input-grad matmul, m32n8 variant.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P88B dormant backend breakdown sweep.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--output-root", default=str(RESULTS_ROOT / "p88_backend_breakdown_sweep"))
    parser.add_argument("--summary-path", default=str(RESULTS_ROOT / "p88_backend_breakdown_sweep_summary.md"))
    parser.add_argument("--only", default="", help="Comma-separated variant labels. Empty runs all.")
    return parser.parse_args()


def selected_variants(only: str) -> list[Variant]:
    if not only.strip():
        return list(VARIANTS)
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    return [variant for variant in VARIANTS if variant.label in wanted]


def run_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for key in VARIANT_ENV_KEYS:
        env.pop(key, None)
    env.update(extra)
    return env


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_range(summary: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if summary is None:
        return None
    for row in summary.get("range_rows", []):
        if row.get("name") == name:
            return row
    return None


def range_kernel(summary: dict[str, Any] | None, name: str) -> float | None:
    row = find_range(summary, name)
    if row is None:
        return None
    value = row.get("kernel_ms")
    return float(value) if value is not None else None


def sum_range_prefix(summary: dict[str, Any] | None, prefix: str) -> float | None:
    if summary is None:
        return None
    values = [
        float(row.get("kernel_ms", 0.0))
        for row in summary.get("range_rows", [])
        if str(row.get("name", "")).startswith(prefix)
    ]
    return sum(values) if values else None


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def run_variant(variant: Variant, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    cmd = [
        PYTHON,
        str(P88_SCRIPT),
        "--device",
        args.device,
        "--limit",
        str(args.limit),
        "--warmup-steps",
        str(args.warmup_steps),
        "--sample-seed",
        str(args.sample_seed),
        "--activation-calibration-limit",
        str(args.activation_calibration_limit),
        "--activation-calibration-seed",
        str(args.activation_calibration_seed),
        "--output-dir",
        str(output_dir),
    ]
    print(f"\n=== {variant.label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=run_env(variant.env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()

    summary_path = output_dir / "summary.json"
    summary = load_json(summary_path)
    backward_total = sum_range_prefix(summary, "saved_pre.backward.")
    return {
        "label": variant.label,
        "notes": variant.notes,
        "env": variant.env,
        "returncode": returncode,
        "output_dir": str(output_dir),
        "summary_json": str(summary_path),
        "summary_md": str(output_dir / "summary.md"),
        "trace": str(output_dir / "trace.json"),
        "forward_kernel_with_pre_ms": range_kernel(summary, "saved_pre.forward.kernel_with_pre"),
        "forward_prepare_ms": range_kernel(summary, "saved_pre.forward.prepare"),
        "forward_save_for_backward_ms": range_kernel(summary, "saved_pre.forward.save_for_backward"),
        "backward_total_ms": backward_total,
        "tail_backward_ms": range_kernel(summary, "saved_pre.backward.tail_backward"),
        "input_grad_torch_linear_ms": range_kernel(summary, "saved_pre.backward.input_grad_torch_linear"),
        "fused_tail_input_grad_dq_ms": range_kernel(summary, "saved_pre.backward.fused_tail_input_grad_dq"),
        "fused_tail_input_grad_int8_ms": range_kernel(summary, "saved_pre.backward.fused_tail_input_grad_int8"),
        "input_grad_cublas_pair_ms": range_kernel(summary, "saved_pre.backward.input_grad_cublas_pair"),
        "input_grad_bmm_ms": range_kernel(summary, "saved_pre.backward.input_grad_bmm"),
        "input_grad_tiled_ms": range_kernel(summary, "saved_pre.backward.input_grad_tiled_matmul"),
        "hit_ranges": [
            row.get("name")
            for row in (summary or {}).get("range_rows", [])
            if str(row.get("name", "")).startswith("saved_pre.backward.")
        ],
    }


def write_summary(records: list[dict[str, Any]], summary_path: Path, args: argparse.Namespace) -> None:
    baseline = next((row for row in records if row["label"] == "baseline_torch_linear"), None)
    baseline_bwd = baseline.get("backward_total_ms") if baseline else None
    lines = [
        "# P88B Dormant Backend Breakdown Sweep",
        "",
        f"- generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- protocol: `limit={args.limit}`, `warmup_steps={args.warmup_steps}`, `sample_seed={args.sample_seed}`",
        "- target: W8A8 saved-pre second-tail on `refine blocks 8-9 + attn blocks 8-9`",
        "- note: profiler timing is for attribution; use E/F/S eval only after a backend wins locally.",
        "",
        "## Result Table",
        "",
        "| backend | rc | backward total ms | vs baseline | tail backward | torch linear | fused dq | fused int8 | cublas pair | bmm | tiled | forward kernel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        bwd = row.get("backward_total_ms")
        ratio = (bwd / baseline_bwd) if bwd is not None and baseline_bwd else None
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    str(row.get("returncode")),
                    fmt(bwd),
                    fmt(ratio),
                    fmt(row.get("tail_backward_ms")),
                    fmt(row.get("input_grad_torch_linear_ms")),
                    fmt(row.get("fused_tail_input_grad_dq_ms")),
                    fmt(row.get("fused_tail_input_grad_int8_ms")),
                    fmt(row.get("input_grad_cublas_pair_ms")),
                    fmt(row.get("input_grad_bmm_ms")),
                    fmt(row.get("input_grad_tiled_ms")),
                    fmt(row.get("forward_kernel_with_pre_ms")),
                ]
            )
            + " |"
        )

    passing = [
        row
        for row in records
        if row.get("returncode") == 0
        and row.get("backward_total_ms") is not None
        and row["label"] != "baseline_torch_linear"
    ]
    best = sorted(passing, key=lambda row: float(row["backward_total_ms"]))[:3]
    lines.extend(["", "## Local Winners", ""])
    if best:
        for row in best:
            bwd = row.get("backward_total_ms")
            ratio = (bwd / baseline_bwd) if bwd is not None and baseline_bwd else None
            lines.append(
                f"- `{row['label']}`: backward `{fmt(bwd)} ms`, ratio `{fmt(ratio)}x`, "
                f"hit ranges `{', '.join(row.get('hit_ranges') or [])}`"
            )
    else:
        lines.append("- No backend beat or completed against baseline.")

    lines.extend(["", "## Details", ""])
    for row in records:
        lines.extend(
            [
                f"- **{row['label']}**",
                f"  - notes: {row.get('notes') or 'n/a'}",
                f"  - env: `{json.dumps(row.get('env', {}), sort_keys=True)}`",
                f"  - summary: `{row['summary_md']}`",
                f"  - trace: `{row['trace']}`",
            ]
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_prefix = args.run_prefix or time.strftime("p88_backend_breakdown_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / run_prefix
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for variant in selected_variants(args.only):
        record = run_variant(variant, output_root / variant.label, args)
        records.append(record)
        (output_root / "partial_records.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_summary(records, Path(args.summary_path), args)

    write_summary(records, Path(args.summary_path), args)
    print(f"\nSummary written to {args.summary_path}", flush=True)
    print(f"Records written to {output_root / 'partial_records.json'}", flush=True)


if __name__ == "__main__":
    main()
