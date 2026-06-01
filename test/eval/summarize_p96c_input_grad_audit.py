from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P96C input-grad F.linear/layout audit JSONL.")
    parser.add_argument("audit_jsonl")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "sum": float(sum(values)),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def tensor_key(record: dict[str, Any], name: str) -> str:
    meta = record[name]
    return f"shape={meta['shape']},stride={meta['stride']},contig={meta['is_contiguous']},dtype={meta['dtype']}"


def summarize(records: list[dict[str, Any]], audit_path: Path) -> dict[str, Any]:
    timing_keys = (
        "core_linear_cpu_wall",
        "core_linear_cuda_event",
        "core_reshape_cpu_wall",
        "gate_linear_cpu_wall",
        "gate_linear_cuda_event",
        "gate_reshape_cpu_wall",
    )
    timing_stats = {
        key: stats([float(record["timings_ms"].get(key, 0.0)) for record in records])
        for key in timing_keys
    }
    module_counts = Counter(str(record["module_name"]) for record in records)
    graph_counts = Counter(str(record.get("graph_group", "unknown")) for record in records)
    rows = [int(record["grad_core_pre"]["shape"][0]) for record in records if record["grad_core_pre"]["shape"]]
    rows_counter = Counter(rows)
    tensor_layouts = {
        name: Counter(tensor_key(record, name) for record in records)
        for name in (
            "grad_core_pre",
            "grad_gate_pre",
            "core_dq_weight_t",
            "gate_dq_weight_t",
            "grad_core_2d",
            "grad_gate_2d",
            "grad_core",
            "grad_gate",
        )
    }
    contiguity = {
        name: {
            "all_contiguous": all(bool(record[name]["is_contiguous"]) for record in records),
            "non_contiguous_count": sum(1 for record in records if not bool(record[name]["is_contiguous"])),
        }
        for name in tensor_layouts
    }
    reshape_view = {
        "core_same_data_ptr_count": sum(1 for record in records if record.get("core_reshape_same_data_ptr") is True),
        "gate_same_data_ptr_count": sum(1 for record in records if record.get("gate_reshape_same_data_ptr") is True),
        "core_has_base_count": sum(1 for record in records if record.get("core_reshape_has_base") is True),
        "gate_has_base_count": sum(1 for record in records if record.get("gate_reshape_has_base") is True),
    }
    per_module = defaultdict(lambda: {key: [] for key in timing_keys})
    for record in records:
        bucket = per_module[str(record["module_name"])]
        for key in timing_keys:
            bucket[key].append(float(record["timings_ms"].get(key, 0.0)))

    return {
        "audit_path": str(audit_path),
        "record_count": len(records),
        "module_counts": dict(module_counts),
        "graph_counts": dict(graph_counts),
        "row_counts_top": rows_counter.most_common(20),
        "timing_stats": timing_stats,
        "contiguity": contiguity,
        "reshape_view": reshape_view,
        "tensor_layouts_top": {
            name: counter.most_common(8)
            for name, counter in tensor_layouts.items()
        },
        "per_module_timing": {
            module: {key: stats(values) for key, values in values_by_key.items()}
            for module, values_by_key in sorted(per_module.items())
        },
    }


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    timing = summary["timing_stats"]
    lines = [
        "# P96C Input-Grad F.linear/Layout Audit",
        "",
        f"- audit jsonl: `{summary['audit_path']}`",
        f"- records: `{summary['record_count']}`",
        "",
        "## Timing",
        "",
        "| op | sum ms | mean ms | min ms | max ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in timing.items():
        lines.append(
            f"| `{key}` | `{value['sum']:.6f}` | `{value['mean']:.6f}` | `{value['min']:.6f}` | `{value['max']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Layout",
            "",
            "| tensor | all contiguous | non-contiguous count |",
            "|---|---:|---:|",
        ]
    )
    for name, value in summary["contiguity"].items():
        lines.append(f"| `{name}` | `{str(value['all_contiguous']).lower()}` | `{value['non_contiguous_count']}` |")
    view = summary["reshape_view"]
    lines.extend(
        [
            "",
            "## Reshape View Check",
            "",
            f"- core reshape same data_ptr: `{view['core_same_data_ptr_count']}/{summary['record_count']}`",
            f"- gate reshape same data_ptr: `{view['gate_same_data_ptr_count']}/{summary['record_count']}`",
            f"- core reshape has base: `{view['core_has_base_count']}/{summary['record_count']}`",
            f"- gate reshape has base: `{view['gate_has_base_count']}/{summary['record_count']}`",
            "",
            "## Counts",
            "",
            f"- graph counts: `{summary['graph_counts']}`",
            f"- row counts top: `{summary['row_counts_top']}`",
            "",
            "## Module Counts",
            "",
            "| module | count |",
            "|---|---:|",
        ]
    )
    for module, count in summary["module_counts"].items():
        lines.append(f"| `{module}` | {count} |")
    lines.extend(["", "## Top Tensor Layouts", ""])
    for name, rows in summary["tensor_layouts_top"].items():
        lines.extend([f"### `{name}`", "", "| layout | count |", "|---|---:|"])
        for layout, count in rows:
            lines.append(f"| `{layout}` | {count} |")
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else audit_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(audit_path)
    summary = summarize(records, audit_path)
    (output_dir / "p96c_input_grad_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, output_dir / "p96c_input_grad_audit_summary.md")
    print(json.dumps(summary["timing_stats"], indent=2))


if __name__ == "__main__":
    main()
