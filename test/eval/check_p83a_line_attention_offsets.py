from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from infer_salex_lmdb_quant import build_calculator, configure_precision, sync_if_needed  # noqa: E402
from matris.model.processgraph import process_graphs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P83A line-attention offset metadata validation.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default="p71_latency_pruned_fusion_only")
    parser.add_argument("--fusion-mode", default="p28_p26_all_ffn_mlp_input_grad_only")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=0)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--output-json", default="results/p83a_line_attention_offsets_check.json")
    parser.add_argument("--output-md", default="results/p83a_line_attention_offsets_check.md")
    return parser.parse_args()


def select_group_aligned_keys(dataset_len: int, limit: int, seed: int) -> np.ndarray:
    keys = np.arange(dataset_len)
    state = random.getstate()
    random.seed(seed)
    random.shuffle(keys)
    random.setstate(state)
    if limit > 0:
        keys = keys[:limit]
    return keys


def index_stats(index: torch.Tensor, num_segments: int) -> dict:
    if index.numel() <= 1:
        monotonic = True
    else:
        monotonic = bool((index[1:] >= index[:-1]).all().item())
    unique = torch.unique(index)
    unique_consecutive = torch.unique_consecutive(index)
    lengths = torch.bincount(index, minlength=num_segments)
    nonzero = lengths[lengths > 0].float()
    if nonzero.numel() == 0:
        degree = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    else:
        degree = {
            "mean": float(nonzero.mean().item()),
            "p50": float(torch.quantile(nonzero, 0.50).item()),
            "p95": float(torch.quantile(nonzero, 0.95).item()),
            "max": float(nonzero.max().item()),
        }
    return {
        "is_monotonic": monotonic,
        "is_grouped": int(unique.numel()) == int(unique_consecutive.numel()),
        "unique_segments": int(unique.numel()),
        "unique_consecutive_runs": int(unique_consecutive.numel()),
        "zero_segments": int((lengths == 0).sum().item()),
        "degree": degree,
    }


def validate_offsets(index: torch.Tensor, lengths: torch.Tensor, offsets: torch.Tensor, num_segments: int) -> dict:
    rows = int(index.numel())
    expected_lengths = torch.bincount(index, minlength=num_segments)
    offsets_ok = (
        offsets.dtype == torch.int64
        and int(offsets.numel()) == num_segments + 1
        and int(offsets[0].item()) == 0
        and int(offsets[-1].item()) == rows
        and bool((offsets[1:] >= offsets[:-1]).all().item())
    )
    lengths_ok = (
        lengths.dtype == torch.int64
        and int(lengths.numel()) == num_segments
        and int(lengths.sum().item()) == rows
        and bool(torch.equal(lengths, expected_lengths))
    )
    reconstruct_ok = False
    if offsets_ok and lengths_ok:
        expected_index = torch.repeat_interleave(
            torch.arange(num_segments, device=index.device, dtype=index.dtype),
            lengths,
        )
        reconstruct_ok = bool(torch.equal(index, expected_index))
    return {
        "lengths_ok": lengths_ok,
        "offsets_ok": offsets_ok,
        "reconstruct_sorted_index_ok": reconstruct_ok,
    }


def main() -> int:
    args = parse_args()
    os.environ["MATRIS_P83_LINE_ATTENTION_OFFSETS"] = "1"
    configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config={"src": args.dataset_src})
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    calculator = build_calculator(args)

    records: list[dict] = []
    for graph_id in tqdm(keys, desc="p83a", leave=False):
        atom = structures.get_atoms(int(graph_id))
        calculator._adjust_pbc(atom)
        structure = AseAtomsAdaptor.get_structure(atom)
        graph_cpu = calculator.model.graph_converter(structure)
        graph = graph_cpu.to(args.device)
        batch_graph = process_graphs([graph], compute_stress="s" in args.task)
        sync_if_needed(args.device)
        line_graph = batch_graph["line_graph_dict"]
        if len(line_graph["line_graph"]) == 0:
            records.append({"graph_id": int(graph_id), "has_line_graph": False})
            continue
        num_segments = int(line_graph["num_segment"])
        source_index = line_graph["source_index"]
        target_index = line_graph["target_index"]
        target_validation = validate_offsets(
            target_index,
            line_graph["target_segment_lengths"],
            line_graph["target_segment_offsets"],
            num_segments,
        )
        source_validation = validate_offsets(
            source_index,
            line_graph["source_segment_lengths"],
            line_graph["source_segment_offsets"],
            num_segments,
        )
        records.append(
            {
                "graph_id": int(graph_id),
                "has_line_graph": True,
                "rows": int(target_index.numel()),
                "num_segments": num_segments,
                "target": index_stats(target_index, num_segments),
                "source": index_stats(source_index, num_segments),
                "target_offsets": target_validation,
                "source_offsets": source_validation,
            }
        )

    checked = [item for item in records if item.get("has_line_graph")]
    summary = {
        "limit": int(args.limit),
        "sample_seed": int(args.sample_seed),
        "checked_graphs": len(checked),
        "empty_line_graphs": len(records) - len(checked),
        "target_monotonic": sum(1 for item in checked if item["target"]["is_monotonic"]),
        "target_grouped": sum(1 for item in checked if item["target"]["is_grouped"]),
        "target_reconstruct_ok": sum(1 for item in checked if item["target_offsets"]["reconstruct_sorted_index_ok"]),
        "source_monotonic": sum(1 for item in checked if item["source"]["is_monotonic"]),
        "source_grouped": sum(1 for item in checked if item["source"]["is_grouped"]),
        "source_reconstruct_ok": sum(1 for item in checked if item["source_offsets"]["reconstruct_sorted_index_ok"]),
        "max_rows": max((int(item["rows"]) for item in checked), default=0),
        "max_segments": max((int(item["num_segments"]) for item in checked), default=0),
    }
    output = {"summary": summary, "records": records}

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P83A line-attention offset metadata check",
        "",
        f"- limit: {summary['limit']}",
        f"- sample_seed: {summary['sample_seed']}",
        f"- checked graphs: {summary['checked_graphs']}",
        f"- empty line graphs: {summary['empty_line_graphs']}",
        f"- target monotonic/grouped/reconstruct_ok: {summary['target_monotonic']} / {summary['target_grouped']} / {summary['target_reconstruct_ok']}",
        f"- source monotonic/grouped/reconstruct_ok: {summary['source_monotonic']} / {summary['source_grouped']} / {summary['source_reconstruct_ok']}",
        f"- max rows: {summary['max_rows']}",
        f"- max segments: {summary['max_segments']}",
        "",
        "| graph_id | rows | segments | target sorted | target offset ok | source sorted | source offset ok |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in checked:
        lines.append(
            "| {graph_id} | {rows} | {num_segments} | {target_sorted} | {target_ok} | {source_sorted} | {source_ok} |".format(
                graph_id=item["graph_id"],
                rows=item["rows"],
                num_segments=item["num_segments"],
                target_sorted=item["target"]["is_monotonic"],
                target_ok=item["target_offsets"]["reconstruct_sorted_index_ok"],
                source_sorted=item["source"]["is_monotonic"],
                source_ok=item["source_offsets"]["reconstruct_sorted_index_ok"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if summary["target_reconstruct_ok"] != summary["checked_graphs"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
