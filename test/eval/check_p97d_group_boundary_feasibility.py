from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from fairchem.core.datasets import AseDBDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from infer_salex_lmdb_quant import (  # noqa: E402
    build_calculator,
    configure_precision,
    run_activation_calibration,
    select_group_aligned_keys,
)
from profile_matriscalculator_pipeline import profile_one  # noqa: E402
from profile_p88_w8a8_second_tail_breakdown import (  # noqa: E402
    BASE_ENV_FLAGS,
    FUSION_MODE,
    P8D_QUANT_MODE,
    TARGET_GROUPS,
    selected_target_globs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P97D group-level autograd boundary feasibility check.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--quant-mode", default=P8D_QUANT_MODE)
    parser.add_argument("--fusion-mode", default=FUSION_MODE)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=16)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--target-group", default="combo_8_9", choices=sorted(TARGET_GROUPS))
    parser.add_argument("--target-globs", default="")
    parser.add_argument("--output-dir", default="results/p97d_group_boundary_feasibility")
    return parser.parse_args()


def module_target_from_quant_target(target: str) -> str:
    return target.rsplit(".mlp_", 1)[0]


def parse_module_group(module_name: str) -> dict[str, Any]:
    parts = module_name.split(".")
    block_id = None
    if len(parts) > 1 and parts[0] == "interaction_block":
        try:
            block_id = int(parts[1])
        except ValueError:
            block_id = None
    if ".attn_block_line_graph." in module_name:
        graph_group = "attn_line"
    elif ".refine_block_line_graph." in module_name:
        graph_group = "refine_line"
    elif ".attn_block_atom_graph." in module_name:
        graph_group = "attn_atom"
    elif ".refine_block_atom_graph." in module_name:
        graph_group = "refine_atom"
    else:
        graph_group = "unknown"
    return {"block_id": block_id, "graph_group": graph_group}


def tensor_meta(tensor: Any) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        return {"type": type(tensor).__name__}
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "is_contiguous": bool(tensor.is_contiguous()),
        "data_ptr": int(tensor.data_ptr()) if tensor.is_cuda else None,
    }


class TargetForwardTracer:
    def __init__(self, model: torch.nn.Module, target_modules: set[str]) -> None:
        self.target_modules = target_modules
        self.records: list[dict[str, Any]] = []
        self.handles = []
        self.seq = 0
        for named_module, module in model.named_modules():
            module_name = str(getattr(module, "module_name", named_module))
            if module_name not in target_modules and named_module not in target_modules:
                continue
            display_name = module_name if module_name in target_modules else named_module
            self.handles.append(module.register_forward_hook(self._make_hook(display_name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, module_name: str):
        def hook(module, inputs, output):
            self.seq += 1
            input0 = inputs[0] if inputs else None
            self.records.append(
                {
                    "seq": self.seq,
                    "module_name": module_name,
                    **parse_module_group(module_name),
                    "input": tensor_meta(input0),
                    "output": tensor_meta(output),
                    "output_grad_fn": type(getattr(output, "grad_fn", None)).__name__
                    if isinstance(output, torch.Tensor) and output.grad_fn is not None
                    else None,
                }
            )

        return hook


def load_backward_audit(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_cycle(records: list[dict[str, Any]], size: int = 4) -> list[dict[str, Any]]:
    return records[:size]


def compact_order(records: list[dict[str, Any]]) -> list[tuple[int | None, str]]:
    return [(record.get("block_id"), str(record.get("graph_group"))) for record in records]


def analyze_feasibility(forward_records: list[dict[str, Any]], backward_records: list[dict[str, Any]]) -> dict[str, Any]:
    forward_first = first_cycle(forward_records)
    backward_first = first_cycle(backward_records)
    forward_order = compact_order(forward_first)
    backward_order = compact_order(backward_first)
    expected_forward = [(8, "attn_line"), (8, "refine_line"), (9, "attn_line"), (9, "refine_line")]
    expected_backward = [(9, "refine_line"), (9, "attn_line"), (8, "refine_line"), (8, "attn_line")]
    reverse_match = list(reversed(forward_order)) == backward_order if forward_order and backward_order else False

    crosses_blocks = len({item[0] for item in backward_order if item[0] is not None}) > 1
    contains_attn_and_refine = {"attn_line", "refine_line"}.issubset({item[1] for item in backward_order})
    verdict = "not_natural_for_second_tail_group_boundary"
    reason = [
        "The backward four-call group is the reverse of a sequential forward chain, not a forward-time batch.",
        "Block 9 depends on outputs from block 8 through MatRIS.model.run_interaction_blocks.",
        "Within one Interaction_Block, refine_line runs after attn_line/attn_atom and consumes their outputs.",
        "A second-tail-only group autograd Function would need all four inputs before downstream use, which is not available without wrapping a much larger block.",
    ]
    feasible_next_boundary = "interaction_block_or_edge_update_macro"
    should_integrate_p97_group8_directly = False

    return {
        "forward_first_order": forward_order,
        "backward_first_order": backward_order,
        "expected_forward_order": expected_forward,
        "expected_backward_order": expected_backward,
        "reverse_match": reverse_match,
        "crosses_blocks": crosses_blocks,
        "contains_attn_and_refine": contains_attn_and_refine,
        "verdict": verdict,
        "reason": reason,
        "feasible_next_boundary": feasible_next_boundary,
        "should_integrate_p97_group8_directly": should_integrate_p97_group8_directly,
    }


def write_summary(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "p97d_group_boundary_feasibility.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    analysis = result["analysis"]
    lines = [
        "# P97D Group-Level Autograd Boundary Feasibility",
        "",
        "## Orders",
        "",
        f"- forward first order: `{analysis['forward_first_order']}`",
        f"- backward first order: `{analysis['backward_first_order']}`",
        f"- reverse match: `{str(analysis['reverse_match']).lower()}`",
        f"- crosses blocks: `{str(analysis['crosses_blocks']).lower()}`",
        f"- contains attn/refine: `{str(analysis['contains_attn_and_refine']).lower()}`",
        "",
        "## Verdict",
        "",
        f"- verdict: `{analysis['verdict']}`",
        f"- integrate P97 group8 directly: `{str(analysis['should_integrate_p97_group8_directly']).lower()}`",
        f"- feasible next boundary: `{analysis['feasible_next_boundary']}`",
        "",
        "原因：",
    ]
    lines.extend(f"- {item}" for item in analysis["reason"])
    lines.extend(
        [
            "",
            "结论：",
            "- P97 group8 input-grad 在 replay 里有效，但当前模型没有自然的 second-tail group-level autograd 边界。",
            "- 如果继续做 group8，需要把边界扩大到 interaction block / edge_update macro，已经不是 input-grad 小修。",
            "- 因此不建议把 `MATRIS_P97_GROUP8_INPUT_GRAD=1` 直接接入当前 saved-pre wrapper。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("P97D feasibility check is intended for CUDA.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_globs = selected_target_globs(args)
    target_modules = {module_target_from_quant_target(target) for target in target_globs}
    audit_path = output_dir / "p95_backward_audit.jsonl"

    env_flags = dict(BASE_ENV_FLAGS)
    env_flags.update(
        {
            "MATRIS_QUANT_INCLUDE_GLOBS": ",".join(target_globs),
            "MATRIS_P95_GROUPING_AUDIT_PATH": str(audit_path),
            "MATRIS_P30_SAVED_PRE_FUSED_INPUT_GRAD": "0",
            "MATRIS_P31_TILED_INPUT_GRAD_MATMUL": "0",
            "MATRIS_P33_CUBLAS_GROUPED_INPUT_GRAD": "0",
            "MATRIS_P34_BMM_INPUT_GRAD": "0",
            "MATRIS_P40_SAVED_PRE_DQ_FUSED_INPUT_GRAD": "0",
            "MATRIS_P41_CUBLAS_PAIR_INPUT_GRAD": "0",
            "MATRIS_P43_TAIL_STACK_BMM_INPUT_GRAD": "0",
            "MATRIS_P89_TAIL_BWD_V2": "0",
            "MATRIS_P91_BACKWARD_DRIVER": "0",
            "MATRIS_P91_CUBLAS_PAIR": "0",
        }
    )
    for key, value in env_flags.items():
        os.environ[key] = value
    if audit_path.exists():
        audit_path.unlink()

    configure_precision(args.device, args.precision_mode)
    structures = AseDBDataset(config=dict(src=args.dataset_src))
    calculator = build_calculator(args)
    run_activation_calibration(structures, calculator, args)
    keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    tracer = TargetForwardTracer(calculator.model, target_modules)
    try:
        for graph_id in keys[: max(0, args.warmup_steps)]:
            try:
                profile_one(structures, int(graph_id), calculator, args)
            except Exception:
                pass
        records = []
        for graph_id in keys:
            records.append(profile_one(structures, int(graph_id), calculator, args))
    finally:
        tracer.close()

    backward_records = load_backward_audit(audit_path)
    analysis = analyze_feasibility(tracer.records, backward_records)
    result = {
        "metadata": {
            "target_group": args.target_group,
            "target_globs": target_globs,
            "target_modules": sorted(target_modules),
            "limit": args.limit,
            "sample_seed": args.sample_seed,
            "activation_calibration_limit": args.activation_calibration_limit,
            "audit_path": str(audit_path),
        },
        "profile_records": records,
        "forward_records_count": len(tracer.records),
        "backward_records_count": len(backward_records),
        "forward_first_records": first_cycle(tracer.records),
        "backward_first_records": first_cycle(backward_records),
        "analysis": analysis,
    }
    write_summary(output_dir, result)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
