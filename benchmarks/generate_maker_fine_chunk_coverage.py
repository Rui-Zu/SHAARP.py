from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.compare_maker_fringes_reference import compare_maker_reference_payloads


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_coverage_v1.json"

CHUNKS = [
    {
        "chunk_id": "negative_normal_positive_start_4_count_5",
        "case_id": "fine_maker_negative_normal_positive",
        "start": 4,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_python_output_v1.json",
    },
    {
        "chunk_id": "negative_normal_positive_center_start_48_count_5",
        "case_id": "fine_maker_negative_normal_positive",
        "start": 48,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_normal_center_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_normal_center_python_output_v1.json",
    },
    {
        "chunk_id": "negative_normal_positive_positive_edge_start_92_count_5",
        "case_id": "fine_maker_negative_normal_positive",
        "start": 92,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_positive_edge_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_positive_edge_python_output_v1.json",
    },
    {
        "chunk_id": "high_oblique_low_start_0_count_5",
        "case_id": "fine_maker_high_oblique",
        "start": 0,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_high_oblique_low_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_high_oblique_low_python_output_v1.json",
    },
    {
        "chunk_id": "high_oblique_mid_start_80_count_5",
        "case_id": "fine_maker_high_oblique",
        "start": 80,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_high_oblique_mid_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_high_oblique_mid_python_output_v1.json",
    },
    {
        "chunk_id": "high_oblique_tail_start_156_count_5",
        "case_id": "fine_maker_high_oblique",
        "start": 156,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_high_oblique_tail_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_high_oblique_tail_python_output_v1.json",
    },
    {
        "chunk_id": "fringe_resolution_negative_edge_start_0_count_5",
        "case_id": "fine_maker_fringe_resolution",
        "start": 0,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_fringe_negative_edge_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_fringe_negative_edge_python_output_v1.json",
    },
    {
        "chunk_id": "fringe_resolution_center_start_240_count_5",
        "case_id": "fine_maker_fringe_resolution",
        "start": 240,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_fringe_center_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_fringe_center_python_output_v1.json",
    },
    {
        "chunk_id": "negative_normal_positive_mid_start_24_count_5",
        "case_id": "fine_maker_negative_normal_positive",
        "start": 24,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_neg_mid_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_neg_mid_python_output_v1.json",
    },
    {
        "chunk_id": "negative_normal_positive_upper_start_70_count_5",
        "case_id": "fine_maker_negative_normal_positive",
        "start": 70,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_neg_pos_edge_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_neg_pos_edge_python_output_v1.json",
    },
    {
        "chunk_id": "high_oblique_lowmid_start_40_count_5",
        "case_id": "fine_maker_high_oblique",
        "start": 40,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_high_oblique_lowmid_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_high_oblique_lowmid_python_output_v1.json",
    },
    {
        "chunk_id": "high_oblique_high_start_120_count_5",
        "case_id": "fine_maker_high_oblique",
        "start": 120,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_high_oblique_high_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_high_oblique_high_python_output_v1.json",
    },
    {
        "chunk_id": "fringe_resolution_negative_mid_start_120_count_5",
        "case_id": "fine_maker_fringe_resolution",
        "start": 120,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_fringe_neg_mid_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_fringe_neg_mid_python_output_v1.json",
    },
    {
        "chunk_id": "fringe_resolution_positive_mid_start_360_count_5",
        "case_id": "fine_maker_fringe_resolution",
        "start": 360,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_fringe_pos_mid_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_fringe_pos_mid_python_output_v1.json",
    },
    {
        "chunk_id": "fringe_resolution_positive_tail_start_476_count_5",
        "case_id": "fine_maker_fringe_resolution",
        "start": 476,
        "count": 5,
        "reference": "benchmarks/mathematica_reference/maker_fine_chunk_fringe_pos_tail_reference_v1.json",
        "python": "benchmarks/maker_fine_chunk_fringe_pos_tail_python_output_v1.json",
    },
]


# Fill chunks tiling every fine-Maker angle not covered by the count=5 probe chunks
# above. Together with the 15 entries above these partition each case completely and
# without overlap, reaching all 739 fine angles. Each (short, case_id, start, count)
# maps to live Wolfram references exported by benchmarks/fill_fine_maker_chunks.py.
_FILL_SPECS = [
    ("neg", "fine_maker_negative_normal_positive", 0, 4),
    ("neg", "fine_maker_negative_normal_positive", 9, 15),
    ("neg", "fine_maker_negative_normal_positive", 29, 19),
    ("neg", "fine_maker_negative_normal_positive", 53, 17),
    ("neg", "fine_maker_negative_normal_positive", 75, 17),
    ("hob", "fine_maker_high_oblique", 5, 35),
    ("hob", "fine_maker_high_oblique", 45, 35),
    ("hob", "fine_maker_high_oblique", 85, 35),
    ("hob", "fine_maker_high_oblique", 125, 31),
    ("frng", "fine_maker_fringe_resolution", 5, 60),
    ("frng", "fine_maker_fringe_resolution", 65, 55),
    ("frng", "fine_maker_fringe_resolution", 125, 60),
    ("frng", "fine_maker_fringe_resolution", 185, 55),
    ("frng", "fine_maker_fringe_resolution", 245, 60),
    ("frng", "fine_maker_fringe_resolution", 305, 55),
    ("frng", "fine_maker_fringe_resolution", 365, 60),
    ("frng", "fine_maker_fringe_resolution", 425, 51),
]

CHUNKS += [
    {
        "chunk_id": f"fill_{short}_{start}_{count}",
        "case_id": case_id,
        "start": start,
        "count": count,
        "reference": f"benchmarks/mathematica_reference/maker_fine_chunk_fill_{short}_{start}_{count}_reference_v1.json",
        "python": f"benchmarks/maker_fine_chunk_fill_{short}_{start}_{count}_python_output_v1.json",
    }
    for short, case_id, start, count in _FILL_SPECS
]


def build_coverage(root: Path = ROOT, *, atol: float = 1e-10, rtol: float = 1e-10) -> dict[str, Any]:
    fine_input = json.loads((root / "benchmarks" / "mathematica_reference" / "maker_fine_mathematica_inputs_v1.json").read_text(encoding="utf-8"))
    total_angles = int(fine_input["total_angle_count"])
    chunk_records = [_chunk_record(root, chunk, atol=atol, rtol=rtol) for chunk in CHUNKS]
    verified = sum(record["angle_count"] for record in chunk_records if record["passes_value_comparison"])
    failed = sum(1 for record in chunk_records if not record["passes_value_comparison"])
    full_validated = verified == total_angles and failed == 0
    return {
        "schema_version": 1,
        "status": "complete_fine_maker_chunk_coverage" if full_validated else "partial_fine_maker_chunk_coverage",
        "source": "Live Wolfram fine Maker chunk references compared with matching Python fine-grid slices",
        "full_fine_grid_validated": full_validated,
        "total_fine_angle_count": total_angles,
        "verified_angle_count": verified,
        "remaining_angle_count": total_angles - verified,
        "verified_chunk_count": sum(1 for record in chunk_records if record["passes_value_comparison"]),
        "failed_comparison_count": failed,
        "atol": atol,
        "rtol": rtol,
        "chunks": chunk_records,
    }


def _chunk_record(root: Path, chunk: dict[str, Any], *, atol: float, rtol: float) -> dict[str, Any]:
    reference = json.loads((root / chunk["reference"]).read_text(encoding="utf-8"))
    python = json.loads((root / chunk["python"]).read_text(encoding="utf-8"))
    results = compare_maker_reference_payloads(reference, python, atol=atol, rtol=rtol)
    failed = [result for result in results if not result.passed]
    py_case = python["cases"][0]
    return {
        "chunk_id": chunk["chunk_id"],
        "case_id": chunk["case_id"],
        "start": chunk["start"],
        "count": chunk["count"],
        "angle_count": len(py_case["theta_deg"]),
        "theta_deg": py_case["theta_deg"],
        "reference": chunk["reference"],
        "python": chunk["python"],
        "comparison_count": len(results),
        "failed_comparison_count": len(failed),
        "passes_value_comparison": not failed,
        "max_abs_error": max((result.max_abs_error for result in results), default=0.0),
        "max_rel_error": max((result.max_rel_error for result in results), default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fine Maker chunk coverage report.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()
    payload = build_coverage(ROOT, atol=args.atol, rtol=args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote fine Maker chunk coverage to {args.output}")


if __name__ == "__main__":
    main()
