"""Point-group SHAARP.si single-interface reflected-SHG benchmark generator (sipg2).

This COMPLETES point-group coverage of the SI reflected-SHG path. sipg1 validated
6 crystal classes (3m/-42m/mm2/4mm/32/-6m2) against live SHAARP.si ER2w; the
transmitted/Maker pipeline (pg1+pg2+pg3) is validated for all 13 distinct
buildable SHG d-patterns. sipg2 adds the remaining 8 point-group representatives
NOT in sipg1 -- 2, m, -4, 422, 3, -6, 4, 6 -- so that the SI reflected-SHG path is
validated at FULL PARITY with the transmitted path: every distinct buildable
symmetry-constrained d-pattern is now exercised in BOTH geometries.

It reuses the EXACT machinery of the sipg1 generator (bounded-eps construction,
``_pointgroup_d_voigt_crystal`` symbolic-template substitution, crystal->lab
rotation, and the shared ``_case_mapping`` round-trip recovery of the crystal
``dold`` from the lab ``d``). The ONLY differences from sipg1 are:
  * RAW_CASES -- the 8 remaining crystal classes (with diverse orientation/theta/pol),
  * SEED_OFFSET = 1500 (well beyond v1 0.., ext1 500.., sipg1 900..) so the complex
    eps off-diagonal inputs are genuinely distinct,
  * tag = "sipg2".

It does NOT touch any v1, ext1, or sipg1 artifact and creates only new ``sipg2``
files (benchmark + input mapping into Copy-tree and ORIGINAL-tree dirs).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.generate_solver_stage_benchmarks import (
    complex_array_to_json,
    complex_symmetric_epsilon,
    orientation_matrix,
)
# Reuse sipg1's validated helpers verbatim so the only divergence is the case set.
from benchmarks.generate_solver_stage_benchmarks_si_pg import (
    _eps_axes,
    _pointgroup_d_voigt_crystal,
    _case_mapping_with_pointgroup,
    run_case,
)
from shaarp.tensors import rotate_d_voigt_crystal_to_lab


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
# Optional second working tree to mirror generated references into, e.g. a checkout of the
# original Mathematica-side project. Unset by default: without it this script writes only inside
# its own repository.
ORIGINAL_REF_DIR = (
    Path(os.environ["SHAARP_ORIGINAL_REF_DIR"])
    if os.environ.get("SHAARP_ORIGINAL_REF_DIR")
    else None
)

TAG = "sipg2"
# Placed beyond sipg1 (900..) so the complex eps off-diagonal parts are distinct inputs.
SEED_OFFSET = 1500

# The 8 point-group representatives NOT covered by sipg1, spanning every distinct
# buildable SHG d-pattern that the transmitted/Maker pg1+pg2+pg3 grids cover.
# (point_group, orientation_index, theta_deg, incident_polarization)
RAW_CASES = [
    ("2", 0, 14.0, "p"),
    ("m", 1, 27.5, "s"),
    ("-4", 2, 9.0, "p"),
    ("422", 3, 41.0, "s"),
    ("3", 4, 33.0, "p"),
    ("-6", 5, 19.0, "s"),
    ("4", 6, 52.0, "p"),
    ("6", 7, 6.5, "s"),
]


def build_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases: list[dict] = []
    for point_group, orientation_idx, theta_deg, pol in raw:
        idx = len(cases)
        orientation = orientation_matrix(orientation_idx)
        angle_idx = idx
        eps_w_axes, eps_2w_axes = _eps_axes(orientation_idx, angle_idx)
        seed = SEED_OFFSET + idx
        d_crystal = _pointgroup_d_voigt_crystal(point_group, seed)
        d_voigt_lab = rotate_d_voigt_crystal_to_lab(d_crystal, orientation)
        crystal_nonzero_d = int(np.sum(np.abs(d_crystal) > 1e-12))
        lab_nonzero_d = int(np.sum(np.abs(d_voigt_lab) > 1e-12))
        cases.append(
            {
                "id": f"si_pg2_stage_{idx + 1:02d}",
                "point_group": point_group,
                "theta_deg": theta_deg,
                "incident_polarization": pol,
                "orientation_index": orientation_idx,
                "orientation": orientation,
                "epsilon_omega_crystal": complex_symmetric_epsilon(seed, eps_w_axes, imag_base=0.025),
                "epsilon_2omega_crystal": complex_symmetric_epsilon(seed + 100, eps_2w_axes, imag_base=0.035),
                "d_voigt_crystal": d_crystal,
                "d_voigt_lab": d_voigt_lab,
                "crystal_nonzero_d": crystal_nonzero_d,
                "lab_nonzero_d": lab_nonzero_d,
            }
        )
    return cases


def build_benchmark(limit: int | None = None) -> dict:
    records = [run_case(case) for case in build_cases(limit=limit)]
    return {
        "benchmark_version": 1,
        "tag": TAG,
        "status": "python_solver_stage_residual_regression_si_pointgroup_not_validation",
        "extension_of": "benchmarks/mathematica_reference/solver_stage_benchmarks_si_pg1.json",
        "seed_offset": SEED_OFFSET,
        "case_count": len(records),
        "point_groups": [c["point_group"] for c in build_cases(limit=limit)],
        "notes": [
            "Single-interface reflected-SHG cases for the 8 crystal classes NOT in sipg1 (2/m/-4/422/3/-6/4/6).",
            "Together with sipg1 (3m/-42m/mm2/4mm/32/-6m2), reflected SHG now exercises every distinct buildable SHG d-pattern -- full parity with the transmitted/Maker pg1+pg2+pg3 coverage.",
            "d_voigt_lab = rotate_d_voigt_crystal_to_lab(pointgroup_crystal_d, orientation); the crystal tensor has <18 nonzero entries.",
            "eps construction matches the sipg1/ext1 SI generator exactly; only the d tensor and seed differ.",
            "Python-constructed inputs; the validation is Python solve_shaarp_si_reflected_shg ER2w vs live Mathematica SHAARP.si ER2w.",
        ],
        "cases": records,
    }


def build_mapping(benchmark: dict) -> dict:
    cases = [_case_mapping_with_pointgroup(case) for case in benchmark["cases"]]
    return {
        "schema_version": 1,
        "tag": TAG,
        "source": "SHAARP.si Mathematica input mapping for point-group single-interface solver-stage cases (sipg2: remaining 8 classes)",
        "status": "input_mapping_not_validation_evidence",
        "validation_claim": False,
        "extension_of": "benchmarks/mathematica_reference/shaarp_si_case_input_mapping_sipg1.json",
        "case_count": len(cases),
        "point_groups": [c["point_group"] for c in benchmark["cases"]],
        "convention": {
            "python_orientation": "3x3 real orientation matrix from the sipg2 benchmark",
            "shaarp_si_a": "Transpose[python_orientation] so a.epsilon_crystal.Transpose[a] reconstructs epsilon_lab",
            "epsilon_crystal": "Computed as Transpose[a].epsilon_lab.a",
            "dold": "Crystal-frame point-group d recovered from the lab-frame d (Transpose[a] rotation); symmetry-constrained (<18 nonzero).",
            "polarizer_angle_deg": "s -> 90, p -> 0, matching EIw = {Cos[angle], Sin[angle], 0} before sample tilt.",
        },
        "constant_policy": {
            "Functionality": "SHG Simulation",
            "Eiw": 1.0,
            "w": 1.0,
            "e0": 1.0,
            "mu0": 1.0,
            "mu": "IdentityMatrix[3]",
            "RotatePolarizer": False,
            "RotateAnalyzer": False,
            "AnalyzerAngle_deg": 0.0,
            "Ellipticity_deg": 0.0,
            "DebugFlag": False,
        },
        "cases": cases,
        "notes": [
            "Point-group SI mapping (sipg2); not validation evidence by itself.",
            "dold is the symmetry-constrained crystal d-tensor (<18 nonzero) recovered from the lab d by a pure rotation; reconstruction residuals are recorded per case.",
            "Every tensor conversion includes a reconstruction residual so convention mistakes are visible.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate point-group SHAARP.si single-interface reflected-SHG cases (sipg2: remaining 8 classes)."
    )
    parser.add_argument("--tag", type=str, default=TAG, help="Artifact tag (default: sipg2).")
    parser.add_argument(
        "--no-original-tree",
        action="store_true",
        help="Skip writing the mapping into the ORIGINAL-tree directory.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases (debug).")
    args = parser.parse_args()

    benchmark = build_benchmark(limit=args.limit)
    mapping = build_mapping(benchmark)

    bench_path = REF_DIR / "solver_stage_benchmarks_si_pg2.json"
    mapping_copy_path = REF_DIR / f"shaarp_si_case_input_mapping_{args.tag}.json"
    bench_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    mapping_copy_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Wrote {benchmark['case_count']} sipg2 point-group SI benchmark cases to {bench_path}")
    print(f"Wrote sipg2 input mapping to {mapping_copy_path}")

    if not args.no_original_tree and ORIGINAL_REF_DIR is None:
        print("SHAARP_ORIGINAL_REF_DIR is not set -- skipping the second-tree mirror copy.")
    elif not args.no_original_tree:
        ORIGINAL_REF_DIR.mkdir(parents=True, exist_ok=True)
        mapping_orig_path = ORIGINAL_REF_DIR / f"shaarp_si_case_input_mapping_{args.tag}.json"
        mapping_orig_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        print(f"Wrote sipg2 input mapping (original tree) to {mapping_orig_path}")


if __name__ == "__main__":
    main()
