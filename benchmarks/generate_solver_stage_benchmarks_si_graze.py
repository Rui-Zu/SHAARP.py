"""SI reflected-SHG solver-stage cases at GRAZING INCIDENCE (tag: sigraze) — R1 breadth.

WHY THIS SET EXISTS. Residual risk R1: live-Mathematica agreement is certified only at the
EXPORTED reference points; between and beyond them the claim rests on the structural port plus the
published-equation stage tests. Enumerating what is actually certified showed a specific, reachable
hole in the INCIDENCE ANGLE axis:

    certified theta (deg), all existing SI mappings:
      v1 0.0 8.0 17.5 29.0 41.0 55.0
      ext1 4.0 12.0 20.5 35.0 48.0 62.0 70.0
      sipg1 8.0 12.0 20.5 29.0 35.0 48.0
      sipg2 6.5 9.0 14.0 19.0 27.5 33.0 41.0 52.0
    -> MAX = 70.0 deg.

Above 70 deg nothing is certified — yet the GUI exposes the full range and its own matrix sweep
drives theta = 89 deg, where the sweep's slowest cells sit. This is the band where sin(theta) -> 1,
the transmitted wave approaches grazing, cos(theta_T) gets small, and the boundary determinants are
worst-conditioned; if the port and the original ever diverge on geometry, this is where it shows.

(An earlier framing of this file said "grazing >60 deg is uncertified". That was wrong — ext1
already covers 62 and 70. Checking the data before writing the claim is why the band is stated
exactly.)

CONSTRUCTION. Deliberately a faithful clone of the sipg1/sipg2 generators: the same validated
helpers (`_eps_axes`, `_pointgroup_d_voigt_crystal`, `_case_mapping_with_pointgroup`, `run_case`,
`orientation_matrix`, `complex_symmetric_epsilon`) build the inputs, so the ONLY thing that differs
from an already-validated case set is the incidence angle. That is what makes a disagreement here
attributable to the angle rather than to a new input-construction path. The eps are complex by
construction (imag_base 0.025/0.035), so these cases are absorbing as well as grazing.

This file produces Python-side inputs and a Mathematica input mapping. It is NOT validation
evidence on its own — the evidence is the live SHAARP.si export compared against it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.generate_solver_stage_benchmarks import (
    complex_symmetric_epsilon,
    orientation_matrix,
)
from benchmarks.generate_solver_stage_benchmarks_si_pg import (
    _case_mapping_with_pointgroup,
    _eps_axes,
    _pointgroup_d_voigt_crystal,
    run_case,
)
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"

TAG = "sigraze"
# Beyond sipg2's 1500 block so the seeded complex-eps off-diagonals are distinct inputs again.
SEED_OFFSET = 2200

# (point_group, orientation_index, theta_deg, incident_polarization)
# Angles are the UNCERTIFIED band only (>70 deg), walked to the practical limit the GUI allows.
# Point groups/orientations are spread across already-validated classes so that any disagreement
# is attributable to the ANGLE, not to an untested d-pattern.
RAW_CASES = [
    ("3m", 0, 75.0, "p"),
    ("mm2", 1, 80.0, "s"),
    ("-42m", 2, 85.0, "p"),
    ("4mm", 3, 88.0, "s"),
    ("32", 4, 89.0, "p"),
    ("-6m2", 5, 89.5, "s"),
]


def build_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases: list[dict] = []
    for point_group, orientation_idx, theta_deg, pol in raw:
        idx = len(cases)
        orientation = orientation_matrix(orientation_idx)
        eps_w_axes, eps_2w_axes = _eps_axes(orientation_idx, idx)
        seed = SEED_OFFSET + idx
        d_crystal = _pointgroup_d_voigt_crystal(point_group, seed)
        d_voigt_lab = rotate_d_voigt_crystal_to_lab(d_crystal, orientation)
        cases.append(
            {
                "id": f"si_graze_stage_{idx + 1:02d}",
                "point_group": point_group,
                "theta_deg": theta_deg,
                "incident_polarization": pol,
                "orientation_index": orientation_idx,
                "orientation": orientation,
                "epsilon_omega_crystal": complex_symmetric_epsilon(seed, eps_w_axes, imag_base=0.025),
                "epsilon_2omega_crystal": complex_symmetric_epsilon(seed + 100, eps_2w_axes, imag_base=0.035),
                "d_voigt_crystal": d_crystal,
                "d_voigt_lab": d_voigt_lab,
                "crystal_nonzero_d": int(np.sum(np.abs(d_crystal) > 1e-12)),
                "lab_nonzero_d": int(np.sum(np.abs(d_voigt_lab) > 1e-12)),
            }
        )
    return cases


def build_benchmark(limit: int | None = None) -> dict:
    records = [run_case(case) for case in build_cases(limit=limit)]
    cases = build_cases(limit=limit)
    return {
        "benchmark_version": 1,
        "tag": TAG,
        "status": "python_solver_stage_residual_regression_si_grazing_not_validation",
        "extension_of": "benchmarks/mathematica_reference/solver_stage_benchmarks_si_pg2.json",
        "seed_offset": SEED_OFFSET,
        "case_count": len(records),
        "theta_deg": [c["theta_deg"] for c in cases],
        "point_groups": [c["point_group"] for c in cases],
        "notes": [
            "GRAZING-INCIDENCE single-interface reflected-SHG cases (theta 75..89.5 deg).",
            "Motivation: residual risk R1. Every pre-existing SI mapping tops out at theta = 70 deg "
            "(v1 55, ext1 70, sipg1 48, sipg2 52), while the GUI exposes the full range and its "
            "matrix sweep drives theta = 89.",
            "Inputs are built by the SAME validated helpers as sipg1/sipg2; only the incidence "
            "angle differs, so a disagreement is attributable to the angle.",
            "eps is complex (imag_base 0.025/0.035): these cases are absorbing AND grazing.",
            "Python-constructed inputs; the validation is Python solve_shaarp_si_reflected_shg ER2w "
            "vs live Mathematica SHAARP.si ER2w.",
        ],
        "cases": records,
    }


def build_mapping(benchmark: dict) -> dict:
    cases = [_case_mapping_with_pointgroup(case) for case in benchmark["cases"]]
    return {
        "schema_version": 1,
        "tag": TAG,
        "source": "SHAARP.si Mathematica input mapping for GRAZING-INCIDENCE single-interface solver-stage cases (sigraze)",
        "status": "input_mapping_not_validation_evidence",
        "validation_claim": False,
        "extension_of": "benchmarks/mathematica_reference/shaarp_si_case_input_mapping_sipg2.json",
        "case_count": len(cases),
        "theta_deg": [c["theta_deg"] for c in benchmark["cases"]],
        "point_groups": [c["point_group"] for c in benchmark["cases"]],
        "convention": {
            "python_orientation": "3x3 real orientation matrix from the sigraze benchmark",
            "shaarp_si_a": "Transpose[python_orientation] so a.epsilon_crystal.Transpose[a] reconstructs epsilon_lab",
            "epsilon_crystal": "Computed as Transpose[a].epsilon_lab.a",
            "dold": "Crystal-frame point-group d recovered from the lab-frame d (Transpose[a] rotation); symmetry-constrained (<18 nonzero).",
        },
        # Mirrors sipg1/sipg2 EXACTLY. The first draft of this file renamed the payload key to
        # "records" and dropped constant_policy/notes; the exporter reads `mapping["cases"]`, so it
        # got Missing["KeyAbsent","cases"], mapped caseRecord over a non-list, and the downstream
        # Select[records[[All, ...]]] on that unevaluated expression spun for >90 minutes before
        # being killed — twice. LESSON: when cloning a validated pipeline, match the SCHEMA
        # exactly; "equivalent" is not the same as "identical", and the consumer is not forgiving.
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
            "Grazing-incidence extension of the sipg1/sipg2 SI reflected-SHG mapping (theta 75..89.5 deg).",
            "Schema is identical to sipg2 by construction; only the incidence angles differ.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the grazing-incidence SI solver-stage benchmark + Mathematica input mapping.")
    parser.add_argument("--benchmark-out", type=Path,
                        default=REF_DIR / f"solver_stage_benchmarks_si_{TAG}.json")
    parser.add_argument("--mapping-out", type=Path,
                        default=REF_DIR / f"shaarp_si_case_input_mapping_{TAG}.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N cases (pilot before the full set)")
    args = parser.parse_args()

    benchmark = build_benchmark(limit=args.limit)
    mapping = build_mapping(benchmark)
    for path, payload in ((args.benchmark_out, benchmark), (args.mapping_out, mapping)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {path}  ({payload['case_count']} cases)")


if __name__ == "__main__":
    main()
