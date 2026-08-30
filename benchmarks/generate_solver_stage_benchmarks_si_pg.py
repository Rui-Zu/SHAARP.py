"""Point-group SHAARP.si single-interface reflected-SHG benchmark generator (sipg1).

Every existing single-interface (SI) solver-stage case (the v1 grid and the
``ext1`` coverage extension) drives the reflected-SHG boundary-value solve with a
GENERIC 18/18-nonzero lab-frame ``d`` tensor. SHAARP, however, is a
crystallographic SHG tool: for real crystal classes most ``d`` components vanish
by symmetry, and the lab-frame tensor that enters the solve is a SYMMETRY-
CONSTRAINED crystal tensor ROTATED into the lab frame.

This grid builds a SMALL, independent family of SI reflected-SHG cases whose
active crystal carries a point-group-constrained ``d`` tensor (3m, -42m, mm2,
4mm, 32, -6m2), rotated crystal->lab, to validate the SI reflected-SHG path
(``solve_single_interface_shg`` / ``solve_shaarp_si_reflected_shg``) for real
crystal classes against live Wolfram SHAARP.si ER2w.

It reuses the EXACT eps construction of the ``ext1`` SI extension generator
(bounded refractive-index parametrization, ``complex_symmetric_epsilon`` with the
same imag bases), and the SAME orientation matrices. The ONLY change per case is
the ``d`` tensor: instead of ``complex_d_voigt(seed)`` (generic 18/18 lab d) we
substitute deterministic complex values into the symbolic point-group template
``shaarp.symbolic.d_voigt_symbolic(pg)`` to obtain a symmetry-constrained crystal
tensor, then rotate it crystal->lab with the SAME orientation matrix used for the
eps tensors via ``shaarp.tensors.rotate_d_voigt_crystal_to_lab``.

Because the comparator's ``_case_mapping`` recovers the crystal-frame ``dold`` from
the lab-frame ``d`` (a pure rotation round-trip, exact to ~1e-16), the live
SHAARP.si exporter receives the symmetry-constrained crystal tensor as input,
while the Python solver consumes the lab-frame ``d`` directly. The lab tensor is a
pure input (not part of the boundary solve under test), so no separate PNL export
is required -- identical to the ``ext1`` design.

The script writes:
  * a point-group SI solver-stage benchmark JSON (record of constructed inputs),
  * a point-group SHAARP.si input mapping JSON, into BOTH the Copy-tree
    ``benchmarks/mathematica_reference`` directory and the ORIGINAL-tree directory
    (the Wolfram export scripts read ORIGINAL-tree paths).

It does NOT touch any v1 or ext1 artifact and creates only new ``sipg1`` files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import sympy as sp

from benchmarks.generate_solver_stage_benchmarks import (
    complex_array_to_json,
    complex_symmetric_epsilon,
    orientation_matrix,
)
from benchmarks.generate_shaarp_si_case_input_mapping import _case_mapping
from shaarp.shg import solve_single_interface_shg
from shaarp.symbolic import d_voigt_symbolic
from shaarp.tensors import rotate_d_voigt_crystal_to_lab, rotate_rank2_crystal_to_lab


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

# RNG seed offset placed well beyond v1 (0..) and ext1 (500..) so the complex
# eps off-diagonal parts are genuinely distinct inputs.
SEED_OFFSET = 900

# A small, deliberately diverse set of point-group SI cases.
# (point_group, orientation_index, theta_deg, incident_polarization)
RAW_CASES = [
    ("3m", 2, 12.0, "p"),
    ("-42m", 0, 20.5, "s"),
    ("mm2", 3, 35.0, "p"),
    ("4mm", 1, 8.0, "s"),
    ("32", 4, 48.0, "p"),
    ("-6m2", 5, 29.0, "s"),
]

# Per-(point_group, orientation, theta) refractive-index band, mirroring the
# bounded ext1 parametrization so the materials stay physically reasonable.
def _eps_axes(orientation_idx: int, angle_idx: int) -> tuple[np.ndarray, np.ndarray]:
    n_xx = 1.46 + 0.03 * angle_idx + 0.05 * orientation_idx
    n_yy = 1.70 + 0.04 * orientation_idx + 0.01 * angle_idx
    n_zz = 2.06 + 0.02 * angle_idx + 0.03 * orientation_idx
    eps_w_axes = np.array([n_xx, n_yy, n_zz]) ** 2
    eps_2w_axes = np.array([n_xx + 0.17, n_yy + 0.14, n_zz + 0.19]) ** 2
    return eps_w_axes, eps_2w_axes


def _pointgroup_d_voigt_crystal(point_group: str, seed: int) -> np.ndarray:
    """Numeric symmetry-constrained crystal d-tensor: substitute deterministic
    complex values into the symbolic point-group template (the validated doldExp
    pattern), exactly as in generate_maker_fringes_pointgroup_pg1_benchmarks.py."""
    dsym = d_voigt_symbolic(point_group)
    syms = sorted(dsym.free_symbols, key=str)
    subs = {s: complex(0.30 + 0.07 * (seed % 5) + 0.05 * i, 0.04 * (i + 1)) for i, s in enumerate(syms)}
    return np.array(
        [[complex(sp.N(dsym[r, c].subs(subs))) for c in range(6)] for r in range(3)],
        dtype=complex,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate point-group SHAARP.si single-interface reflected-SHG cases (sipg1)."
    )
    parser.add_argument("--tag", type=str, default="sipg1", help="Artifact tag (default: sipg1).")
    parser.add_argument(
        "--no-original-tree",
        action="store_true",
        help="Skip writing the mapping into the ORIGINAL-tree directory.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases (debug).")
    args = parser.parse_args()

    benchmark = build_benchmark(limit=args.limit)
    mapping = build_mapping(benchmark)

    bench_path = REF_DIR / f"solver_stage_benchmarks_si_pg1.json"
    mapping_copy_path = REF_DIR / f"shaarp_si_case_input_mapping_{args.tag}.json"
    bench_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    mapping_copy_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Wrote {benchmark['case_count']} sipg1 point-group SI benchmark cases to {bench_path}")
    print(f"Wrote sipg1 input mapping to {mapping_copy_path}")

    if not args.no_original_tree and ORIGINAL_REF_DIR is None:
        print("SHAARP_ORIGINAL_REF_DIR is not set -- skipping the second-tree mirror copy.")
    elif not args.no_original_tree:
        ORIGINAL_REF_DIR.mkdir(parents=True, exist_ok=True)
        mapping_orig_path = ORIGINAL_REF_DIR / f"shaarp_si_case_input_mapping_{args.tag}.json"
        mapping_orig_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        print(f"Wrote sipg1 input mapping (original tree) to {mapping_orig_path}")


def build_benchmark(limit: int | None = None) -> dict:
    records = [run_case(case) for case in build_cases(limit=limit)]
    return {
        "benchmark_version": 1,
        "tag": "sipg1",
        "status": "python_solver_stage_residual_regression_si_pointgroup_not_validation",
        "extension_of": "benchmarks/mathematica_reference/solver_stage_benchmarks_si_ext1.json",
        "seed_offset": SEED_OFFSET,
        "case_count": len(records),
        "point_groups": [c["point_group"] for c in build_cases(limit=limit)],
        "notes": [
            "Single-interface reflected-SHG cases whose active crystal carries a SYMMETRY-CONSTRAINED point-group d-tensor (3m/-42m/mm2/4mm/32/-6m2).",
            "d_voigt_lab = rotate_d_voigt_crystal_to_lab(pointgroup_crystal_d, orientation); the crystal tensor has <18 nonzero entries.",
            "eps construction matches the ext1 SI extension generator exactly; only the d tensor differs (point-group crystal d rotated to lab vs generic 18/18 lab d).",
            "These are Python-constructed inputs; the validation is Python solve_shaarp_si_reflected_shg ER2w vs live Mathematica SHAARP.si ER2w.",
        ],
        "cases": records,
    }


def build_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases: list[dict] = []
    for point_group, orientation_idx, theta_deg, pol in raw:
        idx = len(cases)
        orientation = orientation_matrix(orientation_idx)
        # angle_idx used only for the bounded eps band; spread across the set.
        angle_idx = idx
        eps_w_axes, eps_2w_axes = _eps_axes(orientation_idx, angle_idx)
        seed = SEED_OFFSET + idx
        d_crystal = _pointgroup_d_voigt_crystal(point_group, seed)
        d_voigt_lab = rotate_d_voigt_crystal_to_lab(d_crystal, orientation)
        crystal_nonzero_d = int(np.sum(np.abs(d_crystal) > 1e-12))
        lab_nonzero_d = int(np.sum(np.abs(d_voigt_lab) > 1e-12))
        cases.append(
            {
                "id": f"si_pg_stage_{idx + 1:02d}",
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


def run_case(case: dict) -> dict:
    epsilon_omega_lab = rotate_rank2_crystal_to_lab(case["epsilon_omega_crystal"], case["orientation"])
    epsilon_2omega_lab = rotate_rank2_crystal_to_lab(case["epsilon_2omega_crystal"], case["orientation"])
    result = solve_single_interface_shg(
        epsilon_omega_lab,
        epsilon_2omega_lab,
        case["d_voigt_lab"],
        incident_index_omega=1.0,
        incident_index_2omega=1.0,
        incident_theta_rad=np.deg2rad(case["theta_deg"]),
        incident_polarization=case["incident_polarization"],
        mu=1.0,
        eps0=1.0,
    )
    return {
        "id": case["id"],
        "point_group": case["point_group"],
        "theta_deg": case["theta_deg"],
        "incident_polarization": case["incident_polarization"],
        "orientation_index": case["orientation_index"],
        "crystal_nonzero_d": case["crystal_nonzero_d"],
        "lab_nonzero_d": case["lab_nonzero_d"],
        "inputs": {
            "orientation": complex_array_to_json(case["orientation"]),
            "epsilon_omega_lab": complex_array_to_json(epsilon_omega_lab),
            "epsilon_2omega_lab": complex_array_to_json(epsilon_2omega_lab),
            "d_voigt_lab": complex_array_to_json(case["d_voigt_lab"]),
            "d_voigt_crystal": complex_array_to_json(case["d_voigt_crystal"]),
        },
        "metrics": {
            "boundary_residual_l2": round(float(np.linalg.norm(result.boundary_residual)), 15),
            "coefficient_l2": round(float(np.linalg.norm(result.coefficients)), 15),
        },
    }


def build_mapping(benchmark: dict) -> dict:
    cases = [_case_mapping_with_pointgroup(case) for case in benchmark["cases"]]
    return {
        "schema_version": 1,
        "tag": "sipg1",
        "source": "SHAARP.si Mathematica input mapping for point-group single-interface solver-stage cases",
        "status": "input_mapping_not_validation_evidence",
        "validation_claim": False,
        "extension_of": "benchmarks/mathematica_reference/shaarp_si_case_input_mapping_ext1.json",
        "case_count": len(cases),
        "point_groups": [c["point_group"] for c in benchmark["cases"]],
        "convention": {
            "python_orientation": "3x3 real orientation matrix from the sipg1 benchmark",
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
            "Point-group SI mapping; not validation evidence by itself.",
            "dold is the symmetry-constrained crystal d-tensor (<18 nonzero) recovered from the lab d by a pure rotation; reconstruction residuals are recorded per case.",
            "Every tensor conversion includes a reconstruction residual so convention mistakes are visible.",
        ],
    }


def _case_mapping_with_pointgroup(case: dict) -> dict:
    """Wrap the shared _case_mapping and inject point-group provenance fields so
    the comparator/exporter consume an EXT1-shaped record while the manifest also
    records point_group + crystal_nonzero_d."""
    mapped = _case_mapping(case)
    mapped["point_group"] = case["point_group"]
    mapped["crystal_nonzero_d"] = case["crystal_nonzero_d"]
    mapped["lab_nonzero_d"] = case["lab_nonzero_d"]
    return mapped


if __name__ == "__main__":
    main()
