"""Non-destructive SHAARP.si coverage-extension input generator.

This builds a SECOND, independent family of synthetic single-interface SHG
cases at NEW oblique incidence angles (distinct from the original
``solver_stage_benchmarks_v1.json`` grid of {0, 8, 17.5, 29, 41, 55} deg) so the
SHAARP.si compatibility solver can be validated against live Mathematica ER2w
values over a broader oblique-angle range.

It reuses the original orientation matrices and the same complex anisotropic
``Biaxial=True`` material family construction, but with a distinct RNG seed
offset and a bounded refractive-index parametrization so the materials stay
physically reasonable while being genuinely different inputs from the v1 set.

The script writes:
  * an extension solver-stage benchmark JSON (record of the constructed inputs),
  * an extension SHAARP.si input mapping JSON, into BOTH the Copy-tree
    benchmarks/mathematica_reference directory and the ORIGINAL-tree directory
    (the Wolfram export scripts read ORIGINAL-tree paths).

It does NOT touch any v1 artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.generate_solver_stage_benchmarks import (
    ORIENTATION_COUNT,
    complex_array_to_json,
    complex_d_voigt,
    complex_symmetric_epsilon,
    orientation_matrix,
)
from benchmarks.generate_shaarp_si_case_input_mapping import _case_mapping
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_rank2_crystal_to_lab


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

# New oblique grid, deliberately interleaved with (not equal to) the v1 angles.
DEFAULT_ANGLES_DEG = [4.0, 12.0, 20.5, 35.0, 48.0, 62.0, 70.0]
POLARIZATIONS = ["s", "p"]
# RNG seed offset so the complex/off-diagonal material parts differ from v1.
SEED_OFFSET = 500


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAARP.si coverage-extension cases at new oblique angles.")
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=DEFAULT_ANGLES_DEG,
        help="Incidence angles in degrees (default: the full extension grid).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="ext1",
        help="Artifact tag, used in output filenames (default: ext1).",
    )
    parser.add_argument(
        "--both-polarizations",
        action="store_true",
        default=True,
        help="Emit both s and p cases for each (orientation, angle).",
    )
    parser.add_argument(
        "--no-original-tree",
        action="store_true",
        help="Skip writing the mapping into the ORIGINAL-tree directory.",
    )
    args = parser.parse_args()

    benchmark = build_benchmark(args.angles)
    mapping = build_mapping(benchmark, args.angles)

    bench_path = REF_DIR / f"solver_stage_benchmarks_si_{args.tag}.json"
    mapping_copy_path = REF_DIR / f"shaarp_si_case_input_mapping_{args.tag}.json"
    bench_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    mapping_copy_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Wrote {benchmark['case_count']} ext benchmark cases to {bench_path}")
    print(f"Wrote ext input mapping to {mapping_copy_path}")

    if not args.no_original_tree and ORIGINAL_REF_DIR is None:
        print("SHAARP_ORIGINAL_REF_DIR is not set -- skipping the second-tree mirror copy.")
    elif not args.no_original_tree:
        ORIGINAL_REF_DIR.mkdir(parents=True, exist_ok=True)
        mapping_orig_path = ORIGINAL_REF_DIR / f"shaarp_si_case_input_mapping_{args.tag}.json"
        mapping_orig_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        print(f"Wrote ext input mapping (original tree) to {mapping_orig_path}")


def build_benchmark(angles_deg: list[float]) -> dict:
    records = [run_case(case) for case in build_cases(angles_deg)]
    return {
        "benchmark_version": 1,
        "status": "python_solver_stage_residual_regression_si_extension_not_validation",
        "extension_of": "benchmarks/solver_stage_benchmarks_v1.json",
        "angles_deg": angles_deg,
        "seed_offset": SEED_OFFSET,
        "case_count": len(records),
        "notes": [
            "Independent synthetic SHG cases at NEW oblique angles for SHAARP.si coverage breadth.",
            "Distinct RNG seed offset and bounded refractive-index parametrization vs the v1 grid.",
            "These are Python-constructed inputs; the validation is Python vs Mathematica ER2w.",
        ],
        "cases": records,
    }


def build_cases(angles_deg: list[float]) -> list[dict]:
    cases: list[dict] = []
    orientations = [orientation_matrix(i) for i in range(ORIENTATION_COUNT)]
    for orientation_idx, orientation in enumerate(orientations):
        for angle_idx, theta_deg in enumerate(angles_deg):
            for pol in POLARIZATIONS:
                idx = len(cases)
                # Bounded indices keep n in a physical band (~1.46-1.95) regardless
                # of how many angles are swept, unlike the v1 idx-linear scaling.
                n_xx = 1.46 + 0.03 * angle_idx + 0.05 * orientation_idx
                n_yy = 1.70 + 0.04 * orientation_idx + 0.01 * angle_idx
                n_zz = 2.06 + 0.02 * angle_idx + 0.03 * orientation_idx
                eps_w_axes = np.array([n_xx, n_yy, n_zz]) ** 2
                eps_2w_axes = np.array([n_xx + 0.17, n_yy + 0.14, n_zz + 0.19]) ** 2
                seed = SEED_OFFSET + idx
                cases.append(
                    {
                        "id": f"si_ext_stage_{idx + 1:02d}",
                        "theta_deg": theta_deg,
                        "incident_polarization": pol,
                        "orientation_index": orientation_idx,
                        "orientation": orientation,
                        "epsilon_omega_crystal": complex_symmetric_epsilon(seed, eps_w_axes, imag_base=0.025),
                        "epsilon_2omega_crystal": complex_symmetric_epsilon(seed + 100, eps_2w_axes, imag_base=0.035),
                        "d_voigt_lab": complex_d_voigt(seed),
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
        "theta_deg": case["theta_deg"],
        "incident_polarization": case["incident_polarization"],
        "orientation_index": case["orientation_index"],
        "inputs": {
            "orientation": complex_array_to_json(case["orientation"]),
            "epsilon_omega_lab": complex_array_to_json(epsilon_omega_lab),
            "epsilon_2omega_lab": complex_array_to_json(epsilon_2omega_lab),
            "d_voigt_lab": complex_array_to_json(case["d_voigt_lab"]),
        },
        "metrics": {
            "boundary_residual_l2": round(float(np.linalg.norm(result.boundary_residual)), 15),
            "coefficient_l2": round(float(np.linalg.norm(result.coefficients)), 15),
        },
    }


def build_mapping(benchmark: dict, angles_deg: list[float]) -> dict:
    cases = [_case_mapping(case) for case in benchmark["cases"]]
    return {
        "schema_version": 1,
        "source": "SHAARP.si Mathematica input mapping for coverage-extension solver-stage cases",
        "status": "input_mapping_not_validation_evidence",
        "validation_claim": False,
        "extension_of": "benchmarks/mathematica_reference/shaarp_si_case_input_mapping_v1.json",
        "angles_deg": angles_deg,
        "case_count": len(cases),
        "convention": {
            "python_orientation": "3x3 real orientation matrix from the ext benchmark",
            "shaarp_si_a": "Transpose[python_orientation] so a.epsilon_crystal.Transpose[a] reconstructs epsilon_lab",
            "epsilon_crystal": "Computed as Transpose[a].epsilon_lab.a",
            "dold": "Computed so SHAARP.si dSHG rotation with a reconstructs benchmark d_voigt_lab.",
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
            "Coverage-extension mapping at new oblique angles; not validation evidence by itself.",
            "Every tensor conversion includes a reconstruction residual so convention mistakes are visible.",
        ],
    }


if __name__ == "__main__":
    main()
