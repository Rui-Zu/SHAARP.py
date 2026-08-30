from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from benchmarks.reference_paths import resolve_reference_path, to_reference_value
from shaarp.shg import solve_single_interface_shg


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
DEFAULT_OUTPUT = REF_DIR / "shaarp_si_end_to_end_signal_diagnostic_v1.json"
SIGNAL_REFERENCE = REF_DIR / "shaarp_si_signal_stage_reference_v1.json"

_SHAARP_VOIGT_INDEX = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def build_diagnostic_summary(*, atol: float, rtol: float) -> dict[str, Any]:
    reference = json.loads(SIGNAL_REFERENCE.read_text(encoding="utf-8"))
    mapping_path = resolve_reference_path(reference["mapping_path"])
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_by_id = {case["id"]: case for case in mapping["cases"]}

    records = [
        _case_diagnostic(case, mapping_by_id[case["id"]], atol=atol, rtol=rtol)
        for case in reference["cases"]
    ]
    passing = [record for record in records if record["er2w_passed"]]
    failing = [record for record in records if not record["er2w_passed"]]
    max_abs = max((record["er2w_max_abs_error"] for record in records), default=0.0)
    max_rel = max((record["er2w_max_rel_error"] for record in records), default=0.0)
    first_failing = next((record for record in records if not record["er2w_passed"]), None)
    return {
        "schema_version": 1,
        "status": "diagnostic_current_python_si_solver_not_full_shaarp_si_agreement",
        "source": "Current Python solve_single_interface_shg compared against SHAARP.si reflected signal stage ER2w values",
        "full_shaarp_si_agreement_claimed": False,
        "mathematica_reference": str(SIGNAL_REFERENCE.relative_to(ROOT)).replace("\\", "/"),
        "mapping_path": to_reference_value(mapping_path),
        "case_count": len(records),
        "passing_er2w_case_count": len(passing),
        "failing_er2w_case_count": len(failing),
        "max_er2w_abs_error": max_abs,
        "max_er2w_rel_error": max_rel,
        "tolerance": {"atol": atol, "rtol": rtol},
        "likely_first_gap": (
            "ordinary/extraordinary source-mode identity: current Python full-SI solver uses "
            "fast/slow fallback for low-symmetry complex cases, while SHAARP.si uses its active "
            "angle-root and eigensystem branch-label path before nonlinear polarization"
        ),
        "first_failing_case": first_failing,
        "cases": records,
    }


def _case_diagnostic(case: dict[str, Any], mapped_case: dict[str, Any], *, atol: float, rtol: float) -> dict[str, Any]:
    variables = mapped_case["mathematica_variables"]
    a_matrix = numeric_array(variables["a"])
    epsilon_omega_lab = a_matrix @ numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]) @ a_matrix.T
    epsilon_2omega_lab = a_matrix @ numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]) @ a_matrix.T
    d_lab = _rotate_d_like_shaarp_si(a_matrix, numeric_array(variables["dold"]))
    result = solve_single_interface_shg(
        epsilon_omega_lab,
        epsilon_2omega_lab,
        d_lab,
        incident_index_omega=1.0,
        incident_index_2omega=1.0,
        incident_theta_rad=math.radians(float(mapped_case["theta_deg"])),
        incident_polarization=str(mapped_case["incident_polarization"]),
        mu=1.0,
        eps0=1.0,
    )
    mathematica_er2w = numeric_array(case["outputs"]["ER2w"])
    python_er2w = result.reflected_total_2omega.electric
    abs_delta = np.abs(python_er2w - mathematica_er2w)
    scale = np.maximum(np.abs(mathematica_er2w), np.finfo(float).tiny)
    rel_delta = abs_delta / scale
    max_abs = float(np.max(abs_delta))
    max_rel = float(np.max(rel_delta))
    passed = bool(np.allclose(python_er2w, mathematica_er2w, atol=atol, rtol=rtol))
    return {
        "case_id": case["id"],
        "theta_deg": float(mapped_case["theta_deg"]),
        "incident_polarization": str(mapped_case["incident_polarization"]),
        "er2w_passed": passed,
        "er2w_max_abs_error": max_abs,
        "er2w_max_rel_error": max_rel,
        "python_boundary_residual_norm": float(np.linalg.norm(result.boundary_residual)),
        "python_source_mode_basis": result.source_mode_basis,
        "python_source_mode_order": list(result.source_mode_order),
    }


def _rotate_d_like_shaarp_si(a_matrix: np.ndarray, d_old: np.ndarray) -> np.ndarray:
    d_shg = np.zeros((3, 6), dtype=complex)
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):
                column = _SHAARP_VOIGT_INDEX[j, k]
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            d_shg[i, column] += (
                                a_matrix[i, l]
                                * a_matrix[j, m]
                                * a_matrix[k, n]
                                * d_old[l, _SHAARP_VOIGT_INDEX[m, n]]
                            )
    return d_shg


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose current Python full-SI signal agreement against SHAARP.si.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()
    payload = build_diagnostic_summary(atol=args.atol, rtol=args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si end-to-end signal diagnostic to {args.output}")


if __name__ == "__main__":
    main()
