from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from benchmarks.reference_paths import resolve_reference_path
from shaarp.si_compat import solve_shaarp_si_reflected_shg


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
DEFAULT_OUTPUT = REF_DIR / "shaarp_si_compat_signal_agreement_v1.json"
SIGNAL_REFERENCE = REF_DIR / "shaarp_si_signal_stage_reference_v1.json"
PNL_REFERENCE = REF_DIR / "shaarp_si_pnl_stage_reference_v1.json"


def build_compat_signal_agreement(*, atol: float, rtol: float) -> dict[str, Any]:
    signal = json.loads(SIGNAL_REFERENCE.read_text(encoding="utf-8"))
    pnl = json.loads(PNL_REFERENCE.read_text(encoding="utf-8"))
    mapping = json.loads(
        resolve_reference_path(signal["mapping_path"]).read_text(encoding="utf-8"))
    mapping_by_id = {case["id"]: case for case in mapping["cases"]}
    pnl_by_id = {case["id"]: case for case in pnl["cases"]}
    records = [_case_record(case, mapping_by_id[case["id"]], pnl_by_id[case["id"]], atol=atol, rtol=rtol) for case in signal["cases"]]
    passing = [record for record in records if record["er2w_passed"]]
    failing = [record for record in records if not record["er2w_passed"]]
    return {
        "schema_version": 1,
        "status": "si_compat_reflected_signal_matches_exported_shaarp_si_stage_reference",
        "source": "Python SHAARP.si compatibility solver compared against SHAARP.si reflected signal stage ER2w values",
        "mathematica_reference": str(SIGNAL_REFERENCE.relative_to(ROOT)).replace("\\", "/"),
        "pnl_reference": str(PNL_REFERENCE.relative_to(ROOT)).replace("\\", "/"),
        "full_notebook_end_to_end_agreement_claimed": False,
        "full_notebook_end_to_end_caveat": (
            "This validates the self-contained Python compatibility solver against exported SHAARP.si "
            "stage references. It is stronger than the legacy generic SI solver diagnostic, but still "
            "does not prove every GUI/notebook pathway or analytical branch."
        ),
        "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
        "case_count": len(records),
        "passing_er2w_case_count": len(passing),
        "failing_er2w_case_count": len(failing),
        "max_er2w_abs_error": max((record["er2w_max_abs_error"] for record in records), default=0.0),
        "max_er2w_rel_error": max((record["er2w_max_rel_error"] for record in records), default=0.0),
        "tolerance": {"atol": atol, "rtol": rtol},
        "cases": records,
    }


def _case_record(case: dict[str, Any], mapped_case: dict[str, Any], pnl_case: dict[str, Any], *, atol: float, rtol: float) -> dict[str, Any]:
    variables = mapped_case["mathematica_variables"]
    result = solve_shaarp_si_reflected_shg(
        epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
        epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
        orientation_matrix=numeric_array(variables["a"]),
        d_voigt_lab=numeric_array(pnl_case["outputs"]["dSHG"]),
        theta_in_rad=np.deg2rad(float(mapped_case["theta_deg"])),
        incident_polarization=str(mapped_case["incident_polarization"]),
        mu=1.0,
        eps0=1.0,
        normal_incidence_2omega_branch_policy="shaarp_reference_like",
    )
    expected = numeric_array(case["outputs"]["ER2w"])
    actual = result.reflected_total.electric
    abs_delta = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), np.finfo(float).tiny)
    rel_delta = abs_delta / scale
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    return {
        "case_id": case["id"],
        "theta_deg": float(mapped_case["theta_deg"]),
        "incident_polarization": str(mapped_case["incident_polarization"]),
        "er2w_passed": passed,
        "er2w_max_abs_error": float(np.max(abs_delta)),
        "er2w_max_rel_error": float(np.max(rel_delta)),
        "boundary_residual_norm": float(np.linalg.norm(result.boundary_residual)),
        "two_omega_branch_indices": {
            "ext": int(result.homogeneous_2omega.branch_indices.ext_index),
            "ord": int(result.homogeneous_2omega.branch_indices.ord_index),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SI compatibility reflected signal against SHAARP.si stage ER2w.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    payload = build_compat_signal_agreement(atol=args.atol, rtol=args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SI compatibility signal agreement to {args.output}")


if __name__ == "__main__":
    main()
