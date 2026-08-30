"""Compare the SHAARP.si compatibility solver against an EXTENSION ER2w stage
reference (new oblique-angle coverage), feeding the constructed lab-frame d
tensor directly.

The lab-frame d tensor equals SHAARP.si's rotated ``dSHG`` to ~4e-15 (a pure
crystal<->lab rotation round-trip, verified on the v1 set), so no separate PNL
export is required for the extension: the d tensor is a pure input, not part of
the boundary-value solve under test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.si_compat import solve_shaarp_si_reflected_shg


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"

KEY_EPS_W = "\\[CurlyEpsilon]\\[Omega]Cry"
KEY_EPS_2W = "\\[CurlyEpsilon]2\\[Omega]Cry"


def build_agreement(*, tag: str, atol: float, rtol: float) -> dict[str, Any]:
    signal_path = REF_DIR / f"shaarp_si_signal_stage_reference_{tag}.json"
    mapping_path = REF_DIR / f"shaarp_si_case_input_mapping_{tag}.json"
    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_by_id = {case["id"]: case for case in mapping["cases"]}
    records = [_case_record(case, mapping_by_id[case["id"]], atol=atol, rtol=rtol) for case in signal["cases"]]
    passing = [r for r in records if r["er2w_passed"]]
    failing = [r for r in records if not r["er2w_passed"]]
    return {
        "schema_version": 1,
        "status": "si_compat_reflected_signal_matches_extension_shaarp_si_stage_reference",
        "tag": tag,
        "source": "Python SHAARP.si compatibility solver vs SHAARP.si extension ER2w stage reference",
        "mathematica_reference": str(signal_path.relative_to(ROOT)).replace("\\", "/"),
        "input_mapping": str(mapping_path.relative_to(ROOT)).replace("\\", "/"),
        "d_tensor_source": "constructed lab-frame d (equals SHAARP.si dSHG to ~4e-15; pure rotation input)",
        "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
        "case_count": len(records),
        "passing_er2w_case_count": len(passing),
        "failing_er2w_case_count": len(failing),
        "max_er2w_abs_error": max((r["er2w_max_abs_error"] for r in records), default=0.0),
        "max_er2w_rel_error": max((r["er2w_max_rel_error"] for r in records), default=0.0),
        "tolerance": {"atol": atol, "rtol": rtol},
        "cases": records,
    }


def _case_record(case: dict[str, Any], mapped_case: dict[str, Any], *, atol: float, rtol: float) -> dict[str, Any]:
    variables = mapped_case["mathematica_variables"]
    d_voigt_lab = numeric_array(mapped_case["expected_python_lab_inputs"]["d_voigt_lab"])
    result = solve_shaarp_si_reflected_shg(
        epsilon_omega_crystal=numeric_array(variables[KEY_EPS_W]),
        epsilon_2omega_crystal=numeric_array(variables[KEY_EPS_2W]),
        orientation_matrix=numeric_array(variables["a"]),
        d_voigt_lab=d_voigt_lab,
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
        "orientation_index": int(mapped_case["orientation_index"]),
        "er2w_passed": passed,
        "er2w_max_abs_error": float(np.max(abs_delta)),
        "er2w_max_rel_error": float(np.max(rel_delta)),
        "boundary_residual_norm": float(np.linalg.norm(result.boundary_residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SI compat reflected signal against an extension ER2w reference.")
    parser.add_argument("--tag", type=str, default="ext1")
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = build_agreement(tag=args.tag, atol=args.atol, rtol=args.rtol)
    output = args.output or (REF_DIR / f"shaarp_si_compat_signal_agreement_{args.tag}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[{args.tag}] {payload['passing_er2w_case_count']}/{payload['case_count']} pass "
        f"at atol={args.atol}; max_abs_err={payload['max_er2w_abs_error']:.3e}; "
        f"max_rel_err={payload['max_er2w_rel_error']:.3e}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
