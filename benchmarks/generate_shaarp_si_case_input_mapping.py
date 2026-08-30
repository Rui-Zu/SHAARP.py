from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from shaarp.tensors import chi2_to_voigt, voigt_to_chi2


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
REF_DIR = BENCHMARK_DIR / "mathematica_reference"
SOLVER_STAGE_PATH = BENCHMARK_DIR / "solver_stage_benchmarks_v1.json"
OUTPUT_PATH = REF_DIR / "shaarp_si_case_input_mapping_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAARP.si Mathematica input mapping for solver-stage cases.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si case input mapping to {args.output}")


def build_payload() -> dict:
    solver_stage = _load_json(SOLVER_STAGE_PATH)
    cases = [_case_mapping(case) for case in solver_stage["cases"]]
    return {
        "schema_version": 1,
        "source": "SHAARP.si Mathematica input mapping for Python solver-stage benchmark cases",
        "status": "input_mapping_not_validation_evidence",
        "validation_claim": False,
        "source_benchmark_file": "benchmarks/solver_stage_benchmarks_v1.json",
        "case_count": len(cases),
        "convention": {
            "python_orientation": "3x3 real orientation matrix stored in solver_stage_benchmarks_v1.json",
            "shaarp_si_a": "Transpose[python_orientation] so that a.epsilon_crystal.Transpose[a] reconstructs epsilon_lab",
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
            "This file maps Python benchmark inputs into SHAARP.si variable names; it does not run Mathematica.",
            "Every tensor conversion includes a reconstruction residual so convention mistakes are visible.",
            "The a/epsilon/d convention must still be checked against Mathematica values before claiming agreement.",
        ],
    }


def _case_mapping(case: dict) -> dict:
    inputs = case["inputs"]
    orientation = _complex_matrix(inputs["orientation"]).real
    a = orientation.T
    eps_w_lab = _complex_matrix(inputs["epsilon_omega_lab"])
    eps_2w_lab = _complex_matrix(inputs["epsilon_2omega_lab"])
    d_lab = _complex_matrix(inputs["d_voigt_lab"])
    eps_w_crystal = a.T @ eps_w_lab @ a
    eps_2w_crystal = a.T @ eps_2w_lab @ a
    d_crystal = _lab_to_crystal_d_voigt(d_lab, a)
    eps_w_reconstructed = a @ eps_w_crystal @ a.T
    eps_2w_reconstructed = a @ eps_2w_crystal @ a.T
    d_reconstructed = _crystal_to_lab_d_voigt(d_crystal, a)
    pol = case["incident_polarization"]
    polarizer_angle = 90.0 if pol == "s" else 0.0
    return {
        "id": case["id"],
        "theta_deg": case["theta_deg"],
        "incident_polarization": pol,
        "orientation_index": case["orientation_index"],
        "mathematica_variables": {
            "ThetaIn": case["theta_deg"],
            "PolarizerAngle": polarizer_angle,
            "Ellipticity": 0.0,
            "AnalyzerAngle": 0.0,
            "a": _complex_array_to_json(a),
            "\\[CurlyEpsilon]\\[Omega]Cry": _complex_array_to_json(eps_w_crystal),
            "\\[CurlyEpsilon]2\\[Omega]Cry": _complex_array_to_json(eps_2w_crystal),
            "dold": _complex_array_to_json(d_crystal),
        },
        "expected_python_lab_inputs": {
            "epsilon_omega_lab": _complex_array_to_json(eps_w_lab),
            "epsilon_2omega_lab": _complex_array_to_json(eps_2w_lab),
            "d_voigt_lab": _complex_array_to_json(d_lab),
        },
        "reconstruction_checks": {
            "max_abs_epsilon_omega_lab_error": _max_abs(eps_w_reconstructed - eps_w_lab),
            "max_abs_epsilon_2omega_lab_error": _max_abs(eps_2w_reconstructed - eps_2w_lab),
            "max_abs_d_voigt_lab_error": _max_abs(d_reconstructed - d_lab),
            "orientation_orthogonality_error": _max_abs(orientation @ orientation.T - np.eye(3)),
            "orientation_det": round(float(np.linalg.det(orientation)), 15),
        },
    }


def _lab_to_crystal_d_voigt(d_lab: np.ndarray, a: np.ndarray) -> np.ndarray:
    chi_lab = voigt_to_chi2(d_lab)
    chi_crystal = np.einsum("il,jm,kn,ijk->lmn", a, a, a, chi_lab)
    return chi2_to_voigt(chi_crystal)


def _crystal_to_lab_d_voigt(d_crystal: np.ndarray, a: np.ndarray) -> np.ndarray:
    chi_crystal = voigt_to_chi2(d_crystal)
    chi_lab = np.einsum("il,jm,kn,lmn->ijk", a, a, a, chi_crystal)
    return chi2_to_voigt(chi_lab)


def _complex_matrix(value: list) -> np.ndarray:
    return np.asarray([[_complex_scalar(item) for item in row] for row in value], dtype=complex)


def _complex_scalar(value: Any) -> complex:
    if isinstance(value, dict):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    return complex(value)


def _complex_array_to_json(values: np.ndarray) -> list:
    arr = np.asarray(values, dtype=complex)
    return [[_complex_scalar_to_json(item) for item in row] for row in arr]


def _complex_scalar_to_json(value: complex) -> dict:
    return {
        "real": round(float(np.real(value)), 15),
        "imag": round(float(np.imag(value)), 15),
    }


def _max_abs(values: np.ndarray) -> float:
    return round(float(np.max(np.abs(values))), 15)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
