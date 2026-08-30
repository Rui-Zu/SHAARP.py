from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from shaarp.nonlinear import NonlinearSource, solve_inhomogeneous_field


BENCHMARK_VERSION = 1
ANGLES_DEG = [0.0, 12.0, 24.0, 36.0, 48.0, 60.0]
ORIENTATION_COUNT = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate complex-permittivity inhomogeneous-field benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "complex_inhomogeneous_benchmarks_v1.json",
    )
    args = parser.parse_args()

    records = [run_case(case) for case in build_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_solveinhom_complex_anisotropic_residual_regression_not_mathematica_validation",
        "case_count": len(records),
        "notes": [
            "These cases verify the ported solveInhom algebraic stage with complex anisotropic dielectric tensors.",
            "They sweep propagation angle and orientation, with complex nonlinear polarization source vectors.",
            "They are residual/regression benchmarks, not Mathematica-vs-Python validation.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} complex inhomogeneous benchmark cases to {args.output}")


def build_cases() -> list[dict]:
    cases = []
    for orientation_idx in range(ORIENTATION_COUNT):
        orientation = orientation_matrix(orientation_idx)
        for angle_idx, theta_deg in enumerate(ANGLES_DEG):
            idx = len(cases)
            cases.append(
                {
                    "id": f"complex_inhom_{idx + 1:02d}",
                    "theta_deg": theta_deg,
                    "orientation_index": orientation_idx,
                    "orientation": orientation,
                    "epsilon_crystal": complex_symmetric_epsilon(idx),
                    "source": source_for_case(idx, theta_deg),
                }
            )
    return cases


def run_case(case: dict) -> dict:
    epsilon_lab = case["orientation"].T @ case["epsilon_crystal"] @ case["orientation"]
    field = solve_inhomogeneous_field(epsilon_lab, case["source"], mu=1.0, eps0=1.0)
    record = {
        "id": case["id"],
        "theta_deg": case["theta_deg"],
        "orientation_index": case["orientation_index"],
        "inputs": {
            "orientation": complex_array_to_json(case["orientation"]),
            "epsilon_lab": complex_array_to_json(epsilon_lab),
            "source_k": complex_vector_to_json(case["source"].k),
            "source_polarization": complex_vector_to_json(case["source"].polarization),
            "omega": case["source"].omega,
        },
        "outputs": {
            "electric": complex_vector_to_json(field.electric),
            "residual": complex_vector_to_json(field.residual),
        },
        "metrics": {
            "residual_l2": round(float(np.linalg.norm(field.residual)), 15),
            "electric_l2": round(float(np.linalg.norm(field.electric)), 15),
            "max_abs_epsilon_imag": round(float(np.max(np.abs(np.imag(epsilon_lab)))), 15),
            "max_abs_epsilon_offdiag": round(float(max_abs_offdiag(epsilon_lab)), 15),
            "max_abs_source_imag": round(float(np.max(np.abs(np.imag(case["source"].polarization)))), 15),
            "orientation_det": round(float(np.linalg.det(case["orientation"])), 15),
        },
    }
    record["fingerprint"] = fingerprint_record(record)
    return record


def orientation_matrix(idx: int) -> np.ndarray:
    if idx == 0:
        return np.eye(3)
    rng = np.random.default_rng(20260701 + idx)
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs
    if np.linalg.det(q) < 0:
        q[0] *= -1
    return q


def complex_symmetric_epsilon(idx: int) -> np.ndarray:
    rng = np.random.default_rng(20260715 + idx)
    real_diag = np.array([2.0 + 0.03 * idx, 2.6 + 0.02 * idx, 3.3 + 0.01 * idx])
    imag_diag = np.array([0.08, 0.13, 0.19]) + rng.uniform(0.0, 0.04, size=3)
    eps = np.diag(real_diag + 1j * imag_diag)
    off = rng.normal(0.0, 0.04, size=(3, 3)) + 1j * rng.normal(0.0, 0.025, size=(3, 3))
    off = np.triu(off, 1)
    return eps + off + off.T


def source_for_case(idx: int, theta_deg: float) -> NonlinearSource:
    rng = np.random.default_rng(20260801 + idx)
    theta = np.deg2rad(theta_deg)
    k_mag = 1.2 + 0.03 * idx
    k = k_mag * np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=complex)
    polarization = rng.normal(0.0, 0.4, size=3) + 1j * rng.normal(0.0, 0.3, size=3)
    polarization += np.array([0.2, -0.1j, 0.15 + 0.05j])
    return NonlinearSource(label=f"complex_inhom_{idx + 1:02d}", k=k, polarization=polarization, omega=1.1 + 0.01 * idx)


def complex_array_to_json(values: np.ndarray) -> list:
    arr = np.asarray(values, dtype=complex)
    return [[complex_scalar_to_json(item) for item in row] for row in arr]


def complex_vector_to_json(values: np.ndarray) -> list:
    return [complex_scalar_to_json(item) for item in np.asarray(values, dtype=complex)]


def complex_scalar_to_json(value: complex) -> dict:
    return {
        "real": round(float(np.real(value)), 15),
        "imag": round(float(np.imag(value)), 15),
    }


def max_abs_offdiag(matrix: np.ndarray) -> float:
    off = matrix - np.diag(np.diag(matrix))
    return float(np.max(np.abs(off)))


def fingerprint_record(record: dict) -> str:
    comparable = dict(record)
    comparable.pop("fingerprint", None)
    data = json.dumps(comparable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
