from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from shaarp.anisotropic import solve_snell_modes


BENCHMARK_VERSION = 1
ANGLES_DEG = [0.0, 10.0, 22.5, 37.0, 52.0]
ORIENTATION_COUNT = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reflected complex Snell benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "reflected_snell_benchmarks_v1.json",
    )
    args = parser.parse_args()
    records = [run_case(case) for case in build_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_reflected_complex_snell_residual_regression_not_mathematica_validation",
        "case_count": len(records),
        "notes": [
            "These cases verify reflected/backward anisotropic Snell roots with complex dielectric tensors.",
            "They are residual/regression benchmarks, not Mathematica-vs-Python validation.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} reflected Snell benchmark cases to {args.output}")


def build_cases() -> list[dict]:
    cases = []
    for orientation_idx in range(ORIENTATION_COUNT):
        orientation = orientation_matrix(orientation_idx)
        for angle_idx, theta_deg in enumerate(ANGLES_DEG):
            idx = len(cases)
            eps_crystal = complex_symmetric_epsilon(idx)
            cases.append(
                {
                    "id": f"reflected_snell_{idx + 1:02d}",
                    "theta_deg": theta_deg,
                    "orientation_index": orientation_idx,
                    "epsilon_lab": orientation.T @ eps_crystal @ orientation,
                }
            )
    return cases


def run_case(case: dict) -> dict:
    theta = np.deg2rad(case["theta_deg"])
    modes = solve_snell_modes(case["epsilon_lab"], incident_index=1.0, incident_theta_rad=theta, reflected=True)
    target = np.sin(theta)
    mode_records = {}
    residuals = []
    z_signs = []
    for name, mode in (("fast", modes.fast), ("slow", modes.slow)):
        snell_residual = mode.refractive_index * np.sin(mode.theta_rad) - target
        projector = np.eye(3, dtype=complex) - np.outer(mode.direction, mode.direction)
        eigen_residual = projector @ mode.electric_direction - mode.eta * (case["epsilon_lab"] @ mode.electric_direction)
        residuals.append(abs(snell_residual))
        residuals.append(np.linalg.norm(eigen_residual))
        z_signs.append(float(np.real(mode.direction[2])))
        mode_records[name] = {
            "theta_rad": complex_scalar_to_json(mode.theta_rad),
            "refractive_index": complex_scalar_to_json(mode.refractive_index),
            "direction": complex_vector_to_json(mode.direction),
            "electric_direction": complex_vector_to_json(mode.electric_direction),
            "snell_residual": complex_scalar_to_json(snell_residual),
            "eigen_residual_l2": round(float(np.linalg.norm(eigen_residual)), 15),
        }
    record = {
        "id": case["id"],
        "theta_deg": case["theta_deg"],
        "orientation_index": case["orientation_index"],
        "inputs": {
            "epsilon_lab": complex_array_to_json(case["epsilon_lab"]),
        },
        "outputs": mode_records,
        "metrics": {
            "max_residual": round(float(np.max(residuals)), 15),
            "max_real_direction_z": round(float(np.max(z_signs)), 15),
            "max_abs_epsilon_imag": round(float(np.max(np.abs(np.imag(case["epsilon_lab"])))), 15),
            "max_abs_epsilon_offdiag": round(float(max_abs_offdiag(case["epsilon_lab"])), 15),
        },
    }
    record["fingerprint"] = fingerprint_record(record)
    return record


def orientation_matrix(idx: int) -> np.ndarray:
    if idx == 0:
        return np.eye(3)
    rng = np.random.default_rng(20260901 + idx)
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs
    if np.linalg.det(q) < 0:
        q[0] *= -1
    return q


def complex_symmetric_epsilon(idx: int) -> np.ndarray:
    rng = np.random.default_rng(20260915 + idx)
    diag = np.array([2.2 + 0.04 * idx, 2.8 + 0.02 * idx, 3.6 + 0.015 * idx])
    imag = np.array([0.05, 0.08, 0.12]) + rng.uniform(0.0, 0.03, size=3)
    eps = np.diag(diag + 1j * imag)
    off = rng.normal(0.0, 0.025, size=(3, 3)) + 1j * rng.normal(0.0, 0.012, size=(3, 3))
    off = np.triu(off, 1)
    return eps + off + off.T


def complex_array_to_json(values: np.ndarray) -> list:
    arr = np.asarray(values, dtype=complex)
    return [[complex_scalar_to_json(item) for item in row] for row in arr]


def complex_vector_to_json(values: np.ndarray) -> list:
    return [complex_scalar_to_json(item) for item in np.asarray(values, dtype=complex)]


def complex_scalar_to_json(value: complex) -> dict:
    return {"real": round(float(np.real(value)), 15), "imag": round(float(np.imag(value)), 15)}


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
