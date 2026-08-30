from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from shaarp.config import CrystalOrientation, CrystalStructure
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_rank2_crystal_to_lab


BENCHMARK_VERSION = 1
ANGLES_DEG = [0.0, 8.0, 17.5, 29.0, 41.0, 55.0]
POLARIZATIONS = ["s", "p"]
ORIENTATION_COUNT = 6


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full-port solver-stage numeric benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "solver_stage_benchmarks_v1.json",
    )
    args = parser.parse_args()

    records = [run_case(case) for case in build_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_solver_stage_residual_regression_not_mathematica_validation",
        "case_count": len(records),
        "notes": [
            "These cases exercise the ported single-interface SHG boundary solver stage.",
            "They sweep incidence angle, crystal orientation, anisotropic dielectric axes, s/p polarization, and complex SHG coefficients.",
            "These solver-stage cases use complex anisotropic dielectric tensors and complex SHG coefficients.",
            "These are Python residual/regression benchmarks, not Mathematica-vs-Python validation.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} solver-stage benchmark cases to {args.output}")


def build_cases() -> list[dict]:
    cases = []
    orientations = [orientation_matrix(i) for i in range(ORIENTATION_COUNT)]
    for orientation_idx, orientation in enumerate(orientations):
        for angle_idx, theta_deg in enumerate(ANGLES_DEG):
            pol = POLARIZATIONS[(orientation_idx + angle_idx) % len(POLARIZATIONS)]
            idx = len(cases)
            eps_w_axes = np.array([1.45 + 0.02 * idx, 1.68 + 0.015 * orientation_idx, 2.05 + 0.01 * angle_idx]) ** 2
            eps_2w_axes = np.array([1.62 + 0.018 * idx, 1.82 + 0.012 * orientation_idx, 2.24 + 0.012 * angle_idx]) ** 2
            cases.append(
                {
                    "id": f"solver_stage_{idx + 1:02d}",
                    "theta_deg": theta_deg,
                    "incident_polarization": pol,
                    "orientation_index": orientation_idx,
                    "orientation": orientation,
                    "epsilon_omega_crystal": complex_symmetric_epsilon(idx, eps_w_axes, imag_base=0.025),
                    "epsilon_2omega_crystal": complex_symmetric_epsilon(idx + 100, eps_2w_axes, imag_base=0.035),
                    "d_voigt_lab": complex_d_voigt(idx),
                }
            )
    return cases


def orientation_matrix(idx: int) -> np.ndarray:
    if idx == 0:
        return np.eye(3)
    if idx == 4:
        return CrystalOrientation.from_cubic_miller_surface((1, 2, 3), in_plane_hint=(1.0, -1.0, 0.5)).rotation_matrix()
    if idx == ORIENTATION_COUNT - 1:
        return CrystalOrientation.from_miller_surface(
            noncubic_extreme_structure(),
            (1, 2, 3),
            in_plane_direction_uvw=(2.0, -1.0, 1.0),
        ).rotation_matrix()
    rng = np.random.default_rng(20260515 + idx)
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs
    if np.linalg.det(q) < 0:
        q[0] *= -1
    return q


def noncubic_extreme_structure() -> CrystalStructure:
    return CrystalStructure(a=4.1, b=5.2, c=6.3, alpha_deg=76.0, beta_deg=88.0, gamma_deg=103.0)


def complex_d_voigt(idx: int) -> np.ndarray:
    rng = np.random.default_rng(20260601 + idx)
    base = rng.normal(0.0, 0.7, size=(3, 6)) + 1j * rng.normal(0.0, 0.25, size=(3, 6))
    base += np.array(
        [
            [0.2, -0.1j, 0.3, 0.4j, -0.2, 0.1],
            [0.05j, 0.25, -0.2, 0.15, 0.35j, -0.25],
            [0.1, 0.2j, 0.45, -0.3, 0.15j, 0.5],
        ],
        dtype=complex,
    )
    return base


def complex_symmetric_epsilon(idx: int, diagonal: np.ndarray, *, imag_base: float) -> np.ndarray:
    rng = np.random.default_rng(20260620 + idx)
    imag_diag = imag_base + rng.uniform(0.0, 0.035, size=3)
    eps = np.diag(diagonal + 1j * imag_diag)
    off = rng.normal(0.0, 0.012, size=(3, 3)) + 1j * rng.normal(0.0, 0.006, size=(3, 3))
    off = np.triu(off, 1)
    return eps + off + off.T


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
    kx_values = np.array(
        [
            result.inhomogeneous.ee.k[0],
            result.inhomogeneous.oo.k[0],
            result.inhomogeneous.eo.k[0],
            result.reflected_s_2omega.k[0],
            result.transmitted_fast_2omega.k[0],
            result.transmitted_slow_2omega.k[0],
        ],
        dtype=complex,
    )
    record = {
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
        "outputs": {
            "coefficients": complex_vector_to_json(result.coefficients),
            "reflected_total_2omega_electric": complex_vector_to_json(result.reflected_total_2omega.electric),
            "transmitted_total_2omega_electric": complex_vector_to_json(result.transmitted_total_2omega),
            "boundary_residual": complex_vector_to_json(result.boundary_residual),
        },
        "metrics": {
            "boundary_residual_l2": round(float(np.linalg.norm(result.boundary_residual)), 15),
            "inhomogeneous_ee_residual_l2": round(float(np.linalg.norm(result.inhomogeneous.ee.residual)), 15),
            "inhomogeneous_oo_residual_l2": round(float(np.linalg.norm(result.inhomogeneous.oo.residual)), 15),
            "inhomogeneous_eo_residual_l2": round(float(np.linalg.norm(result.inhomogeneous.eo.residual)), 15),
            "tangential_kx_spread": round(float(np.max(np.abs(kx_values - kx_values[0]))), 15),
            "coefficient_l2": round(float(np.linalg.norm(result.coefficients)), 15),
            "orientation_det": round(float(np.linalg.det(case["orientation"])), 15),
            "max_abs_d_imag": round(float(np.max(np.abs(np.imag(case["d_voigt_lab"])))), 15),
            "max_abs_epsilon_omega_imag": round(float(np.max(np.abs(np.imag(epsilon_omega_lab)))), 15),
            "max_abs_epsilon_2omega_imag": round(float(np.max(np.abs(np.imag(epsilon_2omega_lab)))), 15),
            "max_abs_offdiag_epsilon_omega_lab": round(float(max_abs_offdiag(epsilon_omega_lab)), 15),
        },
    }
    record["fingerprint"] = fingerprint_record(record)
    return record


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
