from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shaarp.config import CrystalOrientation, CrystalStructure, Layer, Material, MultilayerSystem, Polarimetry
from shaarp.multilayer_shg_boundary import solve_multilayer_shg_from_system


BENCHMARK_VERSION = 1
ANGLES_DEG = [0.0, 8.0, 18.0, 33.0, 52.0]
ORIENTATION_COUNT = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate staged multilayer-system SHG benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "multilayer_system_benchmarks_v1.json",
    )
    args = parser.parse_args()

    records = [run_case(case) for case in build_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_staged_multilayer_system_regression_not_mathematica_validation",
        "case_count": len(records),
        "notes": [
            "These cases exercise solve_multilayer_shg_from_system, the staged SHAARP.ml-style workflow.",
            "They cover multiple incidence angles, s/p incident polarization, nontrivial orientations, complex non-diagonal tensors, and ten nonlinear source terms per active layer.",
            "These are Python residual/regression benchmarks, not Mathematica-vs-Python validation.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} multilayer-system benchmark cases to {args.output}")


def build_cases() -> list[dict]:
    cases = []
    for orientation_index in range(ORIENTATION_COUNT):
        for angle_index, theta_deg in enumerate(ANGLES_DEG):
            idx = len(cases)
            cases.append(
                {
                    "id": f"multilayer_system_{idx + 1:02d}",
                    "theta_deg": theta_deg,
                    "incident_polarization": "s" if (idx % 2 == 0) else "p",
                    "orientation_index": orientation_index,
                    "system": build_system(idx, theta_deg, orientation_index),
                }
            )
    return cases


def build_system(idx: int, theta_deg: float, orientation_index: int) -> MultilayerSystem:
    structure = CrystalStructure(point_group="1")
    orientation = orientation_for_index(orientation_index)
    top = Material(
        name="top-air",
        structure=structure,
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3),
        epsilon_2omega=np.eye(3) * (1.03 + 0.004j) ** 2,
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    layer = Material(
        name="active-film",
        structure=structure,
        orientation=orientation,
        epsilon_omega=complex_symmetric_epsilon(idx, [1.55**2, 1.74**2, 2.02**2], imag_base=0.012),
        epsilon_2omega=complex_symmetric_epsilon(idx + 100, [2.55, 3.05, 3.65], imag_base=0.045),
        d_voigt_pm_v=complex_d_voigt(idx),
    )
    substrate = Material(
        name="substrate",
        structure=structure,
        orientation=orientation_for_index((orientation_index + 1) % ORIENTATION_COUNT),
        epsilon_omega=complex_symmetric_epsilon(idx + 200, [1.34**2, 1.44**2, 1.56**2], imag_base=0.01),
        epsilon_2omega=complex_symmetric_epsilon(idx + 300, [1.85, 2.15, 2.45], imag_base=0.025),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    return MultilayerSystem(
        wavelength_um=1.064,
        polarimetry=Polarimetry(theta_deg=theta_deg),
        layers=[
            Layer("top", top, shg_active=False),
            Layer("film", layer, thickness_um=0.16 + 0.008 * (idx % 5), shg_active=True),
            Layer("substrate", substrate, shg_active=False),
        ],
    )


def orientation_for_index(index: int) -> CrystalOrientation:
    if index == 0:
        return CrystalOrientation()
    if index == 1:
        return CrystalOrientation.from_cubic_miller_surface((1, 2, 3), in_plane_hint=(1.0, -1.0, 0.5))
    if index == 2:
        return CrystalOrientation.from_miller_surface(
            CrystalStructure(a=4.1, b=5.2, c=6.3, alpha_deg=76.0, beta_deg=88.0, gamma_deg=103.0),
            (1, 2, 3),
            in_plane_direction_uvw=(2.0, -1.0, 1.0),
        )
    angle = np.deg2rad(27.0)
    return CrystalOrientation(
        np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
    )


def complex_symmetric_epsilon(idx: int, diagonal: list[float], *, imag_base: float) -> np.ndarray:
    rng = np.random.default_rng(20260640 + idx)
    eps = np.diag(np.asarray(diagonal, dtype=float) + 1j * (imag_base + rng.uniform(0.0, 0.02, size=3)))
    off = rng.normal(0.0, 0.01, size=(3, 3)) + 1j * rng.normal(0.0, 0.004, size=(3, 3))
    off = np.triu(off, 1)
    return eps + off + off.T


def complex_d_voigt(idx: int) -> np.ndarray:
    rng = np.random.default_rng(20260680 + idx)
    return rng.normal(0.0, 0.45, size=(3, 6)) + 1j * rng.normal(0.0, 0.2, size=(3, 6))


def run_case(case: dict) -> dict:
    result = solve_multilayer_shg_from_system(
        case["system"],
        incident_polarization=case["incident_polarization"],
        mu=1.0,
        eps0=1.0,
    )
    all_fields = [field for layer in result.layer_inhomogeneous_fields for field in layer.as_list()]
    record = {
        "id": case["id"],
        "theta_deg": case["theta_deg"],
        "incident_polarization": case["incident_polarization"],
        "orientation_index": case["orientation_index"],
        "inputs": {
            "wavelength_um": case["system"].wavelength_um,
            "thickness_um": [layer.thickness_um for layer in case["system"].layers[1:-1]],
            "orientation": complex_array_to_json(case["system"].layers[1].material.orientation.rotation_matrix()),
            "epsilon_omega_layer_crystal": complex_array_to_json(case["system"].layers[1].material.eps_w()),
            "epsilon_2omega_layer_crystal": complex_array_to_json(case["system"].layers[1].material.eps_2w()),
            "d_voigt_layer_crystal": complex_array_to_json(case["system"].layers[1].material.d_voigt()),
        },
        "outputs": {
            "fundamental_coefficients": complex_vector_to_json(result.fundamental.coefficients),
            "shg_coefficients": complex_vector_to_json(result.shg.coefficients),
            "first_layer_source_wavevectors": complex_array_to_json(np.array([source.k for source in result.layer_sources[0].as_list()])),
            "first_layer_inhomogeneous_electric": complex_array_to_json(np.array([field.electric for field in result.layer_inhomogeneous_fields[0].as_list()])),
        },
        "metrics": {
            "fundamental_residual_l2": round(float(np.linalg.norm(result.fundamental.residual)), 15),
            "shg_residual_l2": round(float(np.linalg.norm(result.shg.residual)), 15),
            "max_inhomogeneous_residual_l2": round(float(max(np.linalg.norm(field.residual) for field in all_fields)), 15),
            "source_count": len(result.layer_sources[0].as_list()),
            "max_abs_epsilon_omega_imag": round(float(np.max(np.abs(np.imag(case["system"].layers[1].material.eps_w())))), 15),
            "max_abs_epsilon_2omega_imag": round(float(np.max(np.abs(np.imag(case["system"].layers[1].material.eps_2w())))), 15),
            "max_abs_epsilon_omega_offdiag": round(float(max_abs_offdiag(case["system"].layers[1].material.eps_w())), 15),
            "max_abs_d_imag": round(float(np.max(np.abs(np.imag(case["system"].layers[1].material.d_voigt())))), 15),
        },
    }
    record["fingerprint"] = fingerprint_record(record)
    return record


def complex_array_to_json(values: np.ndarray) -> list:
    arr = np.asarray(values, dtype=complex)
    if arr.ndim == 1:
        return complex_vector_to_json(arr)
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
