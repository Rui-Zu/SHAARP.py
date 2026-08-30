from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from shaarp import (
    CrystalOrientation,
    CrystalStructure,
    Layer,
    Material,
    MultilayerSystem,
    Polarimetry,
    multilayer_shg,
    single_interface_intensity,
)


BENCHMARK_VERSION = 1
PHI_DEG = np.linspace(0.0, 360.0, 37)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate numeric SHAARP.py regression benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "numeric_benchmarks_v1.json",
    )
    args = parser.parse_args()

    cases = build_cases()
    records = [run_case(case) for case in cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_regression_baseline_not_mathematica_validation",
        "case_count": len(records),
        "phi_deg": PHI_DEG.tolist(),
        "notes": [
            "All cases use anisotropic dielectric tensors, complex permittivity, and complex SHG coefficients.",
            "These values are generated from the current Python implementation and are intended for regression checks.",
            "They are not a validation against Mathematica SHAARP until Mathematica reference outputs are added.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} benchmark cases to {args.output}")


def build_cases() -> list[dict]:
    cases = []
    for idx in range(20):
        material = make_material(idx)
        pol = Polarimetry(
            theta_deg=10.0 + (idx % 8) * 7.5,
            phi_deg=PHI_DEG,
            psi_deg=(idx * 13.0) % 180.0,
            ellipticity_deg=(idx * 17.0) % 360.0,
        )
        cases.append({"id": f"complex_aniso_{idx + 1:02d}", "material": material, "polarimetry": pol})
    return cases


def make_material(idx: int) -> Material:
    rng = np.random.default_rng(20260514 + idx)
    orientation = random_orientation(rng)
    eps_w = random_complex_symmetric_eps(rng, base=np.array([2.0, 2.4, 3.1]) + 0.03 * idx)
    eps_2w = random_complex_symmetric_eps(rng, base=np.array([2.2, 2.8, 3.6]) + 0.04 * idx)
    d_voigt = rng.normal(0.0, 8.0, size=(3, 6)) + 1j * rng.normal(0.0, 2.0, size=(3, 6))
    d_voigt += np.array(
        [
            [0.0, 0.5, -0.2, 1.0, 0.3, -0.7],
            [0.4, 0.0, 0.6, -0.5, 1.2, 0.1],
            [1.4, -0.8, 2.0, 0.2, -0.1, 0.9],
        ],
        dtype=complex,
    )
    return Material(
        name=f"Benchmark complex anisotropic material {idx + 1}",
        structure=CrystalStructure(
            point_group="1",
            a=4.0 + 0.01 * idx,
            b=4.3 + 0.02 * idx,
            c=5.1 + 0.015 * idx,
            alpha_deg=90.0,
            beta_deg=90.0,
            gamma_deg=90.0,
        ),
        orientation=CrystalOrientation(orientation),
        epsilon_omega=eps_w,
        epsilon_2omega=eps_2w,
        d_voigt_pm_v=d_voigt,
    )


def random_orientation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs
    if np.linalg.det(q) < 0:
        q[0] *= -1
    return q


def random_complex_symmetric_eps(rng: np.random.Generator, base: np.ndarray) -> np.ndarray:
    real_diag = base**2
    imag_diag = rng.uniform(0.01, 0.2, size=3)
    eps = np.diag(real_diag + 1j * imag_diag)
    off_real = rng.normal(0.0, 0.03, size=(3, 3))
    off_imag = rng.normal(0.0, 0.01, size=(3, 3))
    off = off_real + 1j * off_imag
    off = np.triu(off, 1)
    eps = eps + off + off.T
    return eps


def run_case(case: dict) -> dict:
    material = case["material"]
    pol = case["polarimetry"]
    si = single_interface_intensity(material, pol)
    system = MultilayerSystem(
        wavelength_um=1.064,
        polarimetry=pol,
        layers=[
            Layer("incident air", air_like(), shg_active=False),
            Layer("benchmark layer", material, thickness_um=0.25 + 0.05 * int(case["id"][-2:])),
            Layer("exit air", air_like(), shg_active=False),
        ],
    )
    ml = multilayer_shg(system)
    si_intensity = np.asarray(si.intensity, dtype=float)
    ml_r = np.asarray(ml.reflected_intensity, dtype=float)
    ml_t = np.asarray(ml.transmitted_intensity, dtype=float)
    record = {
        "id": case["id"],
        "material_summary": summarize_material(material),
        "polarimetry": {
            "theta_deg": float(pol.theta_deg),
            "psi_deg": float(pol.psi_deg),
            "ellipticity_deg": float(pol.ellipticity_deg),
        },
        "inputs": {
            "orientation": complex_array_to_json(material.orientation.rotation_matrix()),
            "epsilon_omega": complex_array_to_json(material.eps_w()),
            "epsilon_2omega": complex_array_to_json(material.eps_2w()),
            "d_voigt_pm_v": complex_array_to_json(material.d_voigt()),
            "phi_deg": PHI_DEG.round(12).tolist(),
        },
        "outputs": {
            "single_interface_intensity": si_intensity.round(12).tolist(),
            "multilayer_reflected_intensity": ml_r.round(12).tolist(),
            "multilayer_transmitted_intensity": ml_t.round(12).tolist(),
        },
        "single_interface": summarize_array(si_intensity),
        "multilayer_reflected": summarize_array(ml_r),
        "multilayer_transmitted": summarize_array(ml_t),
    }
    record["fingerprint"] = fingerprint_record(record)
    return record


def air_like() -> Material:
    return Material(
        name="Benchmark air",
        structure=CrystalStructure(point_group="1"),
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3),
        epsilon_2omega=np.eye(3),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )


def summarize_material(material: Material) -> dict:
    eps_w = material.eps_w()
    eps_2w = material.eps_2w()
    d = material.d_voigt()
    return {
        "name": material.name,
        "orientation_det": round(float(np.linalg.det(material.orientation.rotation_matrix())), 12),
        "epsilon_omega_diag_real": np.real(np.diag(eps_w)).round(12).tolist(),
        "epsilon_omega_diag_imag": np.imag(np.diag(eps_w)).round(12).tolist(),
        "epsilon_2omega_diag_real": np.real(np.diag(eps_2w)).round(12).tolist(),
        "epsilon_2omega_diag_imag": np.imag(np.diag(eps_2w)).round(12).tolist(),
        "max_abs_offdiag_epsilon_omega": round(float(max_abs_offdiag(eps_w)), 12),
        "max_abs_offdiag_epsilon_2omega": round(float(max_abs_offdiag(eps_2w)), 12),
        "max_abs_d_voigt_imag": round(float(np.max(np.abs(np.imag(d)))), 12),
    }


def summarize_array(values: np.ndarray) -> dict:
    rounded = np.asarray(values, dtype=float)
    return {
        "shape": list(rounded.shape),
        "min": round(float(np.min(rounded)), 12),
        "max": round(float(np.max(rounded)), 12),
        "mean": round(float(np.mean(rounded)), 12),
        "l2": round(float(np.linalg.norm(rounded)), 12),
        "first5": rounded[:5].round(12).tolist(),
        "last5": rounded[-5:].round(12).tolist(),
    }


def complex_array_to_json(values: np.ndarray) -> list:
    arr = np.asarray(values, dtype=complex)
    return [
        [
            {"real": round(float(np.real(item)), 12), "imag": round(float(np.imag(item)), 12)}
            for item in row
        ]
        for row in arr
    ]


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
