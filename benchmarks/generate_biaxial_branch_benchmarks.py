from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from shaarp.anisotropic import solve_snell_modes, track_mode_branches
from shaarp.config import CrystalOrientation, CrystalStructure


BENCHMARK_VERSION = 1
ANGLES_DEG = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0]
ORIENTATION_COUNT = 6


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate biaxial branch-continuity benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "biaxial_branch_benchmarks_v1.json",
    )
    args = parser.parse_args()
    records = [run_sequence(case) for case in build_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_biaxial_branch_tracking_regression_not_mathematica_validation",
        "sequence_count": len(records),
        "angle_count_per_sequence": len(ANGLES_DEG),
        "notes": [
            "These cases track biaxial/non-uniaxial eigenmode branches across incidence-angle sweeps.",
            "They deliberately use branch_1/branch_2 terminology, not ordinary/extraordinary labels.",
            "They are Python regression benchmarks, not Mathematica-vs-Python validation.",
        ],
        "sequences": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} biaxial branch benchmark sequences to {args.output}")


def build_cases() -> list[dict]:
    cases = []
    eps_crystal = complex_biaxial_epsilon()
    for orientation_idx in range(ORIENTATION_COUNT):
        orientation = orientation_matrix(orientation_idx)
        cases.append(
            {
                "id": f"biaxial_branch_orientation_{orientation_idx + 1:02d}",
                "orientation_index": orientation_idx,
                "orientation": orientation,
                "epsilon_lab": orientation.T @ eps_crystal @ orientation,
            }
        )
    return cases


def run_sequence(case: dict) -> dict:
    mode_pairs = [
        solve_snell_modes(case["epsilon_lab"], incident_index=1.0 + 0.01j, incident_theta_rad=np.deg2rad(theta_deg))
        for theta_deg in ANGLES_DEG
    ]
    tracked = track_mode_branches(mode_pairs)
    steps = []
    alignments = []
    ambiguous_count = 0
    residuals = []
    for theta_deg, modes, branches in zip(ANGLES_DEG, mode_pairs, tracked):
        residuals.extend(_residuals(case["epsilon_lab"], modes, theta_deg))
        alignments.extend([branches.branch_1_alignment, branches.branch_2_alignment])
        ambiguous_count += int(branches.ambiguous)
        steps.append(
            {
                "theta_deg": theta_deg,
                "fast_n": complex_scalar_to_json(modes.fast.refractive_index),
                "slow_n": complex_scalar_to_json(modes.slow.refractive_index),
                "branch_1_source_label": branches.branch_1_source_label,
                "branch_2_source_label": branches.branch_2_source_label,
                "branch_1_alignment": round(branches.branch_1_alignment, 15),
                "branch_2_alignment": round(branches.branch_2_alignment, 15),
                "ambiguous": branches.ambiguous,
            }
        )
    record = {
        "id": case["id"],
        "orientation_index": case["orientation_index"],
        "inputs": {
            "orientation": complex_array_to_json(case["orientation"]),
            "epsilon_lab": complex_array_to_json(case["epsilon_lab"]),
            "incident_index": complex_scalar_to_json(1.0 + 0.01j),
            "angles_deg": ANGLES_DEG,
        },
        "steps": steps,
        "metrics": {
            "min_branch_alignment": round(float(np.min(alignments)), 15),
            "ambiguous_count": ambiguous_count,
            "max_residual": round(float(np.max(residuals)), 15),
            "max_abs_epsilon_imag": round(float(np.max(np.abs(np.imag(case["epsilon_lab"])))), 15),
            "max_abs_epsilon_offdiag": round(float(max_abs_offdiag(case["epsilon_lab"])), 15),
        },
    }
    record["fingerprint"] = fingerprint_record(record)
    return record


def _residuals(epsilon_lab: np.ndarray, modes, theta_deg: float) -> list[float]:
    target = (1.0 + 0.01j) * np.sin(np.deg2rad(theta_deg))
    residuals = []
    for mode in (modes.fast, modes.slow):
        snell_residual = mode.refractive_index * np.sin(mode.theta_rad) - target
        projector = np.eye(3, dtype=complex) - np.outer(mode.direction, mode.direction)
        eigen_residual = projector @ mode.electric_direction - mode.eta * (epsilon_lab @ mode.electric_direction)
        residuals.append(abs(snell_residual))
        residuals.append(float(np.linalg.norm(eigen_residual)))
    return residuals


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
    rng = np.random.default_rng(20261001 + idx)
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


def complex_biaxial_epsilon() -> np.ndarray:
    eps = np.diag([2.18 + 0.05j, 2.83 + 0.08j, 3.67 + 0.11j])
    offdiag = np.array(
        [
            [0.0, 0.045 + 0.012j, -0.028 + 0.009j],
            [0.045 + 0.012j, 0.0, 0.034 - 0.015j],
            [-0.028 + 0.009j, 0.034 - 0.015j, 0.0],
        ],
        dtype=complex,
    )
    return eps + offdiag


def complex_array_to_json(values: np.ndarray) -> list:
    arr = np.asarray(values, dtype=complex)
    return [[complex_scalar_to_json(item) for item in row] for row in arr]


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
